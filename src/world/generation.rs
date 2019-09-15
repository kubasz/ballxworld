use crate::math::*;
use crate::world::ecs::{CLoadAnchor, CLocation, Component, ECSHandler};
use crate::world::stdgen::StdGenerator;
use crate::world::{chunkpos_from_blockpos, ChunkPosition, UncompressedChunk, VChunk, World};
use fnv::FnvHashSet;
use parking_lot::Mutex;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::iter::FromIterator;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use thread_local::CachedThreadLocal;

struct ChunkMsg {
    chunk: VChunk,
}

pub struct WorldLoadGen {
    pub world: Arc<World>,
    pub loading_queue: Arc<Mutex<Vec<(i32, ChunkPosition)>>>,
    pub progress_set: Arc<Mutex<FnvHashSet<ChunkPosition>>>,
    pub generator: Arc<StdGenerator>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    thread_killer: Arc<AtomicBool>,
    work_receiver: CachedThreadLocal<mpsc::Receiver<ChunkMsg>>,
}

impl WorldLoadGen {
    pub fn new(world: Arc<World>, seed: u64) -> Self {
        let mut wlg = Self {
            world,
            loading_queue: Arc::new(Mutex::new(Vec::new())),
            progress_set: Arc::new(Mutex::new(FnvHashSet::default())),
            generator: Arc::new(StdGenerator::new(seed)),
            worker_threads: Vec::new(),
            thread_killer: Arc::new(AtomicBool::new(false)),
            work_receiver: CachedThreadLocal::new(),
        };
        wlg.init_worker_threads();
        wlg
    }

    fn init_worker_threads(&mut self) {
        const NUM_WORKERS: usize = 2;
        const STACK_SIZE: usize = 4 * 1024 * 1024;
        if !self.worker_threads.is_empty() {
            return;
        }
        self.worker_threads.reserve_exact(NUM_WORKERS);
        let (tx, rx) = mpsc::channel();
        self.work_receiver.clear();
        self.work_receiver.get_or(move || rx);
        for _ in 0..NUM_WORKERS {
            let tb = thread::Builder::new()
                .name("bxw-worldgen".to_owned())
                .stack_size(STACK_SIZE);
            let tworld = self.world.clone();
            let ttx = tx.clone();
            let tgen = self.generator.clone();
            let tlq = self.loading_queue.clone();
            let tps = self.progress_set.clone();
            let tks = self.thread_killer.clone();
            let thr = tb
                .spawn(move || Self::worldgen_worker(tworld, tgen, tlq, tps, ttx, tks))
                .expect("Could not create worldgen worker thread");
            self.worker_threads.push(thr);
        }
    }

    fn kill_threads(&mut self) {
        self.thread_killer.store(true, Ordering::SeqCst);
        let old_threads = std::mem::replace(&mut self.worker_threads, Vec::new());
        for t in old_threads.into_iter() {
            t.thread().unpark();
            drop(t.join());
        }
        self.thread_killer.store(false, Ordering::SeqCst);
    }

    fn worldgen_worker(
        world_arc: Arc<World>,
        worldgen_arc: Arc<StdGenerator>,
        load_queue_arc: Arc<Mutex<Vec<(i32, ChunkPosition)>>>,
        progress_set_arc: Arc<Mutex<FnvHashSet<ChunkPosition>>>,
        submission: mpsc::Sender<ChunkMsg>,
        killswitch: Arc<AtomicBool>,
    ) {
        let mut done_chunks: Vec<Vector3<i32>> = Vec::new();
        loop {
            let mut load_queue = load_queue_arc.lock();
            let mut progress_set = progress_set_arc.lock();

            for p in done_chunks.iter() {
                progress_set.remove(p);
            }
            done_chunks.clear();

            if load_queue.is_empty() {
                drop(load_queue);
                drop(progress_set);
                thread::park();
                if killswitch.load(Ordering::SeqCst) {
                    return;
                }
                continue;
            }

            let mut pos_to_load = Vec::new();
            let len = load_queue.len().min(10);
            for p in load_queue.iter().rev().take(len) {
                pos_to_load.push(p.1);
                progress_set.insert(p.1);
            }
            let newlen = load_queue.len() - len;
            load_queue.resize(newlen, (0, vec3(0, 0, 0)));
            drop(progress_set);
            drop(load_queue);

            for p in pos_to_load.into_iter() {
                let mut chunk = VChunk::new();
                chunk.position = p;
                let mut ucchunk = UncompressedChunk::new();
                ucchunk.position = p;
                worldgen_arc.generate_chunk(&mut ucchunk, &world_arc.vregistry);
                chunk.compress(&ucchunk);
                if submission.send(ChunkMsg { chunk }).is_err() {
                    return;
                }
                done_chunks.push(p);
            }
        }
    }

    pub fn load_tick(&mut self) {
        let mut voxels = self.world.voxels.write();
        // Chunk loading
        if let Some(rx) = self.work_receiver.get() {
            for vc in rx.try_iter() {
                let cpos = vc.chunk.position;
                voxels.chunks.insert(cpos, vc.chunk);
            }
        }
        let mut req_positions: FnvHashSet<(i32, ChunkPosition)> = FnvHashSet::default();

        let ents_o = self.world.entities.read();
        let ents = &ents_o.ecs;
        let it = ECSHandler::<CLoadAnchor>::iter(ents);
        for anchor in it {
            let loc: Option<&CLocation> = ents.get_component(anchor.entity_id());
            if loc.is_none() {
                continue;
            }
            let loc = loc.unwrap();
            let r = anchor.radius as i32;
            let pos: Vector3<i32> = chunkpos_from_blockpos(loc.position.map(|c| c as i32));
            for xoff in 0..r {
                for yoff in 0..r {
                    for zoff in 0..r {
                        let rr = xoff * xoff + yoff * yoff + zoff * zoff;
                        if rr <= r * r {
                            req_positions.insert((rr, pos + vec3(xoff, yoff, zoff)));

                            req_positions.insert((rr, pos + vec3(-xoff, yoff, zoff)));
                            req_positions.insert((rr, pos + vec3(xoff, -yoff, zoff)));
                            req_positions.insert((rr, pos + vec3(xoff, yoff, -zoff)));

                            req_positions.insert((rr, pos + vec3(-xoff, -yoff, zoff)));
                            req_positions.insert((rr, pos + vec3(xoff, -yoff, -zoff)));
                            req_positions.insert((rr, pos + vec3(-xoff, yoff, -zoff)));

                            req_positions.insert((rr, pos + vec3(-xoff, -yoff, -zoff)));
                        }
                    }
                }
            }
        }

        let mut load_queue = self.loading_queue.lock();
        let progress_set = self.progress_set.lock();
        load_queue.clear();

        req_positions
            .iter()
            .filter(|p| !(voxels.chunks.contains_key(&p.1) || progress_set.contains(&p.1)))
            .copied()
            .for_each(|p| load_queue.push(p));

        // load nearest chunks first
        load_queue.par_sort_by_key(|p| -p.0);

        let req_cpos: FnvHashSet<ChunkPosition> =
            FnvHashSet::from_iter(req_positions.iter().map(|c| c.1));
        let to_remove: SmallVec<[ChunkPosition; 10]> = SmallVec::from_iter(
            voxels
                .chunks
                .keys()
                .filter(|p| !req_cpos.contains(*p))
                .copied(),
        );
        for cp in to_remove {
            voxels.chunks.remove(&cp);
        }

        let has_loads = !load_queue.is_empty();
        drop(progress_set);
        drop(load_queue);
        if has_loads {
            for t in self.worker_threads.iter() {
                t.thread().unpark();
            }
        }
    }
}

impl Drop for WorldLoadGen {
    fn drop(&mut self) {
        self.kill_threads();
    }
}
