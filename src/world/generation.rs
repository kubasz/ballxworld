use crate::world::ecs::{CLoadAnchor, CLocation, Component, ECSHandler};
use crate::world::stdgen::StdGenerator;
use crate::world::{ChunkPosition, UncompressedChunk, VChunk, World, CHUNK_DIM};
use cgmath::prelude::*;
use cgmath::{vec3, Vector3};
use parking_lot::Mutex;
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::iter::FromIterator;
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
    pub progress_set: Arc<Mutex<HashSet<ChunkPosition>>>,
    pub generator: Arc<StdGenerator>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    work_receiver: CachedThreadLocal<mpsc::Receiver<ChunkMsg>>,
}

impl WorldLoadGen {
    pub fn new(world: Arc<World>, seed: u64) -> Self {
        let mut wlg = Self {
            world,
            loading_queue: Arc::new(Mutex::new(Vec::new())),
            progress_set: Arc::new(Mutex::new(HashSet::new())),
            generator: Arc::new(StdGenerator::new(seed)),
            worker_threads: Vec::new(),
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
        self.work_receiver.get_or(move || Box::new(rx));
        for _ in 0..NUM_WORKERS {
            let tb = thread::Builder::new()
                .name("bxw-worldgen".to_owned())
                .stack_size(STACK_SIZE);
            let tworld = self.world.clone();
            let ttx = tx.clone();
            let tgen = self.generator.clone();
            let tlq = self.loading_queue.clone();
            let tps = self.progress_set.clone();
            let thr = tb
                .spawn(move || Self::worldgen_worker(tworld, tgen, tlq, tps, ttx))
                .expect("Could not create worldgen worker thread");
            self.worker_threads.push(thr);
        }
    }

    fn worldgen_worker(
        world_arc: Arc<World>,
        worldgen_arc: Arc<StdGenerator>,
        load_queue_arc: Arc<Mutex<Vec<(i32, ChunkPosition)>>>,
        progress_set_arc: Arc<Mutex<HashSet<ChunkPosition>>>,
        submission: mpsc::Sender<ChunkMsg>,
    ) {
        let registry_arc;
        {
            let world = world_arc.voxels.read();
            registry_arc = world.registry.clone();
        }
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
                worldgen_arc.generate_chunk(&mut ucchunk, &registry_arc);
                chunk.compress(&ucchunk);
                submission.send(ChunkMsg { chunk }).unwrap();
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
        let mut req_positions: HashSet<(i32, ChunkPosition)> = HashSet::new();

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
            let pos: Vector3<i32> = loc
                .position
                .div_element_wise(CHUNK_DIM as f32)
                .map(|c| c as i32);
            for xoff in 0..r {
                for yoff in 0..r {
                    for zoff in 0..r {
                        let rr = xoff * xoff + yoff * yoff * 4 + zoff * zoff;
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

        let to_remove: SmallVec<[ChunkPosition; 10]> = SmallVec::from_iter(
            voxels
                .chunks
                .keys()
                .filter(|p| !req_positions.iter().any(|u| u.1 == **p))
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
