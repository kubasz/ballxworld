use crate::world::ecs::{CLoadAnchor, CLocation, Component, ECSHandler, ECS};
use crate::world::registry::VoxelRegistry;
use crate::world::{ChunkPosition, VoxelChunk, VoxelChunkRef, VOXEL_CHUNK_DIM};
use cgmath::prelude::*;
use cgmath::{vec3, Vector3};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, RwLockWriteGuard};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use thread_local::CachedThreadLocal;

pub trait WorldGenerator {
    fn generate_chunk(&self, cref: VoxelChunkRef, registry: &VoxelRegistry);
}

type ClientWorld = crate::client::world::ClientWorld;
type ChunkMsg = (ChunkPosition, Arc<RwLock<VoxelChunk>>);

pub struct World {
    pub name: String,
    pub loaded_chunks: HashMap<ChunkPosition, Arc<RwLock<VoxelChunk>>>,
    pub loading_queue: Mutex<Vec<(i32, ChunkPosition)>>,
    pub worldgen: Option<Arc<dyn WorldGenerator + Send + Sync>>,
    pub registry: Arc<VoxelRegistry>,
    pub entities: RwLock<ECS>,
    pub client_world: Option<ClientWorld>,
    requesting_write: Arc<AtomicBool>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    work_receiver: CachedThreadLocal<mpsc::Receiver<ChunkMsg>>,
}

impl Debug for World {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "World {{ name: {}, loaded_chunks: {:?}, loading_queue: {:?} }}",
            &self.name,
            self.loaded_chunks.keys(),
            self.loading_queue
        )
    }
}

impl World {
    pub fn new(name: String, registry: Arc<VoxelRegistry>) -> World {
        Self {
            name,
            loaded_chunks: HashMap::new(),
            loading_queue: Mutex::new(Vec::new()),
            worldgen: None,
            registry,
            entities: RwLock::new(ECS::new()),
            client_world: None,
            requesting_write: Arc::new(AtomicBool::new(false)),
            worker_threads: Vec::new(),
            work_receiver: CachedThreadLocal::default(),
        }
    }

    pub fn request_write(w: &Arc<RwLock<Self>>) -> RwLockWriteGuard<Self> {
        let rd = w.read().unwrap();
        rd.requesting_write.store(true, Ordering::SeqCst);
        drop(rd);
        // now we wait...
        let wr = w.write().unwrap();
        wr.requesting_write.store(false, Ordering::SeqCst);
        wr
    }

    pub fn get_write_request(&self) -> Arc<AtomicBool> {
        self.requesting_write.clone()
    }

    pub fn init_worker_threads(w: &Arc<RwLock<Self>>) {
        let wref = w.clone();
        const NUM_WORKERS: usize = 2;
        const STACK_SIZE: usize = 4 * 1024 * 1024;
        let mut w = w.write().unwrap();
        if !w.worker_threads.is_empty() {
            return;
        }
        w.worker_threads.reserve_exact(NUM_WORKERS);
        let (tx, rx) = mpsc::channel();
        w.work_receiver.clear();
        w.work_receiver.get_or(move || Box::new(rx));
        for _ in 0..NUM_WORKERS {
            let tb = thread::Builder::new()
                .name("bxw-worldgen".to_owned())
                .stack_size(STACK_SIZE);
            let tworld = wref.clone();
            let ttx = tx.clone();
            let thr = tb
                .spawn(move || Self::worldgen_worker(tworld, ttx))
                .expect("Could not create worldgen worker thread");
            w.worker_threads.push(thr);
        }
    }

    pub fn change_generator(&mut self, new_generator: Arc<dyn WorldGenerator + Send + Sync>) {
        self.worldgen = Some(new_generator);
    }

    fn worldgen_worker(world_arc: Arc<RwLock<World>>, submission: mpsc::Sender<ChunkMsg>) {
        let mut wr_rq = None;
        loop {
            let world = world_arc.read().unwrap();
            if wr_rq.is_none() {
                wr_rq = Some(world.get_write_request());
            }
            let mut load_queue = world.loading_queue.lock().unwrap();

            if load_queue.is_empty() {
                drop(load_queue);
                drop(world);
                thread::park();
                continue;
            }

            let mut pos_to_load = Vec::new();
            let len = load_queue.len().min(10);
            for p in load_queue.iter().rev().take(len) {
                pos_to_load.push(p.1);
            }
            let newlen = load_queue.len() - len;
            load_queue.resize(newlen, (0, vec3(0, 0, 0)));
            drop(load_queue);

            let mut world = Some(world);
            for p in pos_to_load.into_iter() {
                if world.is_none() {
                    world = Some(world_arc.read().unwrap());
                }
                let worldgen = world.as_ref().unwrap().worldgen.as_ref().unwrap();
                let chunk = Arc::new(RwLock::new(VoxelChunk::new()));
                let cref = VoxelChunkRef {
                    chunk: Arc::downgrade(&chunk),
                    position: p,
                };
                worldgen.generate_chunk(cref, &world.as_ref().unwrap().registry);
                chunk.write().unwrap().dirty += 1;
                submission.send((p, chunk)).unwrap();
                while wr_rq.as_ref().unwrap().load(Ordering::SeqCst) {
                    world = None;
                    thread::yield_now();
                }
            }
        }
    }

    pub fn physics_tick(&mut self) {
        // Chunk loading
        if self.worldgen.is_some() {
            if let Some(rx) = self.work_receiver.get() {
                for vc in rx.try_iter() {
                    self.loaded_chunks.insert(vc.0, vc.1);
                }
            }
            let mut req_positions: HashSet<(i32, ChunkPosition)> = HashSet::new();

            let ents = self.entities.get_mut().unwrap();
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
                    .div_element_wise(VOXEL_CHUNK_DIM as f32)
                    .map(|c| c as i32);
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

            let mut load_queue = self.loading_queue.lock().unwrap();
            load_queue.clear();

            let mut pos_to_load: Vec<(i32, ChunkPosition)> = Vec::new();
            req_positions
                .iter()
                .filter(|p| !self.loaded_chunks.contains_key(&p.1) && !load_queue.contains(p))
                .for_each(|p| {
                    pos_to_load.push(*p);
                });

            let mut pos_to_unload: HashSet<ChunkPosition> = HashSet::new();
            self.loaded_chunks
                .keys()
                .filter(|p| !req_positions.iter().any(|u| u.1 == **p))
                .for_each(|p| {
                    pos_to_unload.insert(*p);
                });

            for p in pos_to_load.into_iter() {
                load_queue.push(p);
            }
            // load nearest chunks first
            load_queue.sort_by_key(|p| -p.0);

            for p in pos_to_unload.into_iter() {
                self.loaded_chunks.remove(&p);
            }

            let has_loads = !load_queue.is_empty();
            drop(load_queue);
            if has_loads {
                for t in self.worker_threads.iter() {
                    t.thread().unpark();
                }
            }
        }
    }
}
