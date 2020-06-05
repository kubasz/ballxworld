use crate::ecs::*;
use crate::*;
use bxw_util::fnv::*;
use bxw_util::math::*;
use bxw_util::smallvec::*;
use bxw_util::taskpool::{Task, TaskPool};
use bxw_util::*;
use parking_lot::*;
use std::any::*;
use std::marker::PhantomData;
use std::sync::atomic::*;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ManagerLocation<'w>(usize, PhantomData<&'w World>);

impl<'w> ManagerLocation<'w> {
    fn new(manager: &'w World, index: usize) -> Self {
        Self(index, PhantomData::default())
    }

    fn get(self) -> usize {
        self.0
    }
}

#[repr(u8)]
pub enum ChunkDataState {
    Unloaded = 0,
    Waiting,
    Loaded,
    Updating,
}

pub const CHUNK_BLOCK_DATA: usize = 0;
pub const CHUNK_LIGHT_DATA: usize = 1;
pub const CHUNK_MESH_DATA: usize = 2;

pub type ChunkBlockData = Option<Arc<VChunk>>;
pub type ChunkLightData = Option<Arc<()>>;

pub type AnyChunkData = Option<Arc<dyn Any + Send + Sync>>;

pub trait ChunkDataHandler {
    fn get_data(&self, world: &World, location: ManagerLocation) -> AnyChunkData;
    /// Returns the old data
    fn swap_data(
        &mut self,
        world: &World,
        location: ManagerLocation,
        new_data: AnyChunkData,
    ) -> AnyChunkData;
    /// Extend with unloaded states or shrink unloaded states (it will never be called on non-Unloaded chunks)
    fn resize_data(&mut self, world: &World, new_size: usize);
    fn create_chunk_update_task(
        &mut self,
        world: &World,
        cpos: ChunkPosition,
        location: ManagerLocation,
        status_field: Arc<AtomicU8>,
    ) -> Task;
}

pub struct ChunkData {
    kind: usize,
    /// (ChunkDataKind, requires 26 populated neighbors)
    depends_on: Option<(usize, bool)>,
    status_array: Vec<Arc<AtomicU8>>,
    handler: Box<dyn ChunkDataHandler>,
}

pub struct World {
    allocation: FnvHashMap<ChunkPosition, usize>,
    chunk_positions: Vec<Option<ChunkPosition>>,
    free_indices: Vec<usize>,
    data: Vec<Arc<Mutex<ChunkData>>>,
    entities: Arc<RwLock<ECS>>,
    load_data: Arc<Mutex<LoadData>>,
    load_data_busy: Arc<AtomicBool>,
    remaining_deltas: Vec<ChunkDelta>,
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct LoadAnchor {
    eid: u64,
    cpos: ChunkPosition,
    radius: u32,
}

impl Ord for LoadAnchor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&self.eid, &other.eid)
    }
}

impl PartialOrd for LoadAnchor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.eid, &other.eid)
    }
}

#[derive(Clone)]
enum ChunkDelta {
    FullUnload(usize),
    PartialUnload {
        id: usize,
        handlers: SmallVec<[usize; 4]>,
    },
    Load {
        cpos: ChunkPosition,
        handlers: SmallVec<[usize; 4]>,
    },
}

struct LoadData {
    busy: Arc<AtomicBool>,
    status_arrays: Vec<Vec<u8>>,
    anchors: Vec<LoadAnchor>,
    remaining_deltas: Vec<ChunkDelta>,
}

impl Default for World {
    fn default() -> Self {
        let default_capacity = 64;
        let busy_arc = Arc::new(AtomicBool::new(false));
        Self {
            allocation: FnvHashMap::with_capacity_and_hasher(default_capacity, Default::default()),
            chunk_positions: Vec::with_capacity(default_capacity),
            free_indices: Vec::with_capacity(default_capacity),
            data: Vec::with_capacity(8),
            entities: Arc::new(Default::default()),
            load_data: Arc::new(Mutex::new(LoadData {
                busy: busy_arc.clone(),
                status_arrays: Vec::new(),
                anchors: Default::default(),
                remaining_deltas: Default::default(),
            })),
            load_data_busy: busy_arc,
            remaining_deltas: Vec::new(),
        }
    }
}

impl World {
    pub fn new() -> Self {
        Default::default()
    }

    fn extend_allocations(&mut self) {
        let ocap = self.chunk_positions.len();
        let ncap = (ocap + 1) * 2;
        if self.allocation.capacity() < ncap {
            self.allocation.reserve(ncap - self.allocation.capacity());
        }
        self.chunk_positions.resize(ncap, None);
        if self.free_indices.capacity() < ncap {
            self.free_indices.reserve(ncap);
        }
        for idx in ocap..ncap {
            self.free_indices.push(idx);
        }
        for d in self.data.iter() {
            let mut d = d.lock();
            d.status_array
                .resize_with(ncap, || Arc::new(AtomicU8::new(0)));
            d.handler.resize_data(self, ncap);
        }
    }

    fn get_or_allocate_chunk(&mut self, cpos: ChunkPosition) -> usize {
        if let Some(&idx) = self.allocation.get(&cpos) {
            idx
        } else {
            if self.free_indices.is_empty() {
                self.extend_allocations();
            }
            let fidx = self.free_indices.pop().unwrap();
            self.chunk_positions[fidx] = Some(cpos);
            self.allocation.insert(cpos, fidx);
            fidx
        }
    }

    pub fn main_loop_tick(&mut self, task_pool: &TaskPool) {
        self.check_load_deltas(task_pool);
    }

    fn check_load_deltas(&mut self, task_pool: &TaskPool) {
        if !self.load_data_busy.load(Ordering::Acquire) {
            let mut load_data = self.load_data.lock();
            if !load_data.remaining_deltas.is_empty() {
                std::mem::swap(&mut self.remaining_deltas, &mut load_data.remaining_deltas);
                load_data.remaining_deltas.clear();
            }
            let ecs = self.entities.read();
            let mut new_anchors = Vec::with_capacity(load_data.anchors.len());
            let it = ECSHandler::<CLoadAnchor>::iter(&*ecs);
            for anchor in it {
                let loc: Option<&CLocation> = ecs.get_component(anchor.entity_id());
                if loc.is_none() {
                    continue;
                }
                let loc = loc.unwrap();
                new_anchors.push(LoadAnchor {
                    eid: anchor.entity_id().u64(),
                    cpos: chunkpos_from_blockpos(blockpos_from_worldpos(loc.position)),
                    radius: anchor.radius,
                });
            }
            drop(ecs);
            new_anchors.sort();
            if new_anchors != load_data.anchors {
                load_data
                    .status_arrays
                    .resize_with(self.data.len(), Vec::new);
                for (i, status_array) in load_data.status_arrays.iter_mut().enumerate() {
                    let orig = self.data[i].lock();
                    status_array.clear();
                    status_array.resize(orig.status_array.len(), 0);
                    status_array
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = orig.status_array[i].load(Ordering::Relaxed));
                }
                self.load_data_busy.store(true, Ordering::SeqCst);
                load_data.anchors = new_anchors;
                drop(load_data);
                let task_load_data = self.load_data.clone();
                task_pool.push_tasks(std::iter::once(Task::new(
                    move || recalculate_load_deltas(task_load_data),
                    true,
                    false,
                )));
            }
        }
    }
}

fn recalculate_load_deltas(load_data: Arc<Mutex<LoadData>>) {
    let mut load_data = load_data.lock();
    load_data.busy.store(false, Ordering::SeqCst);
}
