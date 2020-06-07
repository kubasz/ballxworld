use crate::ecs::*;
use crate::*;
use bxw_util::fnv::*;
use bxw_util::itertools::*;
use bxw_util::math::*;
use bxw_util::smallvec::*;
use bxw_util::taskpool::{Task, TaskPool};
use bxw_util::*;
use parking_lot::*;
use std::any::*;
use std::cell::Cell;
use std::sync::atomic::*;

#[repr(u8)]
pub enum ChunkDataState {
    Unloaded = 0,
    Waiting = 1,
    Loaded = 2,
    Updating = 3,
}

impl ChunkDataState {
    pub fn from_u8(n: u8) -> Option<Self> {
        use ChunkDataState::*;
        match n {
            0 => Some(Unloaded),
            1 => Some(Waiting),
            2 => Some(Loaded),
            3 => Some(Updating),
            _ => None,
        }
    }

    pub fn requires_load_request(n: u8) -> bool {
        n == ChunkDataState::Unloaded as u8
    }
}

pub const CHUNK_BLOCK_DATA: usize = 0;
pub const CHUNK_LIGHT_DATA: usize = 1;
pub const CHUNK_MESH_DATA: usize = 2;

pub type ChunkBlockData = Option<Arc<VChunk>>;
pub type ChunkLightData = Option<Arc<()>>;
type KindsSmallVec = SmallVec<[usize; 4]>;

pub type AnyChunkData = Option<Arc<dyn Any + Send + Sync>>;

pub trait ChunkDataHandler {
    fn get_data(&self, world: &World, index: usize) -> AnyChunkData;
    /// Returns the old data
    fn swap_data(&mut self, world: &World, index: usize, new_data: AnyChunkData) -> AnyChunkData;
    /// Extend with unloaded states or shrink unloaded states (it will never be called on non-Unloaded chunks)
    fn resize_data(&mut self, world: &World, new_size: usize);
    fn create_chunk_update_task(
        &mut self,
        world: &World,
        cpos: ChunkPosition,
        index: usize,
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
    requested_kinds: KindsSmallVec,
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

#[derive(Clone, Eq, PartialEq)]
struct ChunkDelta {
    unload: bool,
    cpos: ChunkPosition,
    handlers: KindsSmallVec,
    min_anchor_distance: i32,
}

impl Ord for ChunkDelta {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        // unloads before loads
        if self.unload != other.unload {
            if self.unload {
                Less
            } else {
                Greater
            }
        } else if self.unload {
            // Unload farthest first
            Ord::cmp(&other.min_anchor_distance, &self.min_anchor_distance)
        } else {
            // Load nearest first
            Ord::cmp(&self.min_anchor_distance, &other.min_anchor_distance)
        }
    }
}

impl PartialOrd for ChunkDelta {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

struct LoadData {
    allocation: FnvHashMap<ChunkPosition, usize>,
    chunk_positions: Vec<Option<ChunkPosition>>,
    busy: Arc<AtomicBool>,
    status_arrays: Vec<Vec<u8>>,
    anchors: Vec<LoadAnchor>,
    remaining_deltas: Vec<ChunkDelta>,
    dependencies: Vec<Option<(usize, bool)>>,
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
                allocation: Default::default(),
                chunk_positions: Default::default(),
                busy: busy_arc.clone(),
                status_arrays: Vec::new(),
                anchors: Default::default(),
                remaining_deltas: Default::default(),
                dependencies: Default::default(),
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
            let kinds_nomesh: KindsSmallVec =
                SmallVec::from_slice(&[CHUNK_BLOCK_DATA, CHUNK_LIGHT_DATA]);
            let kinds_mesh: KindsSmallVec =
                SmallVec::from_slice(&[CHUNK_BLOCK_DATA, CHUNK_LIGHT_DATA, CHUNK_MESH_DATA]);

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
                    requested_kinds: if anchor.load_mesh {
                        kinds_mesh.clone()
                    } else {
                        kinds_nomesh.clone()
                    },
                });
            }
            drop(ecs);
            new_anchors.sort();
            if new_anchors != load_data.anchors {
                load_data
                    .status_arrays
                    .resize_with(self.data.len(), Vec::new);
                load_data.chunk_positions = self.chunk_positions.clone();
                load_data.allocation = self.allocation.clone();
                load_data.dependencies = self.data.iter().map(|d| d.lock().depends_on).collect();
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
    let mut chunks_to_unload: FnvHashMap<ChunkPosition, (KindsSmallVec, i32)> = Default::default();
    chunks_to_unload.reserve(64);
    for (&cpos, &cid) in load_data.allocation.iter() {
        let mut kind_should_be_loaded = [false; 8];
        let mut min_dist = i32::max_value();
        for anchor in load_data.anchors.iter() {
            let dist: i32 = (cpos - anchor.cpos).iter().map(|x| x * x).sum();
            min_dist = dist.min(min_dist);
            if does_anchor_load_coords(anchor, cpos) {
                for &kind in anchor.requested_kinds.iter() {
                    kind_should_be_loaded[kind] = true;
                }
            }
        }
        for (kind, statuses) in load_data.status_arrays.iter().enumerate() {
            let is_loaded = !ChunkDataState::requires_load_request(statuses[cid]);
            if is_loaded && !kind_should_be_loaded[kind] {
                chunks_to_unload.insert(cpos, (Default::default(), min_dist));
            }
        }
    }
    let mut chunks_to_load: FnvHashMap<ChunkPosition, (KindsSmallVec, Cell<i32>)> =
        Default::default();
    chunks_to_load.reserve(64);
    for anchor in load_data.anchors.iter() {
        for cpos in iter_coords_to_load(anchor) {
            let cid = load_data.allocation.get(&cpos).copied();
            let missing: KindsSmallVec = if let Some(cid) = cid {
                let mut missing = SmallVec::new();
                for &kind in anchor.requested_kinds.iter() {
                    if ChunkDataState::requires_load_request(load_data.status_arrays[kind][cid]) {
                        missing.push(kind);
                    }
                }
                missing
            } else {
                anchor.requested_kinds.clone()
            };
            for kind in missing {
                if can_request_load(&load_data, kind, cpos, cid) {
                    let dist: i32 = (cpos - anchor.cpos).iter().map(|x| x * x).sum();
                    let (rkinds, rmindist) = chunks_to_load
                        .entry(cpos)
                        .or_insert((Default::default(), Cell::new(dist)));
                    if rkinds.contains(&kind) {
                        rmindist.set(rmindist.get().min(dist));
                    } else {
                        rkinds.push(kind);
                    }
                }
            }
        }
    }
    load_data.remaining_deltas.clear();
    let old_cap = load_data.remaining_deltas.capacity();
    let new_cap = chunks_to_load.len();
    if old_cap < new_cap {
        load_data.remaining_deltas.reserve(new_cap - old_cap);
    }
    for (cpos, (unload_kinds, min_anchor_distance)) in chunks_to_unload {
        load_data.remaining_deltas.push(ChunkDelta {
            unload: true,
            cpos,
            handlers: unload_kinds,
            min_anchor_distance,
        });
    }
    for (cpos, (load_kinds, min_anchor_distance)) in chunks_to_load {
        load_data.remaining_deltas.push(ChunkDelta {
            unload: false,
            cpos,
            handlers: load_kinds,
            min_anchor_distance: min_anchor_distance.get(),
        });
    }
    load_data.remaining_deltas.sort();
    load_data.busy.store(false, Ordering::SeqCst);
}

fn iter_coords_to_load(loader: &LoadAnchor) -> Box<dyn Iterator<Item = ChunkPosition>> {
    let origin = loader.cpos;
    let r = loader.radius as i32;
    let r2 = r * r;
    let rrange = -r..=r;
    let cube_range = iproduct!(rrange.clone(), rrange.clone(), rrange);
    let sphere_range = cube_range.filter(move |(x, y, z)| x * x + y * y + z * z <= r2);
    Box::new(sphere_range.map(move |(x, y, z)| origin + vec3(x, y, z)))
}

fn does_anchor_load_coords(loader: &LoadAnchor, cpos: ChunkPosition) -> bool {
    (cpos - loader.cpos).map(|x| x * x).sum() <= loader.radius as i32
}

fn iter_neighbors(cpos: ChunkPosition) -> Box<dyn Iterator<Item = ChunkPosition>> {
    Box::new(
        iproduct!(
            cpos.x - 1..=cpos.x + 1,
            cpos.y - 1..=cpos.y + 1,
            cpos.z - 1..=cpos.z + 1
        )
        .filter(move |&(x, y, z)| x != cpos.x || y != cpos.y || z != cpos.z)
        .map(|(x, y, z)| vec3(x, y, z)),
    )
}

fn can_request_load(
    load_data: &LoadData,
    kind: usize,
    cpos: ChunkPosition,
    cid: Option<usize>,
) -> bool {
    if let Some((depkind, neighbours)) = load_data.dependencies[kind] {
        let cid = if let Some(cid) = cid {
            cid
        } else {
            return false;
        };
        let statuses = &load_data.status_arrays[depkind];
        if neighbours {
            !iter_neighbors(cpos).any(|npos| {
                load_data
                    .allocation
                    .get(&npos)
                    .copied()
                    .map(|nid| ChunkDataState::requires_load_request(statuses[nid]))
                    .unwrap_or(true)
            })
        } else {
            !ChunkDataState::requires_load_request(statuses[cid])
        }
    } else {
        true
    }
}
