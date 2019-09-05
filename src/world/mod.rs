pub mod blocks;
pub mod ecs;
pub mod entities;
pub mod generation;
pub mod registry;
pub mod stdgen;

use crate::world::ecs::ECS;
use cgmath::prelude::*;
use cgmath::{vec3, Vector3};
use lru::LruCache;
use parking_lot::RwLock;
pub use registry::VoxelRegistry;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use thread_local::ThreadLocal;
use divrem::{DivFloor, RemFloor};

pub const CHUNK_DIM: usize = 32;
pub const CHUNK_DIM2: usize = CHUNK_DIM * CHUNK_DIM;
pub const CHUNK_DIM3: usize = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM;

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    XMinus = 0,
    XPlus,
    YMinus,
    YPlus,
    ZMinus,
    ZPlus,
}

static ALL_DIRS: [Direction; 6] = {
    use Direction::*;
    [XMinus, XPlus, YMinus, YPlus, ZMinus, ZPlus]
};

impl Direction {
    pub fn all() -> &'static [Direction; 6] {
        &ALL_DIRS
    }

    pub fn opposite(self) -> Self {
        use Direction::*;
        match self {
            XMinus => XPlus,
            XPlus => XMinus,
            YMinus => YPlus,
            YPlus => YMinus,
            ZMinus => ZPlus,
            ZPlus => ZMinus,
        }
    }

    pub fn try_from_vec(v: Vector3<i32>) -> Option<Self> {
        match v {
            Vector3 { x: 1, y: 0, z: 0 } => Some(Direction::XPlus),
            Vector3 { x: -1, y: 0, z: 0 } => Some(Direction::XMinus),
            Vector3 { x: 0, y: 1, z: 0 } => Some(Direction::YPlus),
            Vector3 { x: 0, y: -1, z: 0 } => Some(Direction::YMinus),
            Vector3 { x: 0, y: 0, z: 1 } => Some(Direction::ZPlus),
            Vector3 { x: 0, y: 0, z: -1 } => Some(Direction::ZMinus),
            _ => None,
        }
    }

    pub fn to_vec(self) -> Vector3<i32> {
        use Direction::*;
        match self {
            XMinus => vec3(-1, 0, 0),
            XPlus => vec3(1, 0, 0),
            YMinus => vec3(0, -1, 0),
            YPlus => vec3(0, 1, 0),
            ZMinus => vec3(0, 0, -1),
            ZPlus => vec3(0, 0, 1),
        }
    }
}

pub type ChunkPosition = Vector3<i32>;
pub type BlockPosition = Vector3<i32>;

pub fn chunkpos_from_blockpos(bpos: BlockPosition) -> ChunkPosition {
    let cd = CHUNK_DIM as i32;
    bpos.map(|p|  p.div_floor(cd))
}

pub fn blockidx_from_blockpos(bpos: BlockPosition) -> usize {
    let cd = CHUNK_DIM as i32;
    let innerpos = bpos.map(|p| p.rem_floor(cd) as usize );
    innerpos.x + CHUNK_DIM * innerpos.z + CHUNK_DIM2 * innerpos.y
}

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
pub struct VoxelDatum {
    pub id: u32,
}

#[derive(Clone)]
pub struct UncompressedChunk {
    pub blocks_yzx: [VoxelDatum; CHUNK_DIM3],
    pub position: ChunkPosition,
    pub dirty: u64,
}

impl Default for UncompressedChunk {
    fn default() -> Self {
        Self {
            blocks_yzx: [Default::default(); CHUNK_DIM3],
            position: Zero::zero(),
            dirty: 1,
        }
    }
}

impl UncompressedChunk {
    pub fn new() -> Self {
        Default::default()
    }
}

/// Stored per-thread in the corresponding world object
pub struct VCache {
    uncompressed_chunks: LruCache<ChunkPosition, Arc<UncompressedChunk>>,
}

impl Default for VCache {
    fn default() -> Self {
        Self {
            uncompressed_chunks: LruCache::new(64),
        }
    }
}

impl VCache {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Clone)]
pub enum VChunkData {
    /// Voxels stored for relatively quick access, e.g. RLE-compressed
    QuickCompressed { vox: Vec<u32> },
}

impl VChunkData {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for VChunkData {
    fn default() -> Self {
        VChunkData::QuickCompressed {
            vox: vec![0, 0, CHUNK_DIM3 as u32],
        }
    }
}

#[derive(Clone)]
pub struct VChunk {
    /// Chunk "mip-maps" by level - 0 is VOXEL_CHUNK_DIM-wide, 1 is 1/2 of that, etc.
    pub data: VChunkData,
    /// A number increased after each change to this chunk while it's loaded
    pub dirty: u64,
    pub position: ChunkPosition,
}

impl Default for VChunk {
    fn default() -> Self {
        Self {
            data: Default::default(),
            dirty: 1,
            position: Zero::zero(),
        }
    }
}

fn compress_rle<I: Iterator<Item = u32>>(data: I) -> Vec<u32> {
    let mut outvec = Vec::new();
    let mut rle_elem = None;
    let mut rle_len = 0;
    let mut prev = None;
    for v in data {
        if let Some(r) = rle_elem {
            if r == v {
                rle_len += 1;
            } else {
                outvec.push(rle_len);
                outvec.push(v);
                prev = Some(v);
                rle_elem = None;
            }
        } else {
            outvec.push(v);
            if prev.map(|p| p == v).unwrap_or(false) {
                prev = None;
                rle_len = 0;
                rle_elem = Some(v);
            } else {
                prev = Some(v);
            }
        }
    }
    if rle_elem.is_some() {
        outvec.push(rle_len);
    }
    outvec
}

fn decompress_rle<TF, TT: Copy>(data: &[u32], target: &mut [TT; CHUNK_DIM3], transform: TF)
where
    TF: Fn(u32) -> TT,
{
    let mut ti = 0;
    let mut di = data.iter().copied().enumerate();
    let mut prev = None;
    loop {
        let nopt = di.next();
        let n;
        if nopt.is_none() {
            break;
        } else {
            n = nopt.unwrap();
        }
        let tdat = transform(n.1);
        if ti >= target.len() {
            panic!("{:?} ti={}", n, ti);
        }
        target[ti] = tdat;
        ti += 1;
        if prev.map(|p| p == n.1).unwrap_or(false) {
            prev = None;
            let rn = di.next().unwrap().1;
            for _ in 0..rn {
                target[ti] = tdat;
                ti += 1;
            }
        } else {
            prev = Some(n.1);
        }
    }
    assert_eq!(ti, CHUNK_DIM3);
}

#[cfg(test)]
mod test {
    use crate::world::{compress_rle, decompress_rle, CHUNK_DIM3};

    #[test]
    fn rle_compress_zero_test() {
        let zeroes = [0u32; CHUNK_DIM3];
        let compressed = compress_rle(zeroes.iter().copied());
        assert_eq!(compressed, vec![0, 0, CHUNK_DIM3 as u32 - 2]);
    }

    #[test]
    fn rle_compress_one_test() {
        let ones = [1u32; CHUNK_DIM3];
        let compressed = compress_rle(ones.iter().copied());
        assert_eq!(compressed, vec![1, 1, CHUNK_DIM3 as u32 - 2]);
    }

    #[test]
    fn rle_decompress_zero_test() {
        let mut target = [0xFFFF_FFFFu32; CHUNK_DIM3];
        let compressed = vec![0, 0, CHUNK_DIM3 as u32 - 2];
        decompress_rle(&compressed, &mut target, |x| x);
        assert!(target.iter().copied().all(|e| e == 0));
    }

    #[test]
    fn rle_decompress_one_test() {
        let mut target = [0xFFFF_FFFFu32; CHUNK_DIM3];
        let compressed = vec![1, 1, CHUNK_DIM3 as u32 - 2];
        decompress_rle(&compressed, &mut target, |x| x);
        assert!(target.iter().copied().all(|e| e == 1));
    }

    #[test]
    fn rle_random_cmp() {
        let mut randdata = [0xFFFF_FFFFu32; CHUNK_DIM3];
        {
            use rand::prelude::*;
            use rand_xoshiro::Xoshiro256StarStar;
            let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
            for e in randdata.iter_mut() {
                *e = rng.next_u32() % 16;
            }
        }
        let compdata = compress_rle(randdata.iter().copied());
        let mut decdata = [0xFFFF_FFFFu32; CHUNK_DIM3];
        decompress_rle(&compdata, &mut decdata, |x| x);
        assert_eq!(randdata[..], decdata[..]);
    }
}

impl VChunk {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes the updates from an uncompressed chunk into compressed storage
    pub fn compress(&mut self, from: &UncompressedChunk) {
        debug_assert_eq!(self.position, from.position);
        let voxdat = compress_rle(from.blocks_yzx.iter().map(|v| v.id));
        self.dirty = from.dirty;
        self.data = VChunkData::QuickCompressed { vox: voxdat };
    }

    /// Decompresses the current version of this chunk
    pub fn decompress(&self) -> Arc<UncompressedChunk> {
        let mut uc = Box::new(UncompressedChunk {
            position: self.position,
            dirty: self.dirty,
            blocks_yzx: [Default::default(); CHUNK_DIM3],
        });
        let VChunkData::QuickCompressed { vox } = &self.data;
        decompress_rle(vox, &mut uc.blocks_yzx, |v| VoxelDatum { id: v });
        Arc::from(uc)
    }
}

type VoxelId = u32;

#[derive(Clone, Debug)]
pub enum TextureMapping<T> {
    TiledSingle(T),
    TiledTSB { top: T, side: T, bottom: T },
}

impl Default for TextureMapping<u32> {
    fn default() -> Self {
        TextureMapping::TiledSingle(0)
    }
}

impl<T> TextureMapping<T> {
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> TextureMapping<U> {
        use TextureMapping::*;
        match self {
            TiledSingle(a) => TiledSingle(f(a)),
            TiledTSB { top, side, bottom } => TiledTSB {
                top: f(top),
                side: f(side),
                bottom: f(bottom),
            },
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct VoxelDefinition {
    pub id: VoxelId,
    /// eg. core:air
    pub name: String,
    pub has_mesh: bool,
    pub has_collisions: bool,
    pub has_hitbox: bool,
    pub debug_color: [f32; 3],
    pub texture_mapping: TextureMapping<u32>,
}

impl VoxelDefinition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> u32 {
        self.id
    }
}

type ClientWorld = crate::client::world::ClientWorld;

pub struct WVoxels {
    pub chunks: HashMap<ChunkPosition, VChunk>,
    pub cache: ThreadLocal<RefCell<VCache>>,
    pub registry: Arc<VoxelRegistry>,
}

#[derive(Default)]
pub struct WEntities {
    pub ecs: ECS,
}

pub struct World {
    pub name: String,
    pub voxels: RwLock<WVoxels>,
    pub entities: RwLock<WEntities>,
    pub client_world: Option<RwLock<ClientWorld>>,
}

impl WVoxels {
    pub fn new(registry: Arc<VoxelRegistry>) -> Self {
        Self {
            chunks: HashMap::new(),
            cache: ThreadLocal::new(),
            registry,
        }
    }

    fn ensure_newest_cached(&self, cpos: ChunkPosition) -> Option<()> {
        let newest = self.chunks.get(&cpos)?;
        let cache = self.cache.get_default();
        let mut mcache = cache.borrow_mut();
        let cached = mcache.uncompressed_chunks.get_mut(&cpos);
        if let Some(cached) = cached {
            if cached.dirty != newest.dirty {
                *cached = newest.decompress();
            }
            Some(())
        } else {
            mcache.uncompressed_chunks.put(cpos, newest.decompress());
            Some(())
        }
    }

    /// Cached
    pub fn get_uncompressed_chunk(&self, cpos: ChunkPosition) -> Option<Arc<UncompressedChunk>> {
        self.ensure_newest_cached(cpos)?;
        let cache = self.cache.get_default();
        let mut mcache = cache.borrow_mut();
        Some(mcache.uncompressed_chunks.get(&cpos)?.clone())
    }

    pub fn get_block(&self, bpos: BlockPosition) -> Option<VoxelDatum> {
        let cpos = chunkpos_from_blockpos(bpos);
        self.ensure_newest_cached(cpos)?;
        let cache = self.cache.get_default();
        let mut mcache = cache.borrow_mut();
        Some(mcache.uncompressed_chunks.get(&cpos)?.clone().blocks_yzx[blockidx_from_blockpos(bpos)])
    }
}

impl WEntities {
    pub fn new() -> Self {
        Default::default()
    }
}

impl World {
    pub fn new(
        name: String,
        registry: Arc<VoxelRegistry>,
        client_world: Option<RwLock<ClientWorld>>,
    ) -> Self {
        Self {
            name,
            voxels: RwLock::new(WVoxels::new(registry)),
            entities: RwLock::new(WEntities::new()),
            client_world,
        }
    }

    pub fn physics_tick(&self) {
        //
    }
}
