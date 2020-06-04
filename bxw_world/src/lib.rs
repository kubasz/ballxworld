pub mod blocks;
pub mod ecs;
pub mod entities;
pub mod generation;
pub mod physics;
pub mod raycast;
pub mod registry;
pub mod stdgen;

use crate::ecs::ECS;
use bxw_util::math::*;
use bxw_util::*;
use divrem::{DivFloor, RemFloor};
use fnv::FnvHashMap;
use lru::LruCache;
use parking_lot::RwLock;
pub use registry::VoxelRegistry;
use std::cell::{RefCell, RefMut};
use std::sync::Arc;
use thread_local::ThreadLocal;

pub const CHUNK_DIM: usize = 32;
pub const CHUNK_DIM2: usize = CHUNK_DIM * CHUNK_DIM;
pub const CHUNK_DIM3: usize = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM;

use bxw_util::collider::AABB;
pub use bxw_util::collider::{Direction, ALL_DIRS};

pub type ChunkPosition = Vector3<i32>;
pub type BlockPosition = Vector3<i32>;

pub fn chunkpos_from_blockpos(bpos: BlockPosition) -> ChunkPosition {
    let cd = CHUNK_DIM as i32;
    bpos.map(|p| p.div_floor(&cd))
}

pub fn blockidx_from_blockpos(bpos: BlockPosition) -> usize {
    let cd = CHUNK_DIM as i32;
    let innerpos = bpos.map(|p| p.rem_floor(cd) as usize);
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
            position: vec3(0, 0, 0),
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
    uncompressed_chunks: LruCache<ChunkPosition, Box<UncompressedChunk>>,
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

    pub fn ensure_newest_cached(&mut self, voxels: &WVoxels, cpos: ChunkPosition) -> Option<()> {
        let newest = voxels.chunks.get(&cpos)?;
        let cached = self.uncompressed_chunks.get_mut(&cpos);
        if let Some(cached) = cached {
            if cached.dirty < newest.dirty {
                *cached = newest.decompress();
            }
            Some(())
        } else {
            self.uncompressed_chunks.put(cpos, newest.decompress());
            Some(())
        }
    }

    /// Cached
    pub fn get_uncompressed_chunk(
        &mut self,
        voxels: &WVoxels,
        cpos: ChunkPosition,
    ) -> Option<&UncompressedChunk> {
        self.ensure_newest_cached(voxels, cpos)?;
        self.uncompressed_chunks
            .get(&cpos)
            .map(|x| x as &UncompressedChunk)
    }

    pub fn get_uncompressed_chunk_mut(
        &mut self,
        voxels: &WVoxels,
        cpos: ChunkPosition,
    ) -> Option<&mut UncompressedChunk> {
        self.ensure_newest_cached(voxels, cpos)?;
        self.uncompressed_chunks
            .get_mut(&cpos)
            .map(|x| x as &mut UncompressedChunk)
    }

    pub fn peek_uncompressed_chunk(&self, cpos: ChunkPosition) -> Option<&UncompressedChunk> {
        self.uncompressed_chunks
            .peek(&cpos)
            .map(|x| x as &UncompressedChunk)
    }

    pub fn get_block(&mut self, voxels: &WVoxels, bpos: BlockPosition) -> Option<VoxelDatum> {
        let cpos = chunkpos_from_blockpos(bpos);
        self.ensure_newest_cached(voxels, cpos)?;
        Some(self.uncompressed_chunks.get(&cpos)?.blocks_yzx[blockidx_from_blockpos(bpos)])
    }

    pub fn peek_block(&self, bpos: BlockPosition) -> Option<VoxelDatum> {
        let cpos = chunkpos_from_blockpos(bpos);
        Some(self.uncompressed_chunks.peek(&cpos)?.blocks_yzx[blockidx_from_blockpos(bpos)])
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
            position: vec3(0, 0, 0),
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
        let n = if let Some(n) = nopt {
            n
        } else {
            break;
        };
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
    use crate::{compress_rle, decompress_rle, CHUNK_DIM3};

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
            use bxw_util::*;
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
    pub fn decompress(&self) -> Box<UncompressedChunk> {
        let mut uc: Box<UncompressedChunk> = Box::default();
        uc.position = self.position;
        uc.dirty = self.dirty;
        let VChunkData::QuickCompressed { vox } = &self.data;
        decompress_rle(vox, &mut uc.blocks_yzx, |v| VoxelDatum { id: v });
        uc
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

#[derive(Clone)]
pub struct VoxelDefinition {
    pub id: VoxelId,
    /// eg. core:air
    pub name: String,
    pub has_mesh: bool,
    pub has_collisions: bool,
    pub has_hitbox: bool,
    pub collision_shape: AABB,
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

    pub fn intersect_ray() -> Option<Vector3<f32>> {
        None
    }
}

#[derive(Default)]
pub struct WVoxels {
    pub chunks: FnvHashMap<ChunkPosition, VChunk>,
}

#[derive(Default)]
pub struct WEntities {
    pub ecs: ECS,
}

pub struct World {
    pub name: String,
    pub vregistry: Arc<VoxelRegistry>,
    pub vcache: ThreadLocal<RefCell<VCache>>,
    pub voxels: RwLock<WVoxels>,
    pub entities: RwLock<WEntities>,
}

impl WVoxels {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn dirtify(&mut self, bpos: BlockPosition) {
        let cpos = chunkpos_from_blockpos(bpos);
        let ipos = bpos - (CHUNK_DIM as i32) * cpos;
        let maxdim = CHUNK_DIM as i32 - 1;
        self.chunks
            .get_mut(&cpos)
            .into_iter()
            .for_each(|c| c.dirty += 1);
        if ipos.x == 0 {
            self.chunks
                .get_mut(&(cpos + vec3(-1, 0, 0)))
                .into_iter()
                .for_each(|c| c.dirty += 1);
        }
        if ipos.x == maxdim {
            self.chunks
                .get_mut(&(cpos + vec3(1, 0, 0)))
                .into_iter()
                .for_each(|c| c.dirty += 1);
        }
        if ipos.y == 0 {
            self.chunks
                .get_mut(&(cpos + vec3(0, -1, 0)))
                .into_iter()
                .for_each(|c| c.dirty += 1);
        }
        if ipos.y == maxdim {
            self.chunks
                .get_mut(&(cpos + vec3(0, 1, 0)))
                .into_iter()
                .for_each(|c| c.dirty += 1);
        }
        if ipos.z == 0 {
            self.chunks
                .get_mut(&(cpos + vec3(0, 0, -1)))
                .into_iter()
                .for_each(|c| c.dirty += 1);
        }
        if ipos.z == maxdim {
            self.chunks
                .get_mut(&(cpos + vec3(0, 0, 1)))
                .into_iter()
                .for_each(|c| c.dirty += 1);
        }
    }
}

impl WEntities {
    pub fn new() -> Self {
        Default::default()
    }
}

impl World {
    pub fn new(name: String, vregistry: Arc<VoxelRegistry>) -> Self {
        Self {
            name,
            vregistry,
            vcache: ThreadLocal::new(),
            voxels: RwLock::new(WVoxels::new()),
            entities: RwLock::new(WEntities::new()),
        }
    }

    pub fn get_vcache(&self) -> RefMut<'_, VCache> {
        self.vcache.get_or_default().borrow_mut()
    }

    pub fn physics_tick(&self) {
        physics::world_physics_tick(self);
    }
}
