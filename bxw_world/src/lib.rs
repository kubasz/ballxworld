pub mod blocks;
pub mod ecs;
pub mod entities;
pub mod generation;
pub mod inventory;
pub mod itemregistry;
pub mod physics;
pub mod raycast;
pub mod stdgen;
pub mod voxregistry;
pub mod worldmgr;

use bxw_util::collider::AABB;
pub use bxw_util::direction::{Direction, ALL_DIRS};
use bxw_util::math::*;
use bxw_util::*;
use divrem::{DivFloor, RemFloor};
use lru::LruCache;
use smallvec::SmallVec;
use std::cell::{RefCell, RefMut};
use std::convert::TryFrom;
use std::sync::Arc;
pub use voxregistry::VoxelRegistry;

pub const CHUNK_DIM: usize = 32;
pub const CHUNK_DIM2: usize = CHUNK_DIM * CHUNK_DIM;
pub const CHUNK_DIM3: usize = CHUNK_DIM * CHUNK_DIM * CHUNK_DIM;

pub type ChunkPosition = Vector3<i32>;
pub type BlockPosition = Vector3<i32>;

pub fn blockpos_from_worldpos(wpos: Vector3<f64>) -> BlockPosition {
    wpos.map(|c| (c + 0.5).floor() as i32)
}

pub fn chunkpos_from_blockpos(bpos: BlockPosition) -> ChunkPosition {
    let cd = CHUNK_DIM as i32;
    bpos.map(|p| p.div_floor(&cd))
}

pub fn blockidx_from_blockpos(bpos: BlockPosition) -> usize {
    let cd = CHUNK_DIM as i32;
    let innerpos = bpos.map(|p| p.rem_floor(cd) as usize);
    innerpos.x + CHUNK_DIM * innerpos.z + CHUNK_DIM2 * innerpos.y
}

pub fn dirty_chunkpos_from_blockpos(bpos: BlockPosition) -> SmallVec<[ChunkPosition; 8]> {
    let cpos = chunkpos_from_blockpos(bpos);
    let mut dirty = SmallVec::new();
    dirty.push(cpos);
    let ipos = bpos - (CHUNK_DIM as i32) * cpos;
    let maxdim = CHUNK_DIM as i32 - 1;
    if ipos.x == 0 {
        dirty.push(cpos + vec3(-1, 0, 0));
    }
    if ipos.x == maxdim {
        dirty.push(cpos + vec3(1, 0, 0));
    }
    if ipos.y == 0 {
        dirty.push(cpos + vec3(0, -1, 0));
    }
    if ipos.y == maxdim {
        dirty.push(cpos + vec3(0, 1, 0));
    }
    if ipos.z == 0 {
        dirty.push(cpos + vec3(0, 0, -1));
    }
    if ipos.z == maxdim {
        dirty.push(cpos + vec3(0, 0, 1));
    }
    dirty
}

pub type VoxelId = u16;
pub type VoxelMetadata = u16;

#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
#[repr(transparent)]
pub struct VoxelDatum {
    datum: u32,
}

impl VoxelDatum {
    #[inline(always)]
    pub fn new(id: VoxelId, meta: VoxelMetadata) -> Self {
        Self {
            datum: (id as u32) << 16 | meta as u32,
        }
    }

    #[inline(always)]
    pub fn from_repr(r: u32) -> Self {
        Self { datum: r }
    }

    #[inline(always)]
    pub fn id(self) -> VoxelId {
        VoxelId::try_from((self.datum >> 16) & 0xFFFF).unwrap()
    }

    #[inline(always)]
    pub fn meta(self) -> VoxelMetadata {
        VoxelMetadata::try_from(self.datum & 0xFFFF).unwrap()
    }

    #[inline(always)]
    pub fn repr(self) -> u32 {
        self.datum
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub enum VoxelStdShape {
    /// Standard full voxel cube
    Cube,
    /// "Slab" half-cube, occupying the bottom half of a voxel or reoriented
    HalfCube,
    /// Slope/Wedge shape with top face stretched, front collapsed; left and right faces are triangles
    Slope,
    /// Pentahedral shape to connect two slopes, no top, front and left are the sloped faces
    OuterCornerSlope,
    /// A cube with its top-left-front corner moved down to coincide with bottom-left-front
    InnerCornerSlope,
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
    pub position: ChunkPosition,
}

impl Default for VChunk {
    fn default() -> Self {
        Self {
            data: Default::default(),
            position: vec3(0, 0, 0),
        }
    }
}

fn compress_rle<I: Iterator<Item = u32>>(data: I) -> Vec<u32> {
    let mut outvec = Vec::with_capacity(128);
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
    outvec.shrink_to_fit();
    outvec
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum RleDecompressError {
    TooMuchUncompressedData,
    TooMuchRleData(usize),
    MissingRleRepeatN,
    FinalPosMismatch(usize),
}

fn decompress_rle(data: &[u32]) -> Result<Box<UncompressedChunk>, RleDecompressError> {
    if data.len() < 3 {
        return Err(RleDecompressError::FinalPosMismatch(data.len()));
    }
    let mut target_box: Box<UncompressedChunk> = Box::default();
    let target = &mut target_box.blocks_yzx;
    let first_element = data[0];
    target[0] = VoxelDatum::from_repr(first_element);
    let mut prev = Some(first_element);
    let mut target_pos = 1;
    let mut rle_iterator = data[1..].iter();
    while let Some(&element) = rle_iterator.next() {
        if let Some(e) = target.get_mut(target_pos) {
            *e = VoxelDatum::from_repr(element);
        } else {
            return Err(RleDecompressError::TooMuchUncompressedData);
        }
        target_pos += 1;
        if prev.map_or(false, |prev| prev == element) {
            prev = None;
            let extra_repeat_count = if let Some(&n) = rle_iterator.next() {
                n as usize
            } else {
                return Err(RleDecompressError::MissingRleRepeatN);
            };
            if target_pos + extra_repeat_count > target.len() {
                return Err(RleDecompressError::TooMuchRleData(extra_repeat_count));
            }
            for e in &mut target[target_pos..target_pos + extra_repeat_count] {
                *e = VoxelDatum::from_repr(element);
            }
            target_pos += extra_repeat_count;
        } else {
            prev = Some(element);
        }
    }
    if target_pos != target.len() {
        Err(RleDecompressError::FinalPosMismatch(target_pos))
    } else {
        Ok(target_box)
    }
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
        let compressed = vec![0, 0, CHUNK_DIM3 as u32 - 2];
        let target = decompress_rle(&compressed).unwrap();
        assert!(target.blocks_yzx.iter().copied().all(|e| e.repr() == 0));
    }

    #[test]
    fn rle_decompress_one_test() {
        let compressed = vec![1, 1, CHUNK_DIM3 as u32 - 2];
        let target = decompress_rle(&compressed).unwrap();
        assert!(target.blocks_yzx.iter().copied().all(|e| e.repr() == 1));
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
        let decdata = decompress_rle(&compdata).unwrap();
        assert!(randdata
            .iter()
            .copied()
            .eq(decdata.blocks_yzx.iter().copied().map(|x| x.repr())));
    }
}

impl VChunk {
    pub fn new() -> Self {
        Self::default()
    }

    /// Writes the updates from an uncompressed chunk into compressed storage
    pub fn compress(&mut self, from: &UncompressedChunk) {
        debug_assert_eq!(self.position, from.position);
        let voxdat = compress_rle(from.blocks_yzx.iter().copied().map(VoxelDatum::repr));
        self.data = VChunkData::QuickCompressed { vox: voxdat };
    }

    /// Decompresses the current version of this chunk
    pub fn decompress(&self) -> Box<UncompressedChunk> {
        let VChunkData::QuickCompressed { vox } = &self.data;
        let mut uc = decompress_rle(vox).expect("Invalid compressed chunk stored");
        uc.position = self.position;
        uc
    }
}

#[derive(Clone, Debug)]
pub struct TextureMapping<T> {
    mapping: [T; 6],
}

impl Default for TextureMapping<u32> {
    fn default() -> Self {
        Self { mapping: [0; 6] }
    }
}

impl<T: Clone> TextureMapping<T> {
    pub fn new_single(t: T) -> Self {
        Self {
            mapping: [t.clone(), t.clone(), t.clone(), t.clone(), t.clone(), t],
        }
    }

    pub fn new_tsb(top: T, side: T, bottom: T) -> Self {
        Self {
            mapping: [side.clone(), side.clone(), bottom, top, side.clone(), side],
        }
    }
}

impl<T> TextureMapping<T> {
    pub fn new(by_direction: [T; 6]) -> Self {
        Self {
            mapping: by_direction,
        }
    }

    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> TextureMapping<U> {
        let Self { mapping: m } = self;
        let [m0, m1, m2, m3, m4, m5] = m;
        TextureMapping {
            mapping: [f(m0), f(m1), f(m2), f(m3), f(m4), f(m5)],
        }
    }

    pub fn at_direction(&self, dir: Direction) -> &T {
        &self.mapping[dir.to_signed_axis_index()]
    }
}

#[derive(Clone)]
pub struct VoxelDefinition {
    pub id: VoxelId,
    /// eg. core:air
    pub name: String,
    pub mesh: VoxelMesh,
    pub collision_shape: Option<AABB>,
    pub selection_shape: Option<AABB>,
    pub debug_color: [f32; 3],
    pub texture_mapping: TextureMapping<u32>,
}

impl VoxelDefinition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> VoxelId {
        self.id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum VoxelMesh {
    None,
    CubeAndSlopes,
}

impl Default for VoxelMesh {
    fn default() -> Self {
        VoxelMesh::None
    }
}

impl VoxelMesh {
    pub fn is_none(&self) -> bool {
        matches!(self, VoxelMesh::None)
    }

    pub fn is_some(&self) -> bool {
        !matches!(self, VoxelMesh::None)
    }
}
