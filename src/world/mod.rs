use cgmath::{vec3, Vector3};
use std::sync::{RwLock, Weak};

pub mod blocks;
pub mod ecs;
pub mod entities;
pub mod generation;
pub mod registry;
pub mod stdgen;

pub const VOXEL_CHUNK_DIM: usize = 32;
pub const VOXEL_CHUNK_CUBES: usize = VOXEL_CHUNK_DIM * VOXEL_CHUNK_DIM * VOXEL_CHUNK_DIM;

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

#[derive(Debug, Copy, Clone, Default)]
pub struct VoxelDatum {
    pub id: u32,
}

#[derive(Clone)]
pub struct VoxelChunk {
    /// The raw voxel data
    pub data: [VoxelDatum; VOXEL_CHUNK_CUBES],
    /// A number increased after each change to this chunk while it's loaded
    pub dirty: u64,
    /// References to neighboring chunks, indexed by the Direction enum
    pub neighbor: [Weak<RwLock<Self>>; 6],
}

impl Default for VoxelChunk {
    fn default() -> Self {
        Self::new()
    }
}

pub type ChunkPosition = Vector3<i32>;

pub struct VoxelChunkRef {
    pub chunk: Weak<RwLock<VoxelChunk>>,
    pub position: ChunkPosition,
}

impl VoxelChunk {
    pub fn new() -> VoxelChunk {
        VoxelChunk {
            data: [Default::default(); VOXEL_CHUNK_CUBES],
            dirty: 0,
            neighbor: Default::default(),
        }
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
