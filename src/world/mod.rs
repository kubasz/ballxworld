use cgmath::Vector3;
use std::sync::{RwLock, Weak};

pub mod badgen;
pub mod ecs;
pub mod entities;
pub mod generation;
pub mod registry;

pub const VOXEL_CHUNK_DIM: usize = 32;
pub const VOXEL_CHUNK_CUBES: usize = VOXEL_CHUNK_DIM * VOXEL_CHUNK_DIM * VOXEL_CHUNK_DIM;

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
