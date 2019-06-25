use cgmath::Vector3;

pub mod badgen;
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
    pub data: [VoxelDatum; VOXEL_CHUNK_CUBES],
}

impl Default for VoxelChunk {
    fn default() -> Self {
        Self::new()
    }
}

pub type ChunkPosition = Vector3<i32>;

pub struct VoxelChunkRef<'a> {
    pub chunk: &'a VoxelChunk,
    pub position: ChunkPosition,
}

pub struct VoxelChunkMutRef<'a> {
    pub chunk: &'a mut VoxelChunk,
    pub position: ChunkPosition,
}

impl VoxelChunk {
    pub fn new() -> VoxelChunk {
        VoxelChunk {
            data: [Default::default(); VOXEL_CHUNK_CUBES],
        }
    }
}

type VoxelId = u32;

#[derive(Debug, Clone, Default)]
pub struct VoxelDefinition {
    pub id: VoxelId,
    /// eg. core:air
    pub name: String,
    pub has_mesh: bool,
    pub has_collisions: bool,
    pub has_hitbox: bool,
    pub debug_color: [f32; 3],
}

impl VoxelDefinition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> u32 {
        self.id
    }
}
