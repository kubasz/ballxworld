use crate::world::VoxelChunkMutRef;
use crate::world::registry::VoxelRegistry;

pub trait WorldGenerator {
    fn generate_chunk(&self, cref: VoxelChunkMutRef, registry: &VoxelRegistry);
}
