use crate::world::generation::WorldGenerator;
use crate::world::registry::VoxelRegistry;
use crate::world::{VoxelChunkMutRef, VOXEL_CHUNK_DIM};

const SEA_LEVEL: f32 = 15.0;

#[derive(Debug, Copy, Clone, Default)]
pub struct BadGenerator {}

impl WorldGenerator for BadGenerator {
    fn generate_chunk(&self, cref: VoxelChunkMutRef, registry: &VoxelRegistry) {
        let i_air = registry
            .get_definition_from_name("core:void")
            .expect("No standard air block definition found")
            .id;
        let i_grass = registry
            .get_definition_from_name("core:grass")
            .expect("No standard grass block definition found")
            .id;
        let i_stone = registry
            .get_definition_from_name("core:stone")
            .expect("No standard stone block definition found")
            .id;

        for (vidx, vox) in cref.chunk.data.iter_mut().enumerate() {
            let vcd = VOXEL_CHUNK_DIM as i32;
            let x = (cref.position.x * vcd) as f32 + (vidx % VOXEL_CHUNK_DIM) as f32;
            let y = (cref.position.y * vcd) as f32
                + ((vidx / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as f32;
            let z = (cref.position.z * vcd) as f32
                + ((vidx / VOXEL_CHUNK_DIM / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as f32;

            let sl = SEA_LEVEL + f32::sin((x + z) / 10.0) * 4.0;
            if y < sl - 3.0 {
                vox.id = i_stone;
            } else if y <= sl {
                vox.id = i_grass;
            } else {
                vox.id = i_air;
            }
        }
    }
}
