use crate::world::generation::WorldGenerator;
use crate::world::registry::VoxelRegistry;
use crate::world::{VoxelChunkRef, VOXEL_CHUNK_DIM};

const SEA_LEVEL: f32 = 15.0;

#[derive(Debug, Copy, Clone, Default)]
pub struct BadGenerator {}

impl WorldGenerator for BadGenerator {
    fn generate_chunk(&self, cref: VoxelChunkRef, registry: &VoxelRegistry) {
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
        let i_border = registry
            .get_definition_from_name("core:border")
            .expect("No standard stone block definition found")
            .id;

        let chunkarc = cref.chunk.upgrade();
        if chunkarc.is_none() {
            return;
        }
        let chunkarc = chunkarc.unwrap();
        let mut chunk = chunkarc.write().unwrap();

        for (vidx, vox) in chunk.data.iter_mut().enumerate() {
            let vcd = VOXEL_CHUNK_DIM as i32;
            let xc = (vidx % VOXEL_CHUNK_DIM) as f32;
            let yc = ((vidx / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as f32;
            let zc = ((vidx / VOXEL_CHUNK_DIM / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as f32;
            let x = (cref.position.x * vcd) as f32 + xc;
            let y = (cref.position.y * vcd) as f32 + yc;
            let z = (cref.position.z * vcd) as f32 + zc;

            let sl = SEA_LEVEL + f32::sin((x + z) / 10.0) * 4.0;
            if xc == 0.0 && yc == 0.0 || xc == 0.0 && zc == 0.0 || yc == 0.0 && zc == 0.0 {
                vox.id = i_border;
            } else if y < sl - 3.0 {
                let x = x as u32;
                let y = y as u32;
                let z = z as u32;
                if (x.wrapping_mul(3559) ^ y.wrapping_mul(541) ^ z.wrapping_mul(1223)) % 89 == 1 {
                    vox.id = i_grass;
                } else {
                    vox.id = i_stone;
                }
            } else if y <= sl {
                vox.id = i_grass;
            } else {
                vox.id = i_air;
            }
        }
    }
}
