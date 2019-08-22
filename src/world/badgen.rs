use crate::world::generation::WorldGenerator;
use crate::world::registry::VoxelRegistry;
use crate::world::{VoxelChunkRef, VOXEL_CHUNK_DIM};
use noise::{NoiseFn, Perlin, Seedable};

const SEA_LEVEL: f32 = 15.0;

#[derive(Debug, Copy, Clone)]
pub struct BadGenerator {
    h_noise: Perlin,
    ore_noise: noise::Value,
    dungeon_noise: Perlin,
}

impl Default for BadGenerator {
    fn default() -> Self {
        Self {
            h_noise: Perlin::new().set_seed(1),
            ore_noise: noise::Value::new().set_seed(2),
            dungeon_noise: Perlin::new().set_seed(5),
        }
    }
}

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
        let i_dirt = registry
            .get_definition_from_name("core:dirt")
            .expect("No standard dirt block definition found")
            .id;
        let i_stone = registry
            .get_definition_from_name("core:stone")
            .expect("No standard stone block definition found")
            .id;
        let i_diamond = registry
            .get_definition_from_name("core:diamond_ore")
            .expect("No standard diamond ore block definition found")
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

            // height noise scale factor
            const HNS: f32 = 30.0;
            let height_noise = self.h_noise.get([f64::from(x / HNS), f64::from(y / HNS)]) as f32;
            let sl = (sl + height_noise * 5.0).round();

            if y < sl - 3.0 {
                const ONS: f32 = 2.0;
                let ore_noise =
                    self.ore_noise
                        .get([f64::from(x / ONS), f64::from(y / ONS), f64::from(z / ONS)])
                        as f32;
                if ore_noise < -0.9 {
                    vox.id = i_diamond;
                } else {
                    vox.id = i_stone;
                }
            } else if y < sl - 1.0 {
                vox.id = i_dirt;
            } else if y < sl {
                vox.id = i_grass;
            } else {
                vox.id = i_air;
            }

            const DNS: f32 = 8.0;
            let dungeon_noise = self.dungeon_noise.get([
                f64::from(x / DNS),
                f64::from(y / DNS),
                f64::from(z / DNS),
            ]);
            if dungeon_noise < -0.4 {
                vox.id = i_air;
            }
        }
    }
}
