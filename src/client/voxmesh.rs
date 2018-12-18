use super::super::world::{VoxelChunk, VoxelRegistry};
use super::vulkan::vox::{ChunkBuffers, VoxelVertex};
use cgmath::prelude;
use cgmath::{Vector3, vec3};

pub fn mesh_from_chunk(chunk: &VoxelChunk, registry: &VoxelRegistry) -> ChunkBuffers {
    let vbuf: Vec<VoxelVertex> = Vec::new();
    let ibuf: Vec<u32> = Vec::new();

    struct CubeSide {
        pub center: Vector3<f32>,
        //pub plus_offset: Vector3<f32>,
        //pub minus_offset: Vector3<f32>,
    }

    const SIDES: [CubeSide; 6] = [
        CubeSide {
            center: Vector3{x:0.5, y:0.0, z:0.0}
        },
        CubeSide {
            center: Vector3{x:-0.5, y:0.0, z:0.0}
        },
        CubeSide {
            center: Vector3{x:0.0, y:0.5, z:0.0}
        },
        CubeSide {
            center: Vector3{x:0.0, y:-0.5, z:0.0}
        },
        CubeSide {
            center: Vector3{x:0.0, y:0.0, z:0.5}
        },
        CubeSide {
            center: Vector3{x:0.0, y:0.0, z:-0.5}
        }
    ];

    for side in &SIDES {
        //
    }

    ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    }
}