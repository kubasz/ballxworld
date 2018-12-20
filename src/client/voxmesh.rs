use super::super::world::{VoxelChunk, VoxelRegistry, VOXEL_CHUNK_DIM};
use super::vulkan::vox::{ChunkBuffers, VoxelVertex};

pub fn mesh_from_chunk(chunk: &VoxelChunk, registry: &VoxelRegistry) -> ChunkBuffers {
    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();

    struct CubeSide {
        // counter-clockwise coords of the face
        pub verts: [f32; 3 * 4],
    }

    const SIDES: [CubeSide; 6] = [
        // x+ -> "right"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
            ],
        },
        // x- -> "left"
        CubeSide {
            verts: [
                -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
            ],
        },
        // y+ -> "bottom"
        CubeSide {
            verts: [
                -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
            ],
        },
        // y- -> "top"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
            ],
        },
        // z+ -> "back"
        CubeSide {
            verts: [
                0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
            ],
        },
        // z- -> "front"
        CubeSide {
            verts: [
                -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
            ],
        },
    ];

    let mut vidx: usize = 0;
    for vox in chunk.data.iter() {
        let vdef = registry.get_definition_from_id(&vox);
        if !vdef.has_mesh { continue; }

        let x = (vidx % VOXEL_CHUNK_DIM) as f32;
        let y = ((vidx / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as f32;
        let z = ((vidx / VOXEL_CHUNK_DIM / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as f32;

        for side in &SIDES {
            let voff = vbuf.len() as u32;
            for t in 0..4 {
                let position: [f32; 4] = [
                    x + side.verts[t * 3],
                    y + side.verts[t * 3 + 1],
                    z + side.verts[t * 3 + 2],
                    1.0,
                ];
                vbuf.push(VoxelVertex {
                    position,
                    color: [vdef.debug_color[0], vdef.debug_color[1], vdef.debug_color[2], 1.0],
                });
            }
            ibuf.extend([voff, voff + 1, voff + 2, voff + 2, voff + 3, voff].into_iter());
        }

        vidx += 1;
    }

    ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    }
}
