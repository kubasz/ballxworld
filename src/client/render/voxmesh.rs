use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use crate::math::*;
use crate::world::registry::VoxelRegistry;
use crate::world::{
    blockidx_from_blockpos, ChunkPosition, UncompressedChunk, WVoxels, World, CHUNK_DIM, CHUNK_DIM2,
};
use smallvec::SmallVec;

struct CubeSide {
    // counter-clockwise coords of the face
    pub verts: [f32; 3 * 4],
    // corners of the side, matching up with `verts` above
    pub corners: [[i32; 3]; 4],
    // what to add to position to find neighbor
    pub ioffset: [i32; 3],
    // texture coordinates (u,v) matched with vertex coordinates
    pub texcs: [f32; 2 * 4],
}

pub fn mesh_from_chunk(
    world: &World,
    voxels: &WVoxels,
    cpos: ChunkPosition,
) -> Option<ChunkBuffers> {
    let registry: &VoxelRegistry = &world.vregistry;
    let mut vcache = world.get_vcache();
    let mut chunks: SmallVec<[&UncompressedChunk; 32]> = SmallVec::new();
    for y in -1..=1 {
        for z in -1..=1 {
            for x in -1..=1 {
                vcache.ensure_newest_cached(voxels, cpos + vec3(x, y, z))?;
            }
        }
    }
    for y in -1..=1 {
        for z in -1..=1 {
            for x in -1..=1 {
                chunks.push(
                    vcache
                        .peek_uncompressed_chunk(cpos + vec3(x, y, z))
                        .unwrap(),
                );
            }
        }
    }
    let chunk = &chunks[13];
    // pos relative to Chunk@cpos
    let get_block = |pos: Vector3<i32>| {
        let cp = pos.map(|c| (c + 32) / 32);
        let ch: &UncompressedChunk = &chunks[(cp.x + cp.z * 3 + cp.y * 9) as usize];
        ch.blocks_yzx[blockidx_from_blockpos(pos)]
    };

    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();

    const SIDES: [CubeSide; 6] = [
        // x+ -> "right"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
            ],
            corners: [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
            ioffset: [1, 0, 0],
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // x- -> "left"
        CubeSide {
            verts: [
                -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
            ],
            corners: [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, -1]],
            ioffset: [-1, 0, 0],
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // y+ -> "bottom"
        CubeSide {
            verts: [
                -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
            ],
            corners: [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]],
            ioffset: [0, 1, 0],
            texcs: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        // y- -> "top"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
            ],
            corners: [[1, -1, 0], [1, -1, 1], [-1, -1, 1], [-1, -1, -1]],
            ioffset: [0, -1, 0],
            texcs: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        // z+ -> "back"
        CubeSide {
            verts: [
                0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
            ],
            corners: [[1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]],
            ioffset: [0, 0, 1],
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // z- -> "front"
        CubeSide {
            verts: [
                -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
            ],
            corners: [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]],
            ioffset: [0, 0, -1],
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    ];

    for side in &SIDES {
        let ioffset = Vector3::from_row_slice(&side.ioffset);
        for (vidx, vox) in chunk.blocks_yzx.iter().enumerate() {
            let vox = *vox;
            let vdef = registry.get_definition_from_id(vox);

            if !vdef.has_mesh {
                continue;
            }

            let ipos = vec3(
                (vidx % CHUNK_DIM) as i32,
                ((vidx / CHUNK_DIM2) % CHUNK_DIM) as i32,
                ((vidx / CHUNK_DIM) % CHUNK_DIM) as i32,
            );
            let x = ipos.x as f32;
            let y = ipos.y as f32;
            let z = ipos.z as f32;

            // hidden face removal
            let touchpos = ipos + ioffset;
            let tid = get_block(touchpos);
            let tdef = registry.get_definition_from_id(tid);
            if tdef.has_mesh {
                continue;
            }

            let voff = vbuf.len() as u32;
            let mut corner_ao = [0i32; 4];
            for (t, corner) in side.corners.iter().enumerate() {
                let corner = Vector3::from_row_slice(corner);
                // AO calculation
                let (ao_s1, ao_s2, ao_c): (bool, bool, bool);
                {
                    let p_c = ipos + corner;
                    ao_c = registry.get_definition_from_id(get_block(p_c)).has_mesh;
                    let (p_s1, p_s2);
                    if ioffset.x != 0 {
                        // y,z sides
                        p_s1 = ipos + vec3(corner.x, corner.y, 0);
                        p_s2 = ipos + vec3(corner.x, 0, corner.z);
                    } else if ioffset.y != 0 {
                        // x,z sides
                        p_s1 = ipos + vec3(0, corner.y, corner.z);
                        p_s2 = ipos + vec3(corner.x, corner.y, 0);
                    } else {
                        // x,y sides
                        p_s1 = ipos + vec3(corner.x, 0, corner.z);
                        p_s2 = ipos + vec3(0, corner.y, corner.z);
                    }
                    ao_s1 = registry.get_definition_from_id(get_block(p_s1)).has_mesh;
                    ao_s2 = registry.get_definition_from_id(get_block(p_s2)).has_mesh;
                }
                let ao = if ao_s1 && ao_s2 {
                    3
                } else {
                    ao_s1 as i32 + ao_s2 as i32 + ao_c as i32
                };
                corner_ao[t] = ao;
                let ao: f32 = match ao {
                    0 => 1.0,
                    1 => 0.8,
                    2 => 0.7,
                    _ => 0.6,
                };

                let position: [f32; 4] = [
                    x + side.verts[t * 3],
                    y + side.verts[t * 3 + 1],
                    z + side.verts[t * 3 + 2],
                    1.0,
                ];
                let texid;
                use crate::world::TextureMapping;
                match &vdef.texture_mapping {
                    TextureMapping::TiledSingle(t) => {
                        texid = *t;
                    }
                    TextureMapping::TiledTSB {
                        top,
                        side: tside,
                        bottom,
                    } => {
                        if ioffset.y == 1 {
                            texid = *top;
                        } else if ioffset.y == -1 {
                            texid = *bottom;
                        } else {
                            texid = *tside;
                        }
                    }
                }
                vbuf.push(VoxelVertex {
                    position,
                    color: [
                        vdef.debug_color[0] * ao,
                        vdef.debug_color[1] * ao,
                        vdef.debug_color[2] * ao,
                        1.0,
                    ],
                    texcoord: [side.texcs[t * 2], side.texcs[t * 2 + 1], texid as f32],
                    index: vidx as i32,
                });
            }
            if corner_ao[1] + corner_ao[3] > corner_ao[0] + corner_ao[2] {
                ibuf.extend([voff, voff + 1, voff + 2, voff + 2, voff + 3, voff].iter());
            } else {
                ibuf.extend([voff, voff + 1, voff + 3, voff + 1, voff + 2, voff + 3].iter());
            }
        }
    }

    Some(ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    })
}
