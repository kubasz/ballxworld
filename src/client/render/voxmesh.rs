use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use bxw_util::math::*;
use bxw_util::*;
use itertools::iproduct;
use std::iter::FromIterator;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Instant;
use world::registry::VoxelRegistry;
use world::*;

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

const fn cubed(a: usize) -> usize {
    a * a * a
}

pub fn is_chunk_trivial(chunk: &VChunk, registry: &VoxelRegistry) -> bool {
    let VChunkData::QuickCompressed { vox } = &chunk.data;
    if vox.len() == 3 && vox[0] == vox[1] {
        let vdef = registry.get_definition_from_id(VoxelDatum { id: vox[0] });
        if !vdef.has_mesh {
            return true;
        }
    }
    false
}

#[allow(clippy::cognitive_complexity)]
pub fn mesh_from_chunk(
    registry: &VoxelRegistry,
    chunks: &[Arc<VChunk>],
    _texture_dim: (u32, u32),
) -> Option<ChunkBuffers> {
    assert_eq!(chunks.len(), 27);
    let premesh = Instant::now();
    let ucchunks: Vec<Box<UncompressedChunk>> =
        Vec::from_iter(chunks.iter().map(|c| c.decompress()));
    // Safety: uninitialized array of MaybeUninits is safe
    const INFLATED_DIM: usize = CHUNK_DIM + 2;
    let mut vdefs: [MaybeUninit<&VoxelDefinition>; cubed(INFLATED_DIM)] =
        unsafe { MaybeUninit::uninit().assume_init() };
    for (y, z, x) in iproduct!(0..INFLATED_DIM, 0..INFLATED_DIM, 0..INFLATED_DIM) {
        let rcpos = vec3(x, y, z).map(|c| {
            if c == 0 {
                0
            } else if c <= CHUNK_DIM {
                1
            } else {
                2
            }
        });
        let cidx = rcpos.x + rcpos.z * 3 + rcpos.y * 9;
        let bpos = blockidx_from_blockpos(vec3(x as i32 - 1, y as i32 - 1, z as i32 - 1));
        let dat = ucchunks[cidx].blocks_yzx[bpos];
        let def = registry.get_definition_from_id(dat);
        let idx = x + z * INFLATED_DIM + y * INFLATED_DIM * INFLATED_DIM;
        unsafe {
            vdefs[idx].as_mut_ptr().write(def); // Safety: Initialize every element with a valid reference
        }
    }
    let vdefs: [&VoxelDefinition; cubed(INFLATED_DIM)] = unsafe { std::mem::transmute(vdefs) }; // Safety: The whole array is initialized in the above for loop

    // pos relative to Chunk@cpos
    #[inline(always)]
    fn get_block_idx(pos: Vector3<i32>) -> usize {
        (pos.x + 1) as usize
            + (pos.z + 1) as usize * INFLATED_DIM
            + (pos.y + 1) as usize * INFLATED_DIM * INFLATED_DIM
    };

    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();

    for (cell_y, cell_z, cell_x) in iproduct!(0..CHUNK_DIM, 0..CHUNK_DIM, 0..CHUNK_DIM) {
        let ipos = vec3(cell_x as i32, cell_y as i32, cell_z as i32);
        let vidx = blockidx_from_blockpos(ipos);
        let vdef = vdefs[get_block_idx(ipos)];

        if !vdef.has_mesh {
            continue;
        }

        for side in &SIDES {
            let ioffset = Vector3::from_row_slice(&side.ioffset);

            // hidden face removal
            let touchpos = ipos + ioffset;
            let tdef = vdefs[get_block_idx(touchpos)];
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
                    ao_c = vdefs[get_block_idx(p_c)].has_mesh;
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
                    ao_s1 = vdefs[get_block_idx(p_s1)].has_mesh;
                    ao_s2 = vdefs[get_block_idx(p_s2)].has_mesh;
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
                    ipos.x as f32 + side.verts[t * 3],
                    ipos.y as f32 + side.verts[t * 3 + 1],
                    ipos.z as f32 + side.verts[t * 3 + 2],
                    1.0,
                ];
                let texid;
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
                    texcoord: [
                        side.texcs[t * 2],
                        side.texcs[t * 2 + 1],
                        texid as f32,
                    ],
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

    let postmesh = Instant::now();
    let meshtime = postmesh.saturating_duration_since(premesh);
    bxw_util::debug_data::DEBUG_DATA
        .wmesh_times
        .push_ns(meshtime.as_nanos() as i64);

    Some(ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    })
}

const SIDES: [CubeSide; 6] = [
    // x+ -> "right"
    CubeSide {
        verts: [
            0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
        ],
        corners: [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]],
        ioffset: [1, 0, 0],
        texcs: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    },
    // x- -> "left"
    CubeSide {
        verts: [
            -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
        ],
        corners: [[-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [-1, -1, -1]],
        ioffset: [-1, 0, 0],
        texcs: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    },
    // y+ -> "top"
    CubeSide {
        verts: [
            -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
        ],
        corners: [[-1, 1, -1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]],
        ioffset: [0, 1, 0],
        texcs: [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    },
    // y- -> "bottom"
    CubeSide {
        verts: [
            0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
        ],
        corners: [[1, -1, -1], [1, -1, 1], [-1, -1, 1], [-1, -1, -1]],
        ioffset: [0, -1, 0],
        texcs: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    },
    // z+ -> "back"
    CubeSide {
        verts: [
            0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
        ],
        corners: [[1, -1, 1], [1, 1, 1], [-1, 1, 1], [-1, -1, 1]],
        ioffset: [0, 0, 1],
        texcs: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    },
    // z- -> "front"
    CubeSide {
        verts: [
            -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
        ],
        corners: [[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]],
        ioffset: [0, 0, -1],
        texcs: [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    },
];
