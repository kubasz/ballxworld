use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use bxw_util::math::*;
use bxw_util::*;
use itertools::iproduct;
use std::mem::MaybeUninit;
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

#[allow(clippy::cognitive_complexity)]
pub fn mesh_from_chunk(
    world: &World,
    cpos: ChunkPosition,
    texture_dim: (u32, u32),
) -> Option<ChunkBuffers> {
    let voxels = world.voxels.read();
    let dirty = voxels.chunks.get(&cpos).map(|c| c.dirty).unwrap_or(0);
    let registry: &VoxelRegistry = &world.vregistry;
    {
        let chk = voxels.chunks.get(&cpos)?;
        let VChunkData::QuickCompressed { vox } = &chk.data;
        if vox.len() == 3 && vox[0] == vox[1] {
            let vdef = registry.get_definition_from_id(VoxelDatum { id: vox[0] });
            if !vdef.has_mesh {
                return Some(ChunkBuffers {
                    indices: Vec::new(),
                    vertices: Vec::new(),
                    dirty,
                });
            }
        }
    }
    // only time non-trivial chunks
    let premesh = Instant::now();
    let mut vcache = world.get_vcache();
    // Safety: uninitialized array of MaybeUninits is safe
    const INFLATED_DIM: usize = CHUNK_DIM + 2;
    let mut vdefs: [MaybeUninit<&VoxelDefinition>; cubed(INFLATED_DIM)] =
        unsafe { MaybeUninit::uninit().assume_init() };
    for (y, z, x) in iproduct!(-1..=1, -1..=1, -1..=1) {
        vcache.ensure_newest_cached(&voxels, cpos + vec3(x, y, z))?;
    }
    drop(voxels);
    let cbpos = cpos * CHUNK_DIM as i32;
    for (y, z, x) in iproduct!(0..INFLATED_DIM, 0..INFLATED_DIM, 0..INFLATED_DIM) {
        let dat = vcache
            .peek_block(cbpos + vec3(x as i32 - 1, y as i32 - 1, z as i32 - 1))
            .unwrap();
        let def = registry.get_definition_from_id(dat);
        let idx = x + z * INFLATED_DIM + y * INFLATED_DIM * INFLATED_DIM;
        unsafe {
            vdefs[idx].as_mut_ptr().write(def); // Safety: Initialize every element with a valid reference
        }
    }
    let vdefs: [&VoxelDefinition; cubed(INFLATED_DIM)] = unsafe { std::mem::transmute(vdefs) }; // Safety: The whole array is initialized in the above for loop
                                                                                                // pos relative to Chunk@cpos
    let get_block = |pos: Vector3<i32>| {
        vdefs[(pos.x + 1) as usize
            + (pos.z + 1) as usize * INFLATED_DIM
            + (pos.y + 1) as usize * INFLATED_DIM * INFLATED_DIM]
    };

    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();

    // half-pixel offsets for texels
    let hpo_x = 0.5 / (texture_dim.0 as f32);
    let hpo_y = 0.5 / (texture_dim.1 as f32);
    let apply_hpo = |tc: f32, hpo: f32| {
        if (tc - 0.0).abs() < 1.0e-6 {
            hpo
        } else if (tc - 1.0).abs() < 1.0e-6 {
            1.0f32 - hpo
        } else {
            tc
        }
    };

    for side in &SIDES {
        for (cell_y, cell_z, cell_x) in iproduct!(0..CHUNK_DIM, 0..CHUNK_DIM, 0..CHUNK_DIM) {
            let ioffset = Vector3::from_row_slice(&side.ioffset);

            let ipos = vec3(cell_x as i32, cell_y as i32, cell_z as i32);
            let vidx = blockidx_from_blockpos(ipos);
            let vdef = get_block(ipos);

            if !vdef.has_mesh {
                continue;
            }

            // hidden face removal
            let touchpos = ipos + ioffset;
            let tdef = get_block(touchpos);
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
                    ao_c = get_block(p_c).has_mesh;
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
                    ao_s1 = get_block(p_s1).has_mesh;
                    ao_s2 = get_block(p_s2).has_mesh;
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
                        apply_hpo(side.texcs[t * 2], hpo_x),
                        apply_hpo(side.texcs[t * 2 + 1], hpo_y),
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
        dirty,
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
