use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use bxw_util::*;
use itertools::iproduct;
use math::*;
use smallvec::SmallVec;
use world::registry::VoxelRegistry;
use world::{
    blockidx_from_blockpos, chunkpos_from_blockpos, ChunkPosition, UncompressedChunk, VChunkData,
    VoxelDatum, World, CHUNK_DIM,
};

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

#[allow(clippy::cognitive_complexity)]
pub fn mesh_from_chunk(
    world: &World,
    cpos: ChunkPosition,
    lod: u32,
    texture_dim: (u32, u32),
) -> Option<ChunkBuffers> {
    debug_assert!((1u32 << lod) <= CHUNK_DIM as u32);
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
    let mut vcache = world.get_vcache();
    let mut chunks: SmallVec<[&UncompressedChunk; 32]> = SmallVec::new();
    for (y, z, x) in iproduct!(-1..=1, -1..=1, -1..=1) {
        vcache.ensure_newest_cached(&voxels, cpos + vec3(x, y, z))?;
    }
    drop(voxels);
    for (y, z, x) in iproduct!(-1..=1, -1..=1, -1..=1) {
        chunks.push(
            vcache
                .peek_uncompressed_chunk(cpos + vec3(x, y, z))
                .unwrap(),
        );
    }
    // pos relative to Chunk@cpos
    let get_block = |pos: Vector3<i32>| {
        let cp = chunkpos_from_blockpos(pos) + vec3(1, 1, 1);
        if lod != 0 && cp != vec3(1, 1, 1) {
            return VoxelDatum::default();
        }
        let ch: &UncompressedChunk = &chunks[(cp.x + cp.z * 3 + cp.y * 9) as usize];
        ch.blocks_yzx[blockidx_from_blockpos(pos)]
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

    let cells = (CHUNK_DIM >> lod as usize) as u32;
    let subcells = (1 << lod) as u32;
    for side in &SIDES {
        let suborder = {
            let scs = subcells as i32;
            let mut svec: Vec<Vector3<i32>> = iproduct!(0..scs, 0..scs, 0..scs)
                .map(|(x, y, z)| vec3(x, y, z))
                .collect();
            let ioff = side.ioffset;
            let (dim, dval) = if ioff[0] != 0 {
                (0, ioff[0])
            } else if ioff[1] != 0 {
                (1, ioff[1])
            } else {
                (2, ioff[2])
            };
            svec.sort_by_key(|e| if dval < 0 { e[dim] } else { -e[dim] });
            svec
        };
        for (cell_y, cell_z, cell_x) in iproduct!(0..cells, 0..cells, 0..cells) {
            let (co_y, co_z, co_x) = (cell_y << lod, cell_z << lod, cell_x << lod);
            let (cf_x, cf_y, cf_z) = (co_x as f32, co_y as f32, co_z as f32);

            let ioffset = Vector3::from_row_slice(&side.ioffset);

            let mut ipos = vec3(co_x as i32, co_y as i32, co_z as i32);
            'subfinder: for subp in suborder.iter() {
                ipos = vec3(co_x as i32, co_y as i32, co_z as i32) + subp;
                if registry.get_definition_from_id(get_block(ipos)).has_mesh {
                    break 'subfinder;
                }
            }
            let vox = get_block(ipos);
            let vidx = blockidx_from_blockpos(ipos);
            let vdef = registry.get_definition_from_id(vox);
            //let x = ipos.x as f32;
            //let y = ipos.y as f32;
            //let z = ipos.z as f32;

            if !vdef.has_mesh {
                continue;
            }

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

                let scf = subcells as f32;
                let hf = if lod == 0 { 0.0 } else { scf / 2.0 };
                let position: [f32; 4] = [
                    cf_x + hf + scf * side.verts[t * 3],
                    cf_y + hf + scf * side.verts[t * 3 + 1],
                    cf_z + hf + scf * side.verts[t * 3 + 2],
                    1.0,
                ];
                let texid;
                use world::TextureMapping;
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
