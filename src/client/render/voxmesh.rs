use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use crate::world::registry::VoxelRegistry;
use crate::world::{VoxelChunk, VOXEL_CHUNK_DIM};
use cgmath::{vec3, Vector3};

struct CubeSide {
    // counter-clockwise coords of the face
    pub verts: [f32; 3 * 4],
    // corners of the side, matching up with `verts` above
    pub corners: [Vector3<i32>; 4],
    // what to add to position to find neighbor
    pub ioffset: Vector3<i32>,
    // texture coordinates (u,v) matched with vertex coordinates
    pub texcs: [f32; 2 * 4],
}

pub fn mesh_from_chunk(chunk: &VoxelChunk, registry: &VoxelRegistry) -> Option<ChunkBuffers> {
    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();
    let neighbors: Vec<_> = chunk.neighbor.iter().map(|n| n.upgrade()).collect();
    if !neighbors.iter().all(|n| n.is_some()) {
        return None;
    }
    let neighbors_locks: Vec<_> = neighbors.into_iter().map(|o| o.unwrap()).collect();
    let neighbors: Vec<_> = neighbors_locks.iter().map(|l| l.read().unwrap()).collect();

    const SIDES: [CubeSide; 6] = [
        // x+ -> "right"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
            ],
            corners: [
                vec3(1, -1, -1),
                vec3(1, 1, -1),
                vec3(1, 1, 1),
                vec3(1, -1, 1),
            ],
            ioffset: vec3(1, 0, 0),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // x- -> "left"
        CubeSide {
            verts: [
                -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
            ],
            corners: [
                vec3(-1, -1, 1),
                vec3(-1, 1, 1),
                vec3(-1, 1, -1),
                vec3(-1, -1, -1),
            ],
            ioffset: vec3(-1, 0, 0),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // y+ -> "bottom"
        CubeSide {
            verts: [
                -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
            ],
            corners: [
                vec3(-1, 1, -1),
                vec3(-1, 1, 1),
                vec3(1, 1, 1),
                vec3(1, 1, -1),
            ],
            ioffset: vec3(0, 1, 0),
            texcs: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        // y- -> "top"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
            ],
            corners: [
                vec3(1, -1, 0),
                vec3(1, -1, 1),
                vec3(-1, -1, 1),
                vec3(-1, -1, -1),
            ],
            ioffset: vec3(0, -1, 0),
            texcs: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        // z+ -> "back"
        CubeSide {
            verts: [
                0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
            ],
            corners: [
                vec3(1, -1, 1),
                vec3(1, 1, 1),
                vec3(-1, 1, 1),
                vec3(-1, -1, 1),
            ],
            ioffset: vec3(0, 0, 1),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // z- -> "front"
        CubeSide {
            verts: [
                -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
            ],
            corners: [
                vec3(-1, -1, -1),
                vec3(-1, 1, -1),
                vec3(1, 1, -1),
                vec3(1, -1, -1),
            ],
            ioffset: vec3(0, 0, -1),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    ];

    let get_vox = |vpos: Vector3<i32>| {
        let a = VOXEL_CHUNK_DIM as i32;
        let va: [i32; 3] = vpos.into();
        if !va
            .iter()
            .any(|c| (*c) < 0 || (*c) >= VOXEL_CHUNK_DIM as i32)
        {
            let tidx = (vpos.x + vpos.y * a + vpos.z * a * a) as usize;
            chunk.data[tidx]
        } else {
            let mut sub = va.iter().copied().map(|c| {
                if c < 0 {
                    -1
                } else if c >= a {
                    1
                } else {
                    0
                }
            });
            let x: i32 = sub.next().unwrap();
            let y: i32 = sub.next().unwrap();
            let z: i32 = sub.next().unwrap();
            let cdiff = vec3(x, y, z);
            let vpos: Vector3<i32> = vpos - a * cdiff;
            let tidx = (vpos.x + vpos.y * a + vpos.z * a * a) as usize;
            let nidx = ((x + 1) + 3 * (y + 1) + 9 * (z + 1)) as usize;
            neighbors[nidx].data[tidx]
        }
    };

    for side in &SIDES {
        for (vidx, vox) in chunk.data.iter().enumerate() {
            let vdef = registry.get_definition_from_id(&vox);

            let ipos = vec3(
                (vidx % VOXEL_CHUNK_DIM) as i32,
                ((vidx / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as i32,
                ((vidx / VOXEL_CHUNK_DIM / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as i32,
            );
            let x = ipos.x as f32;
            let y = ipos.y as f32;
            let z = ipos.z as f32;

            if !vdef.has_mesh {
                continue;
            }

            // hidden face removal
            let touchpos = ipos + side.ioffset;
            let tid = get_vox(touchpos);
            let tdef = registry.get_definition_from_id(&tid);
            if tdef.has_mesh {
                continue;
            }

            let voff = vbuf.len() as u32;
            let mut corner_ao = [0i32; 4];
            for t in 0..4 {
                // AO calculation
                let corner = side.corners[t];
                let (ao_s1, ao_s2, ao_c): (bool, bool, bool);
                {
                    let p_c = ipos + corner;
                    ao_c = registry.get_definition_from_id(&get_vox(p_c)).has_mesh;
                    let (p_s1, p_s2);
                    if side.ioffset.x != 0 {
                        // y,z sides
                        p_s1 = ipos + vec3(corner.x, corner.y, 0);
                        p_s2 = ipos + vec3(corner.x, 0, corner.z);
                    } else if side.ioffset.y != 0 {
                        // x,z sides
                        p_s1 = ipos + vec3(0, corner.y, corner.z);
                        p_s2 = ipos + vec3(corner.x, corner.y, 0);
                    } else {
                        // x,y sides
                        p_s1 = ipos + vec3(corner.x, 0, corner.z);
                        p_s2 = ipos + vec3(0, corner.y, corner.z);
                    }
                    ao_s1 = registry.get_definition_from_id(&get_vox(p_s1)).has_mesh;
                    ao_s2 = registry.get_definition_from_id(&get_vox(p_s2)).has_mesh;
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
                        if side.ioffset.y == 1 {
                            texid = *top;
                        } else if side.ioffset.y == -1 {
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
