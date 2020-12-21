use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use bxw_util::direction::OctahedralOrientation;
use bxw_util::math::*;
use bxw_util::*;
use bxw_world::blocks::stdshapes::*;
use bxw_world::voxregistry::VoxelRegistry;
use bxw_world::*;
use itertools::iproduct;
use std::sync::Arc;
use std::time::Instant;

const fn cubed(a: usize) -> usize {
    a * a * a
}

pub fn is_chunk_trivial(chunk: &VChunk, registry: &VoxelRegistry) -> bool {
    let VChunkData::QuickCompressed { vox } = &chunk.data;
    if vox.len() == 3 && vox[0] == vox[1] {
        let vdef = registry.get_definition_from_datum(VoxelDatum::from_repr(vox[0]));
        if vdef.mesh.is_none() {
            return true;
        }
    }
    false
}

const AO_OCCLUSION_FACTOR: f32 = 0.88;

#[allow(clippy::cognitive_complexity)]
pub fn mesh_from_chunk(
    registry: &VoxelRegistry,
    chunks: &[Arc<VChunk>],
    _texture_dim: (u32, u32),
) -> Option<ChunkBuffers> {
    assert_eq!(chunks.len(), 27);
    let premesh = Instant::now();
    let ucchunks: Vec<Box<UncompressedChunk>> = chunks.iter().map(|c| c.decompress()).collect();
    const INFLATED_DIM: usize = CHUNK_DIM + 2;
    const INFLATED_DIM2: usize = INFLATED_DIM * INFLATED_DIM;
    let mut vdecoded: Vec<(
        VoxelDatum,
        &VoxelDefinition,
        &VoxelShapeDef,
        OctahedralOrientation,
    )> = Vec::with_capacity(cubed(INFLATED_DIM));
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
        let vdat = ucchunks[cidx].blocks_yzx[bpos];
        let vdef = registry.get_definition_from_datum(vdat);
        let vshp = block_shape(vdat, vdef);
        let vor = block_orientation(vdat, vdef);
        vdecoded.push((vdat, vdef, vshp, vor));
    }

    // pos relative to Chunk@cpos
    #[inline(always)]
    fn get_block_idx(pos: Vector3<i32>) -> usize {
        (pos.x + 1) as usize
            + (pos.z + 1) as usize * INFLATED_DIM
            + (pos.y + 1) as usize * INFLATED_DIM2
    };

    let mut vbuf: Vec<VoxelVertex> = Vec::with_capacity(6144);
    let mut ibuf: Vec<u32> = Vec::with_capacity(6144);

    for (cell_y, cell_z, cell_x) in iproduct!(0..CHUNK_DIM, 0..CHUNK_DIM, 0..CHUNK_DIM) {
        let ipos = vec3(cell_x as i32, cell_y as i32, cell_z as i32);
        let vidx = get_block_idx(ipos);
        let ic_vidx = blockidx_from_blockpos(ipos);
        let (_vdat, vdef, vshape, vor) = vdecoded[vidx];

        if vdef.mesh.is_none() {
            continue;
        }

        for &side_dir in &ALL_DIRS {
            let rot_side_dir = vor.unapply_to_dir(side_dir);
            let side = &vshape.sides[rot_side_dir.to_signed_axis_index()];
            if side.indices.is_empty() {
                continue;
            }
            let ioffset = side_dir.to_vec();

            // hidden face removal
            let touchside = side_dir.opposite();
            let touchpos = ipos + ioffset;
            let (_tdat, _tdef, tshape, tor) = vdecoded[get_block_idx(touchpos)];
            let touchrotside = tor.unapply_to_dir(touchside);
            let tside = &tshape.sides[touchrotside.to_signed_axis_index()];

            if side.can_be_clipped && tside.can_clip {
                continue;
            }

            let voff = vbuf.len() as u32;
            let mut barycentric_color_sum: Vector4<f32> = zero();
            let vor_mati = vor.to_matrixi();
            let vor_matf = vor_mati.map(|x| x as f32);
            for vtx in side.vertices.iter() {
                // AO calculation
                let mut ao = 1.0;
                for &ao_off in vtx.ao_offsets.iter() {
                    let pos = ipos + vor_mati * ao_off;
                    let idx = get_block_idx(pos);
                    let (_dat, _def, shp, _or) = vdecoded[idx];
                    if shp.causes_ambient_occlusion {
                        ao *= AO_OCCLUSION_FACTOR;
                    }
                }

                let voffset = vor_matf * vtx.offset;
                let position: [f32; 4] = [
                    ipos.x as f32 + voffset.x,
                    ipos.y as f32 + voffset.y,
                    ipos.z as f32 + voffset.z,
                    1.0,
                ];
                let texid = *vdef.texture_mapping.at_direction(rot_side_dir);
                let color = [
                    vdef.debug_color[0] * ao,
                    vdef.debug_color[1] * ao,
                    vdef.debug_color[2] * ao,
                    1.0,
                ];
                barycentric_color_sum += vtx.barycentric_sign as f32 * Vector4::from(color);
                vbuf.push(VoxelVertex {
                    position,
                    color,
                    texcoord: [vtx.texcoord.x, vtx.texcoord.y, texid as f32],
                    index: ic_vidx as i32,
                    barycentric_color_offset: [0.0; 4], // initialized after the loop
                    barycentric: [vtx.barycentric.x, vtx.barycentric.y, vtx.barycentric.z],
                });
            }
            for v in &mut vbuf[voff as usize..] {
                v.barycentric_color_offset = barycentric_color_sum.into();
            }
            ibuf.extend(side.indices.iter().map(|x| x + voff));
        }
    }

    let postmesh = Instant::now();
    let meshtime = postmesh.saturating_duration_since(premesh);
    bxw_util::debug_data::DEBUG_DATA
        .wmesh_times
        .push_ns(meshtime.as_nanos() as i64);
    vbuf.shrink_to_fit();
    ibuf.shrink_to_fit();
    Some(ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    })
}
