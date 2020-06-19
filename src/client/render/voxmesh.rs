use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use bxw_util::math::*;
use bxw_util::*;
use bxw_world::blocks::stdshapes::*;
use bxw_world::registry::VoxelRegistry;
use bxw_world::*;
use itertools::iproduct;
use std::iter::FromIterator;
use std::mem::MaybeUninit;
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
    let ucchunks: Vec<Box<UncompressedChunk>> =
        Vec::from_iter(chunks.iter().map(|c| c.decompress()));
    // Safety: uninitialized array of MaybeUninits is safe
    const INFLATED_DIM: usize = CHUNK_DIM + 2;
    let mut vdata: [MaybeUninit<VoxelDatum>; cubed(INFLATED_DIM)] =
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
        let idx = x + z * INFLATED_DIM + y * INFLATED_DIM * INFLATED_DIM;
        unsafe {
            vdata[idx].as_mut_ptr().write(dat); // Safety: Initialize every element with a valid data copy
        }
    }
    let vdefs: [VoxelDatum; cubed(INFLATED_DIM)] = unsafe { std::mem::transmute(vdata) }; // Safety: The whole array is initialized in the above for loop

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
        let vdat = vdefs[get_block_idx(ipos)];
        let vdef = registry.get_definition_from_datum(vdat);

        if vdef.mesh.is_none() {
            continue;
        }

        let vshape = block_shape(vdat, vdef);

        for side_dir in &ALL_DIRS {
            let side = &vshape.sides[side_dir.to_signed_axis_index()];
            if side.indices.is_empty() {
                continue;
            }
            let ioffset = side_dir.to_vec();

            // hidden face removal
            let touchside = side_dir.opposite();
            let touchpos = ipos + ioffset;
            let tdat = vdefs[get_block_idx(touchpos)];
            let tdef = registry.get_definition_from_datum(tdat);
            let tshape = block_shape(tdat, tdef);
            let tside = &tshape.sides[touchside.to_signed_axis_index()];
            if side.can_be_clipped && tside.can_clip {
                continue;
            }

            let voff = vbuf.len() as u32;
            let mut barycentric_color_sum: Vector4<f32> = zero();
            for vtx in side.vertices.iter() {
                // AO calculation
                let mut ao = 1.0;
                for ao_off in vtx.ao_offsets.iter() {
                    let pos = ipos + ao_off;
                    let idx = get_block_idx(pos);
                    let dat = vdefs[idx];
                    let def = registry.get_definition_from_datum(dat);
                    let shp = block_shape(dat, def);
                    if shp.causes_ambient_occlusion {
                        ao *= AO_OCCLUSION_FACTOR;
                    }
                }

                let position: [f32; 4] = [
                    ipos.x as f32 + vtx.offset.x,
                    ipos.y as f32 + vtx.offset.y,
                    ipos.z as f32 + vtx.offset.z,
                    1.0,
                ];
                let texid = *vdef
                    .texture_mapping
                    .at_direction(Direction::try_from_vec(ioffset).unwrap());
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
                    index: vidx as i32,
                    barycentric_color_offset: [0.0; 4], // initialized after the loop
                    barycentric: [vtx.barycentric.x, vtx.barycentric.y],
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

    Some(ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    })
}
