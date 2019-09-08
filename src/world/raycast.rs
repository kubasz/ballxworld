use crate::math::*;
use crate::world::{
    blockidx_from_blockpos, chunkpos_from_blockpos, BlockPosition, Direction, VoxelDatum,
    WEntities, WVoxels, CHUNK_DIM,
};

#[derive(Clone)]
pub struct RaycastQuery<'q> {
    pub start_point: Vector3<f32>,
    pub direction: Vector3<f32>,
    pub distance_limit: f32,
    pub hit_voxels: Option<&'q WVoxels>,
    pub hit_entities: Option<&'q WEntities>,
}

#[derive(Copy, Clone)]
pub enum Hit {
    Nothing,
    Voxel {
        position: BlockPosition,
        datum: VoxelDatum,
        normal: Direction,
        normal_datum: Option<VoxelDatum>,
    },
    Entity,
}

impl<'q> Default for Hit {
    fn default() -> Self {
        Hit::Nothing
    }
}

#[derive(Clone)]
pub struct RaycastResult {
    pub hit: Hit,
    pub distance: f32,
}

impl<'q> RaycastQuery<'q> {
    pub fn new_directed(
        start_point: Vector3<f32>,
        direction: Vector3<f32>,
        distance_limit: f32,
        hit_voxels: Option<&'q WVoxels>,
        hit_entities: Option<&'q WEntities>,
    ) -> Self {
        Self {
            start_point,
            direction,
            distance_limit,
            hit_voxels,
            hit_entities,
        }
    }

    pub fn new_oriented(
        start_point: Vector3<f32>,
        orientation: UnitQuaternion<f32>,
        distance_limit: f32,
        hit_voxels: Option<&'q WVoxels>,
        hit_entities: Option<&'q WEntities>,
    ) -> Self {
        let direction = glm::quat_rotate_vec3(&orientation, &Vector3::z_axis());
        Self {
            start_point,
            direction,
            distance_limit,
            hit_voxels,
            hit_entities,
        }
    }

    pub fn execute(&self) -> RaycastResult {
        let distance_limit = self.distance_limit.min(1000.0);
        let direction = self
            .direction
            .map(|c| if c == 0.0 { std::f32::EPSILON } else { c });

        // fast voxel traversal
        // https://www.gamedev.net/blogs/entry/2265248-voxel-traversal-algorithm-ray-casting/
        // http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.3443&rep=rep1&type=pdf
        if let Some(voxels) = self.hit_voxels {
            let offset_start: Vector3<f32> = self.start_point + vec3(0.5, 0.5, 0.5);
            let mut bpos: Vector3<i32> = offset_start.map(|c| c.floor() as i32);
            let mut cpos: Vector3<i32> = chunkpos_from_blockpos(bpos);
            let end_bpos: Vector3<i32> =
                (offset_start + direction * self.distance_limit).map(|c| c.floor() as i32);
            let iters: i32 = (end_bpos - bpos).iter().map(|c| c.abs()).sum::<i32>() + 1;

            let step: Vector3<i32> = direction.map(|c| c.signum() as i32);
            let next_vox_boundary: Vector3<f32> =
                (bpos + step.map(|c| if c >= 0 { 1 } else { 0 })).map(|c| c as f32);
            let mut t_max: Vector3<f32> =
                (next_vox_boundary - offset_start).component_div(&direction);
            let t_delta: Vector3<f32> = step.map(|c| c as f32).component_div(&direction);
            let normals: [Direction; 3] = [
                if step.x < 0 {
                    Direction::XPlus
                } else {
                    Direction::XMinus
                },
                if step.y < 0 {
                    Direction::YPlus
                } else {
                    Direction::YMinus
                },
                if step.z < 0 {
                    Direction::ZPlus
                } else {
                    Direction::ZMinus
                },
            ];
            let mut normal = normals[0];

            let ichunk_dim = CHUNK_DIM as i32;
            bpos -= cpos * ichunk_dim;
            let mut chunk = voxels.get_uncompressed_chunk(cpos);
            let mut normal_datum = None;

            for _ in 0..iters {
                // check block
                if let Some(chunk) = &chunk {
                    let bidx = blockidx_from_blockpos(bpos);
                    let datum = chunk.blocks_yzx[bidx];
                    let vdef = voxels.registry.get_definition_from_id(datum);
                    if vdef.has_hitbox {
                        let position = bpos + cpos * ichunk_dim;
                        let distance = (position.map(|c| c as f32) - self.start_point).magnitude();
                        // hit!
                        return RaycastResult {
                            hit: Hit::Voxel {
                                position,
                                datum,
                                normal,
                                normal_datum,
                            },
                            distance,
                        };
                    }
                    normal_datum = Some(datum);
                } else {
                    normal_datum = None;
                }

                // move to next block
                if t_max.x < t_max.y && t_max.x < t_max.z {
                    bpos.x += step.x;
                    if bpos.x < 0 || bpos.x >= ichunk_dim {
                        bpos.x -= ichunk_dim * step.x;
                        cpos.x += step.x;
                        chunk = voxels.get_uncompressed_chunk(cpos);
                    }
                    t_max.x += t_delta.x;
                    normal = normals[0];
                } else if t_max.y < t_max.z {
                    bpos.y += step.y;
                    t_max.y += t_delta.y;
                    if bpos.y < 0 || bpos.y >= ichunk_dim {
                        bpos.y -= ichunk_dim * step.y;
                        cpos.y += step.y;
                        chunk = voxels.get_uncompressed_chunk(cpos);
                    }
                    normal = normals[1];
                } else {
                    bpos.z += step.z;
                    t_max.z += t_delta.z;
                    if bpos.z < 0 || bpos.z >= ichunk_dim {
                        bpos.z -= ichunk_dim * step.z;
                        cpos.z += step.z;
                        chunk = voxels.get_uncompressed_chunk(cpos);
                    }
                    normal = normals[2];
                }
            }
        }

        RaycastResult {
            hit: Hit::Nothing,
            distance: distance_limit,
        }
    }
}
