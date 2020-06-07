use crate::ecs::*;
//use crate::raycast::*;
use crate::generation::WorldBlocks;
use crate::worldmgr::*;
use crate::{blockpos_from_worldpos, chunkpos_from_blockpos, Direction};
use bxw_util::collider::AABB;
use bxw_util::math::*;
use bxw_util::*;
use itertools::Itertools;
use std::time::Instant;

pub const TIMESTEP: f64 = 1.0 / 60.0;
pub const SPEED_LIMIT_MPS: f64 = 1000.0;
pub const SPEED_LIMIT_MPS_SQ: f64 = SPEED_LIMIT_MPS * SPEED_LIMIT_MPS;
pub const DESUFFOCATION_SPEED: f64 = 10.0;
pub const GRAVITY_ACCEL: f64 = 30.0;
pub const AIR_FRICTION_SQ: f64 = 0.5 * 0.5 * 1.2;
// 1/2 * pyramid drag coefficient * air density
pub const SMALL_V_CUTOFF: f64 = 1.0e-6;
pub const WORLD_LIMIT: f64 = i32::max_value() as f64 / 4.0;
pub const TOUCH_EPSILON: f64 = 1.0e-2;

fn check_suffocation(world: &World, voxels: &WorldBlocks, position: Vector3<f64>) -> bool {
    let bpos = blockpos_from_worldpos(position);
    let bidx = voxels.get_vcache().get_block(world, voxels, bpos);
    if let Some(bidx) = bidx {
        let vdef = voxels.voxel_registry.get_definition_from_id(bidx);
        vdef.has_hitbox
    } else {
        false
    }
}

fn drag_force(loc: &CLocation, velocity: Vector3<f64>) -> Vector3<f64> {
    let area_est = match loc.bounding_shape {
        BoundingShape::Point { .. } => 0.05,
        BoundingShape::AxisAlignedBox(aabb) => aabb.size().x * aabb.size().y,
    };
    velocity.component_mul(&velocity) * AIR_FRICTION_SQ * area_est
}

pub fn world_physics_tick(world: &World) {
    let voxels = world.get_handler(CHUNK_BLOCK_DATA).borrow();
    let voxels = voxels.as_any().downcast_ref::<WorldBlocks>().unwrap();
    let mut entities = world.ecs().write();
    let pretick = Instant::now();
    let mut intersections = Vec::with_capacity(10);
    for (phys, loc) in entities.iter_mut_physics() {
        if phys.frozen {
            continue;
        }
        let mass = phys.mass;

        let old_pos = loc.position;
        let old_cpos = chunkpos_from_blockpos(blockpos_from_worldpos(old_pos));
        // don't calculate physics where blocks aren't loaded yet
        if voxels.get_chunk(world, old_cpos).is_none() {
            continue;
        }
        let old_vel = loc.velocity;
        let old_aabb = loc.bounding_shape.aabb(old_pos);
        let mut new_accel = vec3(0.0, 0.0, 0.0);
        // determine wall contacts

        phys.against_wall = determine_wall_contacts(old_aabb, world, &voxels);

        // air friction
        new_accel -= drag_force(loc, old_vel) / mass;
        // gravity
        new_accel += vec3(0.0, -GRAVITY_ACCEL, 0.0);
        // control impulse
        new_accel += phys.control_frame_impulse;
        phys.control_frame_impulse /= 2.0; // exponential backoff
        if phys.control_frame_impulse.magnitude_squared() < SMALL_V_CUTOFF * SMALL_V_CUTOFF {
            phys.control_frame_impulse = zero();
        }
        // control force
        let control_dv = phys.control_target_velocity - old_vel;
        let control_da = {
            let mut control_da = control_dv / TIMESTEP;
            for comp in 0..3 {
                control_da[comp] = control_da[comp]
                    .min(phys.control_max_force[comp] / mass)
                    .max(-phys.control_max_force[comp] / mass);
            }
            control_da
        };
        new_accel += control_da;
        let pred_vel = old_vel + new_accel * TIMESTEP;
        // don't accelerate towards a wall if you're already touching it
        for axis in 0..3 {
            if new_accel[axis] < 0.0 && pred_vel[axis] <= 0.0 && phys.against_wall[axis * 2]
                || new_accel[axis] > 0.0 && pred_vel[axis] >= 0.0 && phys.against_wall[axis * 2 + 1]
            {
                new_accel[axis] = 0.0;
            }
        }
        // velocity integration
        let mut new_vel = old_vel + new_accel * TIMESTEP;
        for axis in 0..new_vel.len() {
            if new_vel[axis].abs() < SMALL_V_CUTOFF {
                new_vel[axis] = 0.0;
            }
            // don't move towards walls
            if new_vel[axis] < 0.0 && phys.against_wall[axis * 2]
                || new_vel[axis] > 0.0 && phys.against_wall[axis * 2 + 1]
            {
                new_vel[axis] = 0.0;
            }
        }
        if new_vel.magnitude_squared() > SPEED_LIMIT_MPS_SQ {
            new_vel = new_vel.normalize() * SPEED_LIMIT_MPS;
        }
        // position integration with collision correction
        let mut new_pos = old_pos + new_vel * TIMESTEP;
        {
            for _iter in 0..4 {
                let new_aabb = loc.bounding_shape.aabb(new_pos);
                if !aabb_voxel_intersection(new_aabb, world, &voxels, Some(&mut intersections)) {
                    break;
                }
                // find the largest intersection volume and use as the heuristic for the most important intersection
                let maxbox = intersections.iter().max_by(|&a, &b| {
                    a.volume()
                        .partial_cmp(&b.volume())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                if let Some(maxbox) = maxbox {
                    // move away along axis with largest perpendicular surface area (which is the axis with the smallest dimension)
                    let smaxis = maxbox.smallest_axis();
                    let dist = maxbox.size()[smaxis]
                        * if maxbox.center()[smaxis] < new_aabb.center()[smaxis] {
                            1.0
                        } else {
                            -1.0
                        };
                    // clamp to move max one block size
                    let dist = dist.max(-1.0).min(1.0);
                    if dist != 0.0 {
                        new_pos[smaxis] += dist;
                        if new_vel[smaxis].signum() == -dist.signum() {
                            new_vel[smaxis] = 0.0;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        // check for suffocation
        let new_suffocation = check_suffocation(world, &voxels, new_pos);
        if new_suffocation {
            let old_suffocation = check_suffocation(world, &voxels, old_pos);
            if old_suffocation {
                use Direction::*;
                let mut desuf_dir = YPlus;
                for dir in &[YPlus, YMinus, XMinus, XPlus, ZMinus, ZPlus] {
                    let dir_suffocation =
                        check_suffocation(world, &voxels, old_pos + dir.to_vec().map(|c| c as f64));
                    if !dir_suffocation {
                        desuf_dir = *dir;
                        break;
                    }
                }
                new_vel = desuf_dir.to_vec().map(|c| c as f64) * DESUFFOCATION_SPEED;
                new_pos = old_pos + new_vel * TIMESTEP;
            } else {
                new_pos = old_pos;
                new_vel = zero();
            }
        }
        // store new values
        loc.position = new_pos;
        loc.velocity = new_vel;
        // clamp position to avoid any overflow problems
        for c in 0..3 {
            loc.position[c] = loc.position[c].max(-WORLD_LIMIT).min(WORLD_LIMIT);
        }
    }

    let posttick = Instant::now();
    let durtick = posttick.saturating_duration_since(pretick);
    bxw_util::debug_data::DEBUG_DATA
        .phys_times
        .push_ns(durtick.as_nanos() as i64);
}

fn determine_wall_contacts(aabb: AABB, world: &World, voxels: &WorldBlocks) -> [bool; 6] {
    let mut contacts = [false; 6];
    for axis in 0..3 {
        let mut minaabb = aabb;
        let mut maxaabb = aabb;
        for maxis in 0..3 {
            if maxis == axis {
                minaabb.maxs[maxis] = minaabb.mins[maxis] + TOUCH_EPSILON;
                minaabb.mins[maxis] -= TOUCH_EPSILON;
                maxaabb.mins[maxis] = maxaabb.maxs[maxis] - TOUCH_EPSILON;
                maxaabb.maxs[maxis] += TOUCH_EPSILON;
            } else {
                minaabb.mins[maxis] += TOUCH_EPSILON;
                minaabb.maxs[maxis] -= TOUCH_EPSILON;
                maxaabb.mins[maxis] += TOUCH_EPSILON;
                maxaabb.maxs[maxis] -= TOUCH_EPSILON;
            }
        }
        contacts[axis * 2] = aabb_voxel_intersection(minaabb, world, &voxels, None);
        contacts[axis * 2 + 1] = aabb_voxel_intersection(maxaabb, world, &voxels, None);
    }
    contacts
}

pub fn aabb_voxel_intersection(
    entity_aabb: AABB,
    world: &World,
    voxels: &WorldBlocks,
    mut out_intersections: Option<&mut Vec<AABB>>,
) -> bool {
    if let Some(ref mut out) = out_intersections {
        out.clear();
    }
    let vx_mins: Vector3<i32> = blockpos_from_worldpos(entity_aabb.mins);
    let vx_maxs: Vector3<i32> = blockpos_from_worldpos(entity_aabb.maxs);
    let mut vcache = voxels.get_vcache();

    let mut intersecting = false;
    for vx_pos in (0..3)
        .map(|x| vx_mins[x]..=vx_maxs[x])
        .multi_cartesian_product()
    {
        let bpos: Vector3<i32> = vec3(vx_pos[0], vx_pos[1], vx_pos[2]);
        let bidx = vcache.get_block(world, voxels, bpos);
        if let Some(bidx) = bidx {
            let bdef = voxels.voxel_registry.get_definition_from_id(bidx);
            if !bdef.has_collisions {
                continue;
            }
            let bshape = bdef.collision_shape.translate(bpos.map(|c| c as f64));
            match AABB::intersection(entity_aabb, bshape) {
                Some(intersection) => {
                    intersecting = true;
                    if let Some(ref mut out) = out_intersections {
                        out.push(intersection);
                    }
                }
                None => continue,
            }
        }
    }
    intersecting
}
