use crate::ecs::*;
use crate::raycast::*;
use crate::{Direction, WVoxels, World, ALL_DIRS};
use bxw_util::math::*;

pub const TIMESTEP: f64 = 1.0 / 60.0;
pub const SPEED_LIMIT_MPS: f32 = 1000.0;
pub const SPEED_LIMIT_MPS_SQ: f32 = SPEED_LIMIT_MPS * SPEED_LIMIT_MPS;
pub const DESUFFOCATION_SPEED: f32 = 10.0;
pub const GRAVITY_ACCEL: f32 = 30.0;
pub const AIR_FRICTION_SQ: f32 = 0.5 * 0.5 * 1.2;
// 1/2 * pyramid drag coefficient * air density
pub const SMALL_V_CUTOFF: f32 = 1.0e-6;

fn check_suffocation(world: &World, voxels: &WVoxels, position: Vector3<f32>) -> bool {
    let bpos = position.map(|c| c.floor() as i32);
    let bidx = world.get_vcache().get_block(voxels, bpos);
    if let Some(bidx) = bidx {
        let vdef = world.vregistry.get_definition_from_id(bidx);
        vdef.has_hitbox
    } else {
        false
    }
}

pub fn world_physics_tick(world: &World) {
    let voxels = world.voxels.read();
    let mut entities = world.entities.write();
    for (phys, loc) in entities.ecs.iter_mut_physics() {
        if phys.frozen {
            continue;
        }
        let mass = phys.mass;
        let area_est = match loc.bounding_shape {
            BoundingShape::Point => 0.05,
            BoundingShape::Ball { r } => r * r * f32::pi(),
            BoundingShape::Capsule { r, h } => 2.0 * r * h,
            BoundingShape::Box { size } => size.x * size.z,
        };
        let bound_length: [f32; 3] = match loc.bounding_shape {
            BoundingShape::Point => [0.05; 3],
            BoundingShape::Ball { r } => [r; 3],
            BoundingShape::Capsule { r, h } => [r, h / 2.0, r],
            BoundingShape::Box { size } => [size.x / 2.0, size.y / 2.0, size.z / 2.0],
        };
        let old_pos = loc.position;
        let old_vel = loc.velocity;
        let mut new_accel = vec3(0.0, 0.0, 0.0);
        // air friction
        let drag_force = old_vel.component_mul(&old_vel) * AIR_FRICTION_SQ * area_est;
        new_accel -= drag_force / mass;
        // gravity
        if !phys.against_wall[2] {
            new_accel += vec3(0.0, -GRAVITY_ACCEL, 0.0);
        }
        // control impulse
        new_accel += phys.control_frame_impulse;
        phys.control_frame_impulse /= 2.0; // exponential backoff
        if phys.control_frame_impulse.magnitude_squared() < SMALL_V_CUTOFF * SMALL_V_CUTOFF {
            phys.control_frame_impulse = zero();
        }
        // control force
        let control_dv = phys.control_target_velocity - old_vel;
        let control_da = {
            let mut control_da = control_dv / TIMESTEP as f32;
            for comp in 0..3 {
                control_da[comp] = control_da[comp]
                    .min(phys.control_max_force[comp] / mass)
                    .max(-phys.control_max_force[comp] / mass);
                let sixaxis = comp as i32 * 2 + (control_da[comp].signum() as i32 + 1) / 2;
            }
            control_da
        };
        new_accel += control_da;
        // velocity integration
        let mut new_vel = old_vel + new_accel * TIMESTEP as f32;
        for comp in 0..new_vel.len() {
            if new_vel[comp].abs() < SMALL_V_CUTOFF {
                new_vel[comp] = 0.0;
            }
        }
        if new_vel.magnitude_squared() > SPEED_LIMIT_MPS_SQ {
            new_vel = new_vel.normalize() * SPEED_LIMIT_MPS;
        }
        // position integration with collision correction
        let mut new_pos = old_pos;
        let collision_probe_count: [u32; 3] = {
            let mut probes = [1; 3];
            for dim in 0..probes.len() {
                probes[dim] = (bound_length[dim].ceil() as u32).max(2);
            }
            probes
        };
        for axis in 0..3 {
            let uaxis = (axis + 1) % 3;
            let vaxis = (axis + 2) % 3;
            phys.against_wall[axis * 2] = false;
            phys.against_wall[axis * 2 + 1] = false;
            let mut anyprobe_hit = false;
            for uprobe in 1..=collision_probe_count[uaxis] {
                for vprobe in 1..=collision_probe_count[vaxis] {
                    let uoffset = ((uprobe as f32 / (collision_probe_count[uaxis] as f32 + 1.0))
                        - 0.5)
                        * bound_length[uaxis];
                    let voffset = ((vprobe as f32 / (collision_probe_count[vaxis] as f32 + 1.0))
                        - 0.5)
                        * bound_length[vaxis];
                    let mut raycast_offset: Vector3<f32> = zero();
                    raycast_offset[uprobe as usize] = uoffset;
                    raycast_offset[vprobe as usize] = voffset;
                    let new_vel_len = new_vel[axis];
                    if new_vel_len == 0.0 {
                        // re-check wall contact
                        for wall_dir in &[-1i32, 1i32] {
                            let sixaxis = ((wall_dir + 1) / 2 + axis as i32 * 2) as usize;
                            let mut vec_dir: Vector3<f32> = zero();
                            vec_dir[axis] = *wall_dir as f32;
                            let rc = RaycastQuery::new_directed(
                                old_pos + raycast_offset,
                                vec_dir,
                                bound_length[axis] + 1.0e-3,
                                world,
                                Some(&voxels),
                                None,
                            )
                                .execute();
                            if let Hit::Nothing = rc.hit {} else {
                                phys.against_wall[sixaxis] = true;
                                if rc.distance < bound_length[axis] {
                                    new_pos =
                                        old_pos - vec_dir * (bound_length[axis] - rc.distance);
                                }
                                anyprobe_hit = true;
                            }
                        }
                        continue;
                    }
                    let mut new_vel_dir: Vector3<f32> = zero();
                    new_vel_dir[axis] = new_vel_len.signum();
                    let move_length = new_vel_len * TIMESTEP as f32;
                    let rc = RaycastQuery::new_directed(
                        old_pos + raycast_offset,
                        new_vel_dir,
                        move_length + bound_length[axis],
                        world,
                        Some(&voxels),
                        None,
                    )
                        .execute();
                    new_pos[axis] = match rc.hit {
                        Hit::Voxel { normal, .. } => {
                            new_vel[normal.to_unsigned_axis_index()] = 0.0;
                            new_vel_dir[normal.to_unsigned_axis_index()] = 0.0;
                            phys.against_wall[normal.opposite().to_signed_axis_index()] = true;
                            anyprobe_hit = true;
                            old_pos[axis] + new_vel_dir[axis] * (rc.distance - bound_length[axis])
                        }
                        Hit::Entity => {
                            unreachable!();
                        }
                        Hit::Nothing => {
                            if anyprobe_hit {
                                new_pos[axis]
                            } else {
                                old_pos[axis] + move_length
                            }
                        }
                    };
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
                        check_suffocation(world, &voxels, old_pos + dir.to_vec().map(|c| c as f32));
                    if !dir_suffocation {
                        desuf_dir = *dir;
                        break;
                    }
                }
                new_vel = desuf_dir.to_vec().map(|c| c as f32) * DESUFFOCATION_SPEED;
                new_pos = old_pos + new_vel * TIMESTEP as f32;
            } else {
                new_pos = old_pos;
                new_vel = zero();
            }
        }
        // store new values
        loc.position = new_pos;
        loc.velocity = new_vel;
    }
}
