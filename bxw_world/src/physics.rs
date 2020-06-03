use crate::ecs::*;
use crate::raycast::*;
use crate::World;
use bxw_util::math::*;

pub const TIMESTEP: f64 = 1.0 / 60.0;
pub const SPEED_LIMIT_MPS: f32 = 1000.0;
pub const SPEED_LIMIT_MPS_SQ: f32 = SPEED_LIMIT_MPS * SPEED_LIMIT_MPS;
pub const GRAVITY_ACCEL: f32 = 30.0;
pub const AIR_FRICTION_SQ: f32 = 0.5 * 0.5 * 1.2;
// 1/2 * pyramid drag coefficient * air density
pub const SMALL_V_CUTOFF: f32 = 1.0e-6;

// TODO: Detect suffocation
// TODO: Handle suffocation

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
            BoundingShape::Capsule { r, h } => [r, h, r],
            BoundingShape::Box { size } => [size.x / 2.0, size.y / 2.0, size.z / 2.0],
        };
        let old_pos = loc.position;
        // check "suffocation" - whether the object is stuck inside a block
        let _suffocation = {
            false // TODO
        };
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
                if phys.against_wall[sixaxis as usize] {
                    control_da[comp] = 0.0;
                }
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
        for axis in 0..3 {
            let new_vel_len = new_vel[axis];
            if new_vel_len == 0.0 {
                // re-check wall contact
                for wall_dir in &[-1i32, 1i32] {
                    let sixaxis = ((wall_dir + 1) / 2 + axis as i32 * 2) as usize;
                    let mut vec_dir: Vector3<f32> = zero();
                    vec_dir[axis] = *wall_dir as f32;
                    let rc = RaycastQuery::new_directed(
                        old_pos,
                        vec_dir,
                        bound_length[axis] + 1.0e-3,
                        world,
                        Some(&voxels),
                        None,
                    )
                    .execute();
                    if let Hit::Nothing = rc.hit {
                        phys.against_wall[sixaxis] = false;
                    } else {
                        phys.against_wall[sixaxis] = true;
                    }
                }
                continue;
            }
            let mut new_vel_dir: Vector3<f32> = zero();
            new_vel_dir[axis] = new_vel_len.signum();
            let move_length = new_vel_len * TIMESTEP as f32;
            let rc = RaycastQuery::new_directed(
                old_pos,
                new_vel_dir,
                move_length + bound_length[axis],
                world,
                Some(&voxels),
                None,
            )
            .execute();
            phys.against_wall[axis * 2] = false;
            phys.against_wall[axis * 2 + 1] = false;
            new_pos[axis] = match rc.hit {
                Hit::Voxel { normal, .. } => {
                    new_vel[normal.to_unsigned_axis_index()] = 0.0;
                    phys.against_wall[normal.opposite().to_signed_axis_index()] = true;
                    old_pos[axis] + new_vel_dir[axis] * (rc.distance - bound_length[axis])
                }
                Hit::Entity => {
                    unreachable!();
                }
                Hit::Nothing => old_pos[axis] + move_length,
            };
        }
        // store new values
        loc.position = new_pos;
        loc.velocity = new_vel;
    }
}
