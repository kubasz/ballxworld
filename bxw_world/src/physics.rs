use crate::ecs::*;
use crate::{VoxelDatum, World};
use bxw_util::math::*;

pub const TIMESTEP: f64 = 1.0 / 60.0;
pub const SPEED_LIMIT_MPS: f32 = 1000.0;
pub const SPEED_LIMIT_MPS_SQ: f32 = SPEED_LIMIT_MPS * SPEED_LIMIT_MPS;
pub const GRAVITY_ACCEL: f32 = 30.0;
pub const AIR_FRICTION_SQ: f32 = 0.5 * 0.5 * 1.2; // 1/2 * pyramid drag coefficient * air density
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
            BoundingShape::Box { size } => size.x * size.z,
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
        new_accel += vec3(0.0, -GRAVITY_ACCEL, 0.0);
        // control impulse
        new_accel += phys.control_frame_impulse;
        phys.control_frame_impulse = zero();
        // control force
        let control_dv = phys.control_target_velocity - old_vel;
        let control_da = {
            let mut control_da = control_dv / TIMESTEP as f32;
            for comp in 0..control_da.len() {
                control_da[comp] = control_da[comp]
                    .min(phys.control_max_force[comp] / mass)
                    .max(-phys.control_max_force[comp] / mass);
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
        // position integration
        let mut new_pos = old_pos + new_vel * TIMESTEP as f32;
        // collision correction
        for axis in 0..3 {
            let dpos: f32 = new_pos[axis] - old_pos[axis];
            if dpos > 1.0 {
                // raycast
            }
            // else
            {
                // TODO: Handle bounding shapes
                let block = world
                    .get_vcache()
                    .get_block(&voxels, new_pos.map(|c| c.round() as i32));
                let id = world
                    .vregistry
                    .get_definition_from_id(block.unwrap_or(VoxelDatum::default()));
                if id.has_hitbox {
                    let dd: f32 = new_pos[axis].floor() - new_pos[axis];
                    let depth = if dpos > 0.0 { dd } else { -dd };
                    new_pos[axis] += depth * -dpos.signum();
                    new_vel[axis] = 0.0;
                }
            }
        }
        // store new values
        loc.position = new_pos;
        loc.velocity = new_vel;
    }
}
