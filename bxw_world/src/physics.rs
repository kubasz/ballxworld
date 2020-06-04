use crate::ecs::*;
//use crate::raycast::*;
use crate::{Direction, WVoxels, World};
use bxw_util::math::*;
use bxw_util::*;
use itertools::Itertools;

pub const TIMESTEP: f64 = 1.0 / 60.0;
pub const SPEED_LIMIT_MPS: f32 = 1000.0;
pub const SPEED_LIMIT_MPS_SQ: f32 = SPEED_LIMIT_MPS * SPEED_LIMIT_MPS;
pub const DESUFFOCATION_SPEED: f32 = 0.1;
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
        let mut new_pos = old_pos + new_vel * TIMESTEP as f32;
        /*let entity_shape: Box<dyn nc::shape::Shape<f32>> = match loc.bounding_shape {
            BoundingShape::Point => Box::new(nc::shape::Ball::new(0.05)),
            BoundingShape::Ball { r } => Box::new(nc::shape::Ball::new(r)),
            BoundingShape::Capsule { r, h } => Box::new(nc::shape::Capsule::new(h / 2.0, r)),
            BoundingShape::Box { size } => Box::new(nc::shape::Cuboid::new(size / 2.0)),
        };
        let mut entity_pos = nc::math::Isometry::translation(new_pos.x, new_pos.y, new_pos.z);
        let entity_aabb = entity_shape.aabb(&entity_pos);
        let vx_mins: Vector3<i32> = entity_aabb
            .mins()
            .to_homogeneous()
            .xyz()
            .map(|c| c.floor() as i32);
        let vx_maxs: Vector3<i32> = entity_aabb
            .maxs()
            .to_homogeneous()
            .xyz()
            .map(|c| c.ceil() as i32);*/
        phys.against_wall = [false; 6];

        /*for vx_pos in (0..3)
            .map(|x| vx_mins[x]..=vx_maxs[x])
            .multi_cartesian_product()
        {
            let bpos: Vector3<i32> = vec3(vx_pos[0], vx_pos[1], vx_pos[2]);
            let bidx = world.get_vcache().get_block(&voxels, bpos);
            if let Some(bidx) = bidx {
                let bdef = world.vregistry.get_definition_from_id(bidx);
                if !bdef.has_collisions {
                    continue;
                }
                let bshape = &bdef.collision_shape;
                let mut bisometry = bdef.collision_offset;
                bisometry.append_translation_mut(&na::Translation3::new(
                    bpos.x as f32,
                    bpos.y as f32,
                    bpos.z as f32,
                ));
                let contact = nc::query::contact(
                    &bisometry,
                    bshape.as_ref(),
                    &entity_pos,
                    entity_shape.as_ref(),
                    0.1,
                );
                match contact {
                    None => continue,
                    Some(contact) => {
                        contact_list.push((bisometry, bshape, contact));
                    }
                }
            }
        }
        contact_list.sort_by_key(|c| (-c.2.depth * 1000.0) as i32);
        for (bisometry, bshape, _orig_contact) in &contact_list {
            if let Some(contact) = nc::query::contact(
                &bisometry,
                bshape.as_ref(),
                &entity_pos,
                entity_shape.as_ref(),
                0.1,
            ) {
                if contact.depth <= -0.05 {
                    continue;
                }
                for axis in 0..3 {
                    if contact.normal[axis].abs() > 0.3 {
                        phys.against_wall
                            [axis * 2 + if contact.normal[axis] < 0.0 { 1 } else { 0 }] = true;
                    }
                }
                contacts += 1;
                if contact.depth <= 0.0 {
                    continue;
                }
                new_pos += contact.normal.as_ref() * contact.depth;
                let vel_len = contact.normal.dot(&new_vel);
                new_vel -= contact.normal.as_ref() * vel_len;
                entity_pos = nc::math::Isometry::translation(new_pos.x, new_pos.y, new_pos.z);
            }
        }*/
        // no contact if moving away from a surface
        for axis in 0..3 {
            if new_vel[axis] > 0.1 {
                phys.against_wall[axis * 2] = false;
            } else if new_vel[axis] < 0.1 {
                phys.against_wall[axis * 2 + 1] = false;
            }
        }
        *bxw_util::debug_data::DEBUG_DATA.custom_string.lock() =
            format!("{:?}\n{:?}", phys.against_wall, old_vel);

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
