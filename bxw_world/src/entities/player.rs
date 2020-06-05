use crate::ecs::*;
use bxw_util::collider::AABB;
use bxw_util::math::*;

pub const PLAYER_WIDTH: f64 = 0.9;
pub const PLAYER_HEIGHT: f64 = 1.95;
pub const PLAYER_EYE_HEIGHT: f64 = 1.8;

pub fn create_player(entities: &mut ECS, local: bool, name: String) -> ValidEntityID {
    let id = entities.add_new_entity(if local {
        EntityDomain::LocalOmnipresent
    } else {
        EntityDomain::SharedOmnipresent
    });
    let mut location = CLocation::new(id);
    location.bounding_shape = BoundingShape::AxisAlignedBox(AABB::from_center_size(
        zero(),
        vec3(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH),
    ));
    entities.set_component(id, location);
    let mut physics = CPhysics::new(id);
    physics.mass = 100.0;
    physics.control_max_force = vec3(30000.0, 0.0, 30000.0);
    entities.set_component(id, physics);
    entities.set_component(id, CDebugInfo::new(id, name));
    entities.set_component(id, CLoadAnchor::new(id, 1, true));
    id
}
