use crate::ecs::*;
use bxw_util::math::*;

pub fn create_player(entities: &mut ECS, local: bool, name: String) -> ValidEntityID {
    let id = entities.add_new_entity(if local {
        EntityDomain::LocalOmnipresent
    } else {
        EntityDomain::SharedOmnipresent
    });
    let mut location = CLocation::new(id);
    location.bounding_shape = BoundingShape::Capsule { r: 0.5, h: 2.0 };
    entities.set_component(id, location);
    let mut physics = CPhysics::new(id);
    physics.mass = 100.0;
    physics.control_max_force = vec3(30000.0, 0.0, 30000.0);
    entities.set_component(id, physics);
    entities.set_component(id, CDebugInfo::new(id, name));
    entities.set_component(id, CLoadAnchor::new(id, 1));
    id
}
