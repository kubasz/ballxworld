use crate::ecs::*;
use bxw_util::change::Change;
use bxw_util::collider::AABB;
use bxw_util::math::*;

pub const PLAYER_WIDTH: f64 = 1.2;
pub const PLAYER_HEIGHT: f64 = 3.90;
pub const PLAYER_EYE_HEIGHT: f64 = 3.80;

pub const PLAYER_INVENTORY_SLOTS_WIDTH: u32 = 9;
pub const PLAYER_INVENTORY_SLOTS_HEIGHT: u32 = 5;
pub const PLAYER_INVENTORY_SLOTS_COUNT: u32 =
    PLAYER_INVENTORY_SLOTS_WIDTH * PLAYER_INVENTORY_SLOTS_HEIGHT;

pub fn create_player(ecs: &ECS, local: bool, name: String) -> EntityChange {
    let domain = if local {
        EntityDomain::LocalOmnipresent
    } else {
        EntityDomain::SharedOmnipresent
    };
    let id = ecs.allocate_id(domain);
    let mut location = CLocation::new(id);
    location.bounding_shape = BoundingShape::AxisAlignedBox(AABB::from_center_size(
        zero(),
        vec3(PLAYER_WIDTH, PLAYER_HEIGHT, PLAYER_WIDTH),
    ));
    let mut physics = CPhysics::new(id);
    physics.mass = 100.0;
    physics.control_max_force = vec3(30000.0, 0.0, 30000.0);
    let debug_info = CDebugInfo::new(id, name);
    let load_anchor = CLoadAnchor::new(id, 1, true);
    let inventory = CInventory::new(id, PLAYER_INVENTORY_SLOTS_COUNT);
    EntityChange {
        kind: EntityChangeKind::NewEntity(id),
        location: Change::Create { new: location },
        physics: Change::Create { new: physics },
        debug_info: Change::Create { new: debug_info },
        load_anchor: Change::Create { new: load_anchor },
        inventory: Change::Create { new: inventory },
    }
}
