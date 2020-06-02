use crate::World;
use bxw_util::*;
use bxw_util::math::*;

pub fn world_physics_tick(world: &World) {
    let voxels = world.voxels.read();
    let mut entities = world.entities.write();
    for (phys, loc) in entities.ecs.iter_mut_physics() {
        //
    }
}
