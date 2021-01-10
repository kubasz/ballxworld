use bxw_world::generation::WorldBlocks;
use bxw_world::worldmgr::*;
use bxw_world::VoxelRegistry;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ServerWorld {}

impl ServerWorld {
    pub fn new_world(name: String, registry: Arc<VoxelRegistry>) -> (World, ServerWorld) {
        let mut world = World::new(name);
        world.replace_handler(CHUNK_BLOCK_DATA, Box::new(WorldBlocks::new(registry, 0)));

        let sw = ServerWorld {};
        (world, sw)
    }
}
