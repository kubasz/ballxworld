use crate::client::world::WorldOpenError;
use bxw_world::generation::WorldBlocks;
use bxw_world::storage::{WorldDiskStorage, WorldSave};
use bxw_world::worldmgr::*;
use bxw_world::VoxelRegistry;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct ServerWorld {}

impl ServerWorld {
    pub fn new_world(
        registry: Arc<VoxelRegistry>,
        save: &WorldSave,
    ) -> Result<(World, ServerWorld), WorldOpenError> {
        let world_disk_storage =
            Box::new(WorldDiskStorage::open(save).map_err(WorldOpenError::StorageError)?);
        let mut world = World::new(save.name(), registry.clone(), world_disk_storage);
        world.replace_handler(CHUNK_BLOCK_DATA, Box::new(WorldBlocks::new(registry, 0)));

        let sw = ServerWorld {};
        Ok((world, sw))
    }
}
