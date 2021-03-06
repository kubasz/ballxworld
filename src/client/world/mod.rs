use bxw_util::change::Change;
use bxw_world::ecs::*;
use bxw_world::generation::WorldBlocks;
use bxw_world::storage::{WorldDiskStorage, WorldSave};
use bxw_world::worldmgr::*;
use bxw_world::VoxelRegistry;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum CameraSettings {
    FPS { pitch: f64, yaw: f64 },
}

#[derive(Clone, Debug)]
pub struct ClientWorld {
    pub local_player: ValidEntityID,
    pub camera_settings: CameraSettings,
}

#[derive(Debug)]
pub enum WorldOpenError {
    StorageError(bxw_world::storage::rusqlite::Error),
}

impl ClientWorld {
    pub fn new_local_world(
        registry: Arc<VoxelRegistry>,
        save: &WorldSave,
    ) -> Result<(World, ClientWorld), WorldOpenError> {
        let world_disk_storage =
            Box::new(WorldDiskStorage::open(save).map_err(WorldOpenError::StorageError)?);
        let mut world = World::new(save.name(), registry.clone(), world_disk_storage);
        world.replace_handler(CHUNK_BLOCK_DATA, Box::new(WorldBlocks::new(registry, 0)));
        let entities = world.ecs();
        let mut local_player = bxw_world::entities::player::create_player(
            entities,
            true,
            String::from("@local_player"),
        );
        let eid;
        match local_player.location {
            Change::Create { ref mut new } => {
                new.position.x = 300.0;
                new.position.y = 32.0;
                new.position.z = 28.0;
                eid = new.entity_id();
            }
            _ => unreachable!(),
        };

        world.apply_entity_changes(&[local_player]);
        let cw = ClientWorld {
            local_player: eid,
            camera_settings: CameraSettings::FPS {
                pitch: 0.0,
                yaw: 0.0,
            },
        };
        Ok((world, cw))
    }
}
