use bxw_util::change::Change;
use std::sync::Arc;
use world::ecs::*;
use world::generation::WorldBlocks;
use world::worldmgr::*;
use world::VoxelRegistry;

#[derive(Clone, Debug)]
pub enum CameraSettings {
    FPS { pitch: f64, yaw: f64 },
}

#[derive(Clone, Debug)]
pub struct ClientWorld {
    pub local_player: ValidEntityID,
    pub camera_settings: CameraSettings,
}

impl ClientWorld {
    pub fn new_world(name: String, registry: Arc<VoxelRegistry>) -> (World, ClientWorld) {
        let mut world = World::new(name);
        world.replace_handler(CHUNK_BLOCK_DATA, Box::new(WorldBlocks::new(registry, 0)));
        let entities = world.ecs();
        let mut local_player =
            world::entities::player::create_player(entities, true, String::from("@local_player"));
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
        (world, cw)
    }
}
