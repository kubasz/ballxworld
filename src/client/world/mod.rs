use std::sync::Arc;
use world::ecs::ValidEntityID;
use world::ecs::{CLocation, ECSHandler};
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
        let mut entities = world.ecs().write();
        let local_player = world::entities::player::create_player(
            &mut entities,
            true,
            String::from("@local_player"),
        );
        let loc: &mut CLocation = entities.get_component_mut(local_player).unwrap();
        loc.position.x = 300.0;
        loc.position.y = 32.0;
        loc.position.z = 28.0;
        let cw = ClientWorld {
            local_player,
            camera_settings: CameraSettings::FPS {
                pitch: 0.0,
                yaw: 0.0,
            },
        };
        drop(entities);
        (world, cw)
    }
}
