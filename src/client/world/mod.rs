use std::sync::Arc;
use world::ecs::ValidEntityID;
use world::ecs::{CLocation, ECSHandler};
use world::{VoxelRegistry, World};

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
        let mut world = World::new(name, registry);
        let entities = world.entities.get_mut();
        let local_player = world::entities::player::create_player(
            &mut entities.ecs,
            true,
            String::from("@local_player"),
        );
        let loc: &mut CLocation = entities.ecs.get_component_mut(local_player).unwrap();
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
        (world, cw)
    }
}
