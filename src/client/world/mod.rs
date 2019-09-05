use crate::world::ecs::ValidEntityID;
use crate::world::{VoxelRegistry, World};
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum CameraSettings {
    FPS { pitch: f32, yaw: f32 },
}

#[derive(Clone, Debug)]
pub struct ClientWorld {
    pub local_player: ValidEntityID,
    pub camera_settings: CameraSettings,
}

impl ClientWorld {
    pub fn new_world(name: String, registry: Arc<VoxelRegistry>) -> World {
        let mut world = World::new(name, registry, None);
        let entities = world.entities.get_mut();
        let local_player = crate::world::entities::player::create_player(
            &mut entities.ecs,
            true,
            String::from("@local_player"),
        );
        let cw = ClientWorld {
            local_player,
            camera_settings: CameraSettings::FPS {
                pitch: 0.0,
                yaw: 0.0,
            },
        };
        world.client_world = Some(RwLock::new(cw));
        world
    }

    pub fn read(w: &World) -> RwLockReadGuard<ClientWorld> {
        w.client_world
            .as_ref()
            .expect("Trying to access client world from server-side")
            .read()
    }

    pub fn write(w: &World) -> RwLockWriteGuard<ClientWorld> {
        w.client_world
            .as_ref()
            .expect("Trying to access client world from server-side")
            .write()
    }
}
