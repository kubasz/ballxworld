use crate::world::ecs::ValidEntityID;
use crate::world::generation::World;

#[derive(Clone, Debug)]
pub enum CameraSettings {
    FPS,
}

#[derive(Clone, Debug)]
pub struct ClientWorld {
    pub local_player: ValidEntityID,
    pub camera_settings: CameraSettings,
}

impl ClientWorld {
    pub fn create_and_attach(world: &mut World) {
        if world.client_world.is_some() {
            return;
        }
        let mut entities = world.entities.write().unwrap();
        let local_player = crate::world::entities::player::create_player(
            &mut entities,
            true,
            String::from("@local_player"),
        );
        world.client_world = Some(ClientWorld {
            local_player,
            camera_settings: CameraSettings::FPS,
        });
    }
}

pub trait ClientWorldMethods {
    fn local_player(&self) -> ValidEntityID;
    fn camera_settings(&self) -> &CameraSettings;
    fn camera_settings_mut(&mut self) -> &mut CameraSettings;
}

fn get_client_world(w: &World) -> &ClientWorld {
    &w.client_world
        .as_ref()
        .expect("Trying to access client world from server-side")
}

fn get_client_world_mut(w: &mut World) -> &mut ClientWorld {
    w.client_world
        .as_mut()
        .expect("Trying to access client world from server-side")
}

impl ClientWorldMethods for World {
    fn local_player(&self) -> ValidEntityID {
        let cw = get_client_world(self);
        cw.local_player
    }

    fn camera_settings(&self) -> &CameraSettings {
        let cw = get_client_world(self);
        &cw.camera_settings
    }

    fn camera_settings_mut(&mut self) -> &mut CameraSettings {
        let cw = get_client_world_mut(self);
        &mut cw.camera_settings
    }
}
