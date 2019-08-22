use crate::world::ecs::*;
use crate::world::generation::World;

#[derive(Clone, Debug)]
pub struct ClientWorld {
    pub local_player: ValidEntityID,
}

impl ClientWorld {
    pub fn create_and_attach(world: &mut World) {
        if world.client_world.is_some() {
            return;
        }
        let mut entities = world.entities.write().unwrap();
        let local_player = entities.add_new_entity(EntityDomain::LocalOmnipresent);
        entities.set_component(local_player, CLocation::new(local_player));
        entities.set_component(
            local_player,
            CDebugInfo::new(local_player, String::from("local_player")),
        );
        world.client_world = Some(ClientWorld { local_player });
    }
}
