use crate::client::render::ui::GuiFrame;
use crate::client::world::ClientWorld;
use bxw_world::worldmgr::World;

pub mod player_inventory;

pub trait UiScreen {
    fn draw(&mut self, gui: &mut GuiFrame, world: Option<(&World, &ClientWorld)>);
}
