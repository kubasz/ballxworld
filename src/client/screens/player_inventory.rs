use crate::client::render::ui::theme::{SLOT_GAP, SLOT_INNER_MARGIN, SLOT_SIZE};
use crate::client::render::ui::z;
use crate::client::render::ui::{
    gv2, gv2a, GuiCmd, GuiControlStyle, GuiFrame, GuiOrderedCmd, GuiRect, GuiVec2, GUI_RED,
    GUI_WHITE,
};
use crate::client::screens::UiScreen;
use crate::client::world::ClientWorld;
use bxw_world::ecs::ECSHandler;
use bxw_world::entities::player::{PLAYER_INVENTORY_SLOTS_HEIGHT, PLAYER_INVENTORY_SLOTS_WIDTH};
use bxw_world::inventory::CInventory;
use bxw_world::worldmgr::World;

pub struct UiPlayerInventory {}

impl UiPlayerInventory {
    fn try_draw(
        &mut self,
        gui: &mut GuiFrame,
        world: Option<(&World, &ClientWorld)>,
    ) -> Result<(), &'static str> {
        let (world, client_world) = world.ok_or("Player UI error: no world")?;
        let _player_inventory: &CInventory = world
            .ecs()
            .get_component(client_world.local_player)
            .ok_or("Player UI error: no inventory")?;
        let total_slots_width =
            PLAYER_INVENTORY_SLOTS_WIDTH as f32 * (SLOT_SIZE + SLOT_GAP) - SLOT_GAP;
        let total_slots_height =
            PLAYER_INVENTORY_SLOTS_HEIGHT as f32 * (SLOT_SIZE + SLOT_GAP) - SLOT_GAP;
        let pos_slots_topleft: GuiVec2 = gv2(
            (0.5, -total_slots_width / 2.0),
            (0.5, -total_slots_height / 2.0),
        );
        // background
        gui.push_cmd(GuiOrderedCmd {
            z_index: z::GUI_Z_LAYER_UI_MEDIUM + z::GUI_Z_OFFSET_BG,
            color: GUI_WHITE,
            cmd: GuiCmd::Rectangle {
                style: GuiControlStyle::Window,
                rect: GuiRect {
                    top_left: pos_slots_topleft - gv2a(SLOT_GAP, SLOT_GAP),
                    bottom_right: pos_slots_topleft
                        + gv2((0.0, total_slots_width), (0.0, total_slots_height))
                        + gv2a(SLOT_GAP, SLOT_GAP),
                },
            },
        });
        // slots background
        for slot_ix in 0..PLAYER_INVENTORY_SLOTS_WIDTH {
            for slot_iy in 0..PLAYER_INVENTORY_SLOTS_HEIGHT {
                let pos_slot_tl = pos_slots_topleft
                    + gv2a(
                        (SLOT_SIZE + SLOT_GAP) * slot_ix as f32,
                        (SLOT_SIZE + SLOT_GAP) * slot_iy as f32,
                    );
                let pos_slot_br = pos_slot_tl + gv2a(SLOT_SIZE, SLOT_SIZE);
                gui.push_cmd(GuiOrderedCmd {
                    z_index: z::GUI_Z_LAYER_UI_MEDIUM + z::GUI_Z_OFFSET_BG + 1,
                    color: GUI_WHITE,
                    cmd: GuiCmd::Rectangle {
                        style: GuiControlStyle::Button,
                        rect: GuiRect {
                            top_left: pos_slot_tl,
                            bottom_right: pos_slot_br,
                        },
                    },
                });
                gui.push_cmd(GuiOrderedCmd {
                    z_index: z::GUI_Z_LAYER_UI_MEDIUM + z::GUI_Z_OFFSET_CONTROL,
                    color: GUI_WHITE,
                    cmd: GuiCmd::VoxelPreview {
                        texture: world
                            .voxel_registry()
                            .get_definition_from_id(1)
                            .texture_mapping,
                        rect: GuiRect {
                            top_left: pos_slot_tl + gv2a(SLOT_INNER_MARGIN, SLOT_INNER_MARGIN),
                            bottom_right: pos_slot_br - gv2a(SLOT_INNER_MARGIN, SLOT_INNER_MARGIN),
                        },
                    },
                });
                gui.push_cmd(GuiOrderedCmd {
                    z_index: z::GUI_Z_LAYER_UI_MEDIUM + z::GUI_Z_OFFSET_CONTROL + 1,
                    color: GUI_WHITE,
                    cmd: GuiCmd::FreeText {
                        text: "123".into(),
                        scale: 0.75,
                        start_at: pos_slot_tl + gv2a(SLOT_INNER_MARGIN, SLOT_INNER_MARGIN),
                    },
                });
            }
        }
        Ok(())
    }
}

impl UiScreen for UiPlayerInventory {
    fn draw(&mut self, gui: &mut GuiFrame, world: Option<(&World, &ClientWorld)>) {
        if self.try_draw(gui, world).is_err() {
            gui.push_cmd(GuiOrderedCmd {
                z_index: z::GUI_Z_LAYER_UI_POPUP,
                color: GUI_RED,
                cmd: GuiCmd::FreeText {
                    text: "ERROR drawing ui player inventory: No world state available".into(),
                    scale: 1.0,
                    start_at: gv2((0.5, 0.0), (0.5, 0.0)),
                },
            });
        }
    }
}
