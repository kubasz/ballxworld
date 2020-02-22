#![allow(unused_variables)]
#![deny(unused_must_use)]
use crate::client::config::Config;
use crate::client::input::InputManager;
use crate::client::render::resources::RenderingResources;
use crate::client::render::ui::{GuiRenderer, GuiOrderedCmd, GUI_WHITE, GuiCmd, GuiControlStyle, GuiRect};
use crate::client::render::{RenderingContext, VoxelRenderer};
use crate::client::world::{CameraSettings, ClientWorld};
use crate::math::*;
use crate::world;
use crate::world::blocks::register_standard_blocks;
use crate::world::ecs::{CLoadAnchor, CLocation, ECSHandler};
use crate::world::generation::WorldLoadGen;
use crate::world::{blockidx_from_blockpos, chunkpos_from_blockpos, BlockPosition};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::io::{Read, Write};
use std::sync::Arc;
use crate::client::render::ui::z::GUI_Z_LAYER_BACKGROUND;

const PHYSICS_FRAME_TIME: f64 = 1.0 / 60.0;

#[derive(Debug, Clone, Default)]
struct InputState {
    /// X (-Left,Right+) Y (-Down,Up+)
    walk: (f32, f32),
    look: (f32, f32),
    capture_mouse: bool,
    do_normal: bool,
}

pub fn client_main() {
    let sdl_ctx = sdl2::init().unwrap();
    let sdl_vid = sdl_ctx.video().unwrap();
    let mut sdl_timer = sdl_ctx.timer().unwrap();

    let mut cfg = Config::new();
    {
        let cfg_file = std::fs::File::open("settings.toml");
        match cfg_file {
            Err(_) => {
                eprintln!("Creating new settings.toml");
            }
            Ok(mut cfg_file) => {
                let mut cfg_text = String::new();
                cfg_file
                    .read_to_string(&mut cfg_text)
                    .expect("Error reading settings.toml");
                cfg.load_from_toml(&cfg_text);
            }
        }
        let cfg_file = std::fs::File::create("settings.toml");
        let cfg_text = cfg.save_toml();
        cfg_file
            .expect("Couldn't open settings.toml for writing")
            .write_all(cfg_text.as_bytes())
            .expect("Couldn't write to settings.toml");
    }
    if std::env::args().any(|a| a == "-renderdoc") {
        cfg.dbg_renderdoc = true;
        cfg.vk_debug_layers = false;
        eprintln!("Adjusting settings for renderdoc");
    }

    let mut rctx = RenderingContext::new(&sdl_vid, &cfg);
    let rres = Arc::new(RenderingResources::load(&cfg, &mut rctx));
    let mut vctx = VoxelRenderer::new(&cfg, &mut rctx, rres.clone());
    let mut guictx = GuiRenderer::new(&cfg, &mut rctx, rres.clone());

    let mut frametimes = VecDeque::new();
    let frametime_count: usize = 100;

    let mut vxreg = world::registry::VoxelRegistry::new();
    register_standard_blocks(&mut vxreg, Some(&vctx)); //FIXME
    let world = ClientWorld::new_world("world".to_owned(), Arc::new(vxreg));
    let world = Arc::new(world);
    {
        let client = ClientWorld::read(&world);
        let lp = client.local_player;
        let mut ents = world.entities.write();
        let anchor: &mut CLoadAnchor = ents.ecs.get_component_mut(lp).unwrap();
        anchor.radius = cfg.performance_load_distance;
    }
    vctx.set_world(world.clone(), &rctx);
    let mut wgen = WorldLoadGen::new(world.clone(), 0);

    let pf_mult = 1.0 / sdl_timer.performance_frequency() as f64;
    let mut previous_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
    let mut physics_accum_time = 0.0f64;
    //

    let mut input_mgr = InputManager::new(&sdl_ctx);
    input_mgr.input_state.capture_input_requested = true;

    let mut click_pos: Option<BlockPosition> = None;
    let mut click_place = false;

    let mut event_pump = sdl_ctx.event_pump().unwrap();
    'running: loop {
        let current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;

        // avoid any nasty divisions by 0
        let frame_delta_time = 1.0e-6f64.max(current_frame_time - previous_frame_time);
        while frametimes.len() >= frametime_count {
            frametimes.pop_front();
        }
        frametimes.push_back(frame_delta_time);

        physics_accum_time += frame_delta_time;
        previous_frame_time = current_frame_time;
        let physics_frames = (physics_accum_time / PHYSICS_FRAME_TIME) as i32;
        if physics_frames > 0 {
            physics_accum_time -= f64::from(physics_frames) * PHYSICS_FRAME_TIME;
            let physics_frames = if physics_frames > 10 {
                eprintln!(
                    "Physics lagging behind, skipping {} ticks",
                    physics_frames - 1
                );
                1
            } else {
                physics_frames
            };

            let mut client = ClientWorld::write(&world);
            let local_player = client.local_player;

            let (dyaw, dpitch) = (input_mgr.input_state.look.x, input_mgr.input_state.look.y);
            input_mgr.input_state.look = zero();
            let CameraSettings::FPS { pitch, yaw } = &mut client.camera_settings;
            *pitch -= dpitch / 60.0;
            *pitch = f32::min(f32::max(*pitch, -PI / 4.0), PI / 4.0);
            *yaw -= dyaw / 60.0;
            *yaw %= 2.0 * PI;
            let (pitch, yaw) = (*pitch, *yaw);

            if let Some(bpos) = click_pos {
                let mut voxels = world.voxels.write();
                let mut vcache = world.get_vcache();
                click_pos = None;
                // place block
                let cpos = chunkpos_from_blockpos(bpos);
                let ch = vcache.get_uncompressed_chunk_mut(&voxels, cpos).unwrap();
                let bidx = blockidx_from_blockpos(bpos);
                let i_place;
                i_place = world
                    .vregistry
                    .get_definition_from_name(if click_place {
                        "core:diamond_ore"
                    } else {
                        "core:void"
                    })
                    .unwrap();
                ch.blocks_yzx[bidx].id = i_place.id;
                let rch = voxels.chunks.get_mut(&cpos).unwrap();
                rch.compress(&ch);
                voxels.dirtify(bpos);
            }

            for _pfrm in 0..physics_frames {
                // do physics tick
                let mut entities = world.entities.write();
                let lp_loc: &mut CLocation = entities.ecs.get_component_mut(local_player).unwrap();
                // position
                let qyaw = Quaternion::from_polar_decomposition(1.0, yaw, Vector3::y_axis());
                let qpitch = Quaternion::from_polar_decomposition(1.0, pitch, Vector3::x_axis());
                lp_loc.orientation = UnitQuaternion::new_normalize(qpitch * qyaw);

                let mview = glm::quat_to_mat3(&-lp_loc.orientation).transpose();

                let mut wvel = Vector3::new(
                    input_mgr.input_state.walk.x,
                    0.0,
                    input_mgr.input_state.walk.y,
                );
                if input_mgr.input_state.sprint.is_active() {
                    wvel *= 3.0;
                }
                lp_loc.position += mview * wvel;
                lp_loc.velocity = (PHYSICS_FRAME_TIME as f32) * wvel;
                drop(entities);
                //
                world.physics_tick();
            }
            wgen.load_tick();
        }

        if let Some(mut fc) = rctx.frame_begin_prepass(&cfg, frame_delta_time) {
            fc.begin_region([0.7, 0.7, 0.1, 1.0], || "vctx.prepass_draw");
            vctx.prepass_draw(&mut fc);
            fc.end_region();
            fc.begin_region([0.5, 0.5, 0.5, 1.0], || "gui.prepass_draw");
            let gui = guictx.prepass_draw(&mut fc);
            gui.push_cmd(GuiOrderedCmd{
                z_index: GUI_Z_LAYER_BACKGROUND,
                color: GUI_WHITE,
                cmd: GuiCmd::Rectangle {
                    style: GuiControlStyle::Window,
                    rect: GuiRect::from_xywh((0.0, 5.0), (0.0, 5.0), (0.4, -5.0), (0.4, -5.0)),
                }
            });
            fc.end_region();
            let mut fc = RenderingContext::frame_goto_pass(fc);
            fc.begin_region([0.3, 0.3, 0.8, 1.0], || "vctx.inpass_draw");
            vctx.inpass_draw(&mut fc);
            fc.end_region();

            fc.begin_region([0.5, 0.5, 0.5, 1.0], || "gui.inpass_draw");
            guictx.inpass_draw(&mut fc);
            fc.end_region();

            let fc = RenderingContext::frame_goto_postpass(fc);
            fc.insert_label([0.1, 0.8, 0.1, 1.0], || "frame_finish");
            RenderingContext::frame_finish(fc);
        }

        for event in event_pump.poll_iter() {
            input_mgr.process(&mut rctx, event);
        }
        input_mgr.post_events_update(&mut rctx);

        if input_mgr.input_state.requesting_exit {
            input_mgr.input_state.requesting_exit = false;
            break 'running;
        }

        let player_pos;
        let player_ang;
        {
            let voxels = world.voxels.read();
            let client = ClientWorld::read(&world);
            let entities = world.entities.read();
            let lp_loc: &CLocation = entities.ecs.get_component(client.local_player).unwrap();
            player_pos = lp_loc.position;
            player_ang = lp_loc.orientation;

            let primary = input_mgr.input_state.primary_action.is_active();
            let secondary = input_mgr.input_state.secondary_action.is_active();
            if primary | secondary {
                use crate::world::raycast;
                let mview = glm::quat_to_mat3(&player_ang).transpose();
                let fwd = mview * vec3(0.0, 0.0, 1.0);
                let rc = raycast::RaycastQuery::new_directed(
                    player_pos,
                    fwd,
                    32.0,
                    &world,
                    Some(&voxels),
                    None,
                )
                .execute();
                click_place = secondary;
                if let raycast::Hit::Voxel {
                    position,
                    normal,
                    normal_datum,
                    ..
                } = &rc.hit
                {
                    if !click_place {
                        click_pos = Some(*position);
                    } else if normal_datum
                        .map(|d| !world.vregistry.get_definition_from_id(d).has_hitbox)
                        .unwrap_or(false)
                    {
                        click_pos = Some(*position + normal.to_vec());
                    }
                }
            }
        }

        if let Some(fps) = cfg.render_fps_lock {
            let end_current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
            let target_ft = 1.0 / f64::from(fps);
            let ms = (target_ft - end_current_frame_time + current_frame_time) * 1000.0;
            if ms > 0.0 {
                sdl_timer.delay(ms as u32);
            }
        }
    }

    vctx.destroy(&rctx.handles);
    guictx.destroy(&rctx.handles);
    Arc::try_unwrap(rres)
        .unwrap_or_else(|_| panic!("Handle still held to resource manager"))
        .destroy(&rctx.handles);
}
