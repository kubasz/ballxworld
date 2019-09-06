use crate::client::config::Config;
use crate::client::render::{RenderingContext, VoxelRenderer};
use crate::client::world::{CameraSettings, ClientWorld};
use crate::world;
use crate::world::blocks::register_standard_blocks;
use crate::world::ecs::{CLoadAnchor, CLocation, ECSHandler};
use crate::world::generation::WorldLoadGen;
use cgmath::prelude::*;
use cgmath::{vec3, Matrix3, Quaternion, Rad};
use conrod_core::widget_ids;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::collections::{HashSet, VecDeque};
use std::f32::consts::PI;
use std::io::{Read, Write};
use std::sync::Arc;

const PHYSICS_FRAME_TIME: f64 = 1.0 / 60.0;

#[derive(Debug, Clone, Default)]
struct InputState {
    /// X (-Left,Right+) Y (-Down,Up+)
    walk: (f32, f32),
    look: (f32, f32),
    capture_mouse: bool,
}

widget_ids! {
struct Ids{canvas, positionlbl}
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

    let mut rctx = RenderingContext::new(&sdl_vid, &cfg);
    let mut vctx = VoxelRenderer::new(&cfg, &mut rctx);

    let mut frametimes = VecDeque::new();
    let frametime_count: usize = 100;

    let mut vxreg = world::registry::VoxelRegistry::new();
    register_standard_blocks(&mut vxreg, Some(&vctx));
    let vxreg = Arc::new(vxreg);
    let world = ClientWorld::new_world("world".to_owned(), vxreg.clone());
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

    let mut input_state = InputState::default();
    input_state.capture_mouse = false;
    let mut pressed_keys: HashSet<Keycode> = HashSet::new();

    let ids = Ids::new(rctx.gui.widget_id_generator());

    sdl_ctx.mouse().set_relative_mouse_mode(input_state.capture_mouse);
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

            let (dyaw, dpitch) = input_state.look;
            let CameraSettings::FPS { pitch, yaw } = &mut client.camera_settings;
            *pitch += dpitch / 60.0;
            *pitch = f32::min(f32::max(*pitch, -PI / 2.0), PI / 2.0);
            *yaw += dyaw / 60.0;
            *yaw %= 2.0 * PI;
            let (pitch, yaw) = (*pitch, *yaw);

            for _pfrm in 0..physics_frames {
                // do physics tick
                let mut entities = world.entities.write();
                let lp_loc: &mut CLocation = entities.ecs.get_component_mut(local_player).unwrap();
                // position
                let qyaw = Quaternion::from_angle_y(Rad(yaw));
                let qpitch = Quaternion::from_angle_x(Rad(pitch));
                lp_loc.orientation = (qpitch * qyaw).normalize();

                let mut mview = Matrix3::from(lp_loc.orientation);
                mview.replace_col(1, -mview.y);
                mview = mview.transpose();

                let wvel = vec3(input_state.walk.0, 0.0, input_state.walk.1) * 1.0;
                lp_loc.position -= mview * wvel;
                lp_loc.velocity = -(PHYSICS_FRAME_TIME as f32) * wvel;
                drop(entities);
                //
                world.physics_tick();
            }
            wgen.load_tick();
        }

        if let Some(mut fc) = rctx.frame_begin_prepass(frame_delta_time) {
            vctx.prepass_draw(&mut fc);
            let mut fc = RenderingContext::frame_goto_pass(fc);
            vctx.inpass_draw(&mut fc);
            RenderingContext::inpass_draw_gui(&mut fc);
            let fc = RenderingContext::frame_goto_postpass(fc);
            RenderingContext::frame_finish(fc);
        }

        input_state.walk = (0.0, 0.0);
        input_state.look = (0.0, 0.0);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(w, h) => {
                        rctx.outdated_swapchain = true;
                        rctx.gui.handle_event(conrod_core::event::Input::Resize(
                            f64::from(w),
                            f64::from(h),
                        ));
                    }
                    WindowEvent::SizeChanged(w, h) => {
                        rctx.outdated_swapchain = true;
                        rctx.gui.handle_event(conrod_core::event::Input::Resize(
                            f64::from(w),
                            f64::from(h),
                        ));
                    }
                    WindowEvent::FocusGained => {
                        sdl_ctx
                            .mouse()
                            .set_relative_mouse_mode(input_state.capture_mouse);
                    }
                    WindowEvent::FocusLost => {
                        sdl_ctx.mouse().set_relative_mouse_mode(false);
                    }
                    _ => {}
                },
                Event::KeyDown { keycode, .. } => {
                    match keycode {
                        Some(sdl2::keyboard::Keycode::F) => {
                            input_state.capture_mouse = !input_state.capture_mouse;
                            sdl_ctx
                                .mouse()
                                .set_relative_mouse_mode(input_state.capture_mouse);
                        }
                        Some(sdl2::keyboard::Keycode::Backquote) => {
                            // rctx.do_dump = true;
                        }
                        Some(sdl2::keyboard::Keycode::Escape) => {
                            break 'running;
                        }
                        _ => {}
                    }
                    if let Some(keycode) = keycode {
                        pressed_keys.insert(keycode);
                    }
                }
                Event::KeyUp { keycode, .. } => {
                    if let Some(keycode) = keycode {
                        pressed_keys.remove(&keycode);
                    }
                }
                Event::MouseMotion {
                    x, y, xrel, yrel, ..
                } => {
                    if input_state.capture_mouse {
                        input_state.look.0 += xrel as f32 * 0.4;
                        input_state.look.1 -= yrel as f32 * 0.3;
                    } else {
                        let wsz = rctx.window.size();
                        let m = conrod_core::input::Motion::MouseCursor {
                            x: f64::from(x - wsz.0 as i32 / 2),
                            y: -f64::from(y - wsz.0 as i32 / 2),
                        };
                        rctx.gui.handle_event(conrod_core::event::Input::Motion(m));
                    }
                }
                _ => {}
            }
        }

        // Simple noclip camera
        if pressed_keys.contains(&Keycode::W) {
            input_state.walk.1 += 1.0;
        }
        if pressed_keys.contains(&Keycode::S) {
            input_state.walk.1 -= 1.0;
        }
        if pressed_keys.contains(&Keycode::A) {
            input_state.walk.0 += 1.0;
        }
        if pressed_keys.contains(&Keycode::D) {
            input_state.walk.0 -= 1.0;
        }
        if pressed_keys.contains(&Keycode::LShift) {
            input_state.walk.0 *= 0.3;
            input_state.walk.1 *= 0.3;
        }
        if pressed_keys.contains(&Keycode::LCtrl) {
            input_state.walk.0 *= 3.0;
            input_state.walk.1 *= 3.0;
        }

        // simple test gui
        let mut ui = rctx.gui.set_widgets();
        use conrod_core::*;
        let avg_ft = frametimes.iter().sum::<f64>() / frametime_count as f64;
        let max_ft = frametimes
            .iter()
            .copied()
            .fold(std::f64::NEG_INFINITY, f64::max);
        widget::Canvas::new()
            .color(conrod_core::color::TRANSPARENT)
            .set(ids.canvas, &mut ui);
        let player_pos;
        let player_ang;
        let loaded_cnum;
        let drawn_cnum = vctx.drawn_chunks_number();
        let pset_cnum = vctx.progress_set_len();
        {
            let voxels = world.voxels.read();
            loaded_cnum = voxels.chunks.len();
            let client = ClientWorld::read(&world);
            let entities = world.entities.read();
            let lp_loc: &CLocation = entities.ecs.get_component(client.local_player).unwrap();
            player_pos = lp_loc.position;
            player_ang = lp_loc.orientation;
        }
        let pos = format!("Position: {:.1}, {:.1}, {:.1}\nAngles: {:#?}\nLast FT (ms): {:.1}\nAvg FT (ms): {:.1}\nMax FT (ms): {:.1}\n Avg FPS: {:.1}\nLoaded chunks: {}\nDrawn chunks: {} (ps{})",
                          player_pos.x, player_pos.y, player_pos.z,
                          player_ang,
                          frame_delta_time * 1000.0,
                          avg_ft * 1000.0,
                          max_ft * 1000.0,
                          1.0 / avg_ft,
                          loaded_cnum,
                          drawn_cnum,
            pset_cnum,
        );
        widget::Text::new(&pos)
            .font_size(14)
            .color(conrod_core::color::WHITE)
            .top_left_of(ids.canvas)
            .set(ids.positionlbl, &mut ui);

        if let Some(fps) = cfg.render_fps_lock {
            let end_current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
            let target_ft = 1.0 / f64::from(fps);
            let ms = (target_ft - end_current_frame_time + current_frame_time) * 1000.0;
            if ms > 0.0 {
                sdl_timer.delay(ms as u32);
            }
        }
    }
}
