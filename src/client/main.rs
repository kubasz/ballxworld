use sdl2::event::{Event, WindowEvent};

use crate::client::render::{RenderingContext, VoxelRenderer};
use crate::world;
use cgmath::prelude::*;
use cgmath::{vec3, Deg, Matrix3};
use sdl2::keyboard::Keycode;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use crate::world::badgen::BadGenerator;

use crate::client::config::Config;
use crate::world::TextureMapping;
use conrod_core::widget_ids;
use std::io::{Read, Write};

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
    vxreg
        .build_definition()
        .name("core:grass")
        .debug_color(1.0, 1.0, 1.0)
        .texture_names(
            &vctx,
            TextureMapping::TiledTSB {
                top: "grass_top",
                side: "grass_side",
                bottom: "grass_bottom",
            },
        )
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:stone")
        .debug_color(1.0, 1.0, 1.0)
        .texture_names(&vctx, TextureMapping::TiledSingle("stone"))
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:border")
        .debug_color(1.0, 1.0, 1.0)
        .texture_names(&vctx, TextureMapping::TiledSingle("unknown"))
        .has_physical_properties()
        .finish()
        .unwrap();
    let vxreg = Arc::new(vxreg);
    let mut world = world::generation::World::new("world".to_owned(), vxreg.clone());
    world.load_anchor.chunk_radius = cfg.performance_load_distance as i32;
    world.change_generator(Arc::new(BadGenerator::default()));
    let world = Arc::new(Mutex::new(world));

    vctx.world = Some(world.clone());

    let pf_mult = 1.0 / sdl_timer.performance_frequency() as f64;
    let mut previous_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
    let mut physics_accum_time = 0.0f64;
    //

    let mut input_state = InputState::default();
    input_state.capture_mouse = true;
    let mut pressed_keys: HashSet<Keycode> = HashSet::new();

    let ids = Ids::new(rctx.gui.widget_id_generator());

    sdl_ctx.mouse().set_relative_mouse_mode(true);
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
            let mut world = world.lock().unwrap();
            for _pfrm in 0..physics_frames {
                // do physics tick
                {
                    // position
                    let mut mview: Matrix3<f32> = Matrix3::identity();
                    mview = Matrix3::from_angle_y(Deg(vctx.angles.1)) * mview;
                    mview = Matrix3::from_angle_x(Deg(vctx.angles.0)) * mview;
                    mview.replace_col(1, -mview.y);
                    mview = mview.transpose();
                    vctx.position -= mview * vec3(input_state.walk.0, 0.0, input_state.walk.1);
                    world.load_anchor.position = vctx.position;
                    //
                    world.physics_tick();
                }
            }
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

        vctx.angles.1 += input_state.look.0; // yaw
        vctx.angles.0 += input_state.look.1; // pitch
        vctx.angles.0 = f32::min(90.0, f32::max(-90.0, vctx.angles.0));
        vctx.angles.1 %= 360.0;

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
        let pos = format!("Position: {:.1}, {:.1}, {:.1}\nAngles: {:.1}, {:.1}\nLast FT (ms): {:.1}\nAvg FT (ms): {:.1}\nMax FT (ms): {:.1}\n Avg FPS: {:.1}",
                          vctx.position.x, vctx.position.y, vctx.position.z,
                          vctx.angles.0, vctx.angles.1,
                          frame_delta_time * 1000.0,
                          avg_ft * 1000.0,
                          max_ft * 1000.0,
                          1.0 / avg_ft
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
