use sdl2::event::{Event, WindowEvent};

use crate::client::voxmesh::mesh_from_chunk;
use crate::client::vulkan::RenderingContext;
use crate::world;
use cgmath::prelude::*;
use cgmath::{vec3, Deg, Matrix3};
use sdl2::keyboard::Keycode;
use std::collections::{HashSet, VecDeque};

use crate::world::badgen::BadGenerator;
use crate::world::generation::WorldGenerator;

use conrod_core::widget_ids;

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
    let sdl_timer = sdl_ctx.timer().unwrap();
    let mut gfx = RenderingContext::new(&sdl_vid);

    let mut frametimes = VecDeque::new();
    let frametime_count: usize = 100;

    let mut vxreg = world::registry::VoxelRegistry::new();
    vxreg
        .build_definition()
        .name("core:grass")
        .debug_color(0.1, 0.8, 0.1)
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:stone")
        .debug_color(0.45, 0.4, 0.4)
        .has_physical_properties()
        .finish()
        .unwrap();
    let mut chunk = world::VoxelChunk::new();
    let gen = BadGenerator {};
    gen.generate_chunk(
        world::VoxelChunkMutRef {
            chunk: &mut chunk,
            position: vec3(0, 0, 0),
        },
        &vxreg,
    );
    gfx.d_reset_buffers(mesh_from_chunk(&chunk, &vxreg));

    let pf_mult = 1.0 / sdl_timer.performance_frequency() as f64;
    let mut previous_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
    let mut physics_accum_time = 0.0f64;
    //

    let mut input_state = InputState::default();
    input_state.capture_mouse = true;
    let mut pressed_keys: HashSet<Keycode> = HashSet::new();

    let ids = Ids::new(gfx.gui.widget_id_generator());

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
            for _pfrm in 0..physics_frames {
                // do physics tick
                {
                    // position
                    let mut mview: Matrix3<f32> = Matrix3::identity();
                    mview = Matrix3::from_angle_y(Deg(gfx.angles.1)) * mview;
                    mview = Matrix3::from_angle_x(Deg(gfx.angles.0)) * mview;
                    mview.replace_col(1, -mview.y);
                    mview = mview.transpose();
                    gfx.position -= mview * vec3(input_state.walk.0, 0.0, input_state.walk.1);
                }
            }
        }

        gfx.draw_next_frame(frame_delta_time);

        input_state.walk = (0.0, 0.0);
        input_state.look = (0.0, 0.0);
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(w, h) => {
                        gfx.outdated_swapchain = true;
                        gfx.gui.handle_event(conrod_core::event::Input::Resize(
                            f64::from(w),
                            f64::from(h),
                        ));
                    }
                    WindowEvent::SizeChanged(w, h) => {
                        gfx.outdated_swapchain = true;
                        gfx.gui.handle_event(conrod_core::event::Input::Resize(
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
                        let wsz = gfx.window.size();
                        let m = conrod_core::input::Motion::MouseCursor {
                            x: f64::from(x - wsz.0 as i32 / 2),
                            y: -f64::from(y - wsz.0 as i32 / 2),
                        };
                        gfx.gui.handle_event(conrod_core::event::Input::Motion(m));
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

        gfx.angles.1 += input_state.look.0; // yaw
        gfx.angles.0 += input_state.look.1; // pitch
        gfx.angles.0 = f32::min(90.0, f32::max(-90.0, gfx.angles.0));
        gfx.angles.1 %= 360.0;

        // simple test gui
        let mut ui = gfx.gui.set_widgets();
        use conrod_core::*;
        let avg_ft = frametimes.iter().sum::<f64>() / frametime_count as f64;
        widget::Canvas::new()
            .color(conrod_core::color::TRANSPARENT)
            .set(ids.canvas, &mut ui);
        let pos = format!("Position: {:.1}, {:.1}, {:.1}\nAngles: {:.1}, {:.1}\nLast FT (ms): {:.1}\nAvg FT (ms): {:.1}\n Avg FPS: {:.1}",
                          gfx.position.x, gfx.position.y, gfx.position.z,
                          gfx.angles.0, gfx.angles.1,
                          frame_delta_time * 1000.0,
                          avg_ft * 1000.0,
                          1.0 / avg_ft
        );
        widget::Text::new(&pos)
            .font_size(14)
            .color(conrod_core::color::WHITE)
            .top_left_of(ids.canvas)
            .set(ids.positionlbl, &mut ui);
    }
}
