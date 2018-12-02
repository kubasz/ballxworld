use sdl2::event::{Event, WindowEvent};

use super::vulkan::RenderingContext;

const PHYSICS_FRAME_TIME: f64 = 1.0 / 60.0;

pub fn client_main() {
    let sdl_ctx = sdl2::init().unwrap();
    let sdl_vid = sdl_ctx.video().unwrap();
    let sdl_timer = sdl_ctx.timer().unwrap();
    let mut gfx = RenderingContext::new(&sdl_vid);

    let pf_mult = 1.0 / sdl_timer.performance_frequency() as f64;
    let mut previous_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
    let mut physics_accum_time = 0.0f64;
    //
    let mut event_pump = sdl_ctx.event_pump().unwrap();
    'running: loop {
        let current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
        // avoid any nasty divisions by 0
        let frame_delta_time = 1.0e-6f64.max(current_frame_time - previous_frame_time);
        physics_accum_time += frame_delta_time;
        previous_frame_time = current_frame_time;
        let physics_frames = (physics_accum_time / PHYSICS_FRAME_TIME) as i32;
        if physics_frames > 0 {
            physics_accum_time -= f64::from(physics_frames) * PHYSICS_FRAME_TIME;
            for _pfrm in 0..physics_frames {
                // do physics tick
            }
        }

        gfx.draw_next_frame(frame_delta_time);

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(_w, _h) => {
                        gfx.outdated_swapchain = true;
                    }
                    WindowEvent::SizeChanged(_w, _h) => {
                        gfx.outdated_swapchain = true;
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }
}
