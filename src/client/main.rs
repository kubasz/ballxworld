use sdl2::event::Event;
use sdl2::keyboard::Keycode;

pub fn client_main() {
    let sdl_ctx = sdl2::init().unwrap();
    let sdl_vid = sdl_ctx.video().unwrap();
    let window = sdl_vid
        .window("BallX World", 1280, 720)
        .position_centered()
        .vulkan()
        .build()
        .unwrap();

    let mut event_pump = sdl_ctx.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                _ => {}
            }
        }
    }
}
