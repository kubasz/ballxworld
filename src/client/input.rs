use crate::client::render::RenderingContext;
use bxw_util::math::*;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::mouse::MouseButton;
use sdl2::Sdl;
use std::collections::HashSet;

#[derive(Copy, Clone, Debug, Default)]
pub struct Input {
    active: bool,
    /// To be reset by the user
    just_pressed: bool,
}

impl Input {
    pub fn update(&mut self, new_value: bool) {
        // don't update just_pressed if input was active last frame
        if !(self.active & new_value) {
            self.just_pressed = new_value;
        }
        self.active = new_value;
    }

    pub fn is_active(self) -> bool {
        self.active
    }

    pub fn get_and_reset_pressed(&mut self) -> bool {
        std::mem::replace(&mut self.just_pressed, false)
    }
}

#[derive(Debug, Clone)]
pub struct InputState {
    pub requesting_exit: bool,
    /// X (-Left,Right+) Y (-Down,Up+)
    /// Range -1 to 1 inclusive
    pub walk: Vector2<f32>,
    pub sprint: Input,
    pub jump: Input,
    pub noclip: bool,
    /// Yaw, Pitch motion
    pub look: Vector2<f32>,

    pub scroller: i32,
    pub primary_action: Input,
    pub secondary_action: Input,

    /// User-available mouse capture switch
    pub capture_mouse_switch: bool,
    /// Whether the game currently requests mouse capture
    pub capture_input_requested: bool,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            requesting_exit: false,

            walk: zero(),
            sprint: Input::default(),
            jump: Input::default(),
            look: zero(),
            noclip: false,

            scroller: 0,
            primary_action: Input::default(),
            secondary_action: Input::default(),

            capture_mouse_switch: false,
            capture_input_requested: false,
        }
    }
}

pub struct InputManager<'i> {
    pub sdl: &'i Sdl,
    pub input_state: InputState,
    pub pressed_keys: HashSet<Keycode>,
    pub just_pressed_keys: HashSet<Keycode>,
    capturing_input: bool,
}

#[allow(unused_variables)]
impl<'i> InputManager<'i> {
    pub fn new(sdl: &'i Sdl) -> Self {
        sdl.mouse().set_relative_mouse_mode(false);
        Self {
            sdl,
            input_state: Default::default(),
            pressed_keys: HashSet::with_capacity(16),
            just_pressed_keys: HashSet::with_capacity(16),
            capturing_input: false,
        }
    }

    pub fn pre_process(&mut self) {
        self.just_pressed_keys.clear();
    }

    pub fn process(&mut self, rctx: &mut RenderingContext, ev: Event) {
        match ev {
            Event::Quit { .. } => self.input_state.requesting_exit = true,
            Event::Window { win_event, .. } => match win_event {
                WindowEvent::Resized(w, h) => {
                    rctx.swapchain.outdated = true;
                    // gui
                }
                WindowEvent::SizeChanged(w, h) => {
                    rctx.swapchain.outdated = true;
                    // gui
                }
                WindowEvent::FocusGained => {
                    self.update_input_capture();
                }
                WindowEvent::FocusLost => {
                    self.capturing_input = false;
                    self.update_input_capture();
                }
                _ => {}
            },
            Event::KeyDown { keycode: Some(keycode), .. } => {
                if keycode == Keycode::Escape {
                    self.input_state.requesting_exit = true;
                    return;
                } else if keycode == Keycode::F {
                    self.input_state.capture_mouse_switch =
                        !self.input_state.capture_mouse_switch;
                    return;
                }
                if self.capturing_input {
                    self.pressed_keys.insert(keycode);
                    self.just_pressed_keys.insert(keycode);
                    self.process_captured_key(true, keycode);
                } else {
                    // gui
                }
            }
            Event::KeyUp { keycode: Some(keycode), .. } => {
                // always release, even when not capturing input
                self.pressed_keys.remove(&keycode);
                if self.capturing_input {
                    self.process_captured_key(false, keycode);
                } else {
                    // gui
                }
            }
            Event::MouseMotion {
                x, y, xrel, yrel, ..
            } => {
                if self.capturing_input {
                    self.input_state.look.x += xrel as f32 * 0.4;
                    self.input_state.look.y += yrel as f32 * 0.3;
                } else {
                    // gui
                }
            }
            Event::MouseWheel { x, y, .. } => {
                if self.capturing_input {
                    self.input_state.scroller += y.signum();
                } else {
                    // gui
                }
            }
            Event::MouseButtonDown {
                x, y, mouse_btn, ..
            } => {
                if self.capturing_input {
                    self.process_captured_mouse_button(true, mouse_btn, x, y);
                } else {
                    // gui
                }
            }
            Event::MouseButtonUp {
                x, y, mouse_btn, ..
            } => {
                if self.capturing_input {
                    self.process_captured_mouse_button(false, mouse_btn, x, y);
                } else {
                    // gui
                }
            }
            _ => {}
        }
    }

    fn process_captured_mouse_button(&mut self, pressed: bool, btn: MouseButton, _x: i32, _y: i32) {
        match btn {
            MouseButton::Left => {
                self.input_state.primary_action.update(pressed);
            }
            MouseButton::Right => {
                self.input_state.secondary_action.update(pressed);
            }
            _ => {}
        }
    }

    #[allow(clippy::single_match)]
    fn process_captured_key(&mut self, pressed: bool, key: Keycode) {
        match key {
            Keycode::Space => {
                self.input_state.jump.update(pressed);
            }
            Keycode::C if pressed => {
                self.input_state.noclip = !self.input_state.noclip;
            }
            _ => {}
        }
    }

    pub fn post_events_update(&mut self, _rctx: &mut RenderingContext) {
        self.update_input_capture();
        self.update_walk();
    }

    fn update_input_capture(&mut self) {
        let should_capture =
            self.input_state.capture_mouse_switch & self.input_state.capture_input_requested;
        if should_capture != self.capturing_input {
            self.capturing_input = should_capture;
            self.sdl.mouse().set_relative_mouse_mode(should_capture);
        }
    }

    fn update_walk(&mut self) {
        self.input_state.walk = vec2(0.0, 0.0);

        let slow_walk = self.pressed_keys.contains(&Keycode::LShift);
        let kbd_walk = if slow_walk { 0.4 } else { 1.0 };

        if self.pressed_keys.contains(&Keycode::W) {
            self.input_state.walk.y += kbd_walk;
        }
        if self.pressed_keys.contains(&Keycode::S) {
            self.input_state.walk.y -= kbd_walk;
        }
        if self.pressed_keys.contains(&Keycode::A) {
            self.input_state.walk.x -= kbd_walk;
        }
        if self.pressed_keys.contains(&Keycode::D) {
            self.input_state.walk.x += kbd_walk;
        }
        self.input_state
            .sprint
            .update(self.pressed_keys.contains(&Keycode::LCtrl));
    }

    /*fn map_sdl_mouse_btn(sbtn: MouseButton) -> gui_input::MouseButton {
        match sbtn {
            MouseButton::Left => gui_input::MouseButton::Left,
            MouseButton::Middle => gui_input::MouseButton::Middle,
            MouseButton::Right => gui_input::MouseButton::Right,
            MouseButton::X1 => gui_input::MouseButton::X1,
            MouseButton::X2 => gui_input::MouseButton::X2,
            _ => gui_input::MouseButton::Unknown,
        }
    }*/
}
