#[macro_use]
extern crate vulkano;
#[macro_use]
extern crate vulkano_shaders;
extern crate sdl2;
extern crate cgmath;
extern crate rand;

pub mod world;
pub mod client;

fn main() {
    client::main::client_main();
}
