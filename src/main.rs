#[macro_use]
extern crate vulkano;
extern crate vulkano_shaders;
extern crate sdl2;

pub mod world;
pub mod client;

fn main() {
    client::main::client_main();
}
