#[macro_use]
extern crate vulkano;
extern crate sdl2;

pub mod client;
pub mod world;

fn main() {
    client::main::client_main();
}
