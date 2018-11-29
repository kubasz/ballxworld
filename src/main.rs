#[macro_use]
extern crate vulkano;
extern crate sdl2;

pub mod client;

fn main() {
    client::main::client_main();
}
