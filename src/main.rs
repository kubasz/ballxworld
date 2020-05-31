extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod bus;
pub mod client;
pub mod math;
pub mod util;
pub mod world;

fn main() {
    client::main::client_main();
}
