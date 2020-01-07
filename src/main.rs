extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod bus;
pub mod client;
pub mod math;
pub mod util;
pub mod world;

use jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    client::main::client_main();
}
