extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod client;
pub mod math;
pub mod world;

use jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    client::main::client_main();
}
