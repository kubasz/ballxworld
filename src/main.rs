extern crate nalgebra as na;
extern crate nalgebra_glm as glm;

pub mod bus;
pub mod client;
pub mod math;
pub mod util;
pub mod world;

#[cfg(not(target_env = "msvc"))]
use jemallocator;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    client::main::client_main();
}
