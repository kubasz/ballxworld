pub mod client;
pub mod world;

use jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    client::main::client_main();
}
