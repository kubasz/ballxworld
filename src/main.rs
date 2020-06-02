pub mod client;
pub mod util;

use bxw_util::debug_data::TrackingAllocator;

#[global_allocator]
static TRACK_ALLOCATOR: TrackingAllocator<std::alloc::System> = TrackingAllocator {
    allocator: std::alloc::System,
};

fn main() {
    client::main::client_main();
}
