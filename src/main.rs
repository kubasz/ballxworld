pub mod client;
pub mod config;
pub mod network;
pub mod server;
pub mod util;

use bxw_util::debug_data::TrackingAllocator;

#[global_allocator]
static TRACK_ALLOCATOR: TrackingAllocator<std::alloc::System> = TrackingAllocator {
    allocator: std::alloc::System,
};

fn main() {
    if std::env::args().any(|a| a == "-server") {
        server::main::server_main();
    } else {
        client::main::client_main();
    }
}
