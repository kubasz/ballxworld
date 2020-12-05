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
    setup_logging();
    bxw_util::sodiumoxide::init().expect("Couldn't initialize cryptography library");
    if std::env::args().any(|a| a == "-server") {
        server::main::server_main();
    } else {
        client::main::client_main();
    }
}

fn setup_logging() {
    log4rs::init_file("settings_logging.toml", Default::default())
        .expect("Couldn't initialize the logging subsystem");
    log::info!("Logging setup complete");
}
