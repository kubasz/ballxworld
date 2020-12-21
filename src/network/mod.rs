pub mod client;
pub mod packets;
pub mod protocol;
pub mod reliability;
pub mod server;

use crate::config::ConfigHandle;

pub(in crate::network) fn new_tokio_runtime(cfg: ConfigHandle) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_name("bxw-netio")
        .worker_threads(cfg.read().performance_network_threads as usize)
        .thread_stack_size(2 * 1024 * 1024)
        .on_thread_start(|| {
            log::info!("Starting network worker thread");
        })
        .on_thread_stop(|| {
            log::info!("Stopping network worker thread");
        })
        .build()
        .expect("Couldn't create network runtime")
}
