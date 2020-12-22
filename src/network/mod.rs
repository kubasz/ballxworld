pub mod client;
pub mod packets;
pub mod protocol;
pub mod reliability;
pub mod server;

use crate::config::ConfigHandle;

use std::sync::Arc;
use bxw_util::parking_lot::Mutex;

static TOKIO_RUNTIME_HANDLE: Mutex<Option<Arc<tokio::runtime::Runtime>>> = Mutex::new(None);

pub(in crate::network) fn get_tokio_runtime(cfg: Option<ConfigHandle>) -> Arc<tokio::runtime::Runtime> {
    Arc::clone(TOKIO_RUNTIME_HANDLE.lock().get_or_insert_with(||{
        Arc::new(tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("bxw-netio")
            .worker_threads(cfg.expect("Network runtime not initialised").read().performance_network_threads as usize)
            .thread_stack_size(2 * 1024 * 1024)
            .on_thread_start(|| {
                log::info!("Starting network worker thread");
            })
            .on_thread_stop(|| {
                log::info!("Stopping network worker thread");
            })
            .build()
            .expect("Couldn't create network runtime"))
    }))
}
