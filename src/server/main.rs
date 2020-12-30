use crate::config::Config;
use crate::network::server::{NetServer, ServerControlMessage};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

static KEEP_RUNNING: AtomicBool = AtomicBool::new(true);

pub fn server_main() {
    ctrlc::set_handler(|| {
        KEEP_RUNNING.store(false, Ordering::SeqCst);
    })
    .unwrap_or_else(|_| log::warn!("Could not install Ctrl-C/SIGTERM handler"));
    log::info!("Starting dedicated server");
    let cfg = Config::standard_load();
    log::debug!(
        "Configuration:\n<START CONFIGURATION>\n{}\n<END CONFIGURATION>\n",
        cfg.write().save_toml()
    );
    let netserver = NetServer::new(cfg.clone()).expect("Couldn't start network server");
    let mut stdin = stdin_reader();
    while KEEP_RUNNING.load(Ordering::SeqCst) {
        if let Ok(cmd) = stdin.try_recv() {
            if cmd == "quit" || cmd == "stop" {
                break;
            } else {
                log::warn!("Unrecognized command: `{}`", cmd);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    log::info!("Shutting down, waiting for netserver...");
    netserver.send_control_message(ServerControlMessage::Stop);
    netserver.wait_for_shutdown();
}

fn stdin_reader() -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("bxw-stdin-reader".into())
        .spawn(|| stdin_reader_worker(tx))
        .expect("Couldn't start stdin reader worker thread");
    rx
}

fn stdin_reader_worker(tx: mpsc::Sender<String>) {
    let mut linebuf = String::with_capacity(128);
    while let Ok(_count) = std::io::stdin().read_line(&mut linebuf) {
        let cmd = linebuf.trim();
        if cmd.is_empty() {
            continue;
        }
        if tx.send(cmd.to_owned()).is_err() {
            break;
        }
        if !KEEP_RUNNING.load(Ordering::SeqCst) {
            break;
        }
        linebuf.clear();
    }
}
