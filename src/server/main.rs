use crate::config::Config;
use crate::network::server::{NetServer, ServerControlMessage};

pub fn server_main() {
    log::info!("Starting dedicated server");
    let cfg = Config::standard_load();
    log::debug!(
        "Configuration:\n<START CONFIGURATION>\n{}\n<END CONFIGURATION>\n",
        cfg.write().save_toml()
    );
    let netserver = NetServer::new(cfg.clone()).expect("Couldn't start network server");
    let mut line = String::with_capacity(32);
    loop {
        std::io::stdin().read_line(&mut line).unwrap();
        if line.contains("quit") {
            netserver.send_control_message(ServerControlMessage::Stop);
            break;
        }
    }
    netserver.wait_for_shutdown();
}
