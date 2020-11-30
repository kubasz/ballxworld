use crate::config::Config;
use crate::network::server::{NetServer, ServerControlMessage};

pub fn server_main() {
    eprintln!("Starting dedicated server");
    let mut cfg = Config::standard_load();
    eprintln!(
        "Configuration:\n<START CONFIGURATION>\n{}\n<END CONFIGURATION>\n",
        cfg.save_toml()
    );
    let netserver = NetServer::new(&cfg).expect("Couldn't start network server");
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
