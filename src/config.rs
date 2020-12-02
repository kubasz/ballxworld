use bxw_util::itertools::Itertools;
use bxw_util::*;
use std::io::prelude::*;
use std::net::{SocketAddr, SocketAddrV4};

#[derive(Clone, Debug)]
pub struct Config {
    pub window_width: u32,
    pub window_height: u32,
    pub window_fullscreen: bool,
    pub window_monitor: u32,

    pub render_samples: u32,
    pub render_wait_for_vsync: bool,
    pub render_fps_lock: Option<u32>,

    pub performance_load_distance: u32,
    pub performance_draw_distance: u32,
    pub performance_threads: u32,
    pub performance_network_threads: u32,

    pub server_listen_addresses: Vec<SocketAddr>,
    pub server_mtu: u16,

    pub debug_logging: bool,
    pub vk_debug_layers: bool,

    /// not in TOML
    pub dbg_renderdoc: bool,

    toml_doc: Option<toml_edit::Document>,
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    pub fn new() -> Self {
        // reserve two cores for the base process and any other threaded activity
        let cpus = bxw_util::num_cpus::get().saturating_sub(2).max(1) as u32;
        Self {
            window_width: 1280,
            window_height: 720,
            window_fullscreen: false,
            window_monitor: 0,

            render_samples: 4,
            render_wait_for_vsync: false,
            render_fps_lock: None,

            performance_load_distance: 10,
            performance_draw_distance: 10,
            performance_threads: cpus,
            performance_network_threads: 3,

            server_listen_addresses: vec![SocketAddr::V4(SocketAddrV4::new(
                std::net::Ipv4Addr::new(0, 0, 0, 0),
                20138,
            ))],
            server_mtu: 1400,

            debug_logging: true,
            vk_debug_layers: false,

            dbg_renderdoc: false,

            toml_doc: None,
        }
    }

    pub fn standard_load() -> Self {
        let mut cfg = Config::new();
        let cfg_file = std::fs::File::open("settings.toml");
        match cfg_file {
            Err(_) => {
                eprintln!("Creating new settings.toml");
            }
            Ok(mut cfg_file) => {
                let mut cfg_text = String::new();
                cfg_file
                    .read_to_string(&mut cfg_text)
                    .expect("Error reading settings.toml");
                cfg.load_from_toml(&cfg_text);
            }
        }
        let cfg_file = std::fs::File::create("settings.toml");
        let cfg_text = cfg.save_toml();
        cfg_file
            .expect("Couldn't open settings.toml for writing")
            .write_all(cfg_text.as_bytes())
            .expect("Couldn't write to settings.toml");
        cfg
    }

    pub fn load_from_toml(&mut self, config: &str) {
        use toml_edit::*;
        let toml_doc: Document = config
            .parse::<Document>()
            .expect("Invalid configuration TOML.");

        self.window_width = toml_doc["window"]["width"]
            .as_integer()
            .map_or(self.window_width, |v| v as u32);
        self.window_height = toml_doc["window"]["height"]
            .as_integer()
            .map_or(self.window_height, |v| v as u32);
        self.window_fullscreen = toml_doc["window"]["width"]
            .as_bool()
            .unwrap_or(self.window_fullscreen);
        self.window_monitor = toml_doc["window"]["monitor"]
            .as_integer()
            .map_or(self.window_monitor, |v| v as u32);

        self.render_samples = toml_doc["render"]["samples"]
            .as_integer()
            .map_or(self.render_samples, |v| v as u32);
        self.render_wait_for_vsync = toml_doc["render"]["wait_for_vsync"]
            .as_bool()
            .unwrap_or(self.render_wait_for_vsync);
        self.render_fps_lock =
            toml_doc["render"]["fps_lock"]
                .as_integer()
                .map_or(self.render_fps_lock, |v| {
                    if v == 0 {
                        None
                    } else {
                        Some(v as u32)
                    }
                });

        self.performance_load_distance = toml_doc["performance"]["load_distance"]
            .as_integer()
            .map_or(self.performance_load_distance, |v| v as u32);
        self.performance_draw_distance = toml_doc["performance"]["draw_distance"]
            .as_integer()
            .map_or(self.performance_draw_distance, |v| v as u32);
        self.performance_threads = toml_doc["performance"]["threads"]
            .as_integer()
            .map_or(self.performance_threads, |v| v as u32);
        self.performance_network_threads = toml_doc["performance"]["network_threads"]
            .as_integer()
            .map_or(self.performance_network_threads, |v| v as u32);

        self.server_listen_addresses = toml_doc["server"]["listen_addresses"]
            .as_array()
            .map_or(std::mem::replace(&mut self.server_listen_addresses, vec![]), |varr| {
                varr.iter().map(|v| v.as_str()
                    .expect("Address not a IP:Port string in config server.listen_addresses [type is not a string]")
                    .parse::<SocketAddr>()
                    .expect("Address not a valid IP:Port string in config server.listen_addresses [can't parse]")).
                    collect_vec()
            });
        self.server_mtu = toml_doc["server"]["mtu"]
            .as_integer()
            .map_or(self.server_mtu, |v| v as u16);

        self.debug_logging = toml_doc["debug"]["enable_logging"]
            .as_bool()
            .unwrap_or(self.debug_logging);
        self.vk_debug_layers = toml_doc["debug"]["enable_vk_layers"]
            .as_bool()
            .unwrap_or(self.vk_debug_layers);

        self.toml_doc = Some(toml_doc);
    }

    #[allow(clippy::cast_lossless)]
    pub fn save_toml(&mut self) -> String {
        use toml_edit::*;
        let mut toml_doc = std::mem::replace(&mut self.toml_doc, None).unwrap_or_default();

        for rootkey in &["window", "render", "performance", "server", "debug"] {
            if toml_doc[rootkey].is_none() {
                toml_doc[rootkey] = Item::Table(Table::new());
            }
        }

        toml_doc["window"]["width"] = Item::Value(Value::from(self.window_width as i64));
        toml_doc["window"]["height"] = Item::Value(Value::from(self.window_height as i64));
        toml_doc["window"]["fullscreen"] = Item::Value(Value::from(self.window_fullscreen));
        toml_doc["window"]["monitor"] = Item::Value(Value::from(self.window_monitor as i64));

        toml_doc["render"]["samples"] = Item::Value(Value::from(self.render_samples as i64));
        toml_doc["render"]["wait_for_vsync"] = Item::Value(Value::from(self.render_wait_for_vsync));
        toml_doc["render"]["fps_lock"] =
            Item::Value(Value::from(self.render_fps_lock.unwrap_or(0) as i64));

        toml_doc["performance"]["load_distance"] =
            Item::Value(Value::from(self.performance_load_distance as i64));
        toml_doc["performance"]["draw_distance"] =
            Item::Value(Value::from(self.performance_draw_distance as i64));
        toml_doc["performance"]["threads"] =
            Item::Value(Value::from(self.performance_threads as i64));
        toml_doc["performance"]["network_threads"] =
            Item::Value(Value::from(self.performance_network_threads as i64));

        toml_doc["server"]["listen_addresses"] = Item::Value(
            self.server_listen_addresses
                .iter()
                .map(|a| format!("{}", a))
                .collect(),
        );
        toml_doc["server"]["mtu"] = Item::Value(Value::from(self.server_mtu as i64));

        toml_doc["debug"]["enable_logging"] = Item::Value(Value::from(self.debug_logging));
        toml_doc["debug"]["enable_vk_layers"] = Item::Value(Value::from(self.vk_debug_layers));

        self.toml_doc = Some(toml_doc);
        self.toml_doc.as_ref().unwrap().to_string()
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::new()
    }
}
