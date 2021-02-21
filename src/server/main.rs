use crate::config::Config;
use crate::network::server::{NetServer, ServerControlMessage};
use crate::server::world::ServerWorld;
use bxw_util::debug_data::DEBUG_DATA;
use bxw_util::log;
use bxw_world::blocks::register_standard_blocks;
use bxw_world::generation::WorldBlocks;
use bxw_world::physics::TIMESTEP as PHYSICS_FRAME_TIME;
use bxw_world::storage::WorldSave;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    let task_pool = bxw_util::taskpool::TaskPool::new(cfg.read().performance_threads as usize);
    let mut vxreg: Box<bxw_world::voxregistry::VoxelRegistry> = Box::default();
    register_standard_blocks(&mut vxreg, &|_| 0);
    let vxreg: Arc<bxw_world::voxregistry::VoxelRegistry> = Arc::from(vxreg);
    let savefile = {
        let name = "serverworld";
        if let Some(ws) = WorldSave::list_existing()
            .expect("Couldn't list world savefiles")
            .into_iter()
            .find(|ws| ws.name() == name)
        {
            ws
        } else {
            WorldSave::new(name).expect("Couldn't create a new world savefile")
        }
    };
    let (mut world, mut _server_world) =
        ServerWorld::new_world(vxreg.clone(), &savefile).expect("Couldn't create a new world");
    let _wgen = WorldBlocks::new(vxreg, 0);

    let mut previous_frame_time = Instant::now();
    let mut physics_accum_time = 0.0f64;

    let netserver = NetServer::new(cfg).expect("Couldn't start network server");
    let stdin = stdin_reader();
    'running: while KEEP_RUNNING.load(Ordering::SeqCst) {
        let current_frame_time = Instant::now();

        // avoid any nasty divisions by 0
        let frame_delta_time = 1.0e-6f64.max(
            current_frame_time
                .saturating_duration_since(previous_frame_time)
                .as_secs_f64(),
        );
        DEBUG_DATA.frame_times.push_sec(frame_delta_time);
        DEBUG_DATA
            .fps
            .store((1.0 / frame_delta_time) as u32, Ordering::Release);

        physics_accum_time += frame_delta_time;
        previous_frame_time = current_frame_time;
        let physics_frames = (physics_accum_time / PHYSICS_FRAME_TIME) as i32;
        if physics_frames > 0 {
            physics_accum_time -= f64::from(physics_frames) * PHYSICS_FRAME_TIME;
            let physics_frames = if physics_frames > 10 {
                log::warn!(
                    "Physics lagging behind, skipping {} ticks",
                    physics_frames - 1
                );
                1
            } else {
                physics_frames
            };

            for _pfrm in 0..physics_frames {
                // do physics tick
                bxw_world::physics::world_physics_tick(&mut world);
            }
        }

        world.main_loop_tick(&task_pool);
        task_pool.main_thread_tick();

        if let Ok(cmd) = stdin.try_recv() {
            if cmd == "quit" || cmd == "stop" {
                break 'running;
            } else {
                log::warn!("Unrecognized command: `{}`", cmd);
            }
        }

        let end_current_frame_time = Instant::now();
        let target_ft = Duration::from_secs_f64(0.25 * PHYSICS_FRAME_TIME);
        let elapsed_ft = end_current_frame_time.saturating_duration_since(current_frame_time);
        if target_ft > elapsed_ft {
            std::thread::sleep(target_ft - elapsed_ft);
        }
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
