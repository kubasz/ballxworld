use crate::ChunkPosition;
use bxw_util::bytemuck::__core::sync::atomic::AtomicI64;
use bxw_util::itertools::Itertools;
use bxw_util::log;
use bxw_util::parking_lot::{Mutex, MutexGuard};
use bxw_util::TracedMutex;
pub use rusqlite;
use rusqlite::Connection;
use std::collections::VecDeque;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

mod schemas;
pub mod serializer;

pub fn saves_folder_path() -> PathBuf {
    PathBuf::from("saves")
}

const SAVEFILE_EXT: &str = "bxw";

pub struct WorldSave(PathBuf);

impl WorldSave {
    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn name(&self) -> String {
        self.0.file_stem().unwrap().to_string_lossy().into_owned()
    }

    pub fn would_be_valid_name(name: &str) -> bool {
        !name.contains(
            &[
                '\\', '/', ':', '<', '>', '"', '\'', '|', '?', '*', '\0', '\n', '\r',
            ][..],
        )
    }

    fn generate_path(name: &str) -> std::io::Result<PathBuf> {
        if !Self::would_be_valid_name(name) {
            Err(std::io::Error::new(
                ErrorKind::InvalidInput,
                "Invalid name for new savefile",
            ))
        } else {
            let mut path = saves_folder_path();
            path.push(name);
            path.set_extension(SAVEFILE_EXT);
            Ok(path)
        }
    }

    pub fn new(name: &str) -> std::io::Result<Self> {
        let path = Self::generate_path(name)?;
        let new_file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        new_file.sync_all()?;
        drop(new_file);
        Ok(Self(path))
    }

    pub fn list_existing() -> std::io::Result<Vec<Self>> {
        saves_folder_listing()
    }
}

fn ensure_saves_folder() -> std::io::Result<()> {
    match std::fs::metadata(saves_folder_path()) {
        Ok(m) if m.is_file() => {
            panic!("Unexpected file named 'saves', it should be a folder");
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            std::fs::create_dir_all(saves_folder_path())?;
            Ok(())
        }
        Err(e) => Err(e),
        _ => Ok(()),
    }
}

pub fn saves_folder_listing() -> std::io::Result<Vec<WorldSave>> {
    ensure_saves_folder()?;
    Ok(std::fs::read_dir(saves_folder_path())?
        .filter_map(|de| {
            de.ok().and_then(|s| {
                let path = s.path();
                if std::fs::metadata(&path)
                    .map(|m| m.is_file())
                    .unwrap_or(false)
                    && path.extension().map(|p| p == SAVEFILE_EXT).unwrap_or(false)
                {
                    Some(WorldSave(path))
                } else {
                    None
                }
            })
        })
        .collect_vec())
}

#[derive(Clone, Eq, PartialEq)]
pub enum ChunkIoRequest {
    TryRead {
        positions: Vec<ChunkPosition>,
    },
    Write {
        positions: Vec<(ChunkPosition, Vec<u8>, Vec<u8>)>,
    },
    Close,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum ChunkIoRequestKind {
    TryRead,
    Write,
    Close,
}

pub type ChunkIoQueue = VecDeque<ChunkIoRequest>;

#[derive(Clone, Eq, PartialEq)]
pub enum ChunkIoResponse {
    ReadOk {
        cpos: ChunkPosition,
        voxel_data: Vec<u8>,
        entity_data: Vec<u8>,
    },
    ReadMissing {
        cpos: ChunkPosition,
    },
    WriteOk {
        cpos: ChunkPosition,
    },
    ClosedOk,
}

impl ChunkIoRequest {
    fn kind(&self) -> ChunkIoRequestKind {
        use ChunkIoRequestKind::*;
        match self {
            Self::TryRead { .. } => TryRead,
            Self::Write { .. } => Write,
            Self::Close => Close,
        }
    }
}

pub type ChunkIoResponseQueue = VecDeque<ChunkIoResponse>;

pub struct WorldDiskStorage {
    db_path: PathBuf,
    db: Arc<Mutex<Connection>>,
    io_requests: Arc<Mutex<ChunkIoQueue>>,
    io_responses: Arc<Mutex<ChunkIoResponseQueue>>,
    worker_kill_switch: Arc<AtomicBool>,
    worker: std::thread::JoinHandle<()>,
}

struct WDSWorkerData {
    db: Arc<Mutex<Connection>>,
    io_requests: Arc<Mutex<ChunkIoQueue>>,
    io_responses: Arc<Mutex<ChunkIoResponseQueue>>,
    worker_kill_switch: Arc<AtomicBool>,
}

impl WorldDiskStorage {
    pub fn open(save: &WorldSave) -> rusqlite::Result<Self> {
        let db_path = save.0.clone();
        use rusqlite::OpenFlags;
        let mut db = Connection::open_with_flags(
            &db_path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )?;
        schemas::db_configure_conn(&mut db)?;
        schemas::db_setup_schema(&mut db)?;
        let db = Arc::new(Mutex::new(db));
        let io_requests = Arc::new(Mutex::new(VecDeque::with_capacity(128)));
        let io_responses = Arc::new(Mutex::new(VecDeque::with_capacity(128)));
        let worker_kill_switch = Arc::new(AtomicBool::new(false));
        let worker_data = WDSWorkerData {
            db: db.clone(),
            io_requests: io_requests.clone(),
            io_responses: io_responses.clone(),
            worker_kill_switch: worker_kill_switch.clone(),
        };
        let worker = std::thread::Builder::new()
            .name("bxw-storage-io".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || wds_worker(worker_data))
            .expect("Couldn't spawn a storage io thread");
        Ok(Self {
            db_path,
            db,
            io_requests,
            io_responses,
            worker_kill_switch,
            worker,
        })
    }
}

impl Drop for WorldDiskStorage {
    fn drop(&mut self) {
        self.worker_kill_switch.store(true, Ordering::SeqCst);
        self.worker.thread().unpark();
    }
}

pub trait WorldStorageBackend {
    fn lock_requests(&mut self) -> MutexGuard<ChunkIoQueue>;
    fn lock_responses(&mut self) -> MutexGuard<ChunkIoResponseQueue>;
    fn notify_worker(&mut self);
}

impl WorldStorageBackend for WorldDiskStorage {
    fn lock_requests(&mut self) -> MutexGuard<ChunkIoQueue> {
        self.io_requests
            .lock_traced("Disk io requests lock", file!(), line!())
    }

    fn lock_responses(&mut self) -> MutexGuard<ChunkIoResponseQueue> {
        self.io_responses
            .lock_traced("Disk io responses lock", file!(), line!())
    }

    fn notify_worker(&mut self) {
        self.worker.thread().unpark();
    }
}

fn wds_worker(data: WDSWorkerData) {
    let WDSWorkerData {
        db,
        io_requests,
        io_responses,
        worker_kill_switch,
    } = data;
    let mut my_requests: ChunkIoQueue = VecDeque::with_capacity(128);
    let mut out_responses: Vec<ChunkIoResponse> = Vec::with_capacity(1024);
    let mut store_chunk_data_buf = Vec::with_capacity(128);
    let mut read_chunk_data_buf = Vec::with_capacity(128);
    let progress_counter = AtomicI64::new(0);
    loop {
        assert!(my_requests.is_empty());
        std::mem::swap(&mut my_requests, &mut io_requests.lock());
        if my_requests.is_empty() {
            if worker_kill_switch.load(Ordering::Acquire) {
                break;
            } else {
                std::thread::park();
            }
        } else {
            for (kind, requests) in &my_requests.drain(..).group_by(ChunkIoRequest::kind) {
                assert!(store_chunk_data_buf.is_empty());
                assert!(out_responses.is_empty());
                match kind {
                    ChunkIoRequestKind::TryRead => {
                        for r in requests {
                            if let ChunkIoRequest::TryRead { mut positions } = r {
                                read_chunk_data_buf.extend(positions.drain(..));
                            } else {
                                unreachable!();
                            }
                        }
                        let mut load_results = schemas::db_load_chunk_data(
                            &mut db.lock(),
                            &read_chunk_data_buf,
                            &progress_counter,
                        )
                        .unwrap_or_else(|e| {
                            log::error!("Error loading chunk data: {}", e);
                            panic!("Load error");
                        });
                        read_chunk_data_buf.clear();
                        for (cpos, opt_data) in load_results.drain(..) {
                            out_responses.push(if let Some((voxel_data, entity_data)) = opt_data {
                                ChunkIoResponse::ReadOk {
                                    cpos,
                                    voxel_data,
                                    entity_data,
                                }
                            } else {
                                ChunkIoResponse::ReadMissing { cpos }
                            });
                        }
                        io_responses.lock().extend(out_responses.drain(..));
                    }
                    ChunkIoRequestKind::Write => {
                        for r in requests {
                            if let ChunkIoRequest::Write { mut positions } = r {
                                out_responses.extend(
                                    positions
                                        .iter()
                                        .map(|(p, _, _)| ChunkIoResponse::WriteOk { cpos: *p }),
                                );
                                store_chunk_data_buf.extend(positions.drain(..));
                            } else {
                                unreachable!();
                            }
                        }
                        schemas::db_store_chunk_data(
                            &mut db.lock(),
                            &store_chunk_data_buf,
                            &progress_counter,
                        )
                        .unwrap_or_else(|e| {
                            log::error!("Error storing chunk data: {}", e);
                            // TODO: Save backup file to allow for unsaved data recovery?
                        });
                        store_chunk_data_buf.clear();
                        io_responses.lock().extend(out_responses.drain(..));
                    }
                    ChunkIoRequestKind::Close => {
                        schemas::db_on_exit(&mut db.lock()).unwrap_or_else(|e| {
                            log::warn!("Error on database pre-close optimization: {}", e);
                        });
                        io_responses.lock().extend(
                            std::iter::repeat(ChunkIoResponse::ClosedOk).take(requests.count()),
                        );
                    }
                }
            }
        }
    }
}
