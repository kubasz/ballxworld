use bxw_util::itertools::Itertools;
use bxw_util::parking_lot::Mutex;
pub use rusqlite;
use rusqlite::Connection;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod schemas;

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

pub struct WorldDiskStorage {
    db_path: PathBuf,
    db: Arc<Mutex<Connection>>,
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
        schemas::setup_db_schema(&mut db)?;
        Ok(Self {
            db_path,
            db: Arc::new(Mutex::new(db)),
        })
    }
}

impl Drop for WorldDiskStorage {
    fn drop(&mut self) {
        schemas::db_on_exit(&mut self.db.lock()).unwrap_or_else(|e| {
            eprintln!("Warning: error on database pre-close optimization: {}", e)
        });
    }
}
