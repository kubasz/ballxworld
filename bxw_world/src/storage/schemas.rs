use rusqlite::{Connection, OptionalExtension, NO_PARAMS};

pub fn db_configure_conn(db: &mut Connection) -> rusqlite::Result<()> {
    db.execute_batch(include_str!("sql/00_conn_pragmas.sql"))
}

pub fn db_on_exit(db: &mut Connection) -> rusqlite::Result<()> {
    db.execute_batch(include_str!("sql/00_conn_on_exit.sql"))
}

pub fn setup_db_schema(db: &mut Connection) -> rusqlite::Result<()> {
    let needs_initial_tables = db
        .query_row(
            "SELECT name FROM sqlite_schema WHERE type='table' AND name='bxw_save_meta';",
            NO_PARAMS,
            |r| r.get::<_, String>(0),
        )
        .optional()?
        .is_none();
    if needs_initial_tables {
        db.execute_batch(include_str!("sql/01_initial_tables.sql"))?;
    }
    Ok(())
}
