use crate::ChunkPosition;
use bxw_util::fnv::FnvHashSet;
use bxw_util::itertools::Itertools;
use rusqlite::{Connection, OptionalExtension, NO_PARAMS};
use std::fmt::Write;
use std::sync::atomic::{AtomicI64, Ordering};

pub fn db_configure_conn(db: &mut Connection) -> rusqlite::Result<()> {
    db.execute_batch(include_str!("sql/00_conn_pragmas.sql"))
}

pub fn db_on_exit(db: &mut Connection) -> rusqlite::Result<()> {
    db.execute_batch(include_str!("sql/00_conn_on_exit.sql"))
}

pub fn db_setup_schema(db: &mut Connection) -> rusqlite::Result<()> {
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

/// chunk_data: `&[(position, serialized voxel data, serialized entity data)]`
pub fn db_store_chunk_data(
    db: &mut Connection,
    chunk_data: &[(ChunkPosition, Vec<u8>, Vec<u8>)],
    chunks_processed_counter: &AtomicI64,
) -> rusqlite::Result<()> {
    let transaction = db.transaction_with_behavior(rusqlite::TransactionBehavior::Exclusive)?;
    {
        let mut stmt = transaction
            .prepare_cached(
                r#"INSERT OR REPLACE INTO bxw_chunk_storage
        (x, y, z, voxel_data, entity_data)
        VALUES
        (:x, :y, :z, :vox, :ent)
        ;"#,
            )
            .expect("Invalid SQL insert/update statement for bxw_chunk_storage rows");
        for (cpos, chunk_data, entity_data) in chunk_data.iter() {
            stmt.execute_named(&[
                (":x", &cpos.x),
                (":y", &cpos.y),
                (":z", &cpos.z),
                (":vox", chunk_data),
                (":ent", entity_data),
            ])?;
            chunks_processed_counter.fetch_add(1 as i64, Ordering::AcqRel);
        }
    }
    transaction.commit()?;
    Ok(())
}

pub fn db_load_chunk_data(
    db: &mut Connection,
    positions: &[ChunkPosition],
    chunks_processed_counter: &AtomicI64,
) -> rusqlite::Result<Vec<(ChunkPosition, Option<(Vec<u8>, Vec<u8>)>)>> {
    use bxw_util::math::vec3;
    let transaction = db.transaction_with_behavior(rusqlite::TransactionBehavior::Deferred)?;
    let sql_prelude = "SELECT x, y, z, voxel_data, entity_data FROM bxw_chunk_storage WHERE (x, y, z) IN (VALUES ";
    let mut query = String::with_capacity(sql_prelude.len() + 16 * positions.len() + 16);
    query.push_str(sql_prelude);
    let mut missing_chunks: FnvHashSet<ChunkPosition> =
        FnvHashSet::with_capacity_and_hasher(2 * positions.len(), Default::default());
    for cpos in positions.iter().with_position() {
        use bxw_util::itertools::Position;
        if matches!(cpos, Position::Middle(_) | Position::Last(_)) {
            query.push_str(",");
        }
        let cv = cpos.into_inner();
        missing_chunks.insert(*cv);
        // SQL Injection Safety: formatting simple integers into the query can't generate non-{digit, +, -} characters
        write!(query, "({},{},{})", cv.x as i32, cv.y as i32, cv.z as i32).unwrap();
    }
    query.push_str(");");
    let mut out_table = Vec::with_capacity(positions.len());
    {
        let mut stmt = transaction
            .prepare(&query)
            .expect("Invalid SQL select statement for bxw_chunk_storage rows");
        {
            let mut rows = stmt.query(rusqlite::NO_PARAMS)?;
            while let Some(row) = rows.next()? {
                let x: i32 = row.get_unwrap(0);
                let y: i32 = row.get_unwrap(1);
                let z: i32 = row.get_unwrap(2);
                let voxel_data: Vec<u8> = row.get_unwrap(3);
                let entity_data: Vec<u8> = row.get_unwrap(4);
                let cpos = vec3(x, y, z);
                missing_chunks.remove(&cpos);
                out_table.push((cpos, Some((voxel_data, entity_data))));
                chunks_processed_counter.fetch_add(1, Ordering::AcqRel);
            }
        }
        stmt.finalize()?;
    }
    let mut missing_no = 0i64;
    for missing_cpos in missing_chunks.into_iter() {
        out_table.push((missing_cpos, None));
        missing_no += 1;
    }
    chunks_processed_counter.fetch_add(missing_no, Ordering::AcqRel);
    transaction.commit()?;
    assert_eq!(out_table.len(), positions.len());
    Ok(out_table)
}

#[cfg(test)]
mod test {
    use super::*;
    use bxw_util::fnv::FnvHashMap;
    use rusqlite::Connection;
    use std::sync::atomic::{AtomicI64, Ordering};

    fn testutil_data_hash(
        data: &[(ChunkPosition, Option<(Vec<u8>, Vec<u8>)>)],
    ) -> FnvHashMap<ChunkPosition, Option<(Vec<u8>, Vec<u8>)>> {
        let mut hash = FnvHashMap::with_capacity_and_hasher(data.len() * 2, Default::default());
        for (cpos, opt) in data.iter() {
            if !hash.insert(*cpos, opt.clone()).is_none() {
                panic!("Duplicate row of data for cpos {:?}", cpos);
            }
        }
        hash
    }

    #[test]
    pub fn db_simple_test() {
        use bxw_util::math::*;

        // Use an in-memory db for testing the SQL queries
        let mut inmem = Connection::open_in_memory().unwrap();
        // Initial setup
        db_configure_conn(&mut inmem).expect("db_configure_conn failed");
        db_setup_schema(&mut inmem).expect("setup_db_schema failed");
        // Insert first set of test data
        let sample_data_1 = [
            (vec3(0, 0, 0), vec![0, 0, 0], vec![1]),
            (vec3(0, 1, 0), vec![0, 1, 0], vec![2]),
            (vec3(0, 0, 1), vec![0, 0, 1], vec![3]),
        ];
        let counter = AtomicI64::new(0);
        db_store_chunk_data(&mut inmem, &sample_data_1, &counter)
            .expect("Couldn't store sample data 1");
        assert_eq!(counter.load(Ordering::SeqCst) as usize, sample_data_1.len());
        counter.store(0, Ordering::SeqCst);
        // Test first set of test data
        let sample_query_1 = [
            vec3(0, 0, 1), //
            vec3(0, 1, 0), //
            vec3(0, 0, 0), //
            vec3(1, 1, 1), //
        ];
        let sample_expected_1 = testutil_data_hash(&[
            (vec3(0, 0, 0), Some((vec![0, 0, 0], vec![1]))),
            (vec3(0, 1, 0), Some((vec![0, 1, 0], vec![2]))),
            (vec3(0, 0, 1), Some((vec![0, 0, 1], vec![3]))),
            (vec3(1, 1, 1), None),
        ]);
        let sample_qresult_1 = db_load_chunk_data(&mut inmem, &sample_query_1, &counter)
            .expect("Couldn't query for initially stored chunks");
        assert_eq!(
            counter.load(Ordering::SeqCst) as usize,
            sample_query_1.len()
        );
        counter.store(0, Ordering::SeqCst);
        assert_eq!(sample_qresult_1.len(), sample_query_1.len());
        let sample_qhash_1 = testutil_data_hash(&sample_qresult_1);
        assert_eq!(sample_expected_1, sample_qhash_1);
        // Insert and replace more data
        let sample_data_2 = [
            (vec3(0, 0, 0), vec![0, 0, 0], vec![1]),
            (vec3(0, 0, 1), vec![0, 6, 1, 5, 3], vec![3, 4]),
            (vec3(1, 1, 1), vec![24, 1, 0, 32], vec![6, 7]),
        ];
        db_store_chunk_data(&mut inmem, &sample_data_2, &counter)
            .expect("Couldn't store sample data 2");
        assert_eq!(counter.load(Ordering::SeqCst) as usize, sample_data_2.len());
        counter.store(0, Ordering::SeqCst);
        // Check updated data
        let sample_query_2 = [
            vec3(0, 0, 1), //
            vec3(0, 1, 0), //
            vec3(0, 0, 0), //
            vec3(1, 1, 1), //
            vec3(1, 1, 2), //
        ];
        let sample_expected_2 = testutil_data_hash(&[
            (vec3(0, 0, 0), Some((vec![0, 0, 0], vec![1]))),
            (vec3(0, 1, 0), Some((vec![0, 1, 0], vec![2]))),
            (vec3(0, 0, 1), Some((vec![0, 6, 1, 5, 3], vec![3, 4]))),
            (vec3(1, 1, 1), Some((vec![24, 1, 0, 32], vec![6, 7]))),
            (vec3(1, 1, 2), None),
        ]);
        let sample_qresult_2 = db_load_chunk_data(&mut inmem, &sample_query_2, &counter)
            .expect("Couldn't query for updated stored chunks");
        assert_eq!(
            counter.load(Ordering::SeqCst) as usize,
            sample_query_2.len()
        );
        counter.store(0, Ordering::SeqCst);
        assert_eq!(sample_qresult_2.len(), sample_query_2.len());
        let sample_qhash_2 = testutil_data_hash(&sample_qresult_2);
        assert_eq!(sample_expected_2, sample_qhash_2);
        // Test finalization SQL
        db_on_exit(&mut inmem).expect("db_on_exit failed");
    }
}
