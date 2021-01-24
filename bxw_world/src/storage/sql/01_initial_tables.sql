BEGIN TRANSACTION

CREATE TABLE bxw_save_meta (
    field_name VARCHAR(64) UNIQUE PRIMARY KEY,
    field_value VARCHAR(1024)
);

INSERT INTO bxw_save_meta(field_name, field_value) VALUES
    ('save_format', 'v1'),
    ('date_created', datetime('now'))
;

CREATE TABLE bxw_chunk_id_position (
    chunk_id INTEGER NOT NULL UNIQUE PRIMARY KEY,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    z INTEGER NOT NULL,
    UNIQUE (x, y, z)
);

CREATE TABLE bxw_chunk_storage (
    chunk_id INTEGER NOT NULL UNIQUE PRIMARY KEY REFERENCES bxw_chunk_id_position(chunk_id) ON DELETE CASCADE ON UPDATE CASCADE,
    voxel_data BLOB,
    entity_data BLOB
);

CREATE TABLE bxw_global_entity_storage (
    entity_id INTEGER NOT NULL UNIQUE PRIMARY KEY,
    entity_data BLOB
);

COMMIT TRANSACTION