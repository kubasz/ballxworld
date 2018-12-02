use std::collections::HashMap;
use std::sync::{Arc, Weak};

const VOXEL_CHUNK_DIM: usize = 32;
const VOXEL_CHUNK_CUBES: usize = VOXEL_CHUNK_DIM * VOXEL_CHUNK_DIM * VOXEL_CHUNK_DIM;

#[derive(Debug, Copy, Clone)]
pub struct VoxelDatum {
    pub id: u32,
}

#[derive(Clone)]
pub struct VoxelChunk {
    pub data: [VoxelDatum; VOXEL_CHUNK_CUBES],
}

type VoxelId = u32;

#[derive(Debug, Clone)]
pub struct VoxelDefinition {
    id: VoxelId,
    /// eg. core:air
    name: String,
}

impl VoxelDefinition {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn id(&self) -> u32 {
        self.id
    }
}

pub struct VoxelDefinitionBuilder<'a> {
    registry: &'a mut VoxelRegistry,
    id: VoxelId,
    name: String,
}

#[derive(Default)]
pub struct VoxelRegistry {
    definitions: HashMap<VoxelId, Arc<VoxelDefinition>>,
    name_lut: HashMap<String, Weak<VoxelDefinition>>,
    last_free_id: VoxelId,
}

impl<'a> VoxelDefinitionBuilder<'a> {
    pub fn name(mut self, v: String) -> Self {
        self.name = v;
        self
    }

    pub fn finish(self) {
        let def = Arc::new(VoxelDefinition {
            id: self.id,
            name: self.name,
        });
        self.registry.definitions.insert(def.id, def.clone());
        self.registry
            .name_lut
            .insert(def.name.clone(), Arc::downgrade(&def));
    }
}

impl VoxelRegistry {
    pub fn new() -> VoxelRegistry {
        let mut reg: VoxelRegistry = Default::default();
        reg.build_definition()
            .name(String::from("core:void"))
            .finish();
        reg
    }

    pub fn build_definition(&mut self) -> VoxelDefinitionBuilder {
        VoxelDefinitionBuilder {
            id: {
                let id = self.last_free_id;
                self.last_free_id += 1;
                id
            },
            name: String::from("unknown:unknown"),
            registry: self,
        }
    }
}
