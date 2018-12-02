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

#[derive(Debug, Clone, Default)]
pub struct VoxelDefinition {
    id: VoxelId,
    /// eg. core:air
    name: String,
    has_mesh: bool,
    has_collisions: bool,
    has_hitbox: bool,
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
    has_mesh: bool,
    has_collisions: bool,
    has_hitbox: bool,
}

#[derive(Default)]
pub struct VoxelRegistry {
    definitions: HashMap<VoxelId, Arc<VoxelDefinition>>,
    name_lut: HashMap<String, Weak<VoxelDefinition>>,
    last_free_id: VoxelId,
}

impl<'a> VoxelDefinitionBuilder<'a> {
    pub fn name(mut self, v: &str) -> Self {
        self.name = String::from(v);
        self
    }

    pub fn has_mesh(mut self) -> Self {
        self.has_mesh = true;
        self
    }

    pub fn has_collisions(mut self) -> Self {
        self.has_collisions = true;
        self
    }

    pub fn has_hitbox(mut self) -> Self {
        self.has_hitbox = true;
        self
    }

    pub fn has_physical_properties(mut self) -> Self {
        self.has_mesh = true;
        self.has_collisions = true;
        self.has_hitbox = true;
        self
    }

    pub fn finish(self) -> Result<(), ()> {
        let def = Arc::new(VoxelDefinition {
            id: self.id,
            name: self.name,
            has_mesh: self.has_mesh,
            has_collisions: self.has_collisions,
            has_hitbox: self.has_hitbox,
        });
        if self.registry.definitions.contains_key(&def.id) {
            return Err(());
        }
        self.registry.definitions.insert(def.id, def.clone());
        self.registry
            .name_lut
            .insert(def.name.clone(), Arc::downgrade(&def));
        Ok(())
    }
}

impl VoxelRegistry {
    pub fn new() -> VoxelRegistry {
        let mut reg: VoxelRegistry = Default::default();
        reg.build_definition()
            .name("core:void")
            .finish().unwrap();
        reg
    }

    pub fn build_definition(&mut self) -> VoxelDefinitionBuilder {
        VoxelDefinitionBuilder {
            id: {
                let id = self.last_free_id;
                self.last_free_id += 1;
                id
            },
            name: String::default(),
            registry: self,
            has_mesh: false,
            has_collisions: false,
            has_hitbox: false,
        }
    }
}
