use crate::world::{VoxelDatum, VoxelDefinition, VoxelId};
use std::collections::HashMap;
use std::sync::Arc;

pub struct VoxelDefinitionBuilder<'a> {
    registry: &'a mut VoxelRegistry,
    id: VoxelId,
    name: String,
    has_mesh: bool,
    has_collisions: bool,
    has_hitbox: bool,
    pub debug_color: [f32; 3],
}

#[derive(Default)]
pub struct VoxelRegistry {
    definitions: HashMap<VoxelId, Arc<VoxelDefinition>>,
    name_lut: HashMap<String, Arc<VoxelDefinition>>,
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

    pub fn debug_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.debug_color = [r, g, b];
        self
    }

    pub fn finish(self) -> Result<(), ()> {
        let def = Arc::new(VoxelDefinition {
            id: self.id,
            name: self.name,
            has_mesh: self.has_mesh,
            has_collisions: self.has_collisions,
            has_hitbox: self.has_hitbox,
            debug_color: self.debug_color,
        });
        if self.registry.definitions.contains_key(&def.id) {
            return Err(());
        }
        self.registry.definitions.insert(def.id, def.clone());
        self.registry.name_lut.insert(def.name.clone(), def.clone());
        Ok(())
    }
}

impl VoxelRegistry {
    pub fn new() -> VoxelRegistry {
        let mut reg: VoxelRegistry = Default::default();
        reg.build_definition()
            .name("core:void")
            .debug_color(0.0, 0.0, 0.0)
            .finish()
            .unwrap();
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
            debug_color: [1.0, 0.0, 1.0],
        }
    }

    pub fn get_definition_from_id(&self, datum: &VoxelDatum) -> &VoxelDefinition {
        &self.definitions[&datum.id]
    }

    pub fn get_definition_from_name(&self, name: &str) -> Option<&VoxelDefinition> {
        self.name_lut.get(name).map(|x| x as &VoxelDefinition)
    }
}
