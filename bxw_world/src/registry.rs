use crate::{TextureMapping, VoxelDatum, VoxelDefinition, VoxelId};
use bxw_util::lazy_static::lazy_static;
use bxw_util::math::*;
use bxw_util::collider::AABB;
use bxw_util::*;
use std::collections::HashMap;
use std::sync::Arc;

lazy_static! {
    pub static ref VOXEL_CUBE_SHAPE: AABB = AABB {
        mins: vec3(-0.5, -0.5, -0.5),
        maxs: vec3(0.5, 0.5, 0.5)
    };
}

pub struct VoxelDefinitionBuilder<'a> {
    registry: &'a mut VoxelRegistry,
    id: VoxelId,
    name: String,
    has_mesh: bool,
    has_collisions: bool,
    has_hitbox: bool,
    collision_shape: AABB,
    debug_color: [f32; 3],
    texture_mapping: TextureMapping<u32>,
}

#[derive(Default)]
pub struct VoxelRegistry {
    definitions: Vec<Option<Arc<VoxelDefinition>>>,
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
            collision_shape: self.collision_shape,
            debug_color: self.debug_color,
            texture_mapping: self.texture_mapping,
        });
        let idx = def.id as usize;
        if self.registry.definitions.len() <= idx {
            self.registry.definitions.resize(idx * 2 + 1, None);
        } else if self.registry.definitions[idx].is_some() {
            return Err(());
        }
        self.registry.definitions[idx] = Some(def.clone());
        self.registry.name_lut.insert(def.name.clone(), def.clone());
        Ok(())
    }

    pub fn texture_names(
        mut self,
        mapper_fn: &dyn Fn(&str) -> u32,
        t: TextureMapping<&str>,
    ) -> Self {
        self.texture_mapping = t.map(mapper_fn);
        self
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
            collision_shape: *VOXEL_CUBE_SHAPE,

            debug_color: [1.0, 1.0, 1.0],
            texture_mapping: TextureMapping::TiledSingle(0),
        }
    }

    pub fn get_definition_from_id(&self, datum: VoxelDatum) -> &VoxelDefinition {
        self.definitions[datum.id as usize].as_ref().unwrap()
    }

    pub fn get_definition_from_name(&self, name: &str) -> Option<&VoxelDefinition> {
        self.name_lut.get(name).map(|x| x as &VoxelDefinition)
    }
}
