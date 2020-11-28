use crate::{TextureMapping, VoxelDatum, VoxelDefinition, VoxelId, VoxelMesh};
use bxw_util::collider::AABB;
use bxw_util::lazy_static::lazy_static;
use bxw_util::math::*;
use std::collections::HashMap;

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
    mesh: VoxelMesh,
    collision_shape: Option<AABB>,
    selection_shape: Option<AABB>,
    debug_color: [f32; 3],
    texture_mapping: TextureMapping<u32>,
}

pub struct VoxelRegistry {
    definitions: Vec<Option<VoxelDefinition>>,
    name_lut: HashMap<String, usize>,
    last_free_id: VoxelId,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum VoxelDefinitionError {
    AlreadyExists,
}

impl<'a> VoxelDefinitionBuilder<'a> {
    pub fn name(mut self, v: &str) -> Self {
        self.name = String::from(v);
        self
    }

    pub fn set_mesh(mut self, mesh: VoxelMesh) -> Self {
        self.mesh = mesh;
        self
    }

    pub fn set_collision_shape(mut self, shape: Option<AABB>) -> Self {
        self.collision_shape = shape;
        self
    }

    pub fn set_selection_shape(mut self, shape: Option<AABB>) -> Self {
        self.selection_shape = shape;
        self
    }

    pub fn debug_color(mut self, r: f32, g: f32, b: f32) -> Self {
        self.debug_color = [r, g, b];
        self
    }

    pub fn finish(self) -> Result<(), VoxelDefinitionError> {
        let def = VoxelDefinition {
            id: self.id,
            name: self.name,
            mesh: self.mesh,
            collision_shape: self.collision_shape,
            selection_shape: self.selection_shape,
            debug_color: self.debug_color,
            texture_mapping: self.texture_mapping,
        };
        let idx = def.id as usize;
        if self.registry.definitions.len() <= idx {
            self.registry.definitions.resize(idx * 2 + 1, None);
        } else if self.registry.definitions[idx].is_some() {
            return Err(VoxelDefinitionError::AlreadyExists);
        }
        self.registry.name_lut.insert(def.name.clone(), idx);
        self.registry.definitions[idx] = Some(def);
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

impl Default for VoxelRegistry {
    fn default() -> Self {
        let mut reg = VoxelRegistry {
            definitions: Default::default(),
            name_lut: Default::default(),
            last_free_id: 0,
        };
        reg.build_definition()
            .name("core:void")
            .debug_color(0.0, 0.0, 0.0)
            .set_mesh(VoxelMesh::None)
            .set_collision_shape(None)
            .set_selection_shape(None)
            .finish()
            .unwrap();
        reg
    }
}

impl VoxelRegistry {
    pub fn new() -> VoxelRegistry {
        Default::default()
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
            mesh: VoxelMesh::CubeAndSlopes,
            collision_shape: Some(*VOXEL_CUBE_SHAPE),
            selection_shape: Some(*VOXEL_CUBE_SHAPE),
            debug_color: [1.0, 1.0, 1.0],
            texture_mapping: TextureMapping::new_single(0),
        }
    }

    #[inline(always)]
    pub fn get_definition_from_datum(&self, datum: VoxelDatum) -> &VoxelDefinition {
        self.get_definition_from_id(datum.id())
    }

    #[inline(always)]
    pub fn get_definition_from_id(&self, id: VoxelId) -> &VoxelDefinition {
        self.definitions[usize::from(id)].as_ref().unwrap()
    }

    pub fn get_definition_from_name(&self, name: &str) -> Option<&VoxelDefinition> {
        self.name_lut
            .get(name)
            .and_then(|x| self.definitions.get(*x)?.as_ref())
    }
}
