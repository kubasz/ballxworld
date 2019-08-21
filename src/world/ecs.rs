use cgmath::prelude::*;
use cgmath::{vec3, Quaternion, Vector3};
use std::collections::HashMap;
use std::marker::PhantomData;

#[repr(u64)]
#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum EntityDomain {
    LocalChunked = 0b00_u64 << 62,
    SharedChunked = 0b01_u64 << 62,
    LocalOmnipresent = 0b10_u64 << 62,
    SharedOmnipresent = 0b11_u64 << 62,
}

impl EntityDomain {
    pub fn number(self) -> usize {
        (self as u64 >> 62) as usize
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct EntityID(u64);

const DOMAIN_MASK: u64 = 0b11_u64 << 62;

impl EntityID {
    const fn const_from_parts(domain: EntityDomain, sub_id: u64) -> Self {
        EntityID(domain as u64 | sub_id)
    }

    pub fn from_parts(domain: EntityDomain, sub_id: u64) -> Self {
        assert_eq!(sub_id & DOMAIN_MASK, 0);
        EntityID(domain as u64 | sub_id)
    }

    pub fn u64(self) -> u64 {
        self.0
    }

    pub fn domain(self) -> EntityDomain {
        unsafe { std::mem::transmute(self.u64() & DOMAIN_MASK) }
    }

    pub fn sub_id(self) -> u64 {
        self.u64() & !DOMAIN_MASK
    }
}

const NULL_EID: EntityID = EntityID::const_from_parts(EntityDomain::SharedOmnipresent, 0);

impl Default for EntityID {
    fn default() -> Self {
        NULL_EID
    }
}

pub trait Component: Default {
    fn name() -> &'static str;
    fn entity_id(&self) -> EntityID;
}

#[derive(Clone, Debug)]
pub enum BoundingShape {
    Point,
    Ball { r: f32 },
    Box { size: Vector3<f32> },
}

#[derive(Clone, Debug)]
pub struct CLocation {
    id: EntityID,
    position: Vector3<f32>,
    velocity: Vector3<f32>,
    orientation: Quaternion<f32>,
    bounding_shape: BoundingShape,
    bounding_offset: Vector3<f32>,
}

impl Default for CLocation {
    fn default() -> Self {
        Self {
            id: NULL_EID,
            position: vec3(0.0, 0.0, 0.0),
            velocity: vec3(0.0, 0.0, 0.0),
            orientation: Quaternion::one(),
            bounding_shape: BoundingShape::Point,
            bounding_offset: vec3(0.0, 0.0, 0.0),
        }
    }
}

impl CLocation {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Component for CLocation {
    fn name() -> &'static str {
        "Location"
    }

    fn entity_id(&self) -> EntityID {
        self.id
    }
}

#[derive(Clone, Debug, Default)]
pub struct CDebugInfo {
    id: EntityID,
    ent_name: String,
}

impl CDebugInfo {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Component for CDebugInfo {
    fn name() -> &'static str {
        "DebugInfo"
    }

    fn entity_id(&self) -> EntityID {
        self.id
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ComponentId<T>(usize, PhantomData<T>);

impl<T> ComponentId<T> {
    pub fn new(i: usize) -> Self {
        Self(i, PhantomData)
    }

    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct Entity {
    pub id: EntityID,
    pub location: Option<ComponentId<CLocation>>,
    pub debug_info: Option<ComponentId<CDebugInfo>>,
}

impl Entity {
    fn new(id: EntityID) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ECS {
    // ECS variables
    entities: HashMap<EntityID, Entity>,
    last_nonfree_ids: [u64; 4],
    // components
    locations: Vec<CLocation>,
    debug_infos: Vec<CDebugInfo>,
}

macro_rules! impl_ecs_fns {
    ( $t:ty, $snake_name:ident, $plural_name:ident ) => {
        impl ECSHandler<$t> for ECS {
            fn has_component(&self, e: EntityID) -> bool {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(|i| &self.$plural_name[i.0]).is_some()
            }

            fn get_component(&self, e: EntityID) -> Option<&$t> {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(|i| &self.$plural_name[i.0])
            }

            fn get_mut_component(&mut self, e: EntityID) -> Option<&mut $t> {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(move |i| &mut self.$plural_name[i.0])
            }

            fn set_component(&mut self, e: EntityID, c: $t) {
                let cid = self
                    .entities
                    .get(&e)
                    .clone()
                    .and_then(|e| e.$snake_name.clone());
                match cid {
                    Some(cid) => {
                        self.$plural_name[cid.0] = c;
                    }
                    None => {
                        let cid = ComponentId::new(self.$plural_name.len());
                        self.$plural_name.push(c);
                        self.entities.get_mut(&e).unwrap().$snake_name = Some(cid);
                    }
                }
            }
        }
    };
}

impl ECS {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn add_new_entity(&mut self, domain: EntityDomain) -> EntityID {
        let nfi = domain.number();
        let mut sub_id = self.last_nonfree_ids[nfi] + 1;
        while self
            .entities
            .contains_key(&EntityID::from_parts(domain, sub_id))
        {
            sub_id += 1;
        }
        self.last_nonfree_ids[nfi] = sub_id;
        let id = EntityID::from_parts(domain, sub_id);
        let iret = self.entities.insert(id, Entity::new(id));
        assert!(iret.is_none());
        id
    }

    #[allow(clippy::map_entry)]
    pub fn add_entity_with_id(&mut self, raw_id: u64) -> Result<EntityID, ()> {
        let id = EntityID(raw_id);
        let sub_id = id.sub_id();
        let nfi = id.domain().number();
        if self.last_nonfree_ids[nfi] < sub_id {
            self.last_nonfree_ids[nfi] = sub_id;
        }
        if self.entities.contains_key(&id) {
            Err(())
        } else {
            self.entities.insert(id, Entity::new(id));
            Ok(id)
        }
    }
}

pub trait ECSHandler<C: Component> {
    fn has_component(&self, e: EntityID) -> bool;
    fn get_component(&self, e: EntityID) -> Option<&C>;
    fn get_mut_component(&mut self, e: EntityID) -> Option<&mut C>;
    fn set_component(&mut self, e: EntityID, c: C);
}

impl_ecs_fns!(CLocation, location, locations);
impl_ecs_fns!(CDebugInfo, debug_info, debug_infos);
