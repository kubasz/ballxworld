use bxw_util::change::Change;
use bxw_util::collider::AABB;
use bxw_util::fnv::*;
use bxw_util::math::*;
use bxw_util::sparsevec::*;
use std::cell::*;
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
pub struct ValidEntityID(u64);

pub type EntityID = Option<ValidEntityID>;

const DOMAIN_MASK: u64 = 0b11_u64 << 62;

impl ValidEntityID {
    pub fn from_raw(r: u64) -> EntityID {
        if r & !DOMAIN_MASK == 0 {
            None
        } else {
            Some(ValidEntityID(r))
        }
    }

    pub fn from_parts(domain: EntityDomain, sub_id: u64) -> EntityID {
        assert_eq!(sub_id & DOMAIN_MASK, 0);
        if sub_id == 0 {
            None
        } else {
            Some(ValidEntityID(domain as u64 | sub_id))
        }
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

pub trait Component: Clone + PartialEq {
    fn name() -> &'static str;
    fn entity_id(&self) -> ValidEntityID;
}

#[derive(Clone, Debug, PartialEq)]
pub enum BoundingShape {
    Point { offset: Vector3<f64> },
    AxisAlignedBox(AABB),
}

impl BoundingShape {
    pub fn aabb(&self, position: Vector3<f64>) -> AABB {
        match self {
            BoundingShape::Point { offset } => {
                AABB::from_center_size(position + offset, vec3(0.05, 0.05, 0.05))
            }
            BoundingShape::AxisAlignedBox(aabb) => aabb.translate(position),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CLocation {
    id: ValidEntityID,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub orientation: UnitQuaternion<f64>,
    pub bounding_shape: BoundingShape,
}

impl CLocation {
    pub fn new(id: ValidEntityID) -> Self {
        Self {
            id,
            position: vec3(0.0, 0.0, 0.0),
            velocity: vec3(0.0, 0.0, 0.0),
            orientation: one(),
            bounding_shape: BoundingShape::Point { offset: zero() },
        }
    }
}

impl Component for CLocation {
    fn name() -> &'static str {
        "Location"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CPhysics {
    id: ValidEntityID,
    pub frozen: bool,
    pub mass: f64,
    /// X-, X+, Y-, Y+, Z-, Z+
    pub against_wall: [bool; 6],
    pub control_target_velocity: Vector3<f64>,
    pub control_max_force: Vector3<f64>,
    /// Acceleration impulse applied on the next physics tick (and then reset to 0)
    pub control_frame_impulse: Vector3<f64>,
}

impl CPhysics {
    pub fn new(id: ValidEntityID) -> Self {
        Self {
            id,
            frozen: false,
            mass: 1.0,
            against_wall: [false; 6],
            control_target_velocity: vec3(0.0, 0.0, 0.0),
            control_max_force: vec3(0.0, 0.0, 0.0),
            control_frame_impulse: vec3(0.0, 0.0, 0.0),
        }
    }
}

impl Component for CPhysics {
    fn name() -> &'static str {
        "Physics"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CDebugInfo {
    id: ValidEntityID,
    pub ent_name: String,
}

impl CDebugInfo {
    pub fn new(id: ValidEntityID, ent_name: String) -> Self {
        Self { id, ent_name }
    }
}

impl Component for CDebugInfo {
    fn name() -> &'static str {
        "DebugInfo"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CLoadAnchor {
    id: ValidEntityID,
    pub radius: u32,
    pub load_mesh: bool,
}

impl CLoadAnchor {
    pub fn new(id: ValidEntityID, radius: u32, load_mesh: bool) -> Self {
        Self {
            id,
            radius,
            load_mesh,
        }
    }
}

impl Component for CLoadAnchor {
    fn name() -> &'static str {
        "LoadAnchor"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub struct ComponentId<T: Component>(usize, PhantomData<T>);

impl<T: Component> Copy for ComponentId<T> {}

impl<T: Component> ComponentId<T> {
    pub fn new(i: usize) -> Self {
        Self(i, PhantomData)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: ValidEntityID,
    pub location: Option<ComponentId<CLocation>>,
    pub physics: Option<ComponentId<CPhysics>>,
    pub debug_info: Option<ComponentId<CDebugInfo>>,
    pub load_anchor: Option<ComponentId<CLoadAnchor>>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum EntityChangeKind {
    NewEntity(ValidEntityID),
    UpdateEntity(ValidEntityID),
    DeleteEntity(ValidEntityID),
}

impl Default for EntityChangeKind {
    fn default() -> Self {
        Self::NewEntity(ValidEntityID(0))
    }
}

#[derive(Default, Clone, PartialEq)]
pub struct EntityChange {
    pub kind: EntityChangeKind,
    pub location: Change<CLocation>,
    pub physics: Change<CPhysics>,
    pub debug_info: Change<CDebugInfo>,
    pub load_anchor: Change<CLoadAnchor>,
}

impl Entity {
    fn new(id: ValidEntityID) -> Self {
        Self {
            id,
            location: None,
            physics: None,
            debug_info: None,
            load_anchor: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ECS {
    // ECS variables
    entities: FnvHashMap<ValidEntityID, Entity>,
    last_nonfree_ids: Cell<[u64; 4]>,
    // components
    locations: SparseVec<CLocation>,
    physicss: SparseVec<CPhysics>,
    debug_infos: SparseVec<CDebugInfo>,
    load_anchors: SparseVec<CLoadAnchor>,
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub enum AddEntityError {
    AlreadyExists,
    InvalidRawID,
}

impl ECS {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn iter(&self) -> std::collections::hash_map::Values<'_, ValidEntityID, Entity> {
        self.entities.values()
    }

    pub fn iter_mut(&mut self) -> std::collections::hash_map::ValuesMut<'_, ValidEntityID, Entity> {
        self.entities.values_mut()
    }

    pub fn allocate_id(&self, domain: EntityDomain) -> ValidEntityID {
        let nfi = domain.number();
        let mut sub_id = self.last_nonfree_ids.get()[nfi] + 1;
        while self
            .entities
            .contains_key(&ValidEntityID::from_parts(domain, sub_id).unwrap())
        {
            sub_id += 1;
        }
        let mut nfis = self.last_nonfree_ids.get();
        nfis[nfi] = sub_id;
        self.last_nonfree_ids.set(nfis);
        ValidEntityID::from_parts(domain, sub_id).unwrap()
    }

    pub fn add_new_entity(&mut self, domain: EntityDomain) -> ValidEntityID {
        let id = self.allocate_id(domain);
        let iret = self.entities.insert(id, Entity::new(id));
        assert!(iret.is_none());
        id
    }

    #[allow(clippy::map_entry)]
    pub fn add_entity_with_id(&mut self, raw_id: u64) -> Result<ValidEntityID, AddEntityError> {
        let id = ValidEntityID::from_raw(raw_id).ok_or(AddEntityError::InvalidRawID)?;
        let sub_id = id.sub_id();
        let nfi = id.domain().number();
        if self.last_nonfree_ids.get()[nfi] < sub_id {
            self.last_nonfree_ids.get_mut()[nfi] = sub_id;
        }
        if self.entities.contains_key(&id) {
            Err(AddEntityError::AlreadyExists)
        } else {
            self.entities.insert(id, Entity::new(id));
            Ok(id)
        }
    }

    fn add_entity_with_preallocated_id(&mut self, id: ValidEntityID) -> ValidEntityID {
        self.add_entity_with_id(id.u64()).unwrap()
    }

    pub fn delete_entity(&mut self, id: ValidEntityID) {
        let ent = self.entities.remove(&id).unwrap();
        if let Some(ComponentId(cid, _)) = ent.physics {
            self.physicss.remove(cid);
        }
        if let Some(ComponentId(cid, _)) = ent.location {
            self.locations.remove(cid);
        }
        if let Some(ComponentId(cid, _)) = ent.debug_info {
            self.debug_infos.remove(cid);
        }
        if let Some(ComponentId(cid, _)) = ent.load_anchor {
            self.load_anchors.remove(cid);
        }
    }

    pub fn apply_entity_changes(&mut self, changes: &[EntityChange]) {
        for change in changes {
            let eid = match change.kind {
                EntityChangeKind::NewEntity(id) => self.add_entity_with_preallocated_id(id),
                EntityChangeKind::UpdateEntity(id) => id,
                EntityChangeKind::DeleteEntity(id) => {
                    if self.entities.contains_key(&id) {
                        self.delete_entity(id);
                    }
                    continue;
                }
            };
            if !self.entities.contains_key(&eid) {
                continue;
            }
            self.change_component(eid, change.location.clone());
            self.change_component(eid, change.physics.clone());
            self.change_component(eid, change.debug_info.clone());
            self.change_component(eid, change.load_anchor.clone());
        }
    }
}

pub trait ECSHandler<C: Component> {
    fn has_component(&self, e: ValidEntityID) -> bool;
    fn get_component(&self, e: ValidEntityID) -> Option<&C>;
    fn get_component_mut(&mut self, e: ValidEntityID) -> Option<&mut C>;
    fn set_component(&mut self, e: ValidEntityID, c: C);
    fn remove_component(&mut self, e: ValidEntityID);
    fn change_component(&mut self, e: ValidEntityID, change: Change<C>);
    fn iter(&self) -> SparseVecIter<C>;
    fn iter_mut(&mut self) -> SparseVecIterMut<C>;
}

macro_rules! impl_ecs_fns {
    ( $t:ty, $snake_name:ident, $plural_name:ident ) => {
        impl ECSHandler<$t> for ECS {
            fn has_component(&self, e: ValidEntityID) -> bool {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(|i| &self.$plural_name[i.0]).is_some()
            }

            fn get_component(&self, e: ValidEntityID) -> Option<&$t> {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(|i| &self.$plural_name[i.0])
            }

            fn get_component_mut(&mut self, e: ValidEntityID) -> Option<&mut $t> {
                let cid = self.entities.get(&e).unwrap().$snake_name.clone();
                cid.map(move |i| &mut self.$plural_name[i.0])
            }

            fn iter(&self) -> SparseVecIter<$t> {
                self.$plural_name.iter()
            }

            fn iter_mut(&mut self) -> SparseVecIterMut<$t> {
                self.$plural_name.iter_mut()
            }

            fn set_component(&mut self, e: ValidEntityID, c: $t) {
                let cid = self.entities.get(&e).and_then(|e| e.$snake_name);
                match cid {
                    Some(cid) => {
                        self.$plural_name[cid.0] = c;
                    }
                    None => {
                        let cid = ComponentId::new(self.$plural_name.add(c));
                        self.entities.get_mut(&e).unwrap().$snake_name = Some(cid);
                    }
                }
            }

            fn remove_component(&mut self, e: ValidEntityID) {
                let cid = self.entities.get(&e).and_then(|e| e.$snake_name);
                match cid {
                    Some(cidv) => {
                        self.$plural_name.remove(cidv.0);
                        self.entities.get_mut(&e).unwrap().$snake_name = None;
                    }
                    None => {
                        panic!(
                            "Trying to remove non-existing component {} on entity {:?}",
                            <$t>::name(),
                            e
                        );
                    }
                }
            }

            fn change_component(&mut self, e: ValidEntityID, change: Change<$t>) {
                let old_value: Option<&$t> = self.get_component(e);
                if change.is_valid(old_value) {
                    let old_some = old_value.is_some();
                    change.apply_with(|new_value| {
                        match (old_some, new_value) {
                            (false, None) => {} // no-op
                            (_, Some(nc)) => {
                                self.set_component(e, nc);
                            } // set
                            (true, None) => {
                                <Self as ECSHandler<$t>>::remove_component(self, e);
                            } // delete
                        };
                    })
                }
            }
        }
    };
}

impl_ecs_fns!(CLocation, location, locations);
impl_ecs_fns!(CPhysics, physics, physicss);
impl_ecs_fns!(CDebugInfo, debug_info, debug_infos);
impl_ecs_fns!(CLoadAnchor, load_anchor, load_anchors);
