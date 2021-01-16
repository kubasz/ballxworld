use crate::ecs::*;
use crate::itemregistry::ItemID;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SlotType {
    Item,
    Fluid,
}

pub type StackSize = u32;
pub const DEFAULT_SLOT_CAPACITY: StackSize = 120;

#[derive(Clone, Debug, PartialEq, Hash)]
pub struct InventorySlot {
    name: [u8; 4],
    type_: SlotType,
    capacity: StackSize,
    held_id: ItemID,
    held_count: StackSize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CInventory {
    id: ValidEntityID,
    slots: Vec<InventorySlot>,
}

impl CInventory {
    pub fn new(id: ValidEntityID, slot_count: u32) -> Self {
        let mut inv = Self {
            id,
            slots: Vec::with_capacity(slot_count as usize),
        };
        for islot in 0..slot_count {
            inv.slots.push(InventorySlot {
                name: islot.to_be_bytes(),
                type_: SlotType::Item,
                capacity: DEFAULT_SLOT_CAPACITY,
                held_id: ItemID::default(),
                held_count: 0,
            });
        }
        inv
    }
}

impl Component for CInventory {
    fn name() -> &'static str {
        "Inventory"
    }

    fn entity_id(&self) -> ValidEntityID {
        self.id
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InventoryChange {}
