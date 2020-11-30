pub type ItemID = u16;

#[derive(Clone, Debug, Hash)]
pub struct ItemDefinition {
    pub id: ItemID,
    pub name: String,
}

#[derive(Clone)]
pub struct ItemRegistry {
    //
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct ItemStack {
    id: ItemID,
    quantity: i64,
}
