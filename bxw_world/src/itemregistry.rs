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

