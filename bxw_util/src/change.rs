/// A type for implementing idempotent change tracking mechanics
#[derive(Clone, PartialEq)]
pub enum Change<T: Clone + PartialEq> {
    Unchanged,
    Create { new: T },
    Update { old: T, new: T },
    Destroy { old: T },
}

impl<T: Clone + PartialEq> Default for Change<T> {
    fn default() -> Self {
        Change::Unchanged
    }
}

impl<T: Clone + PartialEq> Change<T> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn is_valid(&self, current: Option<&T>) -> bool {
        match self {
            Change::Unchanged => true,
            Change::Create { .. } => current.is_none(),
            // Use !ne instead of eq to allow for NaN equality
            Change::Update { old, .. } | Change::Destroy { old } => !Some(old).ne(&current),
        }
    }

    /// Applies the change unconditionally (guard with is_valid if idempotency is wanted)
    pub fn apply_with<F: FnOnce(Option<T>)>(self, mutator: F) {
        match self {
            Change::Unchanged => {}
            Change::Create { new } => (mutator)(Some(new)),
            Change::Update { new, .. } => (mutator)(Some(new)),
            Change::Destroy { .. } => (mutator)(None),
        }
    }

    /// Applies the change unconditionally (guard with is_valid if idempotency is wanted)
    pub fn apply(self, data: &mut Option<T>) {
        self.apply_with(|new| *data = new);
    }
}
