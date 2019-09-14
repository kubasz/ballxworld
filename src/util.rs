#[macro_export]
macro_rules! offset_of {
    ($ty:ty, $field:ident $(,)?) => {
        unsafe {
            let null = 0usize as *const $ty;
            std::mem::transmute::<_, usize>(&(*null).$field as *const _)
        }
    };
}
