#[macro_export]
macro_rules! offset_of {
    ($ty:ty, $field:ident $(,)?) => {
        unsafe {
            let null: *const $ty = std::ptr::null();
            &(*null).$field as *const _ as usize
        }
    };
}
