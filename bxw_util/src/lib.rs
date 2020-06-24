#![allow(clippy::unused_unit)]

// Re-export dependencies
pub use divrem;
pub use fnv;
pub use glm;
pub use itertools;
pub use lazy_static;
pub use lru;
pub use nalgebra as na;
pub use num_cpus;
pub use num_traits;
pub use packed_simd;
pub use parking_lot;
pub use rand;
pub use rand_xoshiro;
pub use rayon;
pub use regex;
pub use simba;
pub use smallvec;
pub use thread_local_crate as thread_local;
pub use toml_edit;

// Own modules
pub mod change;
pub mod collider;
pub mod debug_data;
pub mod direction;
pub mod math;
pub mod sparsevec;
pub mod taskpool;
