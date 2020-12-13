#![allow(clippy::unused_unit)]

// Re-export dependencies
pub use blake3;
pub use bytemuck;
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
pub use rand_distr;
pub use rand_xoshiro;
pub use rayon;
pub use regex;
pub use rmp;
pub use rmp_serde;
pub use rstar;
pub use scopeguard;
pub use serde;
pub use simba;
pub use smallvec;
pub use sodiumoxide;
pub use thread_local_crate as thread_local;
pub use toml;
pub use toml_edit;
pub use zstd;

// Own modules
pub mod change;
pub mod collider;
pub mod debug_data;
pub mod direction;
pub mod math;
pub mod sparsevec;
pub mod taskpool;
