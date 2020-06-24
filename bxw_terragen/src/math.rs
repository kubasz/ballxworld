pub use bxw_util::math::*;

pub fn splitmix64(seed: u64x4) -> u64x4 {
    let mut z = seed + u64x4::splat(0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30u32)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27u32)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}


