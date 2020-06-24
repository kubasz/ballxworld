pub use bxw_util::math::*;

pub type WideF32 = f32x8;
pub type WideF64 = f64x8;
pub type WideI32 = i32x8;
pub type WideU32 = u32x8;
pub type WideI64 = i64x8;
pub type WideU64 = u64x8;

/// Returns (random, new seed)
pub fn splitmix64(seed: u64) -> (u64, u64) {
    let newseed = seed + 0x9e3779b97f4a7c15;
    let mut z = newseed;
    z = (z ^ (z >> 30u32)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27u32)) * 0x94d049bb133111eb;
    (z ^ (z >> 31), newseed)
}

/// Returns (random, new seed)
pub fn splitmix64_wide(seed: WideU64) -> (WideU64, WideU64) {
    let newseed = seed + WideU64::splat(0x9e3779b97f4a7c15);
    let mut z = newseed;
    z = (z ^ (z >> 30u32)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27u32)) * 0x94d049bb133111eb;
    (z ^ (z >> 31), newseed)
}

pub fn scramble_seed(seed: u64) -> u64 {
    let (_r, s) = splitmix64(seed);
    let (r, _s) = splitmix64(s);
    r
}

pub fn random2du64_wide(seed: u64, pos: Vector2<WideI32>) -> WideU64 {
    let pos: Vector2<WideU32> = pos.map(|x| x.cast());
    let pos: Vector2<WideU64> = pos.map(|x| x.cast());
    let p = pos.x << 32 | pos.y;
    splitmix64_wide(p ^ seed).0
}

/// -1..=1
/// Slightly biased due to i32 asymmetry in representation
pub fn normalf32_from_i32_wide(i: WideI32) -> WideF32 {
    let maxval = WideF32::splat(i32::max_value() as f32);
    let f: WideF32 = i.cast();
    f / maxval
}

/// -1..=1
pub fn noise2df32(_seed: u64, _pos: Vector2<WideI32>) -> WideF32 {
    //
    WideF32::splat(0.0)
}
