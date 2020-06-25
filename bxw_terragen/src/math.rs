pub use bxw_util::math::*;

pub type WideCptr<T> = cptrx8<T>;
pub type WideMptr<T> = mptrx8<T>;
pub type WideF32 = f32x8;
pub type WideF64 = f64x8;
pub type WideI8 = i8x8;
pub type WideU8 = u8x8;
pub type WideI16 = i16x8;
pub type WideU16 = u16x8;
pub type WideI32 = i32x8;
pub type WideU32 = u32x8;
pub type WideI64 = i64x8;
pub type WideU64 = u64x8;
pub type WideM32 = m32x8;
pub type WideM64 = m64x8;
pub type WideIsize = isizex8;
pub type WideUsize = usizex8;

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

pub fn widefloor_f64_i32(x: WideF64) -> WideI32 {
    let xi: WideI32 = x.cast();
    let xw: WideF64 = xi.cast();
    let smaller = x.lt(xw);
    if smaller.any() {
        let xim1 = xi - 1i32;
        smaller.select(xim1, xi)
    } else {
        xi
    }
}

#[test]
fn test_widefloor() {
    let xf = WideF64::new(0.0, 0.5, 1.5, -0.5, -1.5, -3.0, 3.0, 8.0);
    let xi = widefloor_f64_i32(xf);
    assert_eq!(xi, WideI32::new(0, 0, 1, -1, -2, -3, 3, 8));
}
