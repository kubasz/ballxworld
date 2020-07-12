pub use bxw_util::blake3::Hasher;
pub use bxw_util::bytemuck::{bytes_of, bytes_of_mut, from_bytes, from_bytes_mut};
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
    let newseed = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = newseed;
    z = (z ^ (z >> 30u32)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27u32)).wrapping_mul(0x94d049bb133111eb);
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

pub fn seeded_hasher(seed: u64, context: &str) -> Hasher {
    let mut h = Hasher::new_derive_key(context);
    h.update(bytes_of(&seed));
    h
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

pub fn perp_vector(v: Vector2<f64>) -> Vector2<f64> {
    vec2(v.y, -v.x)
}

pub fn line_segment_intersection2(s1: Vector2<f64>, s2: Vector2<f64>, l1: Vector2<f64>, l2: Vector2<f64>) -> Option<Vector2<f64>> {
    let t_den: f64 = Matrix2::from_columns(&[s1 - s2, l1 - l2]).determinant();
    if t_den.abs() < 1.0e-6 {
        return None;
    }
    let t_num: f64 = Matrix2::from_columns(&[s1 - l1,l1 - l2]).determinant();
    let t = t_num / t_den;
    if t < 0.0 || t > 1.0 {
        None
    } else {
        Some(s1 + t * (s2 - s1))
    }
}

#[test]
fn test_ls_intersect2() {
    assert_eq!(line_segment_intersection2(vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, -1.0), vec2(0.0, 1.0)), Some(vec2(0.0, 0.0)));
    assert_eq!(line_segment_intersection2(vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0)), Some(vec2(0.0, 0.0)));
    assert_eq!(line_segment_intersection2(vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, -1.0), vec2(0.0, 1.0)), Some(vec2(0.0, 0.0)));
    assert_eq!(line_segment_intersection2(vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0)), Some(vec2(0.0, 0.0)));
    assert_eq!(line_segment_intersection2(vec2(-1.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 0.5), vec2(0.0, 1.0)), Some(vec2(0.0, 0.0)));
    assert_eq!(line_segment_intersection2(vec2(0.5, 0.0), vec2(1.0, 0.0), vec2(0.0, -1.0), vec2(0.0, 1.0)), None);
}
