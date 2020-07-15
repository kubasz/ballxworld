pub use bxw_util::blake3::Hasher;
pub use bxw_util::bytemuck::{bytes_of, bytes_of_mut, from_bytes, from_bytes_mut};
pub use bxw_util::math::*;
use bxw_util::smallvec::SmallVec;

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

pub fn remap_f(v: f64, oldmin: f64, oldmax: f64, newmin: f64, newmax: f64) -> f64 {
    let norm = (v - oldmin) / (oldmax - oldmin);
    norm * (newmax - newmin) + newmin
}

pub fn remap_v(v: WideF64, oldmin: f64, oldmax: f64, newmin: f64, newmax: f64) -> WideF64 {
    let norm = (v - WideF64::splat(oldmin)) / WideF64::splat(oldmax - oldmin);
    norm * WideF64::splat(newmax - newmin) + WideF64::splat(newmin)
}

pub fn smoothstep(v: f64) -> f64 {
    if v <= 0.0 {
        0.0
    } else if v >= 1.0 {
        1.0
    } else {
        let v2 = v * v;
        3.0 * v2 - 2.0 * v2 * v
    }
}

pub fn line_segment_intersection2(
    s1: Vector2<f64>,
    s2: Vector2<f64>,
    l1: Vector2<f64>,
    l2: Vector2<f64>,
) -> Option<Vector2<f64>> {
    let t_den: f64 = Matrix2::from_columns(&[s1 - s2, l1 - l2]).determinant();
    if t_den.abs() < 1.0e-6 {
        return None;
    }
    let t_num: f64 = Matrix2::from_columns(&[s1 - l1, l1 - l2]).determinant();
    let t = t_num / t_den;
    if t < -1.0e-6 || t > 1.0 + 1.0e-6 {
        None
    } else {
        Some(s1 + t * (s2 - s1))
    }
}

pub fn segment_point_distance_sq(s1: Vector2<f64>, s2: Vector2<f64>, p: Vector2<f64>) -> f64 {
    let n = s2 - s1;
    let pa = s1 - p;
    let c = n.dot(&pa);
    if c > 0.0 {
        return pa.dot(&pa);
    }
    let bp = p - s2;
    if n.dot(&bp) > 0.0 {
        return bp.dot(&bp);
    }
    let e = pa - n * (c / n.dot(&n));
    e.dot(&e)
}

#[test]
fn test_ls_intersect2() {
    assert_eq!(
        line_segment_intersection2(
            vec2(-1.0, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, -1.0),
            vec2(0.0, 1.0)
        ),
        Some(vec2(0.0, 0.0))
    );
    assert_eq!(
        line_segment_intersection2(
            vec2(-1.0, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, 1.0),
            vec2(0.0, -1.0)
        ),
        Some(vec2(0.0, 0.0))
    );
    assert_eq!(
        line_segment_intersection2(
            vec2(1.0, 0.0),
            vec2(-1.0, 0.0),
            vec2(0.0, -1.0),
            vec2(0.0, 1.0)
        ),
        Some(vec2(0.0, 0.0))
    );
    assert_eq!(
        line_segment_intersection2(
            vec2(1.0, 0.0),
            vec2(-1.0, 0.0),
            vec2(0.0, 1.0),
            vec2(0.0, -1.0)
        ),
        Some(vec2(0.0, 0.0))
    );
    assert_eq!(
        line_segment_intersection2(
            vec2(-1.0, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, 0.5),
            vec2(0.0, 1.0)
        ),
        Some(vec2(0.0, 0.0))
    );
    assert_eq!(
        line_segment_intersection2(
            vec2(0.5, 0.0),
            vec2(1.0, 0.0),
            vec2(0.0, -1.0),
            vec2(0.0, 1.0)
        ),
        None
    );
}

#[derive(Clone)]
pub struct ConvexHull<D: Clone> {
    /// Start point of an edge and the associated data
    pub half_edges: SmallVec<[(Vector2<f64>, D); 8]>,
}

impl<D: Clone> Default for ConvexHull<D> {
    fn default() -> Self {
        Self {
            half_edges: SmallVec::new(),
        }
    }
}

impl<D: Clone> ConvexHull<D> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_point(&mut self, pos: Vector2<f64>, data: D) {
        self.half_edges.push((pos, data));
    }
    /*
        /// Returns true if an intersection was found, false if the line is outside the hull
        pub fn crop(&mut self, line1: Vector2<f64>, line2: Vector2<f64>, point_in_center: Vector2<f64>, data: D) -> bool {
            let intit = self.half_edges.iter().enumerate().zip(self.half_edges.iter().enumerate().cycle().skip(1))
                .filter_map(|((i1, p1),(i2, p2))| line_segment_intersection2(p1.0, p2.0, line1, line2).map(|inters| (i1, i2, inters)));
            let int1 = intit.next();
            let int2 = intit.next();
            if let Some((int1, int2)) = int1.zip(int2) {
                let (it1i1, it1i2, it1p) = int1;
                let (it2i1, it2i2, it2p) = int2;
                let perpaxis1 = (it1p + it2p) / 2.0;
                let perpaxis2 = point_in_center;
                let perpaxis_inward = perpaxis2 - perpaxis1;
                let it1s1 = self.half_edges[it1i1].0;
                let it1s2 = self.half_edges[it1i2].0;
                let it2s1 = self.half_edges[it2i1].0;
                let it2s2 = self.half_edges[it2i2].0;
                if (it1s1 - perpaxis1).dot(perpaxis_inward) <= 0 {
                    // remove it2i2 ..= it1i1
                    if it2i2 == 0 {
                        // wrapped around - remove until last segment

                    }
                }
                true
            } else {
                false
            }
        }
    */
}
