use crate::math::*;

/// Axis-Aligned Bounding Box
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AABB {
    pub mins: Vector3<f64>,
    pub maxs: Vector3<f64>,
}

impl Default for AABB {
    fn default() -> Self {
        AABB {
            mins: zero(),
            maxs: zero(),
        }
    }
}

impl AABB {
    fn zero() -> Self {
        Self::default()
    }

    pub fn from_min_max(mins: Vector3<f64>, maxs: Vector3<f64>) -> Self {
        AABB { mins, maxs }
    }

    pub fn from_center_size(center: Vector3<f64>, size: Vector3<f64>) -> Self {
        let halfsize = size / 2.0;
        AABB {
            mins: center - halfsize,
            maxs: center + halfsize,
        }
    }

    pub fn size(&self) -> Vector3<f64> {
        self.maxs - self.mins
    }

    /// Axis for which the size is the smallest
    pub fn smallest_axis(&self) -> usize {
        let sz = self.size();
        if sz.x < sz.y {
            if sz.x < sz.z {
                0
            } else {
                2
            }
        } else if sz.y < sz.z {
            1
        } else {
            2
        }
    }

    pub fn center(&self) -> Vector3<f64> {
        (self.maxs + self.mins) / 2.0
    }

    /// Extend the box's dimensions by 2*amount for each axis, preserving the center
    pub fn inflate(&self, amount: f64) -> Self {
        Self {
            mins: self.mins.add_scalar(-amount),
            maxs: self.maxs.add_scalar(amount),
        }
    }

    pub fn volume(&self) -> f64 {
        let sz = self.size();
        sz.x * sz.y * sz.z
    }

    pub fn surface(&self) -> f64 {
        let sz = self.size();
        sz.x * (sz.y + sz.z) + sz.y * sz.z
    }

    pub fn translate(&self, by: Vector3<f64>) -> Self {
        AABB {
            mins: self.mins + by,
            maxs: self.maxs + by,
        }
    }

    pub fn intersection(a: Self, b: Self) -> Option<AABB> {
        let mut prod = AABB::zero();
        // Decompose into per-axis intersection
        for c in 0..3 {
            let (leftmost, rightmost) = if a.mins[c] < b.mins[c] {
                (a, b)
            } else {
                (b, a)
            };
            // disjoint
            if leftmost.maxs[c] < rightmost.mins[c] {
                return None;
            }
            // embedded
            else if leftmost.maxs[c] > rightmost.maxs[c] {
                prod.mins[c] = rightmost.mins[c];
                prod.maxs[c] = rightmost.maxs[c];
            }
            // intersecting
            else {
                prod.mins[c] = rightmost.mins[c];
                prod.maxs[c] = leftmost.maxs[c];
            }
        }
        Some(prod)
    }
}
