use crate::math::*;

#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Direction {
    XMinus = 0,
    XPlus,
    YMinus,
    YPlus,
    ZMinus,
    ZPlus,
}

pub static ALL_DIRS: [Direction; 6] = {
    use Direction::*;
    [XMinus, XPlus, YMinus, YPlus, ZMinus, ZPlus]
};

impl Direction {
    pub fn all() -> &'static [Direction; 6] {
        &ALL_DIRS
    }

    pub fn opposite(self) -> Self {
        use Direction::*;
        match self {
            XMinus => XPlus,
            XPlus => XMinus,
            YMinus => YPlus,
            YPlus => YMinus,
            ZMinus => ZPlus,
            ZPlus => ZMinus,
        }
    }

    pub fn try_from_vec(v: Vector3<i32>) -> Option<Self> {
        let va: [i32; 3] = v.into();
        match va {
            [1, 0, 0] => Some(Direction::XPlus),
            [-1, 0, 0] => Some(Direction::XMinus),
            [0, 1, 0] => Some(Direction::YPlus),
            [0, -1, 0] => Some(Direction::YMinus),
            [0, 0, 1] => Some(Direction::ZPlus),
            [0, 0, -1] => Some(Direction::ZMinus),
            _ => None,
        }
    }

    pub fn to_vec(self) -> Vector3<i32> {
        use Direction::*;
        match self {
            XMinus => vec3(-1, 0, 0),
            XPlus => vec3(1, 0, 0),
            YMinus => vec3(0, -1, 0),
            YPlus => vec3(0, 1, 0),
            ZMinus => vec3(0, 0, -1),
            ZPlus => vec3(0, 0, 1),
        }
    }

    /// 0..=2
    pub fn to_unsigned_axis_index(self) -> usize {
        use Direction::*;
        match self {
            XMinus => 0,
            XPlus => 0,
            YMinus => 1,
            YPlus => 1,
            ZMinus => 2,
            ZPlus => 2,
        }
    }

    /// 0..=5
    pub fn to_signed_axis_index(self) -> usize {
        use Direction::*;
        match self {
            XMinus => 0,
            XPlus => 1,
            YMinus => 2,
            YPlus => 3,
            ZMinus => 4,
            ZPlus => 5,
        }
    }
}

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
