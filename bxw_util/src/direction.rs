use crate::math::*;
use itertools::Itertools;

/// A direction in the left-handed coordinate system of the game
#[repr(i32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Direction {
    /// Left
    XMinus = 0,
    /// Right
    XPlus,
    /// Down/Bottom
    YMinus,
    /// Up/Top
    YPlus,
    /// Front (out of the screen)
    ZMinus,
    /// Back (into the screen)
    ZPlus,
}

pub const DIR_LEFT: Direction = Direction::XMinus;
pub const DIR_RIGHT: Direction = Direction::XPlus;
pub const DIR_DOWN: Direction = Direction::YMinus;
pub const DIR_UP: Direction = Direction::YPlus;
/// Out of the screen
pub const DIR_FRONT: Direction = Direction::ZMinus;
/// Into the screen
pub const DIR_BACK: Direction = Direction::ZPlus;

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

    pub fn from_approx_vecf(v: Vector3<f32>) -> Self {
        if v.is_zero() {
            DIR_UP
        } else {
            let vc: [f32; 3] = v.into();
            let maxaxis = vc
                .iter()
                .map(|x| x.abs())
                .position_max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1);
            Self::from_signed_axis_index(maxaxis * 2 + if vc[maxaxis] < 0.0 { 0 } else { 1 })
                .unwrap()
        }
    }

    pub fn rotate_by_mat(self, m: Matrix3<i32>) -> Option<Self> {
        let v = self.to_vec();
        let nv = m * v;
        Self::try_from_vec(nv)
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

    pub fn from_signed_axis_index(idx: usize) -> Option<Self> {
        use Direction::*;
        match idx {
            0 => Some(XMinus),
            1 => Some(XPlus),
            2 => Some(YMinus),
            3 => Some(YPlus),
            4 => Some(ZMinus),
            5 => Some(ZPlus),
            _ => None,
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

    pub fn is_positive(self) -> bool {
        use Direction::*;
        match self {
            XMinus | YMinus | ZMinus => false,
            XPlus | YPlus | ZPlus => true,
        }
    }

    pub fn is_negative(self) -> bool {
        use Direction::*;
        match self {
            XMinus | YMinus | ZMinus => true,
            XPlus | YPlus | ZPlus => false,
        }
    }

    /// Vector cross product of two directions (left-handed)
    pub fn lh_cross(a: Self, b: Self) -> Option<Self> {
        let aidx = a.to_signed_axis_index();
        let bidx = b.to_signed_axis_index();
        DIRECTION_CROSS_TABLE[aidx * 6 + bidx]
    }
}

/// One of the 24 possible orientations for an octahedron or cube; right cross(left-handed) up == front
#[derive(Copy, Clone, Eq, PartialEq, Debug, Hash)]
pub struct OctahedralOrientation {
    right: Direction,
    up: Direction,
}

impl Default for OctahedralOrientation {
    fn default() -> Self {
        Self {
            right: DIR_RIGHT,
            up: DIR_UP,
        }
    }
}

impl OctahedralOrientation {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_dirs(right: Direction, up: Direction, front: Direction) -> Option<Self> {
        if Direction::lh_cross(right, up) != Some(front) {
            None
        } else {
            Some(Self { up, right })
        }
    }

    pub fn from_right_up(right: Direction, up: Direction) -> Option<Self> {
        if Direction::lh_cross(right, up).is_none() {
            None
        } else {
            Some(Self { up, right })
        }
    }

    pub fn from_up_front(up: Direction, front: Direction) -> Option<Self> {
        if let Some(right) = Direction::lh_cross(up, front) {
            Some(Self { up, right })
        } else {
            None
        }
    }

    pub fn from_front_right(front: Direction, right: Direction) -> Option<Self> {
        if let Some(up) = Direction::lh_cross(front, right) {
            Some(Self { up, right })
        } else {
            None
        }
    }

    /// Converts itself into an index in the range 0..24 (not inclusive)
    pub fn to_index(self) -> usize {
        // 0..6
        let right_idx = self.right.to_signed_axis_index();
        // 0..4
        let up_idx = {
            let i = self.up.to_signed_axis_index();
            if i > right_idx {
                i - 2
            } else {
                i
            }
        };
        // front is always determined by the cross product
        right_idx * 4 + up_idx
    }

    /// Converts an index (0..24, as returned from to_index) to an orientation
    pub fn from_index(i: usize) -> Option<Self> {
        if i >= 24 {
            None
        } else {
            let right_idx = i / 4;
            let right = Direction::from_signed_axis_index(right_idx).unwrap();
            let up_idx = i % 4;
            let up_idx = if (up_idx / 2) >= (right_idx / 2) {
                up_idx + 2
            } else {
                up_idx
            };
            let up = Direction::from_signed_axis_index(up_idx).unwrap();
            Some(Self { right, up })
        }
    }

    pub fn right(self) -> Direction {
        self.right
    }

    pub fn up(self) -> Direction {
        self.up
    }

    pub fn front(self) -> Direction {
        Direction::lh_cross(self.right, self.up).unwrap()
    }

    pub fn left(self) -> Direction {
        self.right.opposite()
    }

    pub fn down(self) -> Direction {
        self.up.opposite()
    }

    pub fn back(self) -> Direction {
        Direction::lh_cross(self.up, self.right).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn orientation_permutation_count() {
        let mut allowed = 0;
        for &d1 in &ALL_DIRS {
            for &d2 in &ALL_DIRS {
                for &d3 in &ALL_DIRS {
                    if let Some(_orientation) = OctahedralOrientation::from_dirs(d1, d2, d3) {
                        allowed += 1;
                    }
                }
            }
        }
        assert_eq!(allowed, 24);
    }

    #[test]
    fn orientation_construction() {
        let mut indices_used = HashSet::new();
        for &d1 in &ALL_DIRS {
            for &d2 in &ALL_DIRS {
                for &d3 in &ALL_DIRS {
                    if let Some(orn) = OctahedralOrientation::from_dirs(d1, d2, d3) {
                        assert_eq!(orn.right(), d1);
                        assert_eq!(orn.up(), d2);
                        assert_eq!(orn.front(), d3);
                        assert_eq!(orn.left(), d1.opposite());
                        assert_eq!(orn.down(), d2.opposite());
                        assert_eq!(orn.back(), d3.opposite());
                        let idx = orn.to_index();
                        assert_eq!(Some(orn), OctahedralOrientation::from_index(idx));
                        assert_eq!(indices_used.insert(idx), true);
                        assert_eq!(
                            Some(orn),
                            OctahedralOrientation::from_dirs(orn.right(), orn.up(), orn.front())
                        );
                        assert_eq!(
                            Some(orn),
                            OctahedralOrientation::from_right_up(orn.right(), orn.up())
                        );
                        assert_eq!(
                            Some(orn),
                            OctahedralOrientation::from_up_front(orn.up(), orn.front())
                        );
                        assert_eq!(
                            Some(orn),
                            OctahedralOrientation::from_front_right(orn.front(), orn.right())
                        );
                    }
                }
            }
        }
        assert_eq!(indices_used.len(), 24);
        assert!(indices_used.iter().all(|&n| n < 24));
        let default_orientation = OctahedralOrientation::default();
        assert_eq!(default_orientation.right(), DIR_RIGHT);
        assert_eq!(default_orientation.up(), DIR_UP);
        assert_eq!(default_orientation.front(), DIR_FRONT);
        assert_eq!(default_orientation.left(), DIR_LEFT);
        assert_eq!(default_orientation.down(), DIR_DOWN);
        assert_eq!(default_orientation.back(), DIR_BACK);
    }

    #[test]
    fn direction_cross_verify() {
        let mut non_zero = 0;
        for &d1 in &ALL_DIRS {
            for &d2 in &ALL_DIRS {
                let v1 = d1.to_vec();
                let v2 = d2.to_vec();
                let vcross = -v1.cross(&v2);
                let vdcross = Direction::try_from_vec(vcross);
                let dcross = Direction::lh_cross(d1, d2);
                //eprintln!("{:?}, // {:?}*{:?}", vdcross, d1, d2);
                assert_eq!(dcross, vdcross);
                if dcross.is_some() {
                    non_zero += 1;
                }
            }
        }
        assert_eq!(non_zero, 24);
        assert_eq!(Direction::lh_cross(DIR_RIGHT, DIR_UP), Some(DIR_FRONT));
        assert_eq!(Direction::lh_cross(DIR_UP, DIR_FRONT), Some(DIR_RIGHT));
        assert_eq!(Direction::lh_cross(DIR_FRONT, DIR_RIGHT), Some(DIR_UP));
    }
}

/// Cross product table for a pair of directions, indexed by 6*a+b (a,b being signed axis indices)
const DIRECTION_CROSS_TABLE: [Option<Direction>; 36] = {
    use Direction::*;
    [
        None,         // XMinus*XMinus
        None,         // XMinus*XPlus
        Some(ZMinus), // XMinus*YMinus
        Some(ZPlus),  // XMinus*YPlus
        Some(YPlus),  // XMinus*ZMinus
        Some(YMinus), // XMinus*ZPlus
        None,         // XPlus*XMinus
        None,         // XPlus*XPlus
        Some(ZPlus),  // XPlus*YMinus
        Some(ZMinus), // XPlus*YPlus
        Some(YMinus), // XPlus*ZMinus
        Some(YPlus),  // XPlus*ZPlus
        Some(ZPlus),  // YMinus*XMinus
        Some(ZMinus), // YMinus*XPlus
        None,         // YMinus*YMinus
        None,         // YMinus*YPlus
        Some(XMinus), // YMinus*ZMinus
        Some(XPlus),  // YMinus*ZPlus
        Some(ZMinus), // YPlus*XMinus
        Some(ZPlus),  // YPlus*XPlus
        None,         // YPlus*YMinus
        None,         // YPlus*YPlus
        Some(XPlus),  // YPlus*ZMinus
        Some(XMinus), // YPlus*ZPlus
        Some(YMinus), // ZMinus*XMinus
        Some(YPlus),  // ZMinus*XPlus
        Some(XPlus),  // ZMinus*YMinus
        Some(XMinus), // ZMinus*YPlus
        None,         // ZMinus*ZMinus
        None,         // ZMinus*ZPlus
        Some(YPlus),  // ZPlus*XMinus
        Some(YMinus), // ZPlus*XPlus
        Some(XMinus), // ZPlus*YMinus
        Some(XPlus),  // ZPlus*YPlus
        None,         // ZPlus*ZMinus
        None,         // ZPlus*ZPlus
    ]
};
