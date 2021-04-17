use crate::math::*;
use itertools::Itertools;
use lazy_static::*;
use std::fmt::Debug;

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
#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct OctahedralOrientation {
    right: Direction,
    up: Direction,
}

impl Debug for OctahedralOrientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OctoOrientation {{right: {:?}, up: {:?}, front: {:?}}}",
            self.right(),
            self.up(),
            self.front()
        )
    }
}

impl Default for OctahedralOrientation {
    fn default() -> Self {
        Self {
            right: DIR_RIGHT,
            up: DIR_UP,
        }
    }
}

lazy_static! {
    static ref APPLY_UNAPPLY_LUT: [(Direction, Direction); 6 * 24] = init_apply_unapply_lut();
}

fn init_apply_unapply_lut() -> [(Direction, Direction); 6 * 24] {
    let mut lut = [(DIR_FRONT, DIR_FRONT); 6 * 24];
    for idir in 0..6 {
        for ior in 0..24 {
            let dir = Direction::from_signed_axis_index(idir).unwrap();
            let or = OctahedralOrientation::from_index(ior).unwrap();
            let applied = Direction::try_from_vec(or.to_matrixi() * dir.to_vec()).unwrap();
            let unapplied =
                Direction::try_from_vec(or.to_matrixi().transpose() * dir.to_vec()).unwrap();
            lut[idir * 24 + ior] = (applied, unapplied);
        }
    }
    lut
}

impl OctahedralOrientation {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_dirs(right: Direction, up: Direction, front: Direction) -> Option<Self> {
        if Direction::lh_cross(right, up) != Some(front) {
            None
        } else {
            Some(Self { right, up })
        }
    }

    pub fn from_right_up(right: Direction, up: Direction) -> Option<Self> {
        if Direction::lh_cross(right, up).is_none() {
            None
        } else {
            Some(Self { right, up })
        }
    }

    pub fn from_up_front(up: Direction, front: Direction) -> Option<Self> {
        Direction::lh_cross(up, front).map(|right| Self { right, up })
    }

    pub fn from_front_right(front: Direction, right: Direction) -> Option<Self> {
        Direction::lh_cross(front, right).map(|up| Self { right, up })
    }

    pub fn from_matrixi(matrix: Matrix3<i32>) -> Option<Self> {
        let vright = matrix * DIR_RIGHT.to_vec();
        let vup = matrix * DIR_UP.to_vec();
        let right = Direction::try_from_vec(vright)?;
        let up = Direction::try_from_vec(vup)?;
        Self::from_right_up(right, up)
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
        // make it so that 0 is the default orientation (X+ right, Y+ up)
        (right_idx * 4 + up_idx + 24 - 5) % 24
    }

    /// Converts an index (0..24, as returned from to_index) to an orientation
    pub fn from_index(i: usize) -> Option<Self> {
        if i >= 24 {
            None
        } else {
            // adjust for default orientation offset
            let i = (i + 5) % 24;
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

    /// M * v will rotate the vector v to match this orientation
    pub fn to_matrixi(self) -> Matrix3<i32> {
        Matrix3::from_columns(&[
            self.right().to_vec(),
            self.up().to_vec(),
            self.back().to_vec(),
        ])
    }

    /// M * v will rotate the vector v to match this orientation
    pub fn to_matrixf(self) -> Matrix3<f32> {
        self.to_matrixi().map(|x| x as f32)
    }

    pub fn apply_to_dir(self, dir: Direction) -> Direction {
        APPLY_UNAPPLY_LUT[dir.to_signed_axis_index() * 24 + self.to_index()].0
    }

    pub fn apply_to_veci(self, vec: Vector3<i32>) -> Vector3<i32> {
        self.to_matrixi() * vec
    }

    pub fn apply_to_vecf(self, vec: Vector3<f32>) -> Vector3<f32> {
        self.to_matrixf() * vec
    }

    pub fn unapply_to_dir(self, dir: Direction) -> Direction {
        APPLY_UNAPPLY_LUT[dir.to_signed_axis_index() * 24 + self.to_index()].1
    }

    pub fn unapply_to_veci(self, vec: Vector3<i32>) -> Vector3<i32> {
        self.to_matrixi().transpose() * vec
    }

    pub fn unapply_to_vecf(self, vec: Vector3<f32>) -> Vector3<f32> {
        self.to_matrixf().transpose() * vec
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
                        assert_eq!(
                            Some(orn),
                            OctahedralOrientation::from_matrixi(orn.to_matrixi())
                        );
                        assert_eq!(orn.apply_to_dir(DIR_FRONT), orn.front());
                        assert_eq!(orn.apply_to_dir(DIR_RIGHT), orn.right());
                        assert_eq!(orn.apply_to_dir(DIR_UP), orn.up());
                        assert_eq!(orn.apply_to_dir(DIR_BACK), orn.back());
                        assert_eq!(orn.apply_to_dir(DIR_LEFT), orn.left());
                        assert_eq!(orn.apply_to_dir(DIR_DOWN), orn.down());
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
        let id3: Matrix3<f32> = one();
        assert_eq!(default_orientation.to_matrixf(), id3);
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
static DIRECTION_CROSS_TABLE: [Option<Direction>; 36] = {
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
