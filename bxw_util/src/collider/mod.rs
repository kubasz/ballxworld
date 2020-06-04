
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
    pub maxs: Vector3<f64>
}
