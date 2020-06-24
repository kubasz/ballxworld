use crate::*;
use bxw_util::direction::*;
use bxw_util::lazy_static::*;
use bxw_util::math::*;
use bxw_util::smallvec::*;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct StdMeta {
    shape: u16,
    orientation: u16,
}

impl StdMeta {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_parts(shape: u16, orientation: u16) -> Option<Self> {
        if shape >= 64 || orientation >= 24 {
            None
        } else {
            Some(Self { shape, orientation })
        }
    }

    pub fn from_meta(meta: VoxelMetadata) -> Self {
        Self {
            shape: meta % 64,
            orientation: meta / 64,
        }
    }

    pub fn to_meta(self) -> VoxelMetadata {
        self.orientation * 64 + self.shape
    }

    pub fn shape(self) -> u16 {
        self.shape
    }

    pub fn orientation(self) -> u16 {
        self.orientation
    }
}

pub const VOX_META_STDSHAPE_CUBE: u16 = 0;
pub const VOX_META_STDSHAPE_SLOPE: u16 = 1;
pub const VOX_META_STDSHAPE_CORNER: u16 = 2;
pub const VOX_META_STDSHAPE_INNER_CORNER: u16 = 3;

pub fn block_shape(datum: VoxelDatum, vdef: &VoxelDefinition) -> &'static VoxelShapeDef {
    match vdef.mesh {
        VoxelMesh::None => &*VOXEL_NO_SHAPE,
        VoxelMesh::CubeAndSlopes => match StdMeta::from_meta(datum.meta()).shape() {
            VOX_META_STDSHAPE_CUBE => &*VOXEL_CUBE_SHAPE,
            VOX_META_STDSHAPE_SLOPE => &*VOXEL_SLOPE_SHAPE,
            VOX_META_STDSHAPE_CORNER => &*VOXEL_CORNER_SHAPE,
            VOX_META_STDSHAPE_INNER_CORNER => &*VOXEL_INNER_CORNER_SHAPE,
            _ => &*VOXEL_CUBE_SHAPE,
        },
    }
}

pub fn block_orientation(datum: VoxelDatum, vdef: &VoxelDefinition) -> OctahedralOrientation {
    match vdef.mesh {
        VoxelMesh::None => OctahedralOrientation::default(),
        VoxelMesh::CubeAndSlopes => OctahedralOrientation::from_index(
            StdMeta::from_meta(datum.meta()).orientation() as usize,
        )
        .unwrap_or_default(),
    }
}

#[derive(Clone, Debug)]
pub struct VoxelShapeDef {
    pub causes_ambient_occlusion: bool,
    pub sides: [VSSide; 6],
}

#[derive(Clone, Debug)]
pub struct VSVertex {
    pub offset: Vector3<f32>,
    pub texcoord: Vector2<f32>,
    pub normal: Vector3<f32>,
    /// https://www.asawicki.info/news_1721_how_to_correctly_interpolate_vertex_attributes_on_a_parallelogram_using_modern_gpus
    /// Archive: https://web.archive.org/web/20200516133048/https://www.asawicki.info/news_1721_how_to_correctly_interpolate_vertex_attributes_on_a_parallelogram_using_modern_gpus
    pub barycentric: Vector3<f32>,
    /// Sign when added to the "extra data" sum for proper quadrilateral interpolation
    pub barycentric_sign: i32,
    pub ao_offsets: SmallVec<[Vector3<i32>; 4]>,
}

#[derive(Clone, Debug)]
pub struct VSSide {
    /// Whether this side covers the entire voxel area in this direction, stopping the neighbor from rendering
    pub can_clip: bool,
    /// Whether this side is within the voxel area in this direction, so it is possible that the neighbor is stopping it from rendering
    pub can_be_clipped: bool,
    pub vertices: SmallVec<[VSVertex; 8]>,
    pub indices: SmallVec<[u32; 8]>,
}

lazy_static! {
    pub static ref VOXEL_NO_SHAPE: VoxelShapeDef = init_no_shape();
    pub static ref VOXEL_CUBE_SHAPE: VoxelShapeDef = init_cube_shape();
    pub static ref VOXEL_SLOPE_SHAPE: VoxelShapeDef = init_slope_shape();
    pub static ref VOXEL_CORNER_SHAPE: VoxelShapeDef = init_corner_shape();
    pub static ref VOXEL_INNER_CORNER_SHAPE: VoxelShapeDef = init_inner_corner_shape();
}

fn init_no_shape() -> VoxelShapeDef {
    let side = VSSide {
        can_clip: false,
        can_be_clipped: true,
        vertices: SmallVec::new(),
        indices: SmallVec::new(),
    };
    VoxelShapeDef {
        causes_ambient_occlusion: false,
        sides: [
            side.clone(),
            side.clone(),
            side.clone(),
            side.clone(),
            side.clone(),
            side,
        ],
    }
}

fn corner_ao_set(corner: Vector3<f32>, inormal: Vector3<i32>) -> SmallVec<[Vector3<i32>; 4]> {
    let icorner = corner.map(|x| if x.abs() < 0.1 { 0 } else { x.signum() as i32 });
    let mut sv = SmallVec::new();
    sv.push(icorner);
    sv.push(inormal);
    if inormal.x == 0 {
        let mut c = icorner;
        c.x = 0;
        sv.push(c);
    }
    if inormal.y == 0 {
        let mut c = icorner;
        c.y = 0;
        sv.push(c);
    }
    if inormal.z == 0 {
        let mut c = icorner;
        c.z = 0;
        sv.push(c);
    }
    sv
}

fn quad_verts(
    center: Vector3<f32>,
    right: Vector3<f32>,
    up: Vector3<f32>,
) -> SmallVec<[VSVertex; 8]> {
    let fnormal = -right.cross(&up);
    let inormal = fnormal.map(|x| if x.abs() < 0.1 { 0 } else { x.signum() as i32 });
    let fnormal = fnormal.normalize();
    smallvec![
        VSVertex {
            offset: center - right - up,
            texcoord: vec2(0.0, 1.0),
            normal: fnormal,
            barycentric: vec3(0.0, 1.0, 1.0),
            barycentric_sign: -1,
            ao_offsets: corner_ao_set(center - right - up, inormal),
        },
        VSVertex {
            offset: center - right + up,
            texcoord: vec2(0.0, 0.0),
            normal: fnormal,
            barycentric: vec3(0.0, 0.0, 1.0),
            barycentric_sign: 1,
            ao_offsets: corner_ao_set(center - right + up, inormal),
        },
        VSVertex {
            offset: center + right + up,
            texcoord: vec2(1.0, 0.0),
            normal: fnormal,
            barycentric: vec3(1.0, 0.0, 1.0),
            barycentric_sign: -1,
            ao_offsets: corner_ao_set(center + right + up, inormal),
        },
        VSVertex {
            offset: center + right - up,
            texcoord: vec2(1.0, 1.0),
            normal: fnormal,
            barycentric: vec3(0.0, 0.0, 1.0),
            barycentric_sign: 1,
            ao_offsets: corner_ao_set(center + right - up, inormal),
        },
    ]
}

const QUAD_INDICES: [u32; 6] = [0, 1, 2, 2, 3, 0];

fn init_cube_shape() -> VoxelShapeDef {
    VoxelShapeDef {
        causes_ambient_occlusion: true,
        sides: [
            // Left X-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(-0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, -0.5),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Right X+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, 0.5),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Bottom Y-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, -0.5, 0.0),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, -0.5),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Top Y+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, 0.5, 0.0),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, 0.5),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Front Z-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, 0.0, -0.5),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Back Z+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, 0.0, 0.5),
                    vec3(-0.5, 0.0, 0.0),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
        ],
    }
}

fn init_slope_shape() -> VoxelShapeDef {
    VoxelShapeDef {
        causes_ambient_occlusion: true,
        sides: [
            // Left X-
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, 1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, -1.0), vec3(-1, 0, 0)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
            // Right X+
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, -1.0), vec3(1, 0, 0))
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(1.0, 0.0, 0.0),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, 1.0, 1.0), vec3(1, 0, 0))
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, 1.0), vec3(1, 0, 0))
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
            // Bottom Y-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, -0.5, 0.0),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, -0.5),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Top Y+
            VSSide {
                can_clip: false,
                can_be_clipped: false,
                vertices: quad_verts(
                    vec3(0.0, 0.0, 0.0),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.5, 0.5),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Front Z-
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![],
                indices: SmallVec::new(),
            },
            // Back Z+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, 0.0, 0.5),
                    vec3(-0.5, 0.0, 0.0),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
        ],
    }
}

fn init_corner_shape() -> VoxelShapeDef {
    VoxelShapeDef {
        causes_ambient_occlusion: true,
        sides: [
            // Left X-
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, 1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, -1.0), vec3(-1, 0, 0)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
            // Right X+
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![],
                indices: smallvec![],
            },
            // Bottom Y-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, -0.5, 0.0),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, -0.5),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Top Y+
            VSSide {
                can_clip: false,
                can_be_clipped: false,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(0.0, 1.0, -1.0).normalize(),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, -1.0), vec3(0, 0, -1)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(0.0, 1.0, -1.0).normalize(),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, -1.0), vec3(0, 0, -1)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(0.0, 1.0, -1.0).normalize(),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, 1.0), vec3(0, 1, 0)),
                    },
                    //
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(1.0, 1.0, 0.0).normalize(),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, 1.0), vec3(0, 1, 0)),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(1.0, 1.0, 0.0).normalize(),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, 1.0), vec3(1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(1.0, 1.0, 0.0).normalize(),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, -1.0), vec3(1, 0, 0)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 3, 4, 5]),
            },
            // Front Z-
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![],
                indices: smallvec![],
            },
            // Back Z+
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(0.0, 0.0, 1.0),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, 1.0), vec3(0, 0, 1)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(0.0, 0.0, 1.0),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(0, 0, 1)),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(0.0, 0.0, 1.0),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, 1.0), vec3(0, 0, 1)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
        ],
    }
}

fn init_inner_corner_shape() -> VoxelShapeDef {
    VoxelShapeDef {
        causes_ambient_occlusion: true,
        sides: [
            // Left X-
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, -1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(-1.0, 0.0, 0.0),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, -1.0), vec3(-1, 0, 0)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
            // Right X+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, 0.5),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Bottom Y-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, -0.5, 0.0),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.0, -0.5),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Top Y+
            VSSide {
                can_clip: false,
                can_be_clipped: false,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(0.0, 1.0, 1.0).normalize(),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, 1.0, -1.0), vec3(0, 1, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(0.0, 1.0, 1.0).normalize(),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, 1.0, -1.0), vec3(0, 1, 0)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        normal: vec3(0.0, 1.0, 1.0).normalize(),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(0, 0, 1)),
                    },
                    //
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(-1.0, 1.0, 0.0).normalize(),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(-1, 0, 0)),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(-1.0, 1.0, 0.0).normalize(),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, 1.0, 1.0), vec3(0, 1, 0)),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(-1.0, 1.0, 0.0).normalize(),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, 1.0, -1.0), vec3(0, 1, 0)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 3, 4, 5]),
            },
            // Front Z-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                vertices: quad_verts(
                    vec3(0.0, 0.0, -0.5),
                    vec3(0.5, 0.0, 0.0),
                    vec3(0.0, 0.5, 0.0),
                ),
                indices: SmallVec::from_slice(&QUAD_INDICES),
            },
            // Back Z+
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        normal: vec3(0.0, 0.0, 1.0),
                        barycentric: vec3(0.0, 0.0, 1.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, 1.0, 1.0), vec3(0, 0, 1)),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        normal: vec3(0.0, 0.0, 1.0),
                        barycentric: vec3(1.0, 0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(-1.0, -1.0, 1.0), vec3(0, 0, 1)),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        normal: vec3(0.0, 0.0, 1.0),
                        barycentric: vec3(0.0, 1.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: corner_ao_set(vec3(1.0, -1.0, 1.0), vec3(0, 0, 1)),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
        ],
    }
}

#[test]
fn print_cube_shape() {
    eprintln!("{:#?}", init_cube_shape());
}
