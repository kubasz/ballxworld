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

pub fn block_shape(datum: VoxelDatum, vdef: &VoxelDefinition) -> &'static VoxelShapeDef {
    match vdef.mesh {
        VoxelMesh::None => &*VOXEL_NO_SHAPE,
        VoxelMesh::CubeAndSlopes => match StdMeta::from_meta(datum.meta()).shape() {
            0 => &*VOXEL_CUBE_SHAPE,
            1 => &*VOXEL_SLOPE_SHAPE,
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

#[derive(Clone)]
pub struct VoxelShapeDef {
    pub causes_ambient_occlusion: bool,
    pub sides: [VSSide; 6],
}

#[derive(Clone)]
pub struct VSVertex {
    pub offset: Vector3<f32>,
    pub texcoord: Vector2<f32>,
    /// https://www.asawicki.info/news_1721_how_to_correctly_interpolate_vertex_attributes_on_a_parallelogram_using_modern_gpus
    /// Archive: https://web.archive.org/web/20200516133048/https://www.asawicki.info/news_1721_how_to_correctly_interpolate_vertex_attributes_on_a_parallelogram_using_modern_gpus
    pub barycentric: Vector2<f32>,
    /// Sign when added to the "extra data" sum for proper quadrilateral interpolation
    pub barycentric_sign: i32,
    pub ao_offsets: SmallVec<[Vector3<i32>; 4]>,
}

#[derive(Clone)]
pub struct VSSide {
    /// Whether this side covers the entire voxel area in this direction, stopping the neighbor from rendering
    pub can_clip: bool,
    /// Whether this side is within the voxel area in this direction, so it is possible that the neighbor is stopping it from rendering
    pub can_be_clipped: bool,
    pub normal: Vector3<f32>,
    pub vertices: SmallVec<[VSVertex; 8]>,
    pub indices: SmallVec<[u32; 8]>,
}

lazy_static! {
    pub static ref VOXEL_NO_SHAPE: VoxelShapeDef = init_no_shape();
    pub static ref VOXEL_CUBE_SHAPE: VoxelShapeDef = init_cube_shape();
    pub static ref VOXEL_SLOPE_SHAPE: VoxelShapeDef = init_slope_shape();
}

fn init_no_shape() -> VoxelShapeDef {
    let side = VSSide {
        can_clip: false,
        can_be_clipped: true,
        normal: vec3(0.0, 1.0, 0.0),
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

fn init_cube_shape() -> VoxelShapeDef {
    VoxelShapeDef {
        causes_ambient_occlusion: true,
        sides: [
            // Left X-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(-1.0, 0.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(-1, -1, 0),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(-1, 1, 0),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, -1),
                            vec3(-1, 1, 0),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(-1, -1, 0),
                            vec3(-1, 0, -1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Right X+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(1.0, 0.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(1, -1, 0),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, -1),
                            vec3(1, 1, 0),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(1, 1, 0),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(1, -1, 0),
                            vec3(1, 0, 1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Bottom Y-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(0.0, -1.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(-1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(-1, -1, 0)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Top Y+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(0.0, 1.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(-1, 1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(-1, 1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(1, 1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(1, 1, 0)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Front Z-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(0.0, 0.0, -1.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(1, 0, -1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Back Z+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(0.0, 0.0, 1.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(-1, 0, 1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
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
                normal: vec3(-1.0, 0.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(-1, -1, 0),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(-1, 1, 0),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(-1, -1, 0),
                            vec3(-1, 0, -1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
            // Right X+
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                normal: vec3(1.0, 0.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(1, -1, 0),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(1, 1, 0),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 0,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(1, -1, 0),
                            vec3(1, 0, 1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2]),
            },
            // Bottom Y-
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(0.0, -1.0, 0.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(-1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(-1, -1, 0)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Top Y+
            VSSide {
                can_clip: false,
                can_be_clipped: false,
                normal: vec3(0.0, 1.0, -1.0).normalize(),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(1, 0, -1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
            // Front Z-
            VSSide {
                can_clip: false,
                can_be_clipped: true,
                normal: vec3(0.0, 1.0, 0.0),
                vertices: smallvec![],
                indices: SmallVec::new(),
            },
            // Back Z+
            VSSide {
                can_clip: true,
                can_be_clipped: true,
                normal: vec3(0.0, 0.0, 1.0),
                vertices: smallvec![
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        barycentric: vec2(0.0, 1.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        barycentric: vec2(1.0, 0.0),
                        barycentric_sign: -1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        barycentric: vec2(0.0, 0.0),
                        barycentric_sign: 1,
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(-1, 0, 1)
                        ]),
                    },
                ],
                indices: SmallVec::from_slice(&[0, 1, 2, 2, 3, 0]),
            },
        ],
    }
}
