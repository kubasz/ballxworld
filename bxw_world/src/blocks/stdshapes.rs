use bxw_util::lazy_static::*;
use bxw_util::math::*;
use bxw_util::smallvec::*;

#[derive(Clone)]
pub struct VoxelShapeDef {
    pub sides: [VSSide; 6],
}

#[derive(Clone)]
pub struct VSVertex {
    offset: Vector3<f32>,
    texcoord: Vector2<f32>,
    ao_offsets: SmallVec<[Vector3<i32>; 4]>,
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
    pub static ref VOXEL_CUBE_SHAPE: VoxelShapeDef = init_cube_shape();
}

fn init_cube_shape() -> VoxelShapeDef {
    VoxelShapeDef {
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
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(-1, -1, 0),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(-1, 1, 0),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, -1),
                            vec3(-1, 1, 0),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
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
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(1, -1, 0),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, -1),
                            vec3(1, 1, 0),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(1, 1, 0),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
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
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(-1, -1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
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
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(-1, 1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(0, 1, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 1.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(1, 1, 0)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
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
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, -1, -1),
                            vec3(0, -1, -1),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, -0.5),
                        texcoord: vec2(0.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(-1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, -0.5),
                        texcoord: vec2(1.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, -1),
                            vec3(0, 1, -1),
                            vec3(1, 0, -1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, -0.5, -0.5),
                        texcoord: vec2(1.0, 1.0),
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
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, -1, 1),
                            vec3(0, -1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(0.5, 0.5, 0.5),
                        texcoord: vec2(0.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, 0.5, 0.5),
                        texcoord: vec2(1.0, 0.0),
                        ao_offsets: SmallVec::from_slice(&[
                            vec3(-1, 1, 1),
                            vec3(0, 1, 1),
                            vec3(-1, 0, 1)
                        ]),
                    },
                    VSVertex {
                        offset: vec3(-0.5, -0.5, 0.5),
                        texcoord: vec2(1.0, 1.0),
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
