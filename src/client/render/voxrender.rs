use crate::client::render::voxmesh::mesh_from_chunk;
use crate::client::render::*;
use crate::world::generation::World;
use crate::world::{ChunkPosition, VoxelChunk, VOXEL_CHUNK_DIM};
use cgmath::prelude::*;
use cgmath::{vec3, Deg, Matrix4, PerspectiveFov, Rad, Vector3};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, Mutex, Weak};
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};

#[allow(clippy::ref_in_deref)] // in impl_vertex! macro
pub mod vox {
    use vulkano::impl_vertex;

    pub struct ChunkBuffers {
        pub vertices: Vec<VoxelVertex>,
        pub indices: Vec<u32>,
    }

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct VoxelVertex {
        pub position: [f32; 4],
        pub color: [f32; 4],
    }

    impl_vertex!(VoxelVertex, position, color);

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct VoxelUBO {
        pub model: [[f32; 4]; 4],
        pub view: [[f32; 4]; 4],
        pub proj: [[f32; 4]; 4],
    }

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct VoxelPC {
        pub chunk_offset: [f32; 3],
    }

    pub mod vs {
        vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/client/shaders/voxel.vert"
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/client/shaders/voxel.frag"
        }
    }
}

struct DrawnChunk {
    pub chunk: Weak<Mutex<VoxelChunk>>,
    pub last_dirty: u64,
    pub vbuffer: Arc<ImmutableBuffer<[vox::VoxelVertex]>>,
    pub ibuffer: Arc<ImmutableBuffer<[u32]>>,
}

impl Debug for DrawnChunk {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "DrawnChunk {{ last_dirty: {} }}", self.last_dirty)
    }
}

pub struct VoxelRenderer {
    pub world: Option<Arc<Mutex<World>>>,
    pub voxel_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    voxel_staging_v: CpuBufferPool<vox::VoxelVertex>,
    voxel_staging_i: CpuBufferPool<u32>,
    drawn_chunks: HashMap<ChunkPosition, Arc<Mutex<DrawnChunk>>>,
    pub ubuffers: CpuBufferPool<vox::VoxelUBO>,
    pub position: Vector3<f32>,
    pub angles: (f32, f32),
}

impl VoxelRenderer {
    pub fn new(rctx: &mut RenderingContext) -> Self {
        let voxel_pipeline = {
            let vs =
                vox::vs::Shader::load(rctx.device.clone()).expect("Failed to create VS module");
            let fs =
                vox::fs::Shader::load(rctx.device.clone()).expect("Failed to create FS module");
            Arc::new(
                GraphicsPipeline::start()
                    .cull_mode_back()
                    .vertex_input_single_buffer::<vox::VoxelVertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(rctx.mainpass.clone(), 0).unwrap())
                    .build(rctx.device.clone())
                    .expect("Could not create voxel graphics pipeline"),
            )
        };

        let ubuffers = CpuBufferPool::new(rctx.device.clone(), BufferUsage::uniform_buffer());

        Self {
            world: None,
            voxel_pipeline,
            voxel_staging_v: CpuBufferPool::upload(rctx.device.clone()),
            voxel_staging_i: CpuBufferPool::upload(rctx.device.clone()),
            drawn_chunks: HashMap::new(),
            ubuffers,
            position: vec3(0.0, 0.0, 90.0),
            angles: (0.0, 0.0),
        }
    }

    pub fn inpass_draw(&mut self, fctx: &mut InPassFrameContext) {
        let mut cmdbufbuild = fctx.replace_cmd();

        let ubo = {
            let mut ubo = vox::VoxelUBO::default();
            let mmdl: Matrix4<f32> = One::one();
            ubo.model = mmdl.into();
            let mut mview: Matrix4<f32> = Matrix4::from_translation(-{
                let mut p = self.position;
                p.y = -p.y;
                p
            });
            mview = Matrix4::from_angle_y(Deg(self.angles.1)) * mview;
            mview = Matrix4::from_angle_x(Deg(self.angles.0)) * mview;
            mview.replace_col(1, -mview.y);
            ubo.view = mview.into();
            let swdim = fctx.dims;
            let sfdim = [swdim[0] as f32, swdim[1] as f32];
            let mproj: Matrix4<f32> = PerspectiveFov {
                fovy: Rad(75.0 * std::f32::consts::PI / 180.0),
                aspect: sfdim[0] / sfdim[1],
                near: 0.1,
                far: 1000.0,
            }
            .into();
            ubo.proj = mproj.into();
            self.ubuffers.next(ubo).unwrap()
        };

        let udset = Arc::new(
            PersistentDescriptorSet::start(self.voxel_pipeline.clone(), 0)
                .add_buffer(ubo.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        for (pos, chunkmut) in self.drawn_chunks.iter() {
            let pc = vox::VoxelPC {
                chunk_offset: pos.map(|x| (x as f32) * (VOXEL_CHUNK_DIM as f32)).into(),
            };
            let chunk = chunkmut.lock().unwrap();
            cmdbufbuild = cmdbufbuild
                .draw_indexed(
                    self.voxel_pipeline.clone(),
                    &fctx.rctx.dynamic_state,
                    vec![chunk.vbuffer.clone()],
                    chunk.ibuffer.clone(),
                    udset.clone(),
                    pc,
                )
                .expect("Failed to submit voxel chunk draw");
        }

        fctx.cmd = Some(cmdbufbuild);
    }

    pub fn prepass_draw(&mut self, fctx: &mut PrePassFrameContext) {
        if self.world.is_none() {
            self.drawn_chunks.clear();
            return;
        }

        let cmd = fctx.replace_cmd();

        let world_some = self.world.as_ref().unwrap();
        let world = world_some.lock().unwrap();

        let mut chunks_to_remove: Vec<ChunkPosition> = Vec::new();
        self.drawn_chunks
            .keys()
            .filter(|p| !world.loaded_chunks.contains_key(p))
            .for_each(|p| chunks_to_remove.push(*p));
        let mut chunks_to_add: Vec<ChunkPosition> = Vec::new();
        for (cpos, chunk) in world.loaded_chunks.iter() {
            if !self.drawn_chunks.contains_key(&cpos) {
                chunks_to_add.push(*cpos);
                continue;
            }
            if self
                .drawn_chunks
                .get(&cpos)
                .unwrap()
                .lock()
                .unwrap()
                .last_dirty
                != chunk.lock().unwrap().dirty
            {
                chunks_to_add.push(*cpos);
            }
        }

        let mut cmd = cmd;
        let ref_pos = self.position; // TODO: Add velocity-based position prediction
        let cposition = ref_pos.map(|c| (c as i32) / 16);
        let dist_key = |p: &Vector3<i32>| {
            let d = cposition - p;
            d.x * d.x + d.y * d.y + d.z * d.z
        };
        // Load nearest chunks first
        chunks_to_add.sort_by_cached_key(&dist_key);
        // Unload farthest chunks first
        chunks_to_remove.sort_by_cached_key(|p| -dist_key(p));

        let mut remover_iter = chunks_to_remove.into_iter();
        for cpos in chunks_to_add.iter().take(3) {
            let cpos = *cpos;
            self.drawn_chunks.remove(&cpos);
            let chunk_arc = world.loaded_chunks.get(&cpos).unwrap();
            let chunk = chunk_arc.lock().unwrap();
            let mesh = mesh_from_chunk(&chunk, &world.registry);

            let vchunk = self
                .voxel_staging_v
                .chunk(mesh.vertices.into_iter())
                .unwrap();
            let ichunk = self
                .voxel_staging_i
                .chunk(mesh.indices.into_iter())
                .unwrap();

            let dchunk = {
                let (vbuffer, vfill) = unsafe {
                    ImmutableBuffer::<[vox::VoxelVertex]>::uninitialized_array(
                        fctx.rctx.device.clone(),
                        vchunk.len(),
                        BufferUsage::vertex_buffer_transfer_destination(),
                    )
                    .unwrap()
                };
                cmd = cmd.copy_buffer(vchunk, vfill).unwrap();
                let (ibuffer, ifill) = unsafe {
                    ImmutableBuffer::<[u32]>::uninitialized_array(
                        fctx.rctx.device.clone(),
                        ichunk.len(),
                        BufferUsage::index_buffer_transfer_destination(),
                    )
                    .unwrap()
                };
                cmd = cmd.copy_buffer(ichunk, ifill).unwrap();

                DrawnChunk {
                    chunk: Arc::downgrade(chunk_arc),
                    last_dirty: chunk.dirty,
                    vbuffer,
                    ibuffer,
                }
            };
            // For each loaded chunk unload 0..# chunks
            for _ in 0..2 {
                if let Some(rpos) = remover_iter.next() {
                    self.drawn_chunks.remove(&rpos);
                }
            }
            self.drawn_chunks.insert(cpos, Arc::new(Mutex::new(dchunk)));
        }

        fctx.cmd = Some(cmd);
    }
}
