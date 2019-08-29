use crate::client::config::Config;
use crate::client::render::voxmesh::mesh_from_chunk;
use crate::client::render::vulkan::Queues;
use crate::client::render::*;
use crate::client::world::{CameraSettings, ClientWorldMethods};
use crate::world::ecs::{CLocation, ECSHandler};
use crate::world::generation::World;
use crate::world::{ChunkPosition, VoxelChunk, VOXEL_CHUNK_DIM};
use cgmath::prelude::*;
use cgmath::{vec3, Matrix4, PerspectiveFov, Rad, Vector3};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::thread;
use thread_local::CachedThreadLocal;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::DescriptorSet;
use vulkano::framebuffer::Subpass;
use vulkano::image::ImmutableImage;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sampler::Sampler;
use vulkano::sync::GpuFuture;

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
        pub texcoord: [f32; 3],
    }

    impl_vertex!(VoxelVertex, position, color, texcoord);

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
    pub chunk: Weak<RwLock<VoxelChunk>>,
    pub last_dirty: u64,
    pub vbuffer: Arc<ImmutableBuffer<[vox::VoxelVertex]>>,
    pub ibuffer: Arc<ImmutableBuffer<[u32]>>,
}

impl Debug for DrawnChunk {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "DrawnChunk {{ last_dirty: {} }}", self.last_dirty)
    }
}

type ChunkMsg = (ChunkPosition, Arc<RwLock<DrawnChunk>>);

pub struct VoxelRenderer {
    _texture_array: Arc<ImmutableImage<vulkano::format::R8G8B8A8Srgb>>,
    texture_name_map: HashMap<String, u32>,
    _texture_sampler: Arc<Sampler>,
    texture_ds: Arc<dyn DescriptorSet + Send + Sync>,
    world: Option<Arc<RwLock<World>>>,
    pub voxel_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    atmosphere_renderer: AtmosphereRenderer,
    drawn_chunks: HashMap<ChunkPosition, Arc<RwLock<DrawnChunk>>>,
    pub ubuffers: CpuBufferPool<vox::VoxelUBO>,
    draw_queue: Arc<Mutex<Vec<ChunkPosition>>>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    work_receiver: CachedThreadLocal<mpsc::Receiver<ChunkMsg>>,
}

impl VoxelRenderer {
    pub fn new(cfg: &Config, rctx: &mut RenderingContext) -> Self {
        // load textures
        let (texture_array, texture_name_map) = Self::new_texture_atlas(cfg, rctx);

        use vulkano::sampler::*;
        let texture_sampler = Sampler::new(
            rctx.device.clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::ClampToEdge,
            0.0,
            4.0,
            0.0,
            0.0,
        )
        .expect("Could not create voxel texture sampler");

        // create pipeline
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
                    .blend_alpha_blending()
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(rctx.mainpass.clone(), 0).unwrap())
                    .build(rctx.device.clone())
                    .expect("Could not create voxel graphics pipeline"),
            )
        };

        let ubuffers = CpuBufferPool::new(rctx.device.clone(), BufferUsage::uniform_buffer());

        let texture_ds = Arc::new(
            PersistentDescriptorSet::start(voxel_pipeline.clone(), 1)
                .add_sampled_image(texture_array.clone(), texture_sampler.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        Self {
            _texture_array: texture_array,
            texture_name_map,
            _texture_sampler: texture_sampler,
            texture_ds,
            world: None,
            voxel_pipeline,
            atmosphere_renderer: AtmosphereRenderer::new(cfg, rctx),
            drawn_chunks: HashMap::new(),
            ubuffers,
            draw_queue: Arc::new(Mutex::new(Vec::new())),
            worker_threads: Vec::new(),
            work_receiver: CachedThreadLocal::new(),
        }
    }

    pub fn get_texture_id(&self, name: &str) -> u32 {
        self.texture_name_map.get(name).copied().unwrap_or(0)
    }

    fn new_texture_atlas(
        cfg: &Config,
        rctx: &mut RenderingContext,
    ) -> (
        Arc<ImmutableImage<vulkano::format::R8G8B8A8Srgb>>,
        HashMap<String, u32>,
    ) {
        use image::RgbaImage;
        use toml_edit::Document;

        let mf_str =
            std::fs::read_to_string("res/manifest.toml").expect("Couldn't read resource manifest");
        let manifest = mf_str
            .parse::<Document>()
            .expect("Invalid configuration TOML.");
        let tkey = manifest["textures"]
            .as_table()
            .expect("manifest.toml[textures] is not a table");
        let mut memimages = Vec::new();
        let mut names = HashMap::new();
        let mut idim = (0, 0);

        // load all textures into memory
        for (nm, path) in tkey.iter() {
            let path: PathBuf = [
                "res",
                "textures",
                path.as_str()
                    .expect("manifest.toml[textures][item] is not a path"),
            ]
            .iter()
            .collect();
            if !(path.exists() && path.is_file()) {
                panic!("Could not find texture file: `{:?}`", &path);
            }
            let img = image::open(&path)
                .unwrap_or_else(|e| panic!("Could not load image from {:?}: {}", path, e));
            let img: RgbaImage = img.to_rgba();
            names.insert(nm.to_owned(), memimages.len() as u32);
            let dim1 = img.dimensions();
            idim.0 = idim.0.max(dim1.0);
            idim.1 = idim.1.max(dim1.1);
            memimages.push(img);
            if cfg.debug_logging {
                eprintln!("Loaded {} from {:?}", nm, path);
            }
        }
        let numimages = memimages.len() as u32;

        // resize all images to max size and put the pixels into a buffer
        let mut rawdata = Vec::new();
        for mut img in memimages.into_iter() {
            if img.dimensions() != idim {
                img = image::imageops::resize(&img, idim.0, idim.1, image::Nearest);
            }
            debug_assert_eq!(img.dimensions(), idim);
            rawdata.append(&mut img.into_raw());
        }

        // create vulkan image array
        let vdim = vulkano::image::Dimensions::Dim2dArray {
            width: idim.0,
            height: idim.1,
            array_layers: numimages,
        };
        let queue = rctx.queues.lock_primary_queue();
        let (vimg, vfuture) = ImmutableImage::from_iter(
            rawdata.iter().copied(),
            vdim,
            vulkano::format::R8G8B8A8Srgb,
            queue.clone(),
        )
        .expect("Could not create voxel texture array");
        vfuture
            .then_signal_fence()
            .wait(None)
            .expect("Could not upload voxel texture array");

        (vimg, names)
    }

    pub fn set_world(&mut self, world: Arc<RwLock<World>>, rctx: &RenderingContext) {
        // create worker threads
        self.worker_threads.clear();
        const NUM_WORKERS: usize = 2;
        const STACK_SIZE: usize = 4 * 1024 * 1024;
        let (tx, rx) = mpsc::channel();
        self.worker_threads.reserve_exact(NUM_WORKERS);
        self.work_receiver.clear();
        self.work_receiver.get_or(move || Box::new(rx));
        for _ in 0..NUM_WORKERS {
            let tb = thread::Builder::new()
                .name("bxw-voxrender".to_owned())
                .stack_size(STACK_SIZE);
            let tworld = Arc::downgrade(&world);
            let ttx = tx.clone();
            let qs = rctx.queues.clone();
            let work_queue = self.draw_queue.clone();
            let thr = tb
                .spawn(move || Self::vox_worker(tworld, qs, work_queue, ttx))
                .expect("Could not create voxrender worker thread");
            self.worker_threads.push(thr);
        }
        self.world = Some(world);
    }

    fn vox_worker(
        world_w: Weak<RwLock<World>>,
        qs: Arc<Queues>,
        work_queue: Arc<Mutex<Vec<ChunkPosition>>>,
        submission: mpsc::Sender<ChunkMsg>,
    ) {
        let device;
        let _q;
        let qfamily;
        {
            let q = qs.lock_gtransfer_queue();
            device = q.device().clone();
            _q = q.clone();
            qfamily = _q.family();
        }
        let voxel_staging_v = CpuBufferPool::upload(device.clone());
        let voxel_staging_i = CpuBufferPool::upload(device.clone());

        let wr_rq;
        let registry;

        {
            let world = world_w.upgrade().unwrap();
            let world = world.read().unwrap();
            wr_rq = world.get_write_request();
            registry = world.registry.clone();
        }

        loop {
            let world = world_w.upgrade();
            if world.is_none() {
                eprintln!("World changing - voxrender worker terminating");
                return;
            }
            let world = world.unwrap();
            let mut work_queue = work_queue.lock().unwrap();

            if work_queue.is_empty() {
                drop(work_queue);
                thread::park();
                continue;
            }

            let mut chunks_to_add = Vec::new();
            let len = work_queue.len().min(10);
            for p in work_queue.iter().rev().take(len) {
                chunks_to_add.push(*p);
            }
            let tgtlen = work_queue.len() - len;
            work_queue.resize(tgtlen, vec3(0, 0, 0));
            drop(work_queue);

            let mut chunk_objs_to_add = Vec::new();
            {
                while wr_rq.load(Ordering::SeqCst) {
                    thread::yield_now();
                }
                let world = world.read().unwrap();
                for cpos in chunks_to_add.into_iter() {
                    let chunk_opt = world.loaded_chunks.get(&cpos);
                    if chunk_opt.is_none() {
                        continue;
                    }
                    let chunk = chunk_opt.unwrap().clone();
                    chunk_objs_to_add.push((cpos, chunk));
                }
            }

            for (cpos, chunk_arc) in chunk_objs_to_add.into_iter() {
                let chunk = chunk_arc.read().unwrap();
                let mesh = mesh_from_chunk(&chunk, &registry);

                let vchunk = voxel_staging_v.chunk(mesh.vertices.into_iter()).unwrap();
                let ichunk = voxel_staging_i.chunk(mesh.indices.into_iter()).unwrap();

                let mut cmd =
                    AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), qfamily)
                        .unwrap();

                let dchunk = {
                    let (vbuffer, vfill) = unsafe {
                        ImmutableBuffer::<[vox::VoxelVertex]>::uninitialized_array(
                            device.clone(),
                            vchunk.len(),
                            BufferUsage::vertex_buffer_transfer_destination(),
                        )
                        .unwrap()
                    };
                    cmd = cmd.copy_buffer(vchunk, vfill).unwrap();
                    let (ibuffer, ifill) = unsafe {
                        ImmutableBuffer::<[u32]>::uninitialized_array(
                            device.clone(),
                            ichunk.len(),
                            BufferUsage::index_buffer_transfer_destination(),
                        )
                        .unwrap()
                    };
                    cmd = cmd.copy_buffer(ichunk, ifill).unwrap();

                    DrawnChunk {
                        chunk: Arc::downgrade(&chunk_arc),
                        last_dirty: chunk.dirty,
                        vbuffer,
                        ibuffer,
                    }
                };
                let cmd = cmd.build().unwrap();
                let q = qs.lock_gtransfer_queue();
                let f = cmd.execute(q.clone()).unwrap();
                f.then_signal_fence().wait(None).unwrap();
                drop(q);
                if submission
                    .send((cpos, Arc::new(RwLock::new(dchunk))))
                    .is_err()
                {
                    eprintln!("World changing - voxrender worker terminating");
                    return;
                }
            }
        }
    }

    pub fn prepass_draw(&mut self, fctx: &mut PrePassFrameContext) {
        if self.world.is_none() {
            self.drawn_chunks.clear();
            return;
        }

        if let Some(work_receiver) = self.work_receiver.get() {
            for (p, dc) in work_receiver.try_iter() {
                self.drawn_chunks.insert(p, dc);
            }
        }

        let cmd = fctx.replace_cmd();

        let world_some = self.world.as_ref().unwrap();
        let world = world_some.read().unwrap();

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
                .read()
                .unwrap()
                .last_dirty
                != chunk.read().unwrap().dirty
            {
                chunks_to_add.push(*cpos);
            }
        }

        let ref_pos; // TODO: Add velocity-based position prediction
        {
            let entities = world.entities.read().unwrap();
            let lp_loc: &CLocation = entities.get_component(world.local_player()).unwrap();
            ref_pos = lp_loc.position;
        }
        let cposition = ref_pos.map(|c| (c as i32) / 16);
        let dist_key = |p: &Vector3<i32>| {
            let d = cposition - p;
            -(d.x * d.x + d.y * d.y + d.z * d.z)
        };
        let mut draw_queue = self.draw_queue.lock().unwrap();
        draw_queue.clear();
        for c in chunks_to_add.iter() {
            if !draw_queue.contains(c) {
                draw_queue.push(*c);
            }
        }
        drop(chunks_to_add);
        // Load nearest chunks first
        draw_queue.sort_by_cached_key(&dist_key);
        // Unload farthest chunks first
        chunks_to_remove.sort_by_cached_key(&dist_key);

        for cpos in chunks_to_remove.into_iter().take(32) {
            self.drawn_chunks.remove(&cpos);
        }

        let new_cmds = !draw_queue.is_empty();
        drop(draw_queue);
        if new_cmds {
            for t in self.worker_threads.iter() {
                t.thread().unpark();
            }
        }

        fctx.cmd = Some(cmd);
    }

    pub fn inpass_draw(&mut self, fctx: &mut InPassFrameContext) {
        let mview = if let Some(world) = &self.world {
            let world = world.read().unwrap();
            let entities = world.entities.read().unwrap();
            let lp_loc: &CLocation = entities.get_component(world.local_player()).unwrap();
            let player_pos = lp_loc.position;
            let player_ang = lp_loc.orientation;
            match world.camera_settings() {
                CameraSettings::FPS { .. } => {
                    let mut mview = Matrix4::from_translation(-{
                        let mut p = player_pos;
                        p.y = -p.y;
                        p
                    });
                    mview = Matrix4::from(player_ang) * mview;
                    mview.replace_col(1, -mview.y);
                    mview
                }
            }
        } else {
            Matrix4::identity()
        };

        let mut cmdbufbuild = fctx.replace_cmd();

        let mproj: Matrix4<f32>;

        let ubo = {
            let mut ubo = vox::VoxelUBO::default();
            let mmdl: Matrix4<f32> = One::one();
            ubo.model = mmdl.into();
            ubo.view = mview.into();
            let swdim = fctx.dims;
            let sfdim = [swdim[0] as f32, swdim[1] as f32];
            mproj = PerspectiveFov {
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
            let chunk = chunkmut.read().unwrap();
            if chunk.ibuffer.len() > 0 {
                cmdbufbuild = cmdbufbuild
                    .draw_indexed(
                        self.voxel_pipeline.clone(),
                        &fctx.rctx.dynamic_state,
                        vec![chunk.vbuffer.clone()],
                        chunk.ibuffer.clone(),
                        (udset.clone(), self.texture_ds.clone()),
                        pc,
                    )
                    .expect("Failed to submit voxel chunk draw");
            }
        }
        fctx.cmd = Some(cmdbufbuild);

        self.atmosphere_renderer.inpass_draw(fctx, mview, mproj);
    }
}
