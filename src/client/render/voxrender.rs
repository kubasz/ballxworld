use crate::client::config::Config;
use crate::client::render::voxmesh::mesh_from_chunk;
use crate::client::render::vulkan::Queues;
use crate::client::render::*;
use crate::client::world::{CameraSettings, ClientWorld};
use crate::math::*;
use crate::world::ecs::{CLocation, ECSHandler};
use crate::world::{blockidx_from_blockpos, chunkpos_from_blockpos, World};
use crate::world::{ChunkPosition, CHUNK_DIM};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::{mpsc, Weak};
use std::thread;
use thread_local::CachedThreadLocal;
use vulkano::buffer::{BufferUsage, CpuBufferPool, ImmutableBuffer, TypedBufferAccess, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::descriptor::DescriptorSet;
use vulkano::framebuffer::Subpass;
use vulkano::image::{ImmutableImage, ImageUsage, ImageLayout, MipmapsCount, ImageAccess};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::sampler::{Sampler, Filter};
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
        pub index: i32,
    }

    impl_vertex!(VoxelVertex, position, color, texcoord, index);

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
        pub highlight_index: i32,
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
    pub cpos: ChunkPosition,
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
    world: Option<Arc<World>>,
    pub voxel_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    atmosphere_renderer: AtmosphereRenderer,
    drawn_chunks: HashMap<ChunkPosition, Arc<RwLock<DrawnChunk>>>,
    pub ubuffers: CpuBufferPool<vox::VoxelUBO>,
    draw_queue: Arc<Mutex<Vec<ChunkPosition>>>,
    progress_set: Arc<Mutex<HashSet<ChunkPosition>>>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    thread_killer: Arc<AtomicBool>,
    work_receiver: CachedThreadLocal<mpsc::Receiver<ChunkMsg>>,
}

impl VoxelRenderer {
    pub fn new(cfg: &Config, rctx: &mut RenderingContext) -> Self {
        // load textures
        let (texture_array, texture_name_map) = Self::new_texture_atlas(cfg, rctx);

        use vulkano::sampler::*;
        let texture_sampler = Sampler::new(
            rctx.device.clone(),
            Filter::Linear,
            Filter::Linear,
            MipmapMode::Linear,
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
                    .cull_mode_front()
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
            progress_set: Arc::new(Mutex::new(HashSet::new())),
            worker_threads: Vec::new(),
            thread_killer: Arc::new(AtomicBool::new(false)),
            work_receiver: CachedThreadLocal::new(),
        }
    }

    fn kill_threads(&mut self) {
        self.thread_killer.store(true, Ordering::SeqCst);
        let old_threads = std::mem::replace(&mut self.worker_threads, Vec::new());
        for t in old_threads.into_iter() {
            t.thread().unpark();
            drop(t.join());
        }
        self.thread_killer.store(false, Ordering::SeqCst);
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
        let vimg =
            {
                let buf = CpuAccessibleBuffer::from_iter(rctx.device.clone(),
                BufferUsage::transfer_source(), rawdata.into_iter())
                    .expect("Could not create voxel texture upload buffer");
                let usage = ImageUsage {
                    transfer_destination: true,
                    transfer_source: true,
                    sampled: true,
                    ..ImageUsage::none()
                };
                let layout = ImageLayout::ShaderReadOnlyOptimal;

                let (img, init) =
                    ImmutableImage::uninitialized(rctx.device.clone(),
                                                  vdim,
                                                  vulkano::format::R8G8B8A8Srgb,
                                                  MipmapsCount::Log2,
                                                  usage,
                                                  layout,
                                                  rctx.device.active_queue_families())
                        .expect("Could not create voxel texture image");

                let mut cb = AutoCommandBufferBuilder::new(rctx.device.clone(), queue.family()).unwrap()
                    .copy_buffer_to_image_dimensions(buf,
                                                     init,
                                                     [0, 0, 0],
                                                     vdim.width_height_depth(),
                                                     0,
                                                     vdim.array_layers_with_cube(),
                                                     0)
                    .unwrap();

                for mip in 1 .. img.mipmap_levels() {
                    let mipw = (idim.0 >> mip).max(1) as i32;
                    let miph = (idim.1 >> mip).max(1) as i32;
                    cb = cb.blit_image(
                        img.clone(),
                        [0,0,0],
                        [mipw*2, miph*2, 1],
                        0,
                        mip-1,
                        img.clone(),
                        [0,0,0],
                        [mipw, miph, 1],
                        0,
                        mip,
                        vdim.array_layers(),
                        Filter::Linear,
                    ).unwrap();
                }

                let cb = cb.build()
                    .unwrap();

                let future = match cb.execute(queue.clone()) {
                    Ok(f) => f,
                    Err(_) => unreachable!(),
                };

                future.then_signal_fence().wait(None).expect("Could not upload voxel texture array");

                img
            };

        (vimg, names)
    }

    pub fn set_world(&mut self, world: Arc<World>, rctx: &RenderingContext) {
        // create worker threads
        self.kill_threads();
        const NUM_WORKERS: usize = 2;
        const STACK_SIZE: usize = 4 * 1024 * 1024;
        let (tx, rx) = mpsc::channel();
        self.worker_threads.reserve_exact(NUM_WORKERS);
        self.work_receiver.clear();
        self.work_receiver.get_or(move || rx);
        for _ in 0..NUM_WORKERS {
            let tb = thread::Builder::new()
                .name("bxw-voxrender".to_owned())
                .stack_size(STACK_SIZE);
            let tworld = Arc::downgrade(&world);
            let ttx = tx.clone();
            let qs = rctx.queues.clone();
            let work_queue = self.draw_queue.clone();
            let progress_set = self.progress_set.clone();
            let killswitch = self.thread_killer.clone();
            let thr = tb
                .spawn(move || {
                    Self::vox_worker(tworld, qs, work_queue, progress_set, ttx, killswitch)
                })
                .expect("Could not create voxrender worker thread");
            self.worker_threads.push(thr);
        }
        self.world = Some(world);
    }

    fn vox_worker(
        world_w: Weak<World>,
        qs: Arc<Queues>,
        work_queue: Arc<Mutex<Vec<ChunkPosition>>>,
        progress_set: Arc<Mutex<HashSet<ChunkPosition>>>,
        submission: mpsc::Sender<ChunkMsg>,
        killswitch: Arc<AtomicBool>,
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

        let mut done_chunks: Vec<Vector3<i32>> = Vec::new();
        loop {
            if killswitch.load(Ordering::SeqCst) {
                return;
            }
            let world = world_w.upgrade();
            if world.is_none() {
                return;
            }
            let world = world.unwrap();
            let mut work_queue = work_queue.lock();
            let mut progress_set = progress_set.lock();

            for p in done_chunks.iter() {
                progress_set.remove(p);
            }
            done_chunks.clear();

            if work_queue.is_empty() {
                drop(work_queue);
                drop(progress_set);
                thread::park();
                if killswitch.load(Ordering::SeqCst) {
                    return;
                }
                continue;
            }

            let mut chunks_to_add = Vec::new();
            let len = work_queue.len().min(10);
            for p in work_queue.iter().rev().take(len) {
                chunks_to_add.push(*p);
                progress_set.insert(*p);
            }
            let tgtlen = work_queue.len() - len;
            work_queue.resize(tgtlen, vec3(0, 0, 0));
            drop(progress_set);
            drop(work_queue);

            let mut chunk_objs_to_add = Vec::new();
            {
                let voxels = world.voxels.read();
                for cpos in chunks_to_add.into_iter() {
                    let chunk_opt = voxels.chunks.get(&cpos);
                    if chunk_opt.is_none() {
                        done_chunks.push(cpos);
                        continue;
                    }
                    chunk_objs_to_add.push(cpos);
                }
            }

            for cpos in chunk_objs_to_add.into_iter() {
                // give a chance to release the lock to a writer on each iteration
                let voxels = world.voxels.read();
                let mesh = mesh_from_chunk(&world, &voxels, cpos);
                if mesh.is_none() {
                    done_chunks.push(cpos);
                    continue;
                }
                let mesh = mesh.unwrap();

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
                        cpos,
                        last_dirty: voxels.chunks.get(&cpos).unwrap().dirty,
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
                    return;
                }
                done_chunks.push(cpos);
            }
        }
    }

    pub fn drawn_chunks_number(&self) -> usize {
        self.drawn_chunks.len()
    }

    pub fn progress_set_len(&self) -> usize {
        self.progress_set.lock().len()
    }

    pub fn prepass_draw(&mut self, fctx: &mut PrePassFrameContext) {
        if self.world.is_none() {
            self.drawn_chunks.clear();
            return;
        }
        let world = self.world.as_ref().unwrap();

        if let Some(work_receiver) = self.work_receiver.get() {
            for (p, dc) in work_receiver.try_iter() {
                self.drawn_chunks.insert(p, dc);
            }
        }

        let cmd = fctx.replace_cmd();

        let voxels = world.voxels.read();

        let mut chunks_to_remove: Vec<ChunkPosition> = Vec::new();
        self.drawn_chunks
            .keys()
            .filter(|p| !voxels.chunks.contains_key(p))
            .for_each(|p| chunks_to_remove.push(*p));
        let mut chunks_to_add: Vec<ChunkPosition> = Vec::new();
        for (cpos, chunk) in voxels.chunks.iter() {
            let cpos = *cpos;
            // check for all neighbors
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let npos = cpos + vec3(dx, dy, dz);
                        if !voxels.chunks.contains_key(&npos) {
                            continue;
                        }
                    }
                }
            }

            if !self.drawn_chunks.contains_key(&cpos) {
                chunks_to_add.push(cpos);
                continue;
            }
            if self.drawn_chunks.get(&cpos).unwrap().read().last_dirty != chunk.dirty {
                chunks_to_add.push(cpos);
            }
        }

        let ref_pos; // TODO: Add velocity-based position prediction
        {
            let client = ClientWorld::read(world);
            let entities = world.entities.read();
            let lp_loc: &CLocation = entities.ecs.get_component(client.local_player).unwrap();
            ref_pos = lp_loc.position;
        }
        let cposition = chunkpos_from_blockpos(ref_pos.map(|x| x as i32));
        let dist_key = |p: &Vector3<i32>| {
            let d = cposition - p;
            -(d.x * d.x + d.y * d.y + d.z * d.z)
        };
        let mut draw_queue = self.draw_queue.lock();
        let progress_set = self.progress_set.lock();
        draw_queue.clear();
        for c in chunks_to_add.iter() {
            if !progress_set.contains(c) {
                draw_queue.push(*c);
            }
        }
        drop(progress_set);
        drop(chunks_to_add);
        // Load nearest chunks first
        draw_queue.sort_by_cached_key(&dist_key);
        // Unload farthest chunks first
        chunks_to_remove.sort_by_cached_key(&dist_key);
        let new_cmds = !draw_queue.is_empty();
        drop(draw_queue);

        for cpos in chunks_to_remove.into_iter().take(32) {
            self.drawn_chunks.remove(&cpos);
        }

        if new_cmds {
            for t in self.worker_threads.iter() {
                t.thread().unpark();
            }
        }

        fctx.cmd = Some(cmd);
    }

    pub fn inpass_draw(&mut self, fctx: &mut InPassFrameContext) {
        let (mut hichunk, mut hiidx) = (vec3(0, 0, 0), -1);
        let mview = if let Some(world) = &self.world {
            let client = ClientWorld::read(world);
            let entities = world.entities.read();
            let lp_loc: &CLocation = entities.ecs.get_component(client.local_player).unwrap();
            let player_pos = lp_loc.position;
            let player_ang = lp_loc.orientation;
            let mview: Matrix4<f32>;
            let mrot: Matrix3<f32>;
            match client.camera_settings {
                CameraSettings::FPS { .. } => {
                    mrot = glm::quat_to_mat3(&-player_ang);
                    mview = mrot.to_homogeneous() * glm::translation(&-player_pos);
                }
            }

            let voxels = world.voxels.read();
            use crate::world::raycast;
            let fwd = mrot.transpose() * vec3(0.0, 0.0, 1.0);
            let rc = raycast::RaycastQuery::new_directed(
                player_pos,
                fwd,
                32.0,
                &world,
                Some(&voxels),
                None,
            )
            .execute();
            if let raycast::Hit::Voxel { position, .. } = rc.hit {
                hichunk = chunkpos_from_blockpos(position);
                hiidx = blockidx_from_blockpos(position) as i32;
            }
            mview
        } else {
            Matrix4::identity()
        };

        let mut cmdbufbuild = fctx.replace_cmd();

        let mproj: Matrix4<f32>;
        let ubo = {
            let mut ubo = vox::VoxelUBO::default();
            let mmdl: Matrix4<f32> = glm::identity();
            ubo.model = mmdl.into();
            ubo.view = mview.into();
            let swdim = fctx.dims;
            let sfdim = [swdim[0] as f32, swdim[1] as f32];
            mproj = {
                let fov = 75.0 * std::f32::consts::PI / 180.0;
                let aspect = sfdim[0] / sfdim[1];
                let (near, far) = (0.1, 1000.0);
                let f = 1.0 / (0.5 * fov).tan();
                let mut mproj: Matrix4<f32> = zero();
                mproj[(0, 0)] = f / aspect;
                mproj[(1, 1)] = -f;
                mproj[(2, 2)] = -far / (near - far);
                mproj[(3, 2)] = 1.0;
                mproj[(2, 3)] = (near * far) / (near - far);
                mproj
            };
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
            let chunk = chunkmut.read();
            let pc = vox::VoxelPC {
                chunk_offset: pos.map(|x| (x as f32) * (CHUNK_DIM as f32)).into(),
                highlight_index: if chunk.cpos == hichunk { hiidx } else { -1 },
            };
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

impl Drop for VoxelRenderer {
    fn drop(&mut self) {
        self.kill_threads();
    }
}
