use crate::client::config::Config;
use crate::client::render::voxmesh::mesh_from_chunk;
use crate::client::render::vulkan::{
    allocation_cbs, OnetimeCmdGuard, OwnedBuffer, OwnedImage, Queues, RenderingHandles,
    INFLIGHT_FRAMES,
};
use crate::client::render::*;
use crate::client::world::{CameraSettings, ClientWorld};
use crate::math::*;
use crate::world::ecs::{CLocation, ECSHandler};
use crate::world::{blockidx_from_blockpos, chunkpos_from_blockpos, World};
use crate::world::{ChunkPosition, CHUNK_DIM};
use ash::version::DeviceV1_0;
use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::fmt::{Debug, Formatter};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::sync::{mpsc, Weak};
use std::thread;
use thread_local::CachedThreadLocal;
use vk_mem as vma;

pub mod vox {
    use crate::offset_of;
    use ash::vk;
    use std::mem;
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

    impl VoxelVertex {
        pub fn description() -> (
            [vk::VertexInputBindingDescription; 1],
            [vk::VertexInputAttributeDescription; 4],
        ) {
            let bind_dsc = [vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(mem::size_of::<Self>() as u32)
                .input_rate(vk::VertexInputRate::VERTEX)
                .build()];
            let attr_dsc = [
                vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: 0,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Self, position) as u32,
                },
                vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: 1,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Self, color) as u32,
                },
                vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: 2,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(Self, texcoord) as u32,
                },
                vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: 3,
                    format: vk::Format::R32_SINT,
                    offset: offset_of!(Self, index) as u32,
                },
            ];
            (bind_dsc, attr_dsc)
        }
    }

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
    impl VoxelPC {
        pub fn pc_ranges() -> [vk::PushConstantRange; 1] {
            [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                offset: 0,
                size: mem::size_of::<Self>() as u32,
            }]
        }
    }
}

struct DrawnChunk {
    pub cpos: ChunkPosition,
    pub last_dirty: u64,
    pub buffer: OwnedBuffer,
    pub istart: usize,
    pub vcount: u32,
    pub icount: u32,
}

impl Debug for DrawnChunk {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "DrawnChunk {{ last_dirty: {} }}", self.last_dirty)
    }
}

impl DrawnChunk {
    fn destroy(&mut self, handles: &RenderingHandles) {
        self.buffer.destroy(&mut handles.vmalloc.lock());
    }
}

type ChunkMsg = (ChunkPosition, DrawnChunk);

pub struct VoxelRenderer {
    texture_array: OwnedImage,
    texture_array_view: vk::ImageView,
    texture_name_map: HashMap<String, u32>,
    texture_sampler: vk::Sampler,
    texture_ds: vk::DescriptorSet,
    texture_ds_layout: vk::DescriptorSetLayout,
    uniform_ds_layout: vk::DescriptorSetLayout,
    uniform_dss: Vec<vk::DescriptorSet>,
    ds_pool: vk::DescriptorPool,
    world: Option<Arc<World>>,
    pub voxel_pipeline_layout: vk::PipelineLayout,
    pub voxel_pipeline: vk::Pipeline,
    atmosphere_renderer: AtmosphereRenderer,
    drawn_chunks: HashMap<ChunkPosition, DrawnChunk>,
    pub ubuffer: OwnedBuffer,
    draw_queue: Arc<Mutex<Vec<ChunkPosition>>>,
    progress_set: Arc<Mutex<HashSet<ChunkPosition>>>,
    worker_threads: Vec<thread::JoinHandle<()>>,
    thread_killer: Arc<AtomicBool>,
    work_receiver: CachedThreadLocal<mpsc::Receiver<ChunkMsg>>,
}

impl VoxelRenderer {
    pub fn new(cfg: &Config, rctx: &mut RenderingContext) -> Self {
        // load textures
        let (texture_array, texture_array_view, texture_name_map) =
            Self::new_texture_atlas(cfg, rctx);

        let texture_sampler = {
            let sci = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .anisotropy_enable(true)
                .max_anisotropy(4.0);
            unsafe { rctx.handles.device.create_sampler(&sci, allocation_cbs()) }
                .expect("Could not create voxel texture sampler")
        };

        let texture_ds_layout = {
            let samplers = [texture_sampler];
            let binds = [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .immutable_samplers(&samplers)
                .descriptor_count(1)
                .build()];
            let dsci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&binds);
            unsafe {
                rctx.handles
                    .device
                    .create_descriptor_set_layout(&dsci, allocation_cbs())
            }
            .expect("Could not create voxel descriptor set layout")
        };

        let uniform_ds_layout = {
            let binds = [vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .build()];
            let dsci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&binds);
            unsafe {
                rctx.handles
                    .device
                    .create_descriptor_set_layout(&dsci, allocation_cbs())
            }
            .expect("Could not create chunk descriptor set layout")
        };

        let ds_pool = {
            let szs = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 2,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: INFLIGHT_FRAMES * 2,
                },
            ];
            let dspi = vk::DescriptorPoolCreateInfo::builder()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(INFLIGHT_FRAMES * 4)
                .pool_sizes(&szs);
            unsafe {
                rctx.handles
                    .device
                    .create_descriptor_pool(&dspi, allocation_cbs())
            }
            .expect("Could not create voxel descriptor pool")
        };

        let texture_ds = {
            let lay = [texture_ds_layout];
            let dai = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(ds_pool)
                .set_layouts(&lay);
            unsafe { rctx.handles.device.allocate_descriptor_sets(&dai) }
                .expect("Could not allocate voxel texture descriptor")[0]
        };

        // write to texture DS
        {
            let ii = [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture_array_view)
                .build()];
            let dw = [vk::WriteDescriptorSet::builder()
                .dst_set(texture_ds)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&ii)
                .build()];
            unsafe {
                rctx.handles.device.update_descriptor_sets(&dw, &[]);
            }
        }

        let ubo_sz = std::mem::size_of::<vox::VoxelUBO>() as u64;

        let ubuffer = {
            let mut vmalloc = rctx.handles.vmalloc.lock();
            let qfis = [rctx.handles.queues.get_primary_family()];
            let bci = vk::BufferCreateInfo::builder()
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&qfis)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .size(ubo_sz * (INFLIGHT_FRAMES as u64));
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::CpuToGpu,
                flags: vma::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let mut buf = OwnedBuffer::from(&mut vmalloc, &bci, &aci);
            buf.give_name(&rctx.handles, || "voxel renderer UBOs");
            buf
        };

        let uniform_dss = {
            let lay = [uniform_ds_layout; INFLIGHT_FRAMES as usize];
            let dai = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(ds_pool)
                .set_layouts(&lay);
            let v = unsafe { rctx.handles.device.allocate_descriptor_sets(&dai) }
                .expect("Could not allocate voxel UBO descriptors");
            for i in 0..(INFLIGHT_FRAMES as u64) {
                let ii = [vk::DescriptorBufferInfo::builder()
                    .buffer(ubuffer.buffer)
                    .range(ubo_sz)
                    .offset(i * ubo_sz)
                    .build()];
                let dw = [vk::WriteDescriptorSet::builder()
                    .dst_set(v[i as usize])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&ii)
                    .build()];
                unsafe {
                    rctx.handles.device.update_descriptor_sets(&dw, &[]);
                }
            }
            v
        };

        let pipeline_layot = {
            let pc = vox::VoxelPC::pc_ranges();
            let dsls = [uniform_ds_layout, texture_ds_layout];
            let lci = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&dsls)
                .push_constant_ranges(&pc);
            unsafe {
                rctx.handles
                    .device
                    .create_pipeline_layout(&lci, allocation_cbs())
            }
            .expect("Could not create voxel pipeline layout")
        };

        // create pipeline
        let voxel_pipeline = {
            let vs = rctx
                .handles
                .load_shader_module("res/shaders/voxel.vert.spv")
                .expect("Couldn't load vertex voxel shader");
            let fs = rctx
                .handles
                .load_shader_module("res/shaders/voxel.frag.spv")
                .expect("Couldn't load fragment voxel shader");
            //
            let cmain = CString::new("main").unwrap();
            let vss = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs)
                .name(&cmain);
            let fss = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs)
                .name(&cmain);
            let shaders = [vss.build(), fss.build()];
            let vox_desc = vox::VoxelVertex::description();
            let vtxinp = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&vox_desc.0)
                .vertex_attribute_descriptions(&vox_desc.1);
            let inpasm = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);
            let viewport = [rctx.swapchain.dynamic_state.viewport];
            let scissor = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: viewport[0].width as u32,
                    height: viewport[0].height as u32,
                },
            }];
            let vwp_info = vk::PipelineViewportStateCreateInfo::builder()
                .scissors(&scissor)
                .viewports(&viewport);
            let raster = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);
            let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let depthstencil = vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_bounds_test_enable(false)
                .stencil_test_enable(false)
                .min_depth_bounds(0.0)
                .max_depth_bounds(1.0);
            let blendings = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(false)
                .build()];
            let blending = vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(&blendings)
                .blend_constants([0.0, 0.0, 0.0, 0.0]);
            let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dyn_state =
                vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dyn_states);

            let pci = vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shaders)
                .vertex_input_state(&vtxinp)
                .input_assembly_state(&inpasm)
                .viewport_state(&vwp_info)
                .rasterization_state(&raster)
                .multisample_state(&multisampling)
                .depth_stencil_state(&depthstencil)
                .color_blend_state(&blending)
                .dynamic_state(&dyn_state)
                .layout(pipeline_layot)
                .render_pass(rctx.handles.mainpass)
                .subpass(0)
                .base_pipeline_handle(vk::Pipeline::null())
                .base_pipeline_index(-1);
            let pcis = [pci.build()];

            let pipeline = unsafe {
                rctx.handles.device.create_graphics_pipelines(
                    rctx.pipeline_cache,
                    &pcis,
                    allocation_cbs(),
                )
            }
            .expect("Could not create voxel pipeline")[0];

            unsafe {
                rctx.handles
                    .device
                    .destroy_shader_module(vs, allocation_cbs());
                rctx.handles
                    .device
                    .destroy_shader_module(fs, allocation_cbs());
            }
            pipeline
        };

        Self {
            texture_array,
            texture_array_view,
            texture_name_map,
            texture_sampler,
            texture_ds,
            texture_ds_layout,
            uniform_ds_layout,
            uniform_dss,
            ds_pool,
            world: None,
            voxel_pipeline_layout: pipeline_layot,
            voxel_pipeline,
            atmosphere_renderer: AtmosphereRenderer::new(cfg, rctx),
            drawn_chunks: HashMap::new(),
            ubuffer,
            draw_queue: Arc::new(Mutex::new(Vec::new())),
            progress_set: Arc::new(Mutex::new(HashSet::new())),
            worker_threads: Vec::new(),
            thread_killer: Arc::new(AtomicBool::new(false)),
            work_receiver: CachedThreadLocal::new(),
        }
    }

    pub fn destroy(&mut self, handles: &RenderingHandles) {
        self.kill_threads();
        unsafe {
            handles.device.device_wait_idle().unwrap();
        }
        for (_cpos, mut ch) in self.drawn_chunks.drain() {
            ch.destroy(handles);
        }
        unsafe {
            handles
                .device
                .destroy_pipeline(self.voxel_pipeline, allocation_cbs());
            handles
                .device
                .destroy_pipeline_layout(self.voxel_pipeline_layout, allocation_cbs());
            self.uniform_dss.clear();
            handles
                .device
                .destroy_descriptor_pool(self.ds_pool, allocation_cbs());
            handles
                .device
                .destroy_descriptor_set_layout(self.texture_ds_layout, allocation_cbs());
            handles
                .device
                .destroy_descriptor_set_layout(self.uniform_ds_layout, allocation_cbs());
            handles
                .device
                .destroy_sampler(self.texture_sampler, allocation_cbs());
            self.texture_name_map.clear();
            handles
                .device
                .destroy_image_view(self.texture_array_view, allocation_cbs());
            self.texture_array.destroy(&mut handles.vmalloc.lock());
            self.ubuffer.destroy(&mut handles.vmalloc.lock());
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
    ) -> (OwnedImage, vk::ImageView, HashMap<String, u32>) {
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
        let mut vmalloc = rctx.handles.vmalloc.lock();
        let qfis = [rctx.handles.queues.get_primary_family()];
        let mip_lvls = (f64::from(idim.0.min(idim.1)).log2().floor() as u32).max(1);
        let img_extent = vk::Extent3D {
            width: idim.0,
            height: idim.1,
            depth: 1,
        };
        let img = {
            let ici = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .extent(img_extent)
                .mip_levels(mip_lvls)
                .array_layers(numimages)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&qfis)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::GpuOnly,
                flags: vma::AllocationCreateFlags::DEDICATED_MEMORY,
                ..Default::default()
            };

            OwnedImage::from(&mut vmalloc, &ici, &aci)
        };
        // upload main level
        let whole_img = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_lvls,
            base_array_layer: 0,
            layer_count: numimages,
        };
        let queue = rctx.handles.queues.lock_primary_queue();
        {
            let bci = vk::BufferCreateInfo::builder()
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&qfis)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .size((idim.0 * idim.1 * numimages * 4) as vk::DeviceSize);
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::CpuOnly,
                flags: vma::AllocationCreateFlags::DEDICATED_MEMORY
                    | vma::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let mut buf = OwnedBuffer::from(&mut vmalloc, &bci, &aci);
            let ba = buf.allocation.as_ref().unwrap();
            assert!(ba.1.get_size() >= rawdata.len());
            unsafe {
                std::ptr::copy_nonoverlapping(
                    rawdata.as_ptr(),
                    ba.1.get_mapped_data(),
                    rawdata.len(),
                );
            }
            vmalloc.flush_allocation(&ba.0, 0, rawdata.len()).unwrap();
            //
            let cmd = OnetimeCmdGuard::new(&rctx.handles);
            vk_sync::cmd::pipeline_barrier(
                rctx.handles.device.fp_v1_0(),
                cmd.handle(),
                None,
                &[],
                &[vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::Nothing],
                    next_accesses: &[vk_sync::AccessType::TransferWrite],
                    previous_layout: vk_sync::ImageLayout::Optimal,
                    next_layout: vk_sync::ImageLayout::Optimal,
                    discard_contents: false,
                    src_queue_family_index: qfis[0],
                    dst_queue_family_index: qfis[0],
                    image: img.image,
                    range: whole_img,
                }],
            );
            let whole_mip0 = vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: numimages,
            };
            let whole_copy = [vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: whole_mip0,
                image_offset: Default::default(),
                image_extent: img_extent,
            }];
            unsafe {
                rctx.handles.device.cmd_copy_buffer_to_image(
                    cmd.handle(),
                    buf.buffer,
                    img.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &whole_copy,
                );
            }
            let ibarrier = vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferWrite],
                next_accesses: &[vk_sync::AccessType::TransferRead],
                previous_layout: vk_sync::ImageLayout::Optimal,
                next_layout: vk_sync::ImageLayout::Optimal,
                discard_contents: false,
                src_queue_family_index: qfis[0],
                dst_queue_family_index: qfis[0],
                image: img.image,
                range: vk::ImageSubresourceRange {
                    base_mip_level: 0,
                    level_count: 1,
                    ..whole_img
                },
            };
            vk_sync::cmd::pipeline_barrier(
                rctx.handles.device.fp_v1_0(),
                cmd.handle(),
                None,
                &[],
                &[ibarrier.clone()],
            );
            for mip in 1..mip_lvls {
                let prev_mip = mip - 1;
                let prev_sz = (idim.0 >> prev_mip, idim.1 >> prev_mip);
                let now_sz = (idim.0 >> mip, idim.1 >> mip);
                let img_blit = [vk::ImageBlit {
                    src_subresource: vk::ImageSubresourceLayers {
                        mip_level: prev_mip,
                        ..whole_mip0
                    },
                    src_offsets: [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: prev_sz.0 as i32,
                            y: prev_sz.1 as i32,
                            z: 1,
                        },
                    ],
                    dst_subresource: vk::ImageSubresourceLayers {
                        mip_level: mip,
                        ..whole_mip0
                    },
                    dst_offsets: [
                        vk::Offset3D::default(),
                        vk::Offset3D {
                            x: now_sz.0 as i32,
                            y: now_sz.1 as i32,
                            z: 1,
                        },
                    ],
                }];
                unsafe {
                    rctx.handles.device.cmd_blit_image(
                        cmd.handle(),
                        img.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        img.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &img_blit,
                        vk::Filter::LINEAR,
                    );
                    vk_sync::cmd::pipeline_barrier(
                        rctx.handles.device.fp_v1_0(),
                        cmd.handle(),
                        None,
                        &[],
                        &[vk_sync::ImageBarrier {
                            range: vk::ImageSubresourceRange {
                                base_mip_level: mip,
                                level_count: 1,
                                ..whole_img
                            },
                            ..ibarrier.clone()
                        }],
                    );
                }
            }
            vk_sync::cmd::pipeline_barrier(
                rctx.handles.device.fp_v1_0(),
                cmd.handle(),
                None,
                &[],
                &[vk_sync::ImageBarrier {
                    previous_accesses: &[vk_sync::AccessType::TransferRead],
                    next_accesses: &[
                        vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                    ],
                    range: whole_img,
                    ..ibarrier.clone()
                }],
            );
            cmd.execute(&queue);
            buf.destroy(&mut vmalloc);
        }

        img.give_name(&rctx.handles, || "voxel texture array");

        let img_view = {
            let ivci = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(vk::Format::R8G8B8A8_SRGB)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(whole_img)
                .image(img.image);
            unsafe {
                rctx.handles
                    .device
                    .create_image_view(&ivci, allocation_cbs())
            }
            .expect("Couldn't create voxel texture array view")
        };

        (img, img_view, names)
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
            let handles = rctx.handles.clone();
            let work_queue = self.draw_queue.clone();
            let progress_set = self.progress_set.clone();
            let killswitch = self.thread_killer.clone();
            let thr = tb
                .spawn(move || {
                    Self::vox_worker(tworld, handles, work_queue, progress_set, ttx, killswitch)
                })
                .expect("Could not create voxrender worker thread");
            self.worker_threads.push(thr);
        }
        self.world = Some(world);
    }

    #[allow(clippy::cast_ptr_alignment)]
    fn vox_worker(
        world_w: Weak<World>,
        handles: RenderingHandles,
        work_queue: Arc<Mutex<Vec<ChunkPosition>>>,
        progress_set: Arc<Mutex<HashSet<ChunkPosition>>>,
        submission: mpsc::Sender<ChunkMsg>,
        killswitch: Arc<AtomicBool>,
    ) {
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
                let vcount = mesh.vertices.len() as u32;
                let icount = mesh.indices.len() as u32;
                let v_sz = std::mem::size_of::<vox::VoxelVertex>();
                let i_sz = std::mem::size_of::<u32>();
                let v_tot_sz = v_sz * (vcount as usize);
                let i_tot_sz = i_sz * (icount as usize);
                let tot_sz = v_tot_sz + i_tot_sz;

                let dchunk = if tot_sz > 0 {
                    let qfs = [handles.queues.get_primary_family()];
                    let mut staging = {
                        let bi = vk::BufferCreateInfo::builder()
                            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                            .size(tot_sz as u64)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE)
                            .queue_family_indices(&qfs)
                            .build();
                        let ai = vma::AllocationCreateInfo {
                            usage: vma::MemoryUsage::CpuOnly,
                            flags: vma::AllocationCreateFlags::MAPPED,
                            ..Default::default()
                        };
                        OwnedBuffer::from(&mut handles.vmalloc.lock(), &bi, &ai)
                    };
                    let buffer = {
                        let bi = vk::BufferCreateInfo::builder()
                            .usage(
                                vk::BufferUsageFlags::TRANSFER_DST
                                    | vk::BufferUsageFlags::VERTEX_BUFFER
                                    | vk::BufferUsageFlags::INDEX_BUFFER,
                            )
                            .size(tot_sz as u64)
                            .queue_family_indices(&qfs)
                            .sharing_mode(vk::SharingMode::EXCLUSIVE);
                        let ai = vma::AllocationCreateInfo {
                            usage: vma::MemoryUsage::GpuOnly,
                            ..Default::default()
                        };
                        OwnedBuffer::from(&mut handles.vmalloc.lock(), &bi, &ai)
                    };
                    buffer.give_name(&handles, || {
                        format!("chunk({},{},{})", cpos.x, cpos.y, cpos.z)
                    });
                    // write to staging
                    {
                        let ai = staging.allocation.as_ref().unwrap();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                mesh.vertices.as_ptr(),
                                ai.1.get_mapped_data() as *mut vox::VoxelVertex,
                                mesh.vertices.len(),
                            );
                            std::ptr::copy_nonoverlapping(
                                mesh.indices.as_ptr(),
                                ai.1.get_mapped_data().add(v_tot_sz) as *mut u32,
                                mesh.indices.len(),
                            );
                        }
                        handles
                            .vmalloc
                            .lock()
                            .flush_allocation(&ai.0, 0, tot_sz)
                            .unwrap();
                    }
                    // copy from staging to gpu
                    {
                        let cmd = OnetimeCmdGuard::new(&handles);
                        let bci = vk::BufferCopy::builder()
                            .size(tot_sz as vk::DeviceSize)
                            .build();
                        unsafe {
                            handles.device.cmd_copy_buffer(
                                cmd.handle(),
                                staging.buffer,
                                buffer.buffer,
                                &[bci],
                            );
                        }
                        cmd.execute(&handles.queues.lock_gtransfer_queue());
                    }
                    staging.destroy(&mut handles.vmalloc.lock());

                    DrawnChunk {
                        cpos,
                        last_dirty: voxels.chunks.get(&cpos).unwrap().dirty,
                        buffer,
                        istart: v_tot_sz,
                        vcount,
                        icount,
                    }
                } else {
                    DrawnChunk {
                        cpos,
                        last_dirty: voxels.chunks.get(&cpos).unwrap().dirty,
                        buffer: OwnedBuffer::new(),
                        istart: 0,
                        vcount: 0,
                        icount: 0,
                    }
                };

                if submission.send((cpos, dchunk)).is_err() {
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
                if let Some(mut ch) = self.drawn_chunks.insert(p, dc) {
                    ch.destroy(&fctx.rctx.handles);
                }
            }
        }

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
            if self.drawn_chunks.get(&cpos).unwrap().last_dirty != chunk.dirty {
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
            if let Some(mut ch) = self.drawn_chunks.remove(&cpos) {
                ch.destroy(&fctx.rctx.handles);
            }
        }

        if new_cmds {
            for t in self.worker_threads.iter() {
                t.thread().unpark();
            }
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
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
            let ubo_sz = std::mem::size_of::<vox::VoxelUBO>();
            let ubo_start = ubo_sz * fctx.inflight_index;
            let ai = self.ubuffer.allocation.as_ref().unwrap();
            unsafe {
                std::ptr::write(
                    (ai.1.get_mapped_data().add(ubo_start)) as *mut vox::VoxelUBO,
                    ubo,
                );
            }
            fctx.rctx
                .handles
                .vmalloc
                .lock()
                .flush_allocation(&ai.0, ubo_start, ubo_sz)
                .unwrap();
            self.uniform_dss[fctx.inflight_index]
        };

        let device = &fctx.rctx.handles.device;

        unsafe {
            device.cmd_bind_pipeline(
                fctx.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.voxel_pipeline,
            );
        }
        let vwp = fctx.rctx.swapchain.dynamic_state.viewport;
        unsafe {
            device.cmd_set_viewport(fctx.cmd, 0, &[vwp]);
        }
        let sci = vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: vwp.width as u32,
                height: vwp.height as u32,
            },
        };
        unsafe { device.cmd_set_scissor(fctx.cmd, 0, &[sci]) }
        unsafe {
            device.cmd_bind_descriptor_sets(
                fctx.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.voxel_pipeline_layout,
                0,
                &[ubo, self.texture_ds],
                &[],
            );
        }

        for (pos, chunk) in self.drawn_chunks.iter().filter(|c| c.1.icount > 0) {
            let pc = vox::VoxelPC {
                chunk_offset: pos.map(|x| (x as f32) * (CHUNK_DIM as f32)).into(),
                highlight_index: if chunk.cpos == hichunk { hiidx } else { -1 },
            };
            let pc_sz = std::mem::size_of_val(&pc);
            let pc_bytes =
                unsafe { std::slice::from_raw_parts(&pc as *const _ as *const u8, pc_sz) };
            unsafe {
                device.cmd_push_constants(
                    fctx.cmd,
                    self.voxel_pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                )
            }
            unsafe {
                device.cmd_bind_vertex_buffers(fctx.cmd, 0, &[chunk.buffer.buffer], &[0]);
                device.cmd_bind_index_buffer(
                    fctx.cmd,
                    chunk.buffer.buffer,
                    chunk.istart as vk::DeviceSize,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(fctx.cmd, chunk.icount, 1, 0, 0, 0);
            }
        }

        self.atmosphere_renderer.inpass_draw(fctx, mview, mproj);
    }
}

impl Drop for VoxelRenderer {
    fn drop(&mut self) {
        self.kill_threads();
    }
}
