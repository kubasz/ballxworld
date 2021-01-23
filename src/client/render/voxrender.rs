use crate::client::render::resources::RenderingResources;
use crate::client::render::vkhelpers::*;
use crate::client::render::voxmesh::{is_chunk_trivial, mesh_from_chunk};
use crate::client::render::vulkan::{
    allocation_cbs, RenderingHandles, VDODestroyQueue, INFLIGHT_FRAMES,
};
use crate::client::render::*;
use crate::client::world::CameraSettings;
use crate::config::Config;
use ash::version::DeviceV1_0;
use ash::vk;
use bxw_util::math::vec3;
use bxw_util::math::*;
use bxw_util::taskpool::Task;
use bxw_util::*;
use bxw_world::ecs::{CLoadAnchor, CLocation, ECSHandler};
use bxw_world::entities::player::PLAYER_EYE_HEIGHT;
use bxw_world::generation::WorldBlocks;
use bxw_world::worldmgr::*;
use bxw_world::*;
use parking_lot::Mutex;
use smallvec::alloc::rc::Rc;
use std::any::Any;
use std::cell::RefCell;
use std::ffi::CString;
use std::sync::Arc;
use std::sync::Weak;
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
        pub barycentric_color_offset: [f32; 4],
        pub barycentric: [f32; 3],
    }

    impl VoxelVertex {
        pub fn description() -> (
            [vk::VertexInputBindingDescription; 1],
            [vk::VertexInputAttributeDescription; 6],
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
                vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: 4,
                    format: vk::Format::R32G32B32A32_SFLOAT,
                    offset: offset_of!(Self, barycentric_color_offset) as u32,
                },
                vk::VertexInputAttributeDescription {
                    binding: 0,
                    location: 5,
                    format: vk::Format::R32G32B32_SFLOAT,
                    offset: offset_of!(Self, barycentric) as u32,
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

    impl VoxelUBO {
        pub fn aligned_size(limits: &vk::PhysicalDeviceLimits) -> u32 {
            let real_sz = mem::size_of::<Self>() as u32;
            let alignment = limits.min_uniform_buffer_offset_alignment as u32;
            (real_sz + alignment - 1) / alignment * alignment
        }
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
    pub destroy_handle: Weak<Mutex<VDODestroyQueue>>,
}

impl Drop for DrawnChunk {
    fn drop(&mut self) {
        let buffer = std::mem::take(&mut self.buffer);
        let destroy_handle = std::mem::take(&mut self.destroy_handle);
        if let Some(destroy_handle) = destroy_handle.upgrade() {
            destroy_handle.lock().push(Box::new(buffer));
        }
        self.vcount = 0;
        self.icount = 0;
        self.istart = 0;
    }
}

pub struct VoxelRenderer {
    chunk_pool: Arc<AllocatorPool>,
    resources: Arc<RenderingResources>,
    texture_ds: vk::DescriptorSet,
    texture_ds_layout: vk::DescriptorSetLayout,
    uniform_ds_layout: vk::DescriptorSetLayout,
    uniform_dss: Vec<vk::DescriptorSet>,
    ds_pool: vk::DescriptorPool,
    pub voxel_pipeline_layout: vk::PipelineLayout,
    pub voxel_pipeline: vk::Pipeline,
    atmosphere_renderer: AtmosphereRenderer,
    drawn_chunks: Vec<Option<Arc<DrawnChunk>>>,
    pub ubuffer: OwnedBuffer,
}

pub struct MeshDataHandler {
    pub renderer: Rc<RefCell<VoxelRenderer>>,
    pub status_array: Vec<ChunkDataState>,
    pub rendering_handles: RenderingHandles,
}

impl MeshDataHandler {
    pub fn new(renderer: Rc<RefCell<VoxelRenderer>>, handles: &RenderingHandles) -> Self {
        Self {
            renderer,
            status_array: Vec::new(),
            rendering_handles: handles.clone(),
        }
    }
}

impl ChunkDataHandler for MeshDataHandler {
    fn status_array(&self) -> &Vec<ChunkDataState> {
        &self.status_array
    }

    fn status_array_mut(&mut self) -> &mut Vec<ChunkDataState> {
        &mut self.status_array
    }

    fn get_dependency(&self) -> Option<(usize, bool)> {
        Some((CHUNK_BLOCK_DATA, true))
    }

    fn get_data(&self, _world: &World, index: usize) -> AnyChunkData {
        self.renderer.borrow().drawn_chunks[index]
            .clone()
            .map(|x| x as AnyChunkDataArc)
    }

    fn swap_data(&mut self, _world: &World, index: usize, new_data: AnyChunkData) -> AnyChunkData {
        let mut vctx = self.renderer.borrow_mut();
        let new_data = new_data.map(|d| d.downcast::<DrawnChunk>().unwrap());
        let old_data = std::mem::replace(&mut vctx.drawn_chunks[index], new_data);
        old_data.map(|x| x as AnyChunkDataArc)
    }

    fn resize_data(&mut self, _world: &World, new_size: usize) {
        let mut vctx = self.renderer.borrow_mut();
        vctx.drawn_chunks.resize(new_size, None);
    }

    #[allow(clippy::cast_ptr_alignment)]
    fn create_chunk_update_task(
        &mut self,
        world: &World,
        cpos: ChunkPosition,
        index: usize,
    ) -> Option<Task> {
        let blocks = world.get_handler(CHUNK_BLOCK_DATA).borrow();
        let blocks = blocks.as_any().downcast_ref::<WorldBlocks>().unwrap();

        let registry = blocks.voxel_registry.clone();
        let mut neighbors = Vec::new();
        for npos in iter_neighbors(cpos, true) {
            let chunk = match blocks.get_chunk(world, npos) {
                Some(c) => c,
                None => return None,
            };
            neighbors.push(chunk);
        }

        self.status_array[index] = match self.status_array[index] {
            ChunkDataState::Unloaded => ChunkDataState::Loading,
            _ => ChunkDataState::Updating,
        };
        let mut vctx = self.renderer.borrow_mut();
        let texture_dim = vctx.resources.voxel_texture_array_params.dims;
        let handles = self.rendering_handles.clone();
        let destroy_handle = handles.get_destroy_queue_handle();

        if is_chunk_trivial(&neighbors[13], &registry) {
            self.status_array[index] = ChunkDataState::Loaded;
            vctx.drawn_chunks[index] = Some(Arc::new(DrawnChunk {
                cpos,
                last_dirty: 0,
                buffer: OwnedBuffer::new(),
                istart: 0,
                vcount: 0,
                icount: 0,
                destroy_handle,
            }));
            return None;
        }

        let alloc_pool = vctx.chunk_pool.clone();
        let submit_channel = world.get_sync_task_channel();

        Some(Task::new(
            move || {
                let cpci = vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(handles.queues.get_gtransfer_family())
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                let cmd_pool = DroppingCommandPool::new(&handles, &cpci);
                drop(cpci);
                let mesh = match mesh_from_chunk(&registry, &neighbors, texture_dim) {
                    Some(m) => m,
                    None => return,
                };
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
                            flags: vma::AllocationCreateFlags::MAPPED
                                | vma::AllocationCreateFlags::DEDICATED_MEMORY,
                            ..Default::default()
                        };
                        OwnedBuffer::from(&mut handles.vmalloc.lock(), &bi, &ai)
                    };
                    let buffer = {
                        let pool = &alloc_pool.0;
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
                            pool: Some(pool.clone()),
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
                        let cmd = OnetimeCmdGuard::new(&handles, Some(cmd_pool.pool));
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
                    staging.destroy(&mut handles.vmalloc.lock(), &handles);

                    DrawnChunk {
                        cpos,
                        last_dirty: 0,
                        buffer,
                        istart: v_tot_sz,
                        vcount,
                        icount,
                        destroy_handle,
                    }
                } else {
                    DrawnChunk {
                        cpos,
                        last_dirty: 0,
                        buffer: OwnedBuffer::new(),
                        istart: 0,
                        vcount: 0,
                        icount: 0,
                        destroy_handle,
                    }
                };
                let dchunk = Arc::new(dchunk);
                submit_channel
                    .send(Box::new(move |world| {
                        let index = match world.get_chunk_index(cpos) {
                            Some(i) => i,
                            None => return,
                        };
                        let mut vctx = world.get_handler(CHUNK_MESH_DATA).borrow_mut();
                        let vctx: &mut Self = vctx.as_any_mut().downcast_mut().unwrap();
                        // request was cancelled
                        if vctx.status_array[index] == ChunkDataState::Unloaded {
                            return;
                        }
                        vctx.status_array[index] = ChunkDataState::Loaded;
                        vctx.renderer.borrow_mut().drawn_chunks[index] = Some(dchunk);
                    }))
                    .unwrap_or(());
            },
            false,
            false,
        ))
    }

    fn needs_loading_for_anchor(&self, anchor: &CLoadAnchor) -> bool {
        anchor.load_mesh
    }

    fn serializable(&self) -> bool {
        false
    }

    fn serialize_data(&self, _world: &World, _index: usize) -> Option<Vec<u8>> {
        None
    }

    fn deserialize_data(
        &mut self,
        _world: &World,
        _index: usize,
        _data: &[u8],
    ) -> Result<AnyChunkData, &'static str> {
        Err("Trying to serialize chunk mesh data")
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl VoxelRenderer {
    pub fn new(cfg: &Config, rctx: &mut RenderingContext, res: Arc<RenderingResources>) -> Self {
        let chunk_pool = {
            let vmalloc = rctx.handles.vmalloc.lock();
            let qfs = [rctx.handles.queues.get_primary_family()];
            let ex_bi = vk::BufferCreateInfo::builder()
                .usage(
                    vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER,
                )
                .size(1024)
                .queue_family_indices(&qfs)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::GpuOnly,
                ..Default::default()
            };
            let mem_type = vmalloc
                .find_memory_type_index_for_buffer_info(&ex_bi, &aci)
                .unwrap();

            let pci = vma::AllocatorPoolCreateInfo {
                memory_type_index: mem_type,
                flags: vma::AllocatorPoolCreateFlags::BUDDY_ALGORITHM
                    | vma::AllocatorPoolCreateFlags::IGNORE_BUFFER_IMAGE_GRANULARITY,
                block_size: 128 * 1024 * 1024,
                min_block_count: 1,
                max_block_count: 0,
                frame_in_use_count: INFLIGHT_FRAMES,
            };
            let pool = vmalloc
                .create_pool(&pci)
                .expect("Could not create chunk buffer allocation pool");
            Arc::new(AllocatorPool(pool))
        };

        let texture_ds_layout = {
            let samplers = [res.voxel_texture_sampler];
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
                .image_view(res.voxel_texture_array.image_view)
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

        let ubo_sz = vox::VoxelUBO::aligned_size(&rctx.handles.physical_limits) as u64;

        let ubuffer = {
            let mut vmalloc = rctx.handles.vmalloc.lock();
            let qfis = [rctx.handles.queues.get_primary_family()];
            let bci = vk::BufferCreateInfo::builder()
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&qfis)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .size(ubo_sz * u64::from(INFLIGHT_FRAMES));
            let aci = vma::AllocationCreateInfo {
                usage: vma::MemoryUsage::CpuToGpu,
                flags: vma::AllocationCreateFlags::MAPPED,
                ..Default::default()
            };
            let buf = OwnedBuffer::from(&mut vmalloc, &bci, &aci);
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
            for i in 0..u64::from(INFLIGHT_FRAMES) {
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

        let pipeline_layout = {
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
            let viewport = [rctx.swapchain.dynamic_state.get_viewport()];
            let scissor = [rctx.swapchain.dynamic_state.get_scissor()];
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
                .rasterization_samples(rctx.handles.sample_count)
                .sample_shading_enable(true)
                .min_sample_shading(0.2);
            let depthstencil = make_pipe_depthstencil();
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
                .layout(pipeline_layout)
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
            chunk_pool,
            resources: res,
            texture_ds,
            texture_ds_layout,
            uniform_ds_layout,
            uniform_dss,
            ds_pool,
            voxel_pipeline_layout: pipeline_layout,
            voxel_pipeline,
            atmosphere_renderer: AtmosphereRenderer::new(cfg, rctx),
            drawn_chunks: Default::default(),
            ubuffer,
        }
    }

    pub fn destroy(mut self, handles: &RenderingHandles) {
        unsafe {
            handles.device.device_wait_idle().unwrap();
        }
        self.atmosphere_renderer.destroy(handles);
        self.drawn_chunks.clear();
        handles.flush_destroy_queue();
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
            let mut vmalloc = handles.vmalloc.lock();
            self.ubuffer.destroy(&mut vmalloc, handles);
            let vpool = Arc::try_unwrap(self.chunk_pool)
                .or(Err(()))
                .expect("Voxrender vmalloc pool has remaining users on destruction");
            vmalloc
                .destroy_pool(&vpool.0)
                .expect("Couldn't destroy voxrender vmalloc pool");
        }
    }

    pub fn get_texture_id(&self, name: &str) -> u32 {
        self.resources
            .voxel_texture_name_map
            .get(name)
            .copied()
            .unwrap_or(0)
    }

    pub fn drawn_chunks_number(&self) -> usize {
        self.drawn_chunks.len()
    }

    pub fn prepass_draw(&mut self, _fctx: &mut PrePassFrameContext, _world: &World) {}

    #[allow(clippy::cast_ptr_alignment)]
    pub fn inpass_draw(&mut self, fctx: &mut InPassFrameContext, world: &World) {
        let (mut hichunk, mut hiidx) = (vec3(0, 0, 0), -1);
        let fwd: Vector3<f32>;
        let player_pos;
        let player_cpos;
        let player_incpos;
        let mview = {
            let client = fctx.client_world;
            let entities = world.ecs();
            let lp_loc: &CLocation = entities.get_component(client.local_player).unwrap();
            player_pos = lp_loc.position + fctx.delta_time * lp_loc.velocity; // predict with delta-time
            player_cpos = chunkpos_from_blockpos(blockpos_from_worldpos(player_pos));
            player_incpos = player_pos - (player_cpos * CHUNK_DIM as i32).map(|c| c as f64);
            let player_ang = lp_loc.orientation;
            let mview: Matrix4<f32>;
            let mrot: Matrix3<f32>;
            match client.camera_settings {
                CameraSettings::FPS { .. } => {
                    mrot = glm::quat_to_mat3(&player_ang).map(|c| c as f32);
                    mview = mrot.to_homogeneous()
                        * glm::translation(
                            &-(player_incpos.map(|c| c as f32)
                                + vec3(0.0, PLAYER_EYE_HEIGHT as f32 / 2.0, 0.0)),
                        );
                }
            }

            fwd = mrot.transpose() * vec3(0.0, 0.0, 1.0);
            let rc = raycast::RaycastQuery::new_directed(
                player_pos + vec3(0.0, PLAYER_EYE_HEIGHT / 2.0, 0.0),
                fwd.map(|c| c as f64),
                32.0,
                &world,
                true,
                false,
            )
            .execute();
            if let raycast::Hit::Voxel { position, .. } = rc.hit {
                hichunk = chunkpos_from_blockpos(position);
                hiidx = blockidx_from_blockpos(position) as i32;
            }
            mview
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
                let (near, far) = (0.005, 4000.0);
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
            let ubo_stride =
                vox::VoxelUBO::aligned_size(&fctx.rctx.handles.physical_limits) as usize;
            let ubo_start = ubo_stride * fctx.inflight_index;
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
        fctx.rctx
            .swapchain
            .dynamic_state
            .cmd_update_pipeline(device, fctx.cmd);
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

        let always_dist = (CHUNK_DIM * CHUNK_DIM * 5) as f32;
        for chunk in self.drawn_chunks.iter() {
            let (pos, chunk) = match chunk {
                Some(dchunk) => (dchunk.cpos, dchunk),
                None => continue,
            };
            if chunk.icount == 0 {
                continue;
            }
            let ch_offset = (pos - player_cpos).map(|x| (x as f32) * (CHUNK_DIM as f32));
            let rpos = ch_offset - (player_incpos.map(|c| c as f32) - fwd * (CHUNK_DIM as f32));
            let ang = fwd.dot(&rpos);
            if ang < 0.0 && rpos.norm_squared() > always_dist {
                continue;
            }
            let pc = vox::VoxelPC {
                chunk_offset: ch_offset.into(),
                highlight_index: if chunk.cpos == hichunk { hiidx } else { -1 },
            };
            cmd_push_struct_constants(
                device,
                fctx.cmd,
                self.voxel_pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                &pc,
            );
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
