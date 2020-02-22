use crate::client::config::Config;
use crate::client::render::resources::RenderingResources;
use crate::client::render::vkhelpers::{make_pipe_depthstencil, OwnedBuffer, VulkanDeviceObject};
use crate::client::render::vulkan::{allocation_cbs, RenderingHandles, INFLIGHT_FRAMES};
use crate::client::render::{InPassFrameContext, PrePassFrameContext, RenderingContext};
use crate::math::*;
use ash::version::DeviceV1_0;
use ash::vk;
use rayon::prelude::*;
use std::ffi::CString;
use std::sync::Arc;
use vk_mem as vma;

pub mod z {
    pub const GUI_Z_OFFSET_BG: i32 = 0;
    pub const GUI_Z_OFFSET_CONTROL: i32 = 4;
    pub const GUI_Z_OFFSET_FG: i32 = 8;

    pub const GUI_ZFACTOR_LAYER: i32 = 64;

    pub const GUI_Z_LAYER_BACKGROUND: i32 = -1024;
    pub const GUI_Z_LAYER_HUD: i32 = GUI_Z_LAYER_BACKGROUND + GUI_ZFACTOR_LAYER;
    pub const GUI_Z_LAYER_UI_LOW: i32 = GUI_Z_LAYER_HUD + GUI_ZFACTOR_LAYER;
    pub const GUI_Z_LAYER_UI_MEDIUM: i32 = GUI_Z_LAYER_UI_LOW + GUI_ZFACTOR_LAYER;
    pub const GUI_Z_LAYER_UI_HIGH: i32 = GUI_Z_LAYER_UI_MEDIUM + GUI_ZFACTOR_LAYER;
    pub const GUI_Z_LAYER_UI_POPUP: i32 = GUI_Z_LAYER_UI_HIGH + GUI_ZFACTOR_LAYER;
    pub const GUI_Z_LAYER_OVERLAY: i32 = GUI_Z_LAYER_UI_POPUP + GUI_ZFACTOR_LAYER;
    pub const GUI_Z_LAYER_CURSOR: i32 = i32::max_value() - GUI_ZFACTOR_LAYER;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum GuiControlStyle {
    Window,
    Button,
    FullDark,
    FullBorder,
    FullButtonBg,
    FullWindowBg,
    FullBlack,
    FullWhite,
}

/// A single gui coordinate with relative and absolute positioning parts
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiCoord(f32, i32);

/// A gui 2D position/size vector
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiVec2(GuiCoord, GuiCoord);

/// A gui 2D position/size vector
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiRect {
    top_left: GuiVec2,
    bottom_right: GuiVec2,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct GuiColor([f32; 4]);

impl Default for GuiColor {
    fn default() -> Self {
        Self([1.0, 1.0, 1.0, 1.0])
    }
}

pub const GUI_WHITE: GuiColor = GuiColor([1.0, 1.0, 1.0, 1.0]);
pub const GUI_BLACK: GuiColor = GuiColor([0.0, 0.0, 0.0, 1.0]);

#[derive(Debug, Clone)]
pub enum GuiCmd {
    Rectangle {
        style: GuiControlStyle,
        rect: GuiRect,
    },
}

#[derive(Debug, Clone)]
pub struct GuiOrderedCmd {
    pub cmd: GuiCmd,
    pub z_index: i32,
    pub color: GuiColor,
}

#[derive(Debug, Default)]
pub struct GuiFrame {
    cmd_list: Vec<GuiOrderedCmd>,
    total_indices: u32,
}

impl GuiFrame {
    pub fn new() -> GuiFrame {
        Default::default()
    }

    pub fn reset(&mut self) {
        self.cmd_list.clear();
        self.total_indices = 0;
    }

    pub fn sort(&mut self) {
        self.cmd_list
            .par_sort_by(|a, b| Ord::cmp(&a.z_index, &b.z_index));
    }

    pub fn push_cmd(&mut self, cmd: GuiOrderedCmd) {
        self.cmd_list.push(cmd);
    }
}

pub struct GuiRenderer {
    resources: Arc<RenderingResources>,
    texture_ds: vk::DescriptorSet,
    texture_ds_layout: vk::DescriptorSetLayout,
    ds_pool: vk::DescriptorPool,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub gui_frame_pool: Vec<GuiFrame>,
    pub gui_buffers: Vec<(OwnedBuffer, OwnedBuffer)>,
}

impl GuiRenderer {
    pub fn new(
        _cfg: &Config,
        rctx: &mut RenderingContext,
        resources: Arc<RenderingResources>,
    ) -> Self {
        let texture_ds_layout = {
            let samplers = [
                resources.gui_sampler,
                resources.gui_sampler,
                resources.voxel_texture_sampler,
            ];
            let binds = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .immutable_samplers(&samplers[0..=0])
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .immutable_samplers(&samplers[1..=1])
                    .descriptor_count(1)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .immutable_samplers(&samplers[2..=2])
                    .descriptor_count(1)
                    .build(),
            ];
            let dsci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&binds);
            unsafe {
                rctx.handles
                    .device
                    .create_descriptor_set_layout(&dsci, allocation_cbs())
            }
            .expect("Could not create GUI descriptor set layout")
        };

        let ds_pool = {
            let szs = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 3,
            }];
            let dspi = vk::DescriptorPoolCreateInfo::builder()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .max_sets(1)
                .pool_sizes(&szs);
            unsafe {
                rctx.handles
                    .device
                    .create_descriptor_pool(&dspi, allocation_cbs())
            }
            .expect("Could not create GUI descriptor pool")
        };

        let texture_ds = {
            let lay = [texture_ds_layout];
            let dai = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(ds_pool)
                .set_layouts(&lay);
            unsafe { rctx.handles.device.allocate_descriptor_sets(&dai) }
                .expect("Could not allocate GUI texture descriptor")[0]
        };

        // write to texture DS
        {
            let ii = [
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(resources.gui_atlas.image_view)
                    .build(),
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(resources.font_atlas.image_view)
                    .build(),
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(resources.voxel_texture_array.image_view)
                    .build(),
            ];
            let wds = |i: usize| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(texture_ds)
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&ii[i..=i])
                    .build()
            };
            let dw = [wds(0), wds(1), wds(2)];
            unsafe {
                rctx.handles.device.update_descriptor_sets(&dw, &[]);
            }
        }

        let pipeline_layout = {
            let pc = [];
            let dsls = [texture_ds_layout];
            let lci = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&dsls)
                .push_constant_ranges(&pc);
            unsafe {
                rctx.handles
                    .device
                    .create_pipeline_layout(&lci, allocation_cbs())
            }
            .expect("Could not create GUI pipeline layout")
        };

        // create pipeline
        let pipeline = {
            let vs = rctx
                .handles
                .load_shader_module("res/shaders/ui.vert.spv")
                .expect("Couldn't load vertex GUI shader");
            let fs = rctx
                .handles
                .load_shader_module("res/shaders/ui.frag.spv")
                .expect("Couldn't load fragment GUI shader");
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
            let vtxinp = vk::PipelineVertexInputStateCreateInfo::builder();
            let inpasm = vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
                .primitive_restart_enable(true);
            let viewport = [rctx.swapchain.dynamic_state.get_viewport()];
            let scissor = [rctx.swapchain.dynamic_state.get_scissor()];
            let vwp_info = vk::PipelineViewportStateCreateInfo::builder()
                .scissors(&scissor)
                .viewports(&viewport);
            let raster = vk::PipelineRasterizationStateCreateInfo::builder()
                .depth_clamp_enable(false)
                .polygon_mode(vk::PolygonMode::FILL)
                .line_width(1.0)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::CLOCKWISE)
                .depth_bias_enable(false);
            let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(rctx.handles.sample_count)
                .sample_shading_enable(false);
            let mut depthstencil = make_pipe_depthstencil();
            depthstencil.depth_test_enable = vk::FALSE;
            depthstencil.depth_write_enable = vk::FALSE;
            let blendings = [vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::all())
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD)
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
            .expect("Could not create GUI pipeline")[0];

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

        let mut gui_frame_pool = Vec::with_capacity(INFLIGHT_FRAMES as usize);
        let mut gui_buffers = Vec::with_capacity(INFLIGHT_FRAMES as usize);

        for _ in 0..INFLIGHT_FRAMES {
            gui_frame_pool.push(GuiFrame::new());
            gui_buffers.push(Self::new_buffers(rctx, 16 * 1024));
        }

        Self {
            resources,
            texture_ds,
            texture_ds_layout,
            ds_pool,
            pipeline_layout,
            pipeline,
            gui_frame_pool,
            gui_buffers,
        }
    }

    fn new_buffers(rctx: &RenderingContext, num_squares: usize) -> (OwnedBuffer, OwnedBuffer) {
        let qfs = [rctx.handles.queues.get_primary_family()];
        let vbi = vk::BufferCreateInfo::builder()
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .size((num_squares * 4 * std::mem::size_of::<shaders::UiVertex>()) as u64)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&qfs)
            .build();
        let ibi = vk::BufferCreateInfo {
            usage: vk::BufferUsageFlags::INDEX_BUFFER,
            size: (num_squares * 5 * std::mem::size_of::<u32>()) as u64,
            ..vbi
        };
        let ai = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::CpuToGpu,
            flags: vma::AllocationCreateFlags::MAPPED,
            ..Default::default()
        };
        let mut vmalloc = rctx.handles.vmalloc.lock();
        let vb = OwnedBuffer::from(&mut vmalloc, &vbi, &ai);
        let ib = OwnedBuffer::from(&mut vmalloc, &ibi, &ai);
        (vb, ib)
    }

    pub fn destroy(mut self, handles: &RenderingHandles) {
        unsafe {
            handles
                .device
                .destroy_pipeline(self.pipeline, allocation_cbs());
            handles
                .device
                .destroy_pipeline_layout(self.pipeline_layout, allocation_cbs());
            handles
                .device
                .destroy_descriptor_pool(self.ds_pool, allocation_cbs());
            handles
                .device
                .destroy_descriptor_set_layout(self.texture_ds_layout, allocation_cbs());
            let mut vmalloc = handles.vmalloc.lock();
            for (mut b1, mut b2) in self.gui_buffers.drain(..) {
                b1.destroy(&mut vmalloc, handles);
                b2.destroy(&mut vmalloc, handles);
            }
        }
    }

    pub fn prepass_draw(&mut self, fctx: &mut PrePassFrameContext) {
        let frame = &mut self.gui_frame_pool[fctx.inflight_index];
        frame.reset();
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn inpass_draw(&mut self, fctx: &mut InPassFrameContext) {
        let frame = &self.gui_frame_pool[fctx.inflight_index];

        if frame.total_indices == 0 {
            return;
        }

        let (vbuf, ibuf) = &self.gui_buffers[fctx.inflight_index];

        let device = &fctx.rctx.handles.device;

        unsafe {
            device.cmd_bind_pipeline(fctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        }
        fctx.rctx
            .swapchain
            .dynamic_state
            .cmd_update_pipeline(device, fctx.cmd);
        unsafe {
            device.cmd_bind_descriptor_sets(
                fctx.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.texture_ds],
                &[],
            );
            device.cmd_bind_vertex_buffers(fctx.cmd, 0, &[vbuf.buffer], &[0]);
            device.cmd_bind_index_buffer(fctx.cmd, ibuf.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(fctx.cmd, frame.total_indices, 1, 0, 0, 0);
        }
    }
}

pub mod shaders {
    use crate::offset_of;
    use ash::vk;
    use std::mem;

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct UiVertex {
        pub position: [f32; 2],
        pub color: [f32; 4],
        pub texcoord: [f32; 3],
        pub texselect: i32,
    }

    impl UiVertex {
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
                    format: vk::Format::R32G32_SFLOAT,
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
                    offset: offset_of!(Self, texselect) as u32,
                },
            ];
            (bind_dsc, attr_dsc)
        }
    }
}
