use crate::client::config::Config;
use crate::client::render::vkhelpers::{cmd_push_struct_constants, make_pipe_depthstencil, OwnedImage};
use crate::client::render::vulkan::{allocation_cbs, RenderingHandles};
use crate::client::render::{InPassFrameContext, RenderingContext};
use crate::math::*;
use ash::version::DeviceV1_0;
use ash::vk;
use std::ffi::CString;
use crate::client::render::resources::RenderingResources;
use std::sync::Arc;

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

pub struct GuiRenderer {
    resources: Arc<RenderingResources>,
    texture_ds: vk::DescriptorSet,
    texture_ds_layout: vk::DescriptorSetLayout,
    ds_pool: vk::DescriptorPool,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl GuiRenderer {
    pub fn new(_cfg: &Config, rctx: &mut RenderingContext, resources: Arc<RenderingResources>) -> Self {
        let pipeline_layot = {
            let pc = [];
            let dsls = [];
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
        let gui_pipeline = {
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
                .expect("Could not create atmosphere pipeline")[0];

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
            resources,
            texture_ds: Default::default(),
            texture_ds_layout: Default::default(),
            ds_pool: Default::default(),
            pipeline_layout,
            pipeline
        }
    }

    pub fn destroy(mut self, handles: &RenderingHandles) {
        unsafe {
            handles
                .device
                .destroy_pipeline(self.pipeline, allocation_cbs());
            handles
                .device
                .destroy_pipeline_layout(self.pipeline_layout, allocation_cbs());
            handles.device.destroy_descriptor_pool(self.ds_pool, allocation_cbs());
            handles.device.destroy_descriptor_set_layout(self.texture_ds_layout, allocation_cbs());
        }
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn inpass_draw(
        &mut self,
        fctx: &mut InPassFrameContext,
        mview: Matrix4<f32>,
        mproj: Matrix4<f32>,
    ) {
        let ubo = {
            let mut ubo = shaders::SkyUBO::default();
            let proj = mproj;
            let mut view = mview;
            view.set_column(3, &vec4(0.0, 0.0, 0.0, 1.0));
            ubo.viewproj = (proj * view).into();
            ubo
        };

        let device = &fctx.rctx.handles.device;

        unsafe {
            device.cmd_bind_pipeline(fctx.cmd, vk::PipelineBindPoint::GRAPHICS, self.sky_pipeline);
        }
        fctx.rctx
            .swapchain
            .dynamic_state
            .cmd_update_pipeline(device, fctx.cmd);

        cmd_push_struct_constants(
            device,
            fctx.cmd,
            self.sky_pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            &ubo,
        );
        unsafe {
            device.cmd_draw(fctx.cmd, 36, 1, 0, 0);
        }
    }
}

