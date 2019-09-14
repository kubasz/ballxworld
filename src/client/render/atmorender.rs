use crate::client::config::Config;
use crate::client::render::vkhelpers::{cmd_push_struct_constants, make_pipe_depthstencil};
use crate::client::render::vulkan::{allocation_cbs, RenderingHandles};
use crate::client::render::{InPassFrameContext, RenderingContext};
use crate::math::*;
use ash::version::DeviceV1_0;
use ash::vk;
use std::ffi::CString;

pub mod shaders {
    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct SkyUBO {
        pub viewproj: [[f32; 4]; 4],
    }
}

pub struct AtmosphereRenderer {
    pub sky_pipeline_layout: vk::PipelineLayout,
    pub sky_pipeline: vk::Pipeline,
}

impl AtmosphereRenderer {
    pub fn new(_cfg: &Config, rctx: &mut RenderingContext) -> Self {
        let pipeline_layot = {
            let pc = [vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: 4 * 4 * 4,
            }];
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
        let sky_pipeline = {
            let vs = rctx
                .handles
                .load_shader_module("res/shaders/atmosphere.vert.spv")
                .expect("Couldn't load vertex atmosphere shader");
            let fs = rctx
                .handles
                .load_shader_module("res/shaders/atmosphere.frag.spv")
                .expect("Couldn't load fragment atmosphere shader");
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
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
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
            sky_pipeline_layout: pipeline_layot,
            sky_pipeline,
        }
    }

    pub fn destroy(&self, handles: &RenderingHandles) {
        unsafe {
            handles
                .device
                .destroy_pipeline(self.sky_pipeline, allocation_cbs());
            handles
                .device
                .destroy_pipeline_layout(self.sky_pipeline_layout, allocation_cbs());
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
