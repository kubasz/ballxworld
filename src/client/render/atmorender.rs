use crate::client::config::Config;
use crate::client::render::{InPassFrameContext, RenderingContext};
use crate::math::*;
use std::sync::Arc;
use ash::prelude::*;
use ash::vk;

pub mod shaders {
    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct SkyUBO {
        pub view: [[f32; 4]; 4],
        pub proj: [[f32; 4]; 4],
    }
}

pub struct AtmosphereRenderer {
    pub sky_pipeline: vk::Pipeline,
    //pub ubuffers: CpuBufferPool<shaders::SkyUBO>,
}

impl AtmosphereRenderer {
    pub fn new(_cfg: &Config, rctx: &mut RenderingContext) -> Self {
        let sky_pipeline: Arc<SkyPipeline> = {
            let vs =
                shaders::vs::Shader::load(rctx.device.clone()).expect("Failed to create VS module");
            let fs =
                shaders::fs::Shader::load(rctx.device.clone()).expect("Failed to create FS module");

            let mut ds = DepthStencil::simple_depth_test();
            ds.depth_write = false;
            ds.depth_compare = Compare::LessOrEqual;
            Arc::new(
                GraphicsPipeline::start()
                    .cull_mode_front()
                    .vertex_shader(vs.main_entry_point(), ())
                    .vertex_input(BufferlessDefinition {})
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .blend_alpha_blending()
                    .depth_stencil(ds)
                    .render_pass(Subpass::from(rctx.mainpass.clone(), 0).unwrap())
                    .build(rctx.device.clone())
                    .expect("Could not create voxel graphics pipeline"),
            )
        };

        let ubuffers = CpuBufferPool::new(rctx.device.clone(), BufferUsage::uniform_buffer());

        Self {
            sky_pipeline,
            ubuffers,
        }
    }

    pub fn inpass_draw(
        &mut self,
        fctx: &mut InPassFrameContext,
        mview: Matrix4<f32>,
        mproj: Matrix4<f32>,
    ) {
        let ubo = {
            let mut ubo = shaders::SkyUBO::default();
            ubo.proj = mproj.into();
            ubo.view = mview.into();
            self.ubuffers.next(ubo).unwrap()
        };
        let mut cmd = fctx.replace_cmd();

        let udset = Arc::new(
            PersistentDescriptorSet::start(self.sky_pipeline.clone(), 0)
                .add_buffer(ubo.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        let vsrc = vulkano::pipeline::vertex::BufferlessVertices {
            vertices: 36,
            instances: 1,
        };

        cmd = cmd
            .draw(
                self.sky_pipeline.clone(),
                &fctx.rctx.dynamic_state,
                vsrc,
                udset.clone(),
                (),
            )
            .expect("Couldn't draw skybox");

        fctx.cmd = Some(cmd);
    }
}
