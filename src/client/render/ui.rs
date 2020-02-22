use crate::client::config::Config;
use crate::client::render::resources::RenderingResources;
use crate::client::render::ui::shaders::UiVertex;
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

#[derive(Clone, Debug)]
enum ControlStyleRenderInfo {
    /// A simple left,right,top,bottom textured rectangle
    LRTB([f32; 4]),
}

impl GuiControlStyle {
    fn render_info(self) -> ControlStyleRenderInfo {
        let p = |x| (x as f32) / 128.0;
        match self {
            _ => ControlStyleRenderInfo::LRTB([p(48), p(80), p(0), p(32)]),
        }
    }
}

/// A single gui coordinate with relative and absolute positioning parts
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiCoord(f32, f32);

impl GuiCoord {
    pub fn to_absolute_from_dim(self, dimension: u32) -> f32 {
        let dim = dimension as f32;
        let unaligned = self.0 + (self.1 / dim);
        let screen = (unaligned * dim).floor() / dim;
        screen * 2.0 - 1.0
    }
}

/// A gui 2D position/size vector
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiVec2(GuiCoord, GuiCoord);

impl GuiVec2 {
    pub fn to_absolute_from_dim(self, dimensions: (u32, u32)) -> Vector2<f32> {
        vec2(
            self.0.to_absolute_from_dim(dimensions.0),
            self.1.to_absolute_from_dim(dimensions.1),
        )
    }

    pub fn to_absolute_from_rctx(self, rctx: &RenderingContext) -> Vector2<f32> {
        let dim = rctx.swapchain.swapimage_size;
        self.to_absolute_from_dim((dim.width, dim.height))
    }
}

pub fn gv2(x: (f32, f32), y: (f32, f32)) -> GuiVec2 {
    GuiVec2(GuiCoord(x.0, x.1), GuiCoord(y.0, y.1))
}

/// A gui 2D position/size vector
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiRect {
    top_left: GuiVec2,
    bottom_right: GuiVec2,
}

impl GuiRect {
    pub fn from_xywh(x: (f32, f32), y: (f32, f32), w: (f32, f32), h: (f32, f32)) -> Self {
        Self {
            top_left: gv2(x, y),
            bottom_right: gv2((x.0 + w.0, x.1 + w.1), (y.0 + h.0, y.1 + h.1)),
        }
    }

    pub fn to_absolute_from_dim(self, dimensions: (u32, u32)) -> (Vector2<f32>, Vector2<f32>) {
        (
            self.top_left.to_absolute_from_dim(dimensions),
            self.bottom_right.to_absolute_from_dim(dimensions),
        )
    }

    pub fn to_absolute_from_rctx(self, rctx: &RenderingContext) -> (Vector2<f32>, Vector2<f32>) {
        (
            self.top_left.to_absolute_from_rctx(rctx),
            self.bottom_right.to_absolute_from_rctx(rctx),
        )
    }
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
}

#[derive(Default)]
struct GuiVtxWriter {
    verts: Vec<shaders::UiVertex>,
    indxs: Vec<u32>,
}

impl GuiVtxWriter {
    fn reset(&mut self) {
        self.verts.clear();
        self.indxs.clear();
    }

    fn put_rect(
        &mut self,
        top_left: Vector2<f32>,
        bottom_right: Vector2<f32>,
        texture_tl: Vector2<f32>,
        texture_br: Vector2<f32>,
        texture_z: f32,
        texselect: i32,
        color: [f32; 4],
    ) {
        let idx = self.verts.len() as u32;
        // top-left
        self.verts.push(UiVertex {
            position: [top_left.x, top_left.y],
            color,
            texcoord: [texture_tl.x, texture_tl.y, texture_z],
            texselect,
        });
        // bottom-left
        self.verts.push(UiVertex {
            position: [top_left.x, bottom_right.y],
            color,
            texcoord: [texture_tl.x, texture_br.y, texture_z],
            texselect,
        });
        // top-right
        self.verts.push(UiVertex {
            position: [bottom_right.x, top_left.y],
            color,
            texcoord: [texture_br.x, texture_tl.y, texture_z],
            texselect,
        });
        // bottom-right
        self.verts.push(UiVertex {
            position: [bottom_right.x, bottom_right.y],
            color,
            texcoord: [texture_br.x, texture_br.y, texture_z],
            texselect,
        });
        self.indxs
            .extend_from_slice(&[idx, idx + 1, idx + 2, idx + 3, u32::max_value()]);
    }
}

const TEXSELECT_GUI: i32 = 0;
const TEXSELECT_FONT: i32 = 1;
const TEXSELECT_VOX: i32 = 2;

impl GuiOrderedCmd {
    fn handle(&self, writer: &mut GuiVtxWriter, rctx: &RenderingContext) {
        let color = self.color.0;
        match self.cmd {
            GuiCmd::Rectangle { style, rect } => {
                let absrect = rect.to_absolute_from_rctx(rctx);
                match style.render_info() {
                    ControlStyleRenderInfo::LRTB(lrtb) => {
                        writer.put_rect(
                            absrect.0,
                            absrect.1,
                            vec2(lrtb[0], lrtb[2]),
                            vec2(lrtb[1], lrtb[3]),
                            0.0,
                            TEXSELECT_GUI,
                            color,
                        );
                    }
                }
            }
        }
    }
}

impl GuiFrame {
    pub fn new() -> GuiFrame {
        Default::default()
    }

    pub fn reset(&mut self) {
        self.cmd_list.clear();
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
    gui_vtx_write: GuiVtxWriter,
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
                resources.font_sampler,
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
            let vox_desc = shaders::UiVertex::description();
            let vtxinp = vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&vox_desc.0)
                .vertex_attribute_descriptions(&vox_desc.1);
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
            gui_vtx_write: Default::default(),
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
            flags: vma::AllocationCreateFlags::NONE,
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

    pub fn prepass_draw(&mut self, fctx: &mut PrePassFrameContext) -> &mut GuiFrame {
        let frame = &mut self.gui_frame_pool[fctx.inflight_index];
        frame.reset();
        frame
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn inpass_draw(&mut self, fctx: &mut InPassFrameContext) {
        let frame = &mut self.gui_frame_pool[fctx.inflight_index];

        self.gui_vtx_write.reset();
        frame.sort();
        for cmd in frame.cmd_list.iter() {
            cmd.handle(&mut self.gui_vtx_write, fctx.rctx);
        }
        if self.gui_vtx_write.indxs.is_empty() {
            return;
        }

        let (vbuf, ibuf) = &self.gui_buffers[fctx.inflight_index];
        {
            let vmalloc = fctx.rctx.handles.vmalloc.lock();
            let (va, _vai) = vbuf.allocation.as_ref().unwrap();
            let (ia, _iai) = ibuf.allocation.as_ref().unwrap();
            let vmap = vmalloc.map_memory(va).unwrap() as *mut shaders::UiVertex;
            let imap = vmalloc.map_memory(ia).unwrap() as *mut u32;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.gui_vtx_write.verts.as_ptr(),
                    vmap,
                    self.gui_vtx_write.verts.len(),
                );
                std::ptr::copy_nonoverlapping(
                    self.gui_vtx_write.indxs.as_ptr(),
                    imap,
                    self.gui_vtx_write.indxs.len(),
                );
            }
            vmalloc.unmap_memory(va).unwrap();
            vmalloc.unmap_memory(ia).unwrap();
        }

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
            device.cmd_draw_indexed(fctx.cmd, self.gui_vtx_write.indxs.len() as u32, 1, 0, 0, 0);
        }
    }
}

pub mod shaders {
    use crate::offset_of;
    use ash::vk;
    use std::mem;

    #[derive(Copy, Clone, Debug, Default)]
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
