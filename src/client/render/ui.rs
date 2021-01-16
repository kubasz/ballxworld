use crate::client::render::resources::RenderingResources;
use crate::client::render::ui::shaders::UiVertex;
use crate::client::render::vkhelpers::{make_pipe_depthstencil, OwnedBuffer, VulkanDeviceObject};
use crate::client::render::vulkan::{allocation_cbs, RenderingHandles, INFLIGHT_FRAMES};
use crate::client::render::{InPassFrameContext, PrePassFrameContext, RenderingContext};
use crate::config::Config;
use ash::version::DeviceV1_0;
use ash::vk;
use bxw_util::direction::*;
use bxw_util::math::*;
use bxw_util::*;
use bxw_world::TextureMapping;
use itertools::zip;
use std::borrow::Cow;
use std::ffi::CString;
use std::ops::{Add, Div, Mul, Neg, Sub};
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

pub mod theme {
    pub const SLOT_SIZE: f32 = 48.0;
    pub const SLOT_GAP: f32 = 10.0;
    pub const SLOT_INNER_MARGIN: f32 = 5.0;
}

const GUI_ATLAS_DIM: f32 = 128.0;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum GuiControlStyle {
    Window,
    Button,
    XButton,
    Cursor,
    Crosshair,
    Typing,
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
    /// A "9-box" render: left, left-mid, right-mid, right, top, top-mid, bot-mid, bottom, midpoint locs in pixels
    Box9([f32; 8], [i32; 4]),
}

impl GuiControlStyle {
    fn render_info(self) -> ControlStyleRenderInfo {
        // can't be const due to float arithmetic not allowed :(
        fn p(x: i32) -> f32 {
            (x as f32) / GUI_ATLAS_DIM
        }
        fn p4(a: i32, b: i32, c: i32, d: i32) -> ControlStyleRenderInfo {
            ControlStyleRenderInfo::LRTB([p(a), p(b), p(c), p(d)])
        }
        #[allow(clippy::many_single_char_names)]
        #[allow(clippy::too_many_arguments)]
        #[rustfmt::skip]
        fn p8(a: i32, b: i32, c: i32, d: i32,
              e: i32, f: i32, g: i32, h: i32) -> ControlStyleRenderInfo {
            ControlStyleRenderInfo::Box9(
                [p(a), p(b), p(c), p(d), p(e), p(f), p(g), p(h)],
                [b - a + 1, d - c + 1, f - e + 1, h - g + 1],
            )
        }
        match self {
            GuiControlStyle::FullBlack => p4(2, 6, 4, 8),
            GuiControlStyle::FullWhite => p4(6, 6, 8, 8),
            GuiControlStyle::FullDark => p4(2, 2, 4, 4),
            GuiControlStyle::FullBorder => p4(4, 2, 6, 4),
            GuiControlStyle::FullButtonBg => p4(10, 2, 12, 4),
            GuiControlStyle::FullWindowBg => p4(14, 2, 16, 4),
            GuiControlStyle::XButton => p4(0, 9, 33, 42),
            GuiControlStyle::Cursor => p4(10, 19, 34, 46),
            GuiControlStyle::Crosshair => p4(20, 35, 33, 48),
            GuiControlStyle::Typing => p4(36, 39, 33, 49),
            GuiControlStyle::Button => p8(17, 22, 44, 49, 0, 5, 27, 32),
            GuiControlStyle::Window => p8(50, 59, 73, 82, 0, 9, 23, 32),
        }
    }

    /// Dimensions in pixels of the texture associated with this style
    pub fn dimensions(self) -> (i32, i32) {
        let ri = self.render_info();
        fn p(x: f32) -> i32 {
            (x * GUI_ATLAS_DIM).round() as i32
        }
        match ri {
            ControlStyleRenderInfo::LRTB([l, r, t, b]) => (p(r - l), p(b - t)),
            ControlStyleRenderInfo::Box9(_, [lm, rm, tm, bm]) => (lm + rm + 2, tm + bm + 2),
        }
    }

    pub fn gui_rect_centered(self, center: GuiVec2) -> GuiRect {
        let (w, h) = self.dimensions();
        let hoff = GuiVec2(GuiCoord(0.0, (w / 2) as f32), GuiCoord(0.0, (h / 2) as f32));
        GuiRect {
            top_left: center - hoff,
            bottom_right: center + hoff,
        }
    }
}

/// A single gui coordinate with relative and absolute positioning parts
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiCoord(pub f32, pub f32);

impl GuiCoord {
    pub fn to_absolute_from_dim(self, dimension: u32) -> f32 {
        let dim = dimension as f32;
        let unaligned = self.0 + (self.1 / dim);
        let screen = (unaligned * dim).floor() / dim;
        screen * 2.0 - 1.0
    }
}

impl Add for GuiCoord {
    type Output = GuiCoord;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for GuiCoord {
    type Output = GuiCoord;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Neg for GuiCoord {
    type Output = GuiCoord;

    fn neg(self) -> Self::Output {
        Self(-self.0, -self.1)
    }
}

impl Mul<f32> for GuiCoord {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

impl Div<f32> for GuiCoord {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs)
    }
}

/// A gui 2D position/size vector
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiVec2(pub GuiCoord, pub GuiCoord);

impl Add for GuiVec2 {
    type Output = GuiVec2;

    fn add(self, rhs: Self) -> Self::Output {
        GuiVec2(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for GuiVec2 {
    type Output = GuiVec2;

    fn sub(self, rhs: Self) -> Self::Output {
        GuiVec2(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Neg for GuiVec2 {
    type Output = GuiVec2;

    fn neg(self) -> Self::Output {
        GuiVec2(-self.0, -self.1)
    }
}

impl Mul<f32> for GuiVec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs, self.1 * rhs)
    }
}

impl Div<f32> for GuiVec2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self(self.0 / rhs, self.1 / rhs)
    }
}

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

pub fn gv2a(x: f32, y: f32) -> GuiVec2 {
    GuiVec2(GuiCoord(0.0, x), GuiCoord(0.0, y))
}

pub fn gv2r(x: f32, y: f32) -> GuiVec2 {
    GuiVec2(GuiCoord(0.0, x), GuiCoord(0.0, y))
}

/// A gui 2D position/size vector
#[derive(Copy, Clone, Default, Debug, PartialEq)]
pub struct GuiRect {
    pub top_left: GuiVec2,
    pub bottom_right: GuiVec2,
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

impl GuiColor {
    fn rgbmul(self, factor: f32) -> Self {
        let [r, g, b, a] = self.0;
        Self([r * factor, g * factor, b * factor, a])
    }
}

pub const GUI_WHITE: GuiColor = GuiColor([1.0, 1.0, 1.0, 1.0]);
pub const GUI_RED: GuiColor = GuiColor([0.9, 0.1, 0.1, 1.0]);
pub const GUI_BLACK: GuiColor = GuiColor([0.0, 0.0, 0.0, 1.0]);

#[derive(Debug, Clone)]
pub enum GuiCmd {
    Rectangle {
        style: GuiControlStyle,
        rect: GuiRect,
    },
    FreeText {
        text: Cow<'static, str>,
        scale: f32,
        start_at: GuiVec2,
    },
    VoxelPreview {
        texture: TextureMapping<u32>,
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

    #[allow(clippy::too_many_arguments)]
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

    #[allow(clippy::too_many_arguments)]
    fn put_quad(
        &mut self,
        verts: [Vector2<f32>; 4],
        texture: [Vector2<f32>; 4],
        texture_z: f32,
        texselect: i32,
        color: [f32; 4],
    ) {
        let idx = self.verts.len() as u32;
        // top-left
        for i in 0..4 {
            self.verts.push(UiVertex {
                position: [verts[i].x, verts[i].y],
                color,
                texcoord: [texture[i].x, texture[i].y, texture_z],
                texselect,
            });
        }
        self.indxs
            .extend_from_slice(&[idx, idx + 1, idx + 2, idx + 3, u32::max_value()]);
    }
}

const TEXSELECT_GUI: i32 = 0;
const TEXSELECT_FONT: i32 = 1;
const TEXSELECT_VOX: i32 = 2;

impl GuiOrderedCmd {
    fn handle(&self, writer: &mut GuiVtxWriter, res: &RenderingResources, rctx: &RenderingContext) {
        let color = self.color.0;
        let (sw, sh) = (
            rctx.swapchain.swapimage_size.width as f32,
            rctx.swapchain.swapimage_size.height as f32,
        );
        match &self.cmd {
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
                    ControlStyleRenderInfo::Box9(tc, ic) => {
                        let dim = rctx.swapchain.swapimage_size;
                        let dim = [2.0 / dim.width as f32, 2.0 / dim.height as f32];
                        let ic = [
                            ic[0] as f32 * dim[0],
                            ic[1] as f32 * dim[0],
                            ic[2] as f32 * dim[1],
                            ic[3] as f32 * dim[1],
                        ];
                        let idx = writer.verts.len() as u32;
                        for (ty, iy) in zip(
                            &[tc[4], tc[5], tc[6], tc[7]],
                            &[
                                absrect.0.y,
                                absrect.0.y + ic[2],
                                absrect.1.y - ic[3],
                                absrect.1.y,
                            ],
                        ) {
                            for (tx, ix) in zip(
                                &[tc[0], tc[1], tc[2], tc[3]],
                                &[
                                    absrect.0.x,
                                    absrect.0.x + ic[0],
                                    absrect.1.x - ic[1],
                                    absrect.1.x,
                                ],
                            ) {
                                writer.verts.push(UiVertex {
                                    position: [*ix, *iy],
                                    color,
                                    texcoord: [*tx, *ty, 0.0],
                                    texselect: TEXSELECT_GUI,
                                });
                            }
                        }
                        #[rustfmt::skip]
                            writer.indxs.extend(
                            [0, 4, 1, 5, 2, 6, 3, 7, u32::max_value(),
                                4, 8, 5, 9, 6, 10, 7, 11, u32::max_value(),
                                8, 12, 9, 13, 10, 14, 11, 15, u32::max_value(),
                            ]
                                .iter()
                                .map(|x| x.saturating_add(idx)),
                        );
                    }
                }
            }
            GuiCmd::FreeText {
                text,
                scale,
                start_at,
            } => {
                let start = start_at.to_absolute_from_rctx(rctx);
                res.font.text_op(text, *scale, |p| {
                    writer.put_rect(
                        start + vec2(2.0 * p.x / sw, 2.0 * p.y / sh),
                        start + vec2(2.0 * (p.x + p.w) / sw, 2.0 * (p.y + p.h) / sh),
                        vec2(p.bmchar.x, p.bmchar.y),
                        vec2(p.bmchar.rightx, p.bmchar.bottomy),
                        0.0,
                        TEXSELECT_FONT,
                        color,
                    );
                });
            }
            GuiCmd::VoxelPreview { texture, rect } => {
                let (tl, br) = rect.to_absolute_from_rctx(rctx);
                let w = br.x - tl.x;
                let h = br.y - tl.y;
                let v_top = tl + vec2(w * 0.6, 0.0);
                let v_tl = tl + vec2(0.0, h * 0.1);
                let v_mid = tl + vec2(w * 0.3, h * 0.3);
                let v_tr = tl + vec2(w, h * 0.15);
                let v_bl = tl + vec2(0.0, h * 0.65);
                let v_bot = tl + vec2(w * 0.3, h);
                let v_br = tl + vec2(w, h * 0.8);
                // top
                writer.put_quad(
                    [v_tl, v_top, v_mid, v_tr],
                    [
                        vec2(0.0, 0.0),
                        vec2(1.0, 0.0),
                        vec2(0.0, 1.0),
                        vec2(1.0, 1.0),
                    ],
                    *texture.at_direction(DIR_UP) as f32,
                    TEXSELECT_VOX,
                    self.color.rgbmul(0.9).0,
                );
                // left
                writer.put_quad(
                    [v_tl, v_mid, v_bl, v_bot],
                    [
                        vec2(0.0, 0.0),
                        vec2(1.0, 0.0),
                        vec2(0.0, 1.0),
                        vec2(1.0, 1.0),
                    ],
                    *texture.at_direction(DIR_LEFT) as f32,
                    TEXSELECT_VOX,
                    self.color.rgbmul(0.8).0,
                );
                // front/right
                writer.put_quad(
                    [v_mid, v_tr, v_bot, v_br],
                    [
                        vec2(0.0, 0.0),
                        vec2(1.0, 0.0),
                        vec2(0.0, 1.0),
                        vec2(1.0, 1.0),
                    ],
                    *texture.at_direction(DIR_FRONT) as f32,
                    TEXSELECT_VOX,
                    color,
                );
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
            .sort_by(|a, b| Ord::cmp(&a.z_index, &b.z_index));
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
            cmd.handle(&mut self.gui_vtx_write, &self.resources, fctx.rctx);
        }
        if self.gui_vtx_write.indxs.is_empty() {
            return;
        }

        let (vbuf, ibuf) = &self.gui_buffers[fctx.inflight_index];
        {
            let (_va, vai) = vbuf.allocation.as_ref().unwrap();
            let (_ia, iai) = ibuf.allocation.as_ref().unwrap();
            let vmap = vai.get_mapped_data() as *mut shaders::UiVertex;
            let imap = iai.get_mapped_data() as *mut u32;
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
                let ranges = [
                    vk::MappedMemoryRange::builder()
                        .size(vai.get_size() as u64)
                        .offset(vai.get_offset() as u64)
                        .memory(vai.get_device_memory())
                        .build(),
                    vk::MappedMemoryRange::builder()
                        .size(iai.get_size() as u64)
                        .offset(iai.get_offset() as u64)
                        .memory(iai.get_device_memory())
                        .build(),
                ];
                fctx.rctx
                    .handles
                    .device
                    .flush_mapped_memory_ranges(&ranges)
                    .unwrap();
            }
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
