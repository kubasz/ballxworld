use crate::client::config::Config;
use crate::client::render::vkhelpers::*;
use crate::client::render::vulkan::{allocation_cbs, RenderingHandles};
use crate::client::render::RenderingContext;
use crate::math::*;
use ash::version::DeviceV1_0;
use ash::vk;
use fnv::FnvHashMap;
use image::RgbaImage;
use regex::{Match, Regex};
use std::path::{Path, PathBuf};
use vk_mem as vma;

pub struct RenderingResources {
    pub gui_atlas: OwnedImage,
    pub font_atlas: OwnedImage,
    pub gui_sampler: vk::Sampler,
    pub font_sampler: vk::Sampler,
    pub font: Box<BMFont>,
    pub voxel_texture_array: OwnedImage,
    pub voxel_texture_array_params: TextureArrayParams,
    pub voxel_texture_name_map: FnvHashMap<String, u32>,
    pub voxel_texture_sampler: vk::Sampler,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct BMChar {
    pub x: f32,
    pub y: f32,
    pub rightx: f32,
    pub bottomy: f32,
    pub width: i32,
    pub height: i32,
    pub xoffset: i32,
    pub yoffset: i32,
    pub xadvance: i32,
}

#[derive(Clone, Default)]
pub struct BMFont {
    pub line_height: i32,
    pub base: i32,
    pub scale_w: i32,
    pub scale_h: i32,
    pub low_present: Vec<bool>,
    pub low_chars: Vec<BMChar>,
    pub high_chars: FnvHashMap<u32, BMChar>,
}

pub struct TextureArrayParams {
    pub dims: (u32, u32),
    pub mip_levels: u32,
}

impl RenderingResources {
    pub fn load(cfg: &Config, rctx: &mut RenderingContext) -> Self {
        let (voxel_texture_array, voxel_texture_array_params, voxel_texture_name_map) =
            Self::new_texture_atlas(cfg, rctx);
        let gui_atlas = load_owned_image("res/ui.png", rctx);
        gui_atlas.give_name(&rctx.handles, || "gui atlas");
        let font_atlas = load_owned_image("res/fonts/cascadia_0.png", rctx);
        font_atlas.give_name(&rctx.handles, || "font atlas");
        let voxel_texture_sampler = {
            let sci = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                .min_lod(0.0)
                .max_lod(voxel_texture_array_params.mip_levels as f32)
                .mip_lod_bias(0.0)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .anisotropy_enable(true)
                .max_anisotropy(4.0);
            unsafe { rctx.handles.device.create_sampler(&sci, allocation_cbs()) }
                .expect("Could not create voxel texture sampler")
        };
        let gui_sampler = {
            let sci = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::NEAREST)
                .mag_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .min_lod(0.0)
                .max_lod(0.0)
                .mip_lod_bias(0.0)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .anisotropy_enable(false)
                .max_anisotropy(0.0);
            unsafe { rctx.handles.device.create_sampler(&sci, allocation_cbs()) }
                .expect("Could not create voxel texture sampler")
        };
        let font_sampler = {
            let sci = vk::SamplerCreateInfo::builder()
                .min_filter(vk::Filter::LINEAR)
                .mag_filter(vk::Filter::LINEAR)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .min_lod(0.0)
                .max_lod(0.0)
                .mip_lod_bias(0.0)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .unnormalized_coordinates(false)
                .anisotropy_enable(false)
                .max_anisotropy(0.0);
            unsafe { rctx.handles.device.create_sampler(&sci, allocation_cbs()) }
                .expect("Could not create voxel texture sampler")
        };
        let font_desc = std::fs::read_to_string("res/fonts/cascadia.fnt")
            .expect("Couldn't read res/fonts/cascadia.fnt");
        let font = BMFont::parse(&font_desc);
        Self {
            gui_atlas,
            font_atlas,
            gui_sampler,
            font_sampler,
            font,
            voxel_texture_array,
            voxel_texture_array_params,
            voxel_texture_name_map,
            voxel_texture_sampler,
        }
    }

    pub fn destroy(mut self, handles: &RenderingHandles) {
        unsafe {
            handles
                .device
                .destroy_sampler(self.voxel_texture_sampler, allocation_cbs());
            handles
                .device
                .destroy_sampler(self.font_sampler, allocation_cbs());
            handles
                .device
                .destroy_sampler(self.gui_sampler, allocation_cbs());
        }
        self.voxel_texture_name_map.clear();
        let mut vmalloc = handles.vmalloc.lock();
        self.voxel_texture_array.destroy(&mut vmalloc, handles);
        self.gui_atlas.destroy(&mut vmalloc, handles);
        self.font_atlas.destroy(&mut vmalloc, handles);
    }

    fn new_texture_atlas(
        _cfg: &Config,
        rctx: &mut RenderingContext,
    ) -> (OwnedImage, TextureArrayParams, FnvHashMap<String, u32>) {
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
        let mut names = FnvHashMap::default();
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
            let img = load_rgba(&path);
            names.insert(nm.to_owned(), memimages.len() as u32);
            let dim1 = img.dimensions();
            idim.0 = idim.0.max(dim1.0);
            idim.1 = idim.1.max(dim1.1);
            memimages.push(img);
        }
        let numimages = memimages.len() as u32;

        // resize all images to max size and put the pixels into a buffer
        let mut rawdata = Vec::new();
        for mut img in memimages.into_iter() {
            if img.dimensions() != idim {
                img = image::imageops::resize(&img, idim.0, idim.1, image::imageops::Nearest);
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

            OwnedImage::from(
                &mut vmalloc,
                &rctx.handles,
                &ici,
                &aci,
                vk::ImageViewType::TYPE_2D_ARRAY,
                vk::ImageAspectFlags::COLOR,
            )
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
            let mut buf = OwnedBuffer::new_single_upload(&mut vmalloc, &rawdata);
            //
            let cmd = OnetimeCmdGuard::new(&rctx.handles, None);
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
            buf.destroy(&mut vmalloc, &rctx.handles);
        }

        img.give_name(&rctx.handles, || "voxel texture array");

        (
            img,
            TextureArrayParams {
                dims: idim,
                mip_levels: mip_lvls,
            },
            names,
        )
    }
}

fn load_rgba<P: AsRef<Path>>(path: P) -> RgbaImage {
    let path: &Path = path.as_ref();
    if !(path.exists() && path.is_file()) {
        panic!("Could not find texture file: `{:?}`", &path);
    }
    let img = image::open(&path)
        .unwrap_or_else(|e| panic!("Could not load image from {:?}: {}", path, e));
    img.to_rgba()
}

fn load_owned_image<P: AsRef<Path>>(path: P, rctx: &RenderingContext) -> OwnedImage {
    let rgba = load_rgba(path);
    rgba_to_owned_image(&rgba, rctx)
}

fn rgba_to_owned_image(rgba: &RgbaImage, rctx: &RenderingContext) -> OwnedImage {
    let mut vmalloc = rctx.handles.vmalloc.lock();
    let qfis = [rctx.handles.queues.get_primary_family()];
    let img_extent = vk::Extent3D {
        width: rgba.dimensions().0,
        height: rgba.dimensions().1,
        depth: 1,
    };
    let img = {
        let ici = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .extent(img_extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&qfis)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let aci = vma::AllocationCreateInfo {
            usage: vma::MemoryUsage::GpuOnly,
            flags: vma::AllocationCreateFlags::DEDICATED_MEMORY,
            ..Default::default()
        };

        OwnedImage::from(
            &mut vmalloc,
            &rctx.handles,
            &ici,
            &aci,
            vk::ImageViewType::TYPE_2D,
            vk::ImageAspectFlags::COLOR,
        )
    };
    // upload main level
    let whole_img = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let queue = rctx.handles.queues.lock_primary_queue();
    let mut buf = OwnedBuffer::new_single_upload(&mut vmalloc, rgba.as_ref());
    //
    let cmd = OnetimeCmdGuard::new(&rctx.handles, None);
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
        layer_count: 1,
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
        next_accesses: &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        previous_layout: vk_sync::ImageLayout::Optimal,
        next_layout: vk_sync::ImageLayout::Optimal,
        discard_contents: false,
        src_queue_family_index: qfis[0],
        dst_queue_family_index: qfis[0],
        image: img.image,
        range: whole_img,
    };
    vk_sync::cmd::pipeline_barrier(
        rctx.handles.device.fp_v1_0(),
        cmd.handle(),
        None,
        &[],
        &[ibarrier.clone()],
    );
    cmd.execute(&queue);
    buf.destroy(&mut vmalloc, &rctx.handles);
    img
}

#[derive(Copy, Clone)]
pub struct TextOpParams<'f> {
    pub bmchar: &'f BMChar,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct TextMeasurement {
    pub minx: f32,
    pub miny: f32,
    pub maxx: f32,
    pub maxy: f32,
}

impl TextMeasurement {
    pub fn min(&self) -> Vector2<f32> {
        vec2(self.minx, self.miny)
    }

    pub fn max(&self) -> Vector2<f32> {
        vec2(self.maxx, self.maxy)
    }
}

impl BMFont {
    pub fn parse(desc: &str) -> Box<Self> {
        let common_re = Regex::new(r#"^common lineHeight=(?P<lh>\d+) base=(?P<b>\d+) scaleW=(?P<sw>\d+) scaleH=(?P<sh>\d+)"#).unwrap();
        let char_re = Regex::new(r#"^char\s+id=(?P<id>[0-9-]+)\s+x=(?P<x>[0-9-]+)\s+y=(?P<y>[0-9-]+)\s+width=(?P<w>[0-9-]+)\s+height=(?P<h>[0-9-]+)\s+xoffset=(?P<xo>[0-9-]+)\s+yoffset=(?P<yo>[0-9-]+)\s+xadvance=(?P<xa>[0-9-]+)"#).unwrap();
        let mut font: Box<BMFont> = Box::default();
        font.low_chars.resize_with(256, Default::default);
        font.low_present.resize(256, false);
        let mi32 = |m: Option<Match>| -> i32 { m.unwrap().as_str().parse().unwrap() };
        let mut scale: Option<(f32, f32)> = None;
        for line in desc.lines() {
            let line = line.trim();
            if line.starts_with("common ") {
                let m = common_re.captures(line).expect("Invalid font common");
                font.line_height = mi32(m.name("lh"));
                font.base = mi32(m.name("b"));
                font.scale_w = mi32(m.name("sw"));
                font.scale_h = mi32(m.name("sh"));
                scale = Some((1.0 / font.scale_w as f32, 1.0 / font.scale_h as f32));
            } else if line.starts_with("char id") {
                let capt = char_re.captures(line).expect("Invalid font char");
                let id = mi32(capt.name("id")).max(0) as u32;
                let x = mi32(capt.name("x"));
                let y = mi32(capt.name("y"));
                let w = mi32(capt.name("w"));
                let h = mi32(capt.name("h"));
                let xoff = mi32(capt.name("xo"));
                let yoff = mi32(capt.name("yo"));
                let xadv = mi32(capt.name("xa"));
                let chr = BMChar {
                    x: x as f32 * scale.unwrap().0,
                    y: y as f32 * scale.unwrap().0,
                    rightx: (x + w) as f32 * scale.unwrap().0,
                    bottomy: (y + h) as f32 * scale.unwrap().0,
                    width: w,
                    height: h,
                    xoffset: xoff,
                    yoffset: yoff,
                    xadvance: xadv,
                };
                if id < 256 {
                    font.low_present[id as usize] = true;
                    font.low_chars[id as usize] = chr;
                } else {
                    font.high_chars.insert(id, chr);
                }
            }
        }
        assert_ne!(font.line_height, 0);
        font
    }

    pub fn get_char(&self, chr: char) -> &BMChar {
        let cid = chr as u32;
        if cid < 256 {
            if self.low_present[cid as usize] {
                &self.low_chars[cid as usize]
            } else {
                &self.low_chars[0]
            }
        } else {
            self.high_chars.get(&cid).unwrap_or(&self.low_chars[0])
        }
    }

    pub fn text_op<'f, F: FnMut(TextOpParams<'f>) -> ()>(
        &'f self,
        text: &str,
        scale: f32,
        mut op: F,
    ) {
        let s = |x: i32| x as f32 * scale;
        let mut x = 0.0f32;
        let mut y = 0.0f32;
        for c in text.chars() {
            if c == '\r' {
                continue;
            }
            if c == '\n' {
                y += s(self.line_height);
                continue;
            }
            let bmchar = self.get_char(c);
            let cx = x + s(bmchar.xoffset);
            let cy = y + s(bmchar.yoffset);
            let cw = s(bmchar.width);
            let ch = s(bmchar.height);
            op(TextOpParams {
                bmchar,
                x: cx,
                y: cy,
                w: cw,
                h: ch,
            });
            x += s(bmchar.xadvance);
        }
    }

    pub fn measure_text(&self, text: &str, scale: f32) -> TextMeasurement {
        let mut m: TextMeasurement = Default::default();
        m.maxy = self.line_height as f32 * scale;
        m.maxx = scale;
        self.text_op(text, scale, |p| {
            m.minx = m.minx.min(p.x);
            m.miny = m.miny.min(p.y);
            m.maxx = m.maxx.max(p.x + p.w);
            m.maxy = m.maxy.max(p.y + p.h);
        });
        m
    }
}
