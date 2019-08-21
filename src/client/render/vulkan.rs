use std::ffi::CString;
use std::marker::PhantomData;
use std::sync::Arc;

use sdl2::video::Window;

use crate::client::config::Config;

use std::cmp::max;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{
    AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainAcquireFuture,
    SwapchainCreationError,
};
use vulkano::sync::{FenceSignalFuture, FlushError, GpuFuture, NowFuture};
use vulkano::{app_info_from_cargo_toml, single_pass_renderpass};

pub enum Queues {
    /// Combined Graphics+Transfer+Compute queue
    Combined(Arc<Queue>),
}

pub trait WaitableFuture: GpuFuture {
    fn wait_or_noop(&mut self);
}

impl<F> WaitableFuture for FenceSignalFuture<F>
where
    F: GpuFuture,
{
    fn wait_or_noop(&mut self) {
        self.wait(None).unwrap();
    }
}

impl WaitableFuture for NowFuture {
    fn wait_or_noop(&mut self) {
        // noop
    }
}

pub struct RenderingContext {
    // basic handles
    pub window: Window,
    pub instance_extensions: InstanceExtensions,
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface<()>>,
    pub device: Arc<Device>,
    pub queues: Queues,
    // swapchain
    pub swapchain: Arc<Swapchain<()>>,
    pub swapimages: Vec<Arc<SwapchainImage<()>>>,
    previous_frame_future: Box<dyn WaitableFuture>,
    pub outdated_swapchain: bool,
    // default render set
    pub mainpass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pub framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    pub dynamic_state: DynamicState,
    // GUI-related handles
    pub gui_renderer: conrod_vulkano::Renderer,
    pub gui: conrod_core::Ui,
    pub gui_image_map: conrod_core::image::Map<conrod_vulkano::Image>,
}

pub trait FrameStage {}

pub struct PrePassStage {}

impl FrameStage for PrePassStage {}

pub struct InPassStage {}

impl FrameStage for InPassStage {}

pub struct PostPassStage {}

impl FrameStage for PostPassStage {}

#[must_use]
pub struct FrameContext<'r, Stage: FrameStage> {
    pub rctx: &'r mut RenderingContext,
    pub cmd: Option<AutoCommandBufferBuilder>,
    pub delta_time: f64,
    pub dims: [u32; 2],
    pub image_num: usize,
    pub queue: Arc<Queue>,
    acquire_future: SwapchainAcquireFuture<()>,
    _phantom: PhantomData<Stage>,
}

impl<'r, Stage: FrameStage> FrameContext<'r, Stage> {
    /// Replaces cmd with None, so that it can be modified and put back
    #[must_use]
    pub fn replace_cmd(&mut self) -> AutoCommandBufferBuilder {
        std::mem::replace(&mut self.cmd, None).unwrap()
    }
}

pub type PrePassFrameContext<'r> = FrameContext<'r, PrePassStage>;
pub type InPassFrameContext<'r> = FrameContext<'r, InPassStage>;
pub type PostPassFrameContext<'r> = FrameContext<'r, PostPassStage>;

fn generate_updated_framebuffers(
    swapimages: &[Arc<SwapchainImage<()>>],
    mainpass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = swapimages[0].dimensions();

    let depth_image = AttachmentImage::transient(
        mainpass.device().clone(),
        dimensions,
        vulkano::format::D32Sfloat_S8Uint,
    )
    .unwrap();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0..1.0,
    };
    dynamic_state.viewports = Some(vec![viewport]);

    swapimages
        .iter()
        .map(|img| {
            Arc::new(
                Framebuffer::start(mainpass.clone())
                    .add(img.clone())
                    .unwrap()
                    .add(depth_image.clone())
                    .unwrap()
                    .build()
                    .unwrap(),
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

impl RenderingContext {
    pub fn new(sdl_video: &sdl2::VideoSubsystem, cfg: &Config) -> RenderingContext {
        sdl_video.vulkan_load_library_default().unwrap();
        let mut window = sdl_video.window("BallX World", cfg.window_width, cfg.window_height);
        window
            .position_centered()
            .vulkan()
            .allow_highdpi()
            .resizable();
        if cfg.window_fullscreen {
            window.fullscreen();
        }
        let window = window.build().expect("Failed to create the game window");
        let instance_extensions;
        let instance = {
            let sdl_exts = window
                .vulkan_instance_extensions()
                .expect("Couldn't get a list of the required VK instance extensions");
            let raw_exts = RawInstanceExtensions::new(
                sdl_exts
                    .into_iter()
                    .map(|x| CString::new(x).expect("Invalid required VK instance extension")),
            );
            instance_extensions = InstanceExtensions::from(&raw_exts);
            Instance::new(
                Some(&app_info_from_cargo_toml!()),
                &instance_extensions,
                None,
            )
        }
        .expect("Failed to create Vulkan instance");
        let surface = Arc::new(unsafe {
            use vulkano::VulkanObject;
            let sdlvki = instance.internal_object() as sdl2::video::VkInstance;
            let rsurf = window
                .vulkan_create_surface(sdlvki)
                .expect("Couldn't create VK surface") as u64;
            Surface::from_raw_surface(instance.clone(), rsurf, ())
        });
        let physical = PhysicalDevice::enumerate(&instance)
            .next()
            .expect("no device available");

        println!("Choosing device {}", physical.name());
        for family in physical.queue_families() {
            println!(
                "Found a queue family with {:?} queue(s)",
                family.queues_count()
            );
        }

        let queue_family = physical
            .queue_families()
            .find(|&q| {
                q.supports_graphics()
                    && q.supports_compute()
                    && surface.is_supported(q).unwrap_or(false)
            })
            .expect("couldn't find a graphical queue family");
        let (device, mut queues) = {
            let device_ext = vulkano::device::DeviceExtensions {
                khr_swapchain: true,
                ..vulkano::device::DeviceExtensions::none()
            };
            Device::new(
                physical,
                physical.supported_features(),
                &device_ext,
                [(queue_family, 0.5)].iter().cloned(),
            )
            .expect("failed to create device")
        };
        let queue = queues.next().expect("Couldn't create rendering queue");

        let (swapchain, swapimages) = {
            let caps = surface
                .capabilities(physical)
                .expect("failed to get surface capabilities");
            let dimensions = caps.current_extent.unwrap_or([1280, 720]);
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;
            Swapchain::new(
                device.clone(),
                surface.clone(),
                max(3, 1 + caps.min_image_count),
                format,
                dimensions,
                1,
                caps.supported_usage_flags,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                if caps.present_modes.mailbox && !cfg.render_wait_for_vsync {
                    PresentMode::Mailbox
                } else {
                    PresentMode::Fifo
                },
                true,
                None,
            )
            .expect("failed to create swapchain")
        };

        let mainpass = Arc::new(
            single_pass_renderpass!(device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: swapchain.format(),
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D32Sfloat_S8Uint,
                        samples: 1,
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {depth}
                }
            )
            .unwrap(),
        );

        let mut dynamic_state = DynamicState {
            line_width: None,
            viewports: None,
            scissors: None,
        };

        let framebuffers =
            generate_updated_framebuffers(&swapimages, mainpass.clone(), &mut dynamic_state);

        let previous_frame_future = Box::new(vulkano::sync::now(device.clone()));

        let dims = swapimages[0].dimensions();

        let gui_renderer = conrod_vulkano::Renderer::new(
            device.clone(),
            Subpass::from(mainpass.clone(), 0).unwrap(),
            queue.family(),
            dims,
            1.0, // FIXME: SDL2 DPI factor needs to be put here
        )
        .unwrap();

        let mut gui = conrod_core::UiBuilder::new([f64::from(dims[0]), f64::from(dims[1])]).build();

        let gui_image_map = conrod_core::image::Map::new();

        gui.fonts
            .insert_from_file("res/fonts/LiberationSans-Regular.ttf")
            .expect("Couldn't load Liberation Sans font");

        RenderingContext {
            window,
            instance_extensions,
            instance,
            device: device.clone(),
            surface,
            queues: Queues::Combined(queue),
            swapchain,
            swapimages,
            mainpass,
            framebuffers,
            dynamic_state,
            previous_frame_future,
            outdated_swapchain: false,
            gui_renderer,
            gui,
            gui_image_map,
        }
    }

    /// Returns None if e.g. swapchain is in the process of being recreated
    pub fn frame_begin_prepass(&mut self, delta_time: f64) -> Option<PrePassFrameContext> {
        self.previous_frame_future.cleanup_finished();

        let dims = {
            let d = self.window.vulkan_drawable_size();
            [max(16, d.0), max(16, d.1)]
        };

        if self.outdated_swapchain {
            let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimension(dims) {
                Ok(r) => r,
                // occurs on manual resizes, just ignore
                Err(SwapchainCreationError::UnsupportedDimensions) => return None,
                Err(err) => panic!("{:?}", err),
            };

            self.swapchain = new_swapchain;
            self.swapimages = new_images;
            self.framebuffers = generate_updated_framebuffers(
                &self.swapimages,
                self.mainpass.clone(),
                &mut self.dynamic_state,
            );

            self.outdated_swapchain = false;
        }
        self.gui.win_w = f64::from(dims[0]);
        self.gui.win_h = f64::from(dims[1]);

        let Queues::Combined(ref queue) = self.queues;
        let queue = queue.clone();
        let (image_num, acquire_future) =
            match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.outdated_swapchain = true;
                    return None;
                }
                Err(err) => panic!("{:?}", err),
            };

        let mut cmdbufbuild =
            AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), queue.family())
                .unwrap();

        let gui_primitives = self.gui.draw();
        let gui_viewport = [0.0, 0.0, dims[0] as f32, dims[1] as f32];
        let dpi_factor = 1.0f64; // FIXME: DPI factor
        if let Some(cmd) = self
            .gui_renderer
            .fill(
                &self.gui_image_map,
                gui_viewport,
                dpi_factor,
                gui_primitives,
            )
            .unwrap()
        {
            let buffer = cmd
                .glyph_cpu_buffer_pool
                .chunk(cmd.glyph_cache_pixel_buffer.iter().cloned())
                .unwrap();
            cmdbufbuild = cmdbufbuild
                .copy_buffer_to_image(buffer, cmd.glyph_cache_texture)
                .expect("Failed to submit glyph cache update");
        }

        Some(PrePassFrameContext {
            rctx: self,
            cmd: Some(cmdbufbuild),
            delta_time,
            dims,
            image_num,
            acquire_future,
            queue,
            _phantom: PhantomData,
        })
    }

    pub fn frame_goto_pass(fctx: PrePassFrameContext) -> InPassFrameContext {
        let PrePassFrameContext {
            rctx: me,
            cmd,
            delta_time,
            dims,
            image_num,
            acquire_future,
            queue,
            _phantom,
        } = fctx;
        let mut cmdbufbuild = cmd.unwrap();
        cmdbufbuild = cmdbufbuild
            .begin_render_pass(
                me.framebuffers[image_num].clone(),
                false,
                vec![[0.1, 0.1, 0.1, 1.0].into(), (1f32, 0u32).into()],
            )
            .expect("Failed to begin render pass");

        InPassFrameContext {
            rctx: me,
            cmd: Some(cmdbufbuild),
            delta_time,
            dims,
            image_num,
            queue,
            acquire_future,
            _phantom: PhantomData,
        }
    }

    pub fn inpass_draw_gui(fctx: &mut InPassFrameContext) {
        let mut cmdbufbuild = fctx.replace_cmd();

        let gui_viewport = [0.0, 0.0, fctx.dims[0] as f32, fctx.dims[1] as f32];
        let gui_draw_cmds = fctx
            .rctx
            .gui_renderer
            .draw(fctx.queue.clone(), &fctx.rctx.gui_image_map, gui_viewport)
            .unwrap();
        for cmd in gui_draw_cmds {
            let conrod_vulkano::DrawCommand {
                graphics_pipeline,
                dynamic_state,
                vertex_buffer,
                descriptor_set,
            } = cmd;
            cmdbufbuild = cmdbufbuild
                .draw(
                    graphics_pipeline,
                    &dynamic_state,
                    vec![vertex_buffer],
                    descriptor_set,
                    (),
                )
                .expect("Failed to submit GUI draw command");
        }

        fctx.cmd = Some(cmdbufbuild);
    }

    pub fn frame_goto_postpass(fctx: InPassFrameContext) -> PostPassFrameContext {
        PostPassFrameContext {
            rctx: fctx.rctx,
            cmd: fctx
                .cmd
                .map(|c| c.end_render_pass().expect("Failed to end render pass")),
            delta_time: fctx.delta_time,
            dims: fctx.dims,
            image_num: fctx.image_num,
            acquire_future: fctx.acquire_future,
            queue: fctx.queue,
            _phantom: PhantomData,
        }
    }

    pub fn frame_finish(fctx: PostPassFrameContext) {
        let PostPassFrameContext {
            rctx: me,
            cmd,
            image_num,
            acquire_future,
            queue,
            ..
        } = fctx;

        let cmdbuf = cmd
            .unwrap()
            .build()
            .expect("Failed to build frame draw commands");

        me.previous_frame_future.wait_or_noop();

        let future = vulkano::sync::now(me.device.clone())
            .join(acquire_future)
            .then_execute(queue.clone(), cmdbuf)
            .unwrap()
            .then_swapchain_present(queue.clone(), me.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                me.previous_frame_future = Box::new(future);
            }
            Err(FlushError::OutOfDate) => {
                me.previous_frame_future = Box::new(vulkano::sync::now(me.device.clone()));
                me.outdated_swapchain = true;
            }
            Err(e) => {
                me.previous_frame_future = Box::new(vulkano::sync::now(me.device.clone()));
                println!("{:?}", e);
            }
        }
    }
}
