use std::ffi::CString;
use std::sync::Arc;

use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, PerspectiveFov, vec3, Vector3, Deg};

use sdl2::video::Window;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{SwapchainImage, AttachmentImage};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{
    AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::sync::{FlushError, GpuFuture, FenceSignalFuture, NowFuture};
use vulkano::{app_info_from_cargo_toml, single_pass_renderpass};
use std::cmp::max;

pub mod vox {
    use vulkano::impl_vertex;

    pub struct ChunkBuffers {
        pub vertices: Vec<VoxelVertex>,
        pub indices: Vec<u32>,
    }

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct VoxelVertex {
        pub position: [f32; 4],
        pub color: [f32; 4],
    }
    impl_vertex!(VoxelVertex, position, color);

    #[derive(Copy, Clone, Default)]
    #[repr(C)]
    pub struct VoxelUBO {
        pub model: [f32; 16],
        pub view: [f32; 16],
        pub proj: [f32; 16],
    }

    pub mod vs {
        vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/client/shaders/voxel.vert"
        }
    }

    pub mod fs {
        vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/client/shaders/voxel.frag"
        }
    }
}

pub enum Queues {
    /// Combined Graphics+Transfer+Compute queue
    Combined(Arc<Queue>),
}

trait WaitableFuture : GpuFuture {
    fn wait_or_noop(&mut self);
}

impl<F> WaitableFuture for FenceSignalFuture<F> where F: GpuFuture {
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
    previous_frame_future: Box<WaitableFuture>,
    pub outdated_swapchain: bool,
    // default render set
    pub mainpass: Arc<RenderPassAbstract + Send + Sync>,
    pub framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    pub dynamic_state: DynamicState,
    // test voxel render stuff
    pub voxel_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub vbuffer: Arc<CpuAccessibleBuffer<[vox::VoxelVertex]>>,
    pub ibuffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub ubuffers: CpuBufferPool<vox::VoxelUBO>,
    pub position: Vector3<f32>,
    pub angles: (f32, f32),
    // GUI-related handles
    pub gui_renderer: conrod_vulkano::Renderer,
    pub gui: conrod_core::Ui,
    pub gui_image_map: conrod_core::image::Map<conrod_vulkano::Image>,
}

fn generate_updated_framebuffers(
    swapimages: &[Arc<SwapchainImage<()>>],
    mainpass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = swapimages[0].dimensions();

    let depth_image =
        AttachmentImage::transient(mainpass.device().clone(),
                                   dimensions,
                                   vulkano::format::D32Sfloat_S8Uint).unwrap()
        ;

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
            ) as Arc<FramebufferAbstract + Send + Sync>
        })
        .collect::<Vec<_>>()
}

impl RenderingContext {
    pub fn new(sdl_video: &sdl2::VideoSubsystem) -> RenderingContext {
        sdl_video.vulkan_load_library_default().unwrap();
        let window = sdl_video
            .window("BallX World", 1280, 720)
            .position_centered()
            .vulkan()
            .allow_highdpi()
            .resizable()
            .build()
            .expect("Failed to create the game window");
        let instance_extensions;
        let instance = {
            let sdl_exts = window
                .vulkan_instance_extensions()
                .expect("Couldn't get a list of the required VK instance extensions");
            let raw_exts = RawInstanceExtensions::new(
                sdl_exts
                    .into_iter()
                    .map(|x| CString::new(x)
                        .expect("Invalid required VK instance extension")),
            );
            instance_extensions = InstanceExtensions::from(&raw_exts);
            Instance::new(Some(&app_info_from_cargo_toml!()),
                          &instance_extensions, None)
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
                if caps.present_modes.mailbox { PresentMode::Mailbox } else { PresentMode::Fifo },
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

        let vbuffer = {
            let v1 = vox::VoxelVertex {
                position: [-0.5, -0.5, 0.0, 0.0],
                color: [1.0, 0.0, 0.0, 1.0],
            };
            let v2 = vox::VoxelVertex {
                position: [0.0, 0.5, 0.0, 0.0],
                color: [0.0, 1.0, 0.0, 1.0],
            };
            let v3 = vox::VoxelVertex {
                position: [0.5, -0.25, 0.0, 0.0],
                color: [0.0, 0.0, 1.0, 1.0],
            };
            CpuAccessibleBuffer::from_iter(
                device.clone(),
                BufferUsage::vertex_buffer(),
                vec![v1, v2, v3].into_iter(),
            )
                .unwrap()
        };

        let ibuffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::index_buffer(),
            vec![0, 1, 2].into_iter(),
        ).unwrap();

        let voxel_pipeline = {
            let vs = vox::vs::Shader::load(device.clone()).expect("Failed to create VS module");
            let fs = vox::fs::Shader::load(device.clone()).expect("Failed to create FS module");
            Arc::new(
                GraphicsPipeline::start()
                    .vertex_input_single_buffer::<vox::VoxelVertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(mainpass.clone(), 0).unwrap())
                    .build(device.clone())
                    .expect("Could not create voxel graphics pipeline")
            )
        };

        let ubuffers = CpuBufferPool::new(device.clone(), BufferUsage::uniform_buffer());

        let previous_frame_future = Box::new(vulkano::sync::now(device.clone()));

        let dims = swapimages[0].dimensions();

        let gui_renderer = conrod_vulkano::Renderer::new(
            device.clone(),
            Subpass::from(mainpass.clone(), 0).unwrap(),
            queue.family(),
            dims,
            1.0, // FIXME: SDL2 DPI factor needs to be put here
        ).unwrap();

        let mut gui = conrod_core::UiBuilder::new(
            [dims[0] as f64, dims[1] as f64]
        ).build();

        let gui_image_map = conrod_core::image::Map::new();

        gui.fonts.insert_from_file("res/fonts/LiberationSans-Regular.ttf").expect("Couldn't load Liberation Sans font");

        RenderingContext {
            window,
            instance_extensions,
            instance,
            device,
            surface,
            queues: Queues::Combined(queue),
            swapchain,
            swapimages,
            mainpass,
            framebuffers,
            dynamic_state,
            previous_frame_future,
            outdated_swapchain: false,
            voxel_pipeline,
            vbuffer,
            ibuffer,
            ubuffers,
            position: vec3(0.0, 0.0, 90.0),
            angles: (0.0, 0.0),
            gui_renderer,
            gui,
            gui_image_map,
        }
    }

    pub fn d_reset_buffers(&mut self, new_data: vox::ChunkBuffers) {
        self.vbuffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::vertex_buffer(),
            new_data.vertices.into_iter(),
        ).unwrap();

        self.ibuffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::index_buffer(),
            new_data.indices.into_iter(),
        ).unwrap();
    }

    pub fn draw_next_frame(&mut self, _delta_time: f64) {
        self.previous_frame_future.cleanup_finished();

        let dims = {
            let d = self.window.vulkan_drawable_size();
            [d.0, d.1]
        };

        if self.outdated_swapchain {
            let (new_swapchain, new_images) =
                match self.swapchain.recreate_with_dimension(dims) {
                    Ok(r) => r,
                    // occurs on manual resizes, just ignore
                    Err(SwapchainCreationError::UnsupportedDimensions) => return,
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
            self.gui.win_w = dims[0] as f64;
            self.gui.win_h = dims[1] as f64;
        }

        let Queues::Combined(ref queue) = self.queues;
        let (image_num, acquire_future) =
            match vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.outdated_swapchain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        let ubo = {
            let mut ubo = vox::VoxelUBO::default();
            let mmat: Matrix4<f32> = One::one();
            ubo.model = AsRef::<[f32; 16]>::as_ref(&mmat).clone();
            let mut mview: Matrix4<f32> = Matrix4::from_translation(-self.position);
            mview = Matrix4::from_angle_y(Deg(self.angles.1)) * mview;
            mview = Matrix4::from_angle_x(Deg(self.angles.0)) * mview;
            ubo.view = AsRef::<[f32; 16]>::as_ref(&mview).clone();
            let swdim = self.swapchain.dimensions();
            let sfdim = [swdim[0] as f32, swdim[1] as f32];
            let mproj: Matrix4<f32> = PerspectiveFov {
                fovy: Rad(75.0 * 3.14 / 180.0),
                aspect: sfdim[0] / sfdim[1],
                near: 0.1,
                far: 1000.0,
            }.into();
            ubo.proj = AsRef::<[f32; 16]>::as_ref(&mproj).clone();
            self.ubuffers.next(ubo).unwrap()
        };

        let udset =
            Arc::new(PersistentDescriptorSet::start(self.voxel_pipeline.clone(), 0)
                .add_buffer(ubo.clone())
                .unwrap()
                .build().unwrap());

        let mut cmdbufbuild =
            AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(),
                                                              queue.family())
                .unwrap();

        let gui_primitives = self.gui.draw();
        let gui_viewport = [0.0, 0.0, dims[0] as f32, dims[1] as f32];
        let dpi_factor = 1.0f64; // FIXME: DPI factor
        if let Some(cmd) = self.gui_renderer.fill(
            &self.gui_image_map, gui_viewport, dpi_factor, gui_primitives).unwrap() {
            let buffer = cmd.glyph_cpu_buffer_pool.chunk(cmd.glyph_cache_pixel_buffer.iter().cloned()).unwrap();
            cmdbufbuild = cmdbufbuild.copy_buffer_to_image(buffer, cmd.glyph_cache_texture)
                .expect("Failed to submit glyph cache update");
        }

        cmdbufbuild = cmdbufbuild.begin_render_pass(
            self.framebuffers[image_num].clone(),
            false,
            vec![[0.1, 0.1, 0.1, 1.0].into(),
                 (1f32, 0u32).into()],
        )
            .expect("Failed to begin render pass")
            //
            .draw_indexed(
                self.voxel_pipeline.clone(),
                &self.dynamic_state,
                vec![self.vbuffer.clone()],
                self.ibuffer.clone(),
                udset.clone(),
                (),
            )
            .expect("Failed to submit voxel chunk draw");

        let gui_draw_cmds = self.gui_renderer.draw(
            queue.clone(), &self.gui_image_map, gui_viewport).unwrap();
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

        let cmdbuf = cmdbufbuild.end_render_pass()
            .expect("Failed to end render pass")
            //
            .build()
            .expect("Failed to build frame draw commands");

        self.previous_frame_future.wait_or_noop();

        let future = vulkano::sync::now(self.device.clone())
            .join(acquire_future)
            .then_execute(queue.clone(), cmdbuf)
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                self.previous_frame_future = Box::new(future);
            }
            Err(FlushError::OutOfDate) => {
                self.previous_frame_future = Box::new(vulkano::sync::now(self.device.clone()));
                self.outdated_swapchain = true;
            }
            Err(e) => {
                self.previous_frame_future = Box::new(vulkano::sync::now(self.device.clone()));
                println!("{:?}", e);
            }
        }
    }
}

