use std::ffi::CString;
use std::sync::Arc;

use cgmath::prelude::*;
use cgmath::{Matrix4, Rad, PerspectiveFov, vec3, Vector3, Deg};

use sdl2::video::Window;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, Queue};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::SwapchainImage;
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice, RawInstanceExtensions};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{
    AcquireError, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::descriptor::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::sync::{FlushError, GpuFuture, FenceSignalFuture, NowFuture};
use std::cmp::max;

pub mod vox {
    pub struct ChunkBuffers {
        pub vertices: Vec<VoxelVertex>,
        pub indices: Vec<u32>,
    }

    #[derive(Copy, Clone, Default)]
    pub struct VoxelVertex {
        pub position: [f32; 4],
        pub color: [f32; 4],
    }
    impl_vertex!(VoxelVertex, position, color);

    #[derive(Copy, Clone, Default)]
    pub struct VoxelUBO {
        pub model: [f32; 16],
        pub view: [f32; 16],
        pub proj: [f32; 16],
    }

    pub mod vs {
        shader! {
        ty: "vertex",
        path: "src/client/shaders/voxel.vert"
        }
    }

    pub mod fs {
        shader! {
        ty: "fragment",
        path: "src/client/shaders/voxel.frag"
        }
    }
}

pub enum Queues {
    /// Combined Graphics+Transfer+Compute queue
    Combined(Arc<Queue>),
}

pub trait WaitableFuture: GpuFuture {
    fn wait_unwrap(&self);
}
impl<F> WaitableFuture for FenceSignalFuture<F> where F: GpuFuture {
    fn wait_unwrap(&self) {
        self.wait(None).unwrap();
    }
}
impl WaitableFuture for NowFuture {
    fn wait_unwrap(&self) {
        // do nothing
    }
}

pub struct RenderingContext {
    pub window: Window,
    pub instance_extensions: InstanceExtensions,
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface<()>>,
    pub device: Arc<Device>,
    pub queues: Queues,
    pub swapchain: Arc<Swapchain<()>>,
    pub swapimages: Vec<Arc<SwapchainImage<()>>>,
    pub mainpass: Arc<RenderPassAbstract + Send + Sync>,
    pub framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    pub dynamic_state: DynamicState,
    pub frame_fences: Vec<Option<Box<WaitableFuture>>>,
    pub outdated_swapchain: bool,
    pub voxel_pipeline: Arc<GraphicsPipelineAbstract + Send + Sync>,
    pub vbuffer: Arc<CpuAccessibleBuffer<[vox::VoxelVertex]>>,
    pub ibuffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub ubuffers: Vec<Arc<CpuAccessibleBuffer<vox::VoxelUBO>>>,
    pub udsets: Vec<Arc<DescriptorSet + Send + Sync>>,
    pub position: Vector3<f32>,
    pub angles: (f32, f32),
}

fn generate_updated_framebuffers(
    swapimages: &[Arc<SwapchainImage<()>>],
    mainpass: Arc<RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState,
) -> Vec<Arc<FramebufferAbstract + Send + Sync>> {
    let dimensions = swapimages[0].dimensions();

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
                max(3,1+caps.min_image_count),
                format,
                dimensions,
                1,
                caps.supported_usage_flags,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                if caps.present_modes.mailbox {PresentMode::Mailbox} else {PresentMode::Fifo},
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
                    }
                },
                pass: {
                    color: [color],
                    depth_stencil: {}
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

        let mut frame_fences = Vec::new();
        for _ in &swapimages {
            frame_fences.push(None);
        }

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
                    .render_pass(Subpass::from(mainpass.clone(), 0).unwrap())
                    .build(device.clone())
                    .expect("Could not create voxel graphics pipeline")
            )
        };

        let mut ubuffers = Vec::new();
        let mut udsets: Vec<Arc<DescriptorSet + Send + Sync>> = Vec::new();
        for _ in 0..swapimages.len() {
            let ubuffer: Arc<CpuAccessibleBuffer<vox::VoxelUBO>> = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer(),
                Default::default(),
            ).unwrap();
            ubuffers.push(ubuffer.clone());
            let udset =
                Arc::new(PersistentDescriptorSet::start(voxel_pipeline.clone(), 0)
                    .add_buffer(ubuffer.clone())
                    .unwrap()
                    .build().unwrap());
            udsets.push(udset);
        }

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
            frame_fences,
            outdated_swapchain: false,
            voxel_pipeline,
            vbuffer,
            ibuffer,
            ubuffers,
            udsets,
            position: vec3(0.0,0.0,-90.0),
            angles: (0.0, 0.0)
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
        if self.outdated_swapchain {
            unsafe {
                self.device.wait().unwrap();
            }

            let dims = {
                let d = self.window.vulkan_drawable_size();
                [d.0, d.1]
            };

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
        if let Some(ref mut fence) = self.frame_fences[image_num] {
            fence.wait_unwrap();
            fence.cleanup_finished();
        }

        {
            let mut ubo = self.ubuffers[image_num].write().unwrap();
            let mmat: Matrix4<f32> = One::one();
            ubo.model = AsRef::<[f32; 16]>::as_ref(&mmat).clone();
            let mut mview: Matrix4<f32> = Matrix4::from_translation(self.position);
            mview = mview * Matrix4::from_angle_x( Deg(self.angles.0));
            mview = mview * Matrix4::from_angle_y(Deg(-self.angles.1));
            ubo.view = AsRef::<[f32; 16]>::as_ref(&mview).clone();
            let swdim = self.swapchain.dimensions();
            let sfdim = [swdim[0] as f32, swdim[1] as f32];
            let mproj: Matrix4<f32> = Matrix4::from(PerspectiveFov {
                fovy: Rad(75.0*3.14/180.0),
                aspect: sfdim[0] / sfdim[1],
                near: 0.1,
                far: 1000.0,
            });
            ubo.proj = AsRef::<[f32; 16]>::as_ref(&mproj).clone();
        }

        let cmdbuf =
            AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(),
                                                              queue.family())
                .unwrap()
                .begin_render_pass(
                    self.framebuffers[image_num].clone(),
                    false,
                    vec![[0.1, 0.1, 0.1, 1.0].into()],
                )
                .unwrap()
                //
                .draw_indexed(
                    self.voxel_pipeline.clone(),
                    &self.dynamic_state,
                    vec![self.vbuffer.clone()],
                    self.ibuffer.clone(),
                    self.udsets[image_num].clone(),
                    (),
                )
                .unwrap()
                //
                .end_render_pass()
                .unwrap()
                //
                .build()
                .unwrap();
        let myfuture = std::mem::replace(
            &mut self.frame_fences[image_num],
            None,
        ).unwrap_or(Box::new(vulkano::sync::now(self.device.clone())));

        let future = myfuture
            .join(acquire_future)
            .then_execute(queue.clone(), cmdbuf)
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                self.frame_fences[image_num] = Some(Box::new(future));
            }
            Err(FlushError::OutOfDate) => {
                self.outdated_swapchain = true;
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
}
