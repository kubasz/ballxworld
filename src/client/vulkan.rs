use sdl2::video::Window;
use std::ffi::CString;
use std::sync::Arc;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::device::Device;
use vulkano::device::DeviceExtensions;
use vulkano::device::Features;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract};
use vulkano::image::SwapchainImage;
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::instance::RawInstanceExtensions;
use vulkano::swapchain::{PresentMode, Surface, SurfaceTransform, Swapchain};
use vulkano::sync::{FlushError, GpuFuture};

pub enum Queues {
    /// Combined Graphics+Transfer+Compute queue
    Combined(Arc<Queue>),
}

pub struct RenderingContext {
    window: Window,
    instance_extensions: InstanceExtensions,
    instance: Arc<Instance>,
    surface: Arc<Surface<()>>,
    device: Arc<Device>,
    queues: Queues,
    swapchain: Arc<Swapchain<()>>,
    swapimages: Vec<Arc<SwapchainImage<()>>>,
    mainpass: Arc<RenderPassAbstract + Send + Sync>,
    framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>>,
    previous_frame_end: Box<GpuFuture>,
}

impl RenderingContext {
    pub fn new(sdl_video: &sdl2::VideoSubsystem) -> RenderingContext {
        sdl_video.vulkan_load_library_default().unwrap();
        let window = sdl_video
            .window("BallX World", 1280, 720)
            .position_centered()
            .vulkan()
            .allow_highdpi()
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
                    .map(|x| CString::new(x).expect("Invalid required VK instance extension")),
            );
            instance_extensions = InstanceExtensions::from(&raw_exts);
            Instance::new(None, &instance_extensions, None)
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
                caps.min_image_count,
                format,
                dimensions,
                1,
                caps.supported_usage_flags,
                &queue,
                SurfaceTransform::Identity,
                alpha,
                PresentMode::Fifo,
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

        let framebuffers: Vec<_> = swapimages
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
            .collect();

        let previous_frame_end = Box::new(vulkano::sync::now(device.clone()));

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
            previous_frame_end,
        }
    }

    pub fn draw_next_frame(&mut self) {
        self.previous_frame_end.cleanup_finished();
        let Queues::Combined(ref queue) = self.queues;
        let (image_num, acquire_future) =
            vulkano::swapchain::acquire_next_image(self.swapchain.clone(), None).unwrap();
        let cmdbuf =
            AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(
                    self.framebuffers[image_num].clone(),
                    false,
                    vec![[0.0, 0.0, 1.0, 1.0].into()],
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();
        let myfuture = std::mem::replace(
            &mut self.previous_frame_end,
            Box::new(vulkano::sync::now(self.device.clone())),
        );
        let future = myfuture
            .join(acquire_future)
            .then_execute(queue.clone(), cmdbuf)
            .unwrap()
            .then_swapchain_present(queue.clone(), self.swapchain.clone(), image_num)
            .then_signal_fence_and_flush();
        match future {
            Ok(future) => {
                self.previous_frame_end = Box::new(future);
            }
            Err(FlushError::OutOfDate) => {
                // out of date
            }
            Err(e) => {
                println!("{:?}", e);
            }
        }
    }
}
