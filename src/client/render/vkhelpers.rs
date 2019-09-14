use crate::client::render::vulkan::{allocation_cbs, QueueGuard, RenderingHandles};
use ash::prelude::VkResult;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::vk::Handle;
use parking_lot::MutexGuard;
use std::ffi::CString;
use vk_mem as vma;

pub fn name_vk_object<F: FnOnce() -> S, S>(
    handles: &RenderingHandles,
    name_fn: F,
    raw_handle: u64,
    obj_type: vk::ObjectType,
) where
    S: Into<Vec<u8>>,
{
    if let Some(ext_debug) = handles.ext_debug.as_ref() {
        let name_slice = name_fn();
        let name = CString::new(name_slice).unwrap();
        let ni = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_handle(raw_handle)
            .object_type(obj_type)
            .object_name(name.as_c_str());
        unsafe {
            ext_debug
                .utils
                .debug_utils_set_object_name(handles.device.handle(), &ni)
        }
        .unwrap();
    }
}

#[derive(Default)]
pub struct OwnedImage {
    pub image: vk::Image,
    pub allocation: Option<(vma::Allocation, vma::AllocationInfo)>,
}

impl OwnedImage {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from(
        vmalloc: &mut vma::Allocator,
        img_info: &vk::ImageCreateInfo,
        mem_info: &vma::AllocationCreateInfo,
    ) -> Self {
        let r = vmalloc
            .create_image(img_info, mem_info)
            .expect("Could not create Vulkan image");
        Self {
            image: r.0,
            allocation: Some((r.1, r.2)),
        }
    }

    pub fn give_name<F: FnOnce() -> S, S>(&self, handles: &RenderingHandles, name_fn: F)
    where
        S: Into<Vec<u8>>,
    {
        name_vk_object(handles, name_fn, self.image.as_raw(), vk::ObjectType::IMAGE);
    }

    pub fn destroy(&mut self, vmalloc: &mut vma::Allocator) {
        if self.allocation.is_some() {
            vmalloc
                .destroy_image(self.image, &self.allocation.take().unwrap().0)
                .unwrap();
            self.image = vk::Image::null();
        }
    }
}

#[derive(Default)]
pub struct OwnedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<(vma::Allocation, vma::AllocationInfo)>,
}

impl OwnedBuffer {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from(
        vmalloc: &mut vma::Allocator,
        buf_info: &vk::BufferCreateInfo,
        mem_info: &vma::AllocationCreateInfo,
    ) -> Self {
        let r = vmalloc
            .create_buffer(buf_info, mem_info)
            .unwrap_or_else(|e| panic!("Could not create Vulkan buffer: {:#?}", e));
        Self {
            buffer: r.0,
            allocation: Some((r.1, r.2)),
        }
    }

    pub fn give_name<F: FnOnce() -> S, S>(&self, handles: &RenderingHandles, name_fn: F)
    where
        S: Into<Vec<u8>>,
    {
        name_vk_object(
            handles,
            name_fn,
            self.buffer.as_raw(),
            vk::ObjectType::BUFFER,
        );
    }

    pub fn destroy(&mut self, vmalloc: &mut vma::Allocator) {
        if self.allocation.is_some() {
            vmalloc
                .destroy_buffer(self.buffer, &self.allocation.take().unwrap().0)
                .unwrap();
            self.buffer = vk::Buffer::null();
        }
    }
}

pub struct FenceGuard<'h> {
    handles: &'h RenderingHandles,
    fence: vk::Fence,
}

impl<'h> FenceGuard<'h> {
    pub fn new<'s, F: FnOnce() -> &'s str>(
        handles: &'h RenderingHandles,
        signaled: bool,
        name_fn: F,
    ) -> Self {
        let fci = vk::FenceCreateInfo::builder().flags(if signaled {
            vk::FenceCreateFlags::SIGNALED
        } else {
            vk::FenceCreateFlags::empty()
        });
        let fence = unsafe { handles.device.create_fence(&fci, allocation_cbs()) }
            .expect("Couldn't create Vulkan fence");
        if let Some(ext_debug) = handles.ext_debug.as_ref() {
            let name_slice = name_fn();
            let name = CString::new(name_slice).unwrap();
            let ni = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_handle(fence.as_raw())
                .object_type(vk::ObjectType::FENCE)
                .object_name(name.as_c_str());
            unsafe {
                ext_debug
                    .utils
                    .debug_utils_set_object_name(handles.device.handle(), &ni)
            }
            .unwrap();
        }
        Self { handles, fence }
    }

    pub fn handle(&self) -> vk::Fence {
        self.fence
    }

    pub fn signaled(&self) -> bool {
        match unsafe { self.handles.device.get_fence_status(self.fence) } {
            Ok(_) => true,
            Err(vk::Result::NOT_READY) => false,
            Err(e) => panic!(e),
        }
    }

    pub fn wait(&self, timeout_ns: Option<u64>) -> VkResult<()> {
        unsafe {
            self.handles.device.wait_for_fences(
                &[self.fence],
                true,
                timeout_ns.unwrap_or(u64::max_value()),
            )
        }
    }

    pub fn reset(&self) {
        unsafe { self.handles.device.reset_fences(&[self.fence]) }.expect("Couldn't reset fence");
    }
}

impl<'h> Drop for FenceGuard<'h> {
    fn drop(&mut self) {
        if self.fence != vk::Fence::null() {
            unsafe {
                self.handles
                    .device
                    .destroy_fence(self.fence, allocation_cbs());
            }
            self.fence = vk::Fence::null();
        }
    }
}

pub struct OnetimeCmdGuard<'h> {
    fence: FenceGuard<'h>,
    custom_pool: Option<vk::CommandPool>,
    pool_lock: Option<MutexGuard<'h, vk::CommandPool>>,
    cmd: vk::CommandBuffer,
}

impl<'h> OnetimeCmdGuard<'h> {
    pub fn new(handles: &'h RenderingHandles, custom_pool: Option<vk::CommandPool>) -> Self {
        let (pool_lock, pool) = if let Some(pool) = custom_pool {
            (None, pool)
        } else {
            let lock = handles.oneoff_cmd_pool.lock();
            let pool = *lock;
            (Some(lock), pool)
        };
        let bai = vk::CommandBufferAllocateInfo::builder()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { handles.device.allocate_command_buffers(&bai) }
            .expect("Couldn't allocate one-time cmd buffer")[0];
        let bgi = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { handles.device.begin_command_buffer(cmd, &bgi) }
            .expect("Couldn't begin recording one-time cmd buffer");
        Self {
            fence: FenceGuard::new(handles, false, || "one-time cmd"),
            custom_pool,
            pool_lock,
            cmd,
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.cmd
    }

    pub fn execute(mut self, queue: &QueueGuard) {
        let handles = self.fence.handles;
        unsafe { handles.device.end_command_buffer(self.cmd) }
            .expect("Couldn't end recording one-time cmd buffer");

        let bufs = [self.cmd];

        let pool_lock = self.pool_lock.take();
        let si = vk::SubmitInfo::builder().command_buffers(&bufs);
        let sis = [si.build()];
        unsafe { handles.device.queue_submit(**queue, &sis, self.fence.fence) }
            .expect("Couldn't submit one-time cmd buffer");
        self.fence
            .wait(None)
            .expect("Failed waiting for one-time cmd fence");

        unsafe {
            handles.device.free_command_buffers(
                pool_lock.map_or_else(|| self.custom_pool.unwrap(), |p| *p),
                &bufs,
            );
        }
        std::mem::forget(self);
    }
}

impl<'h> Drop for OnetimeCmdGuard<'h> {
    fn drop(&mut self) {
        panic!("One-time command buffer not invoked");
    }
}

#[derive(Copy, Clone, Default)]
pub struct DynamicState {
    pub viewport: vk::Viewport,
}

impl DynamicState {
    pub fn get_viewport(&self) -> vk::Viewport {
        self.viewport
    }

    pub fn get_scissor(&self) -> vk::Rect2D {
        vk::Rect2D {
            offset: Default::default(),
            extent: vk::Extent2D {
                width: self.viewport.width as u32,
                height: self.viewport.height as u32,
            },
        }
    }

    pub fn cmd_update_pipeline(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_set_viewport(cmd, 0, &[self.get_viewport()]);
            device.cmd_set_scissor(cmd, 0, &[self.get_scissor()]);
        }
    }
}

pub fn make_pipe_depthstencil() -> vk::PipelineDepthStencilStateCreateInfo {
    vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .build()
}

pub fn identity_components() -> vk::ComponentMapping {
    vk::ComponentMapping {
        r: vk::ComponentSwizzle::IDENTITY,
        g: vk::ComponentSwizzle::IDENTITY,
        b: vk::ComponentSwizzle::IDENTITY,
        a: vk::ComponentSwizzle::IDENTITY,
    }
}

pub fn cmd_push_struct_constants<S>(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    pipe_layout: vk::PipelineLayout,
    stages: vk::ShaderStageFlags,
    offset: u32,
    constants: &S,
) {
    let pc_sz = std::mem::size_of::<S>();
    let pc_bytes = unsafe { std::slice::from_raw_parts(constants as *const _ as *const u8, pc_sz) };
    unsafe {
        device.cmd_push_constants(cmd, pipe_layout, stages, offset, pc_bytes);
    }
}

pub struct DroppingCommandPool {
    pub pool: vk::CommandPool,
    device: ash::Device,
}

impl DroppingCommandPool {
    pub fn new(handles: &RenderingHandles, pci: &vk::CommandPoolCreateInfo) -> Self {
        let pool = unsafe { handles.device.create_command_pool(pci, allocation_cbs()) }
            .expect("Couldn't create command pool");
        Self {
            pool,
            device: handles.device.clone(),
        }
    }
}

impl Drop for DroppingCommandPool {
    fn drop(&mut self) {
        if self.pool != vk::CommandPool::null() {
            unsafe {
                self.device
                    .destroy_command_pool(self.pool, allocation_cbs());
            }
            self.pool = vk::CommandPool::null();
        }
    }
}
