use smallvec::alloc::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::*;

/// Debug view data for monitoring the behaviour of the game
#[derive(Debug, Default)]
pub struct DebugData {
    // Memory data
    pub heap_usage_bytes: AtomicI64,
    pub gpu_usage_bytes: AtomicI64,
    // Rendering performance -- not updated on server side
    pub fps: AtomicU32,
    pub ft_max_us: AtomicU32,
    pub ft_avg_us: AtomicU32,
    // Coordinate info in tenths of a meter
    pub local_player_x: AtomicI64,
    pub local_player_y: AtomicI64,
    pub local_player_z: AtomicI64,
}

pub fn format_bytes(bytes: i64) -> String {
    const UNITS: &[&str] = &["b", "kB", "MB", "GB", "TB"];
    let mut unit = 0usize;
    let mut value = bytes.abs();
    while value > 4096 && unit < UNITS.len() - 1 {
        value >>= 10;
        unit += 1;
    }
    if bytes < 0 {
        value = -value;
    }
    format!("{} {}", value, UNITS[unit])
}

impl DebugData {
    pub fn hud_format(&self) -> String {
        format!(
            r#"FPS: {fps}
FT max ms: {ftmax:.1}
FT avg ms: {ftavg:.1}
Pos: {lpx:.1} {lpy:.1} {lpz:.1}

Heap usage: {heap}
GPU heap usage: {gpuheap}"#,
            fps = self.fps.load(Ordering::Acquire),
            ftmax = self.ft_max_us.load(Ordering::Acquire) as f32 / 1000.0,
            ftavg = self.ft_avg_us.load(Ordering::Acquire) as f32 / 1000.0,
            lpx = self.local_player_x.load(Ordering::Acquire) as f32 / 10.0,
            lpy = self.local_player_y.load(Ordering::Acquire) as f32 / 10.0,
            lpz = self.local_player_z.load(Ordering::Acquire) as f32 / 10.0,
            heap = format_bytes(self.heap_usage_bytes.load(Ordering::Acquire)),
            gpuheap = format_bytes(self.gpu_usage_bytes.load(Ordering::Acquire)),
        )
    }
}

pub static DEBUG_DATA: DebugData = DebugData {
    heap_usage_bytes: AtomicI64::new(0),
    gpu_usage_bytes: AtomicI64::new(0),
    fps: AtomicU32::new(0),
    ft_max_us: AtomicU32::new(0),
    ft_avg_us: AtomicU32::new(0),
    local_player_x: AtomicI64::new(0),
    local_player_y: AtomicI64::new(0),
    local_player_z: AtomicI64::new(0),
};

pub struct TrackingAllocator<A: GlobalAlloc> {
    pub allocator: A,
}

unsafe impl<A: GlobalAlloc> GlobalAlloc for TrackingAllocator<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let data = self.allocator.alloc(layout);
        if !data.is_null() {
            DEBUG_DATA
                .heap_usage_bytes
                .fetch_add(layout.size() as i64, Ordering::SeqCst);
        }
        data
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.allocator.dealloc(ptr, layout);
        DEBUG_DATA
            .heap_usage_bytes
            .fetch_sub(layout.size() as i64, Ordering::SeqCst);
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let data = self.allocator.alloc_zeroed(layout);
        if !data.is_null() {
            DEBUG_DATA
                .heap_usage_bytes
                .fetch_add(layout.size() as i64, Ordering::SeqCst);
        }
        data
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let data = self.allocator.realloc(ptr, layout, new_size);
        if !data.is_null() {
            DEBUG_DATA
                .heap_usage_bytes
                .fetch_sub(layout.size() as i64, Ordering::SeqCst);
            DEBUG_DATA
                .heap_usage_bytes
                .fetch_add(new_size as i64, Ordering::SeqCst);
        }
        data
    }
}
