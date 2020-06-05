use lazy_static::lazy_static;
use parking_lot::Mutex;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::*;
use std::fmt::*;
use std::mem::MaybeUninit;

/// Debug view data for monitoring the behaviour of the game
#[derive(Debug, Default)]
pub struct DebugData {
    // Memory data
    pub heap_usage_bytes: AtomicI64,
    pub gpu_usage_bytes: AtomicI64,
    // Rendering performance -- not updated on server side
    pub fps: AtomicU32,
    pub frame_times: TimingRing,
    pub wgen_times: TimingRing,
    pub wmesh_times: TimingRing,
    // Coordinate info in tenths of a meter
    pub local_player_x: AtomicI64,
    pub local_player_y: AtomicI64,
    pub local_player_z: AtomicI64,
    /// Custom string that can be set to anything while debugging a piece of code
    pub custom_string: Mutex<String>,
}

#[derive(Default, Debug, Copy, Clone, Hash)]
pub struct FmtBytes(pub i64);

#[derive(Default, Debug, Copy, Clone, Hash)]
pub struct FmtNanoseconds(pub i64);

impl Display for FmtBytes {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let bytes = self.0;
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
        write!(f, "{} {}", value, UNITS[unit])
    }
}

impl Display for FmtNanoseconds {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let ns = self.0;
        const UNITS: &[&str] = &["ns", "us", "ms", "s"];
        let mut unit = 0usize;
        let mut value = ns.abs();
        while value > 8000 && unit < UNITS.len() - 1 {
            value /= 1000;
            unit += 1;
        }
        if ns < 0 {
            value = -value;
        }
        write!(f, "{} {}", value, UNITS[unit])
    }
}

impl DebugData {
    pub fn hud_format(&self) -> String {
        format!(
            r#"FPS: {fps}
FT: {ft}
Generate: {gen}
Mesh: {mesh}
Pos: {lpx:.1} {lpy:.1} {lpz:.1}

Heap usage: {heap}
GPU heap usage: {gpuheap}
{custom_string}"#,
            fps = self.fps.load(Ordering::Acquire),
            ft = &self.frame_times,
            gen = &self.wgen_times,
            mesh = &self.wmesh_times,
            lpx = self.local_player_x.load(Ordering::Acquire) as f32 / 10.0,
            lpy = self.local_player_y.load(Ordering::Acquire) as f32 / 10.0,
            lpz = self.local_player_z.load(Ordering::Acquire) as f32 / 10.0,
            heap = FmtBytes(self.heap_usage_bytes.load(Ordering::Acquire)),
            gpuheap = FmtBytes(self.gpu_usage_bytes.load(Ordering::Acquire)),
            custom_string = &self.custom_string.lock()
        )
    }
}

lazy_static! {
    pub static ref DEBUG_DATA: DebugData = DebugData::default();
}

/// A Send+Sync ring buffer for keeping the last N times it took for a specific task to complete
pub struct TimingRing {
    nanoseconds: [AtomicI64; 256],
    write_ptr: AtomicUsize,
}

impl Default for TimingRing {
    fn default() -> Self {
        Self {
            // Safety: Atomic integer types have transparent representation, so this will zero-initialize the array
            nanoseconds: unsafe { MaybeUninit::zeroed().assume_init() },
            write_ptr: Default::default(),
        }
    }
}

impl TimingRing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_ns(&self, ns: i64) {
        let mut wpos = self.write_ptr.fetch_add(1, Ordering::Acquire);
        if wpos >= self.nanoseconds.len() {
            let mask = self.nanoseconds.len() - 1;
            wpos &= mask;
            self.write_ptr.fetch_and(mask, Ordering::Relaxed);
        }
        self.nanoseconds[wpos].store(ns, Ordering::Release);
    }

    pub fn push_sec(&self, sec: f64) {
        self.push_ns((sec * 1.0e9) as i64)
    }

    pub fn min_avg_max_ns(&self) -> (i64, i64, i64) {
        let (mut min, mut avg, mut max) = (i64::max_value(), 0, i64::min_value());
        for val in self.nanoseconds.iter() {
            let val = val.load(Ordering::Relaxed);
            if val < min {
                min = val;
            }
            if val > max {
                max = val;
            }
            avg += val;
        }
        avg /= self.nanoseconds.len() as i64;
        (min, avg, max)
    }
}

impl Display for TimingRing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let (min, avg, max) = self.min_avg_max_ns();
        write!(f, "min {} avg {} max {}", FmtNanoseconds(min), FmtNanoseconds(avg), FmtNanoseconds(max))
    }
}

impl Debug for TimingRing {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

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
