use crate::math::*;
use crate::world::registry::VoxelRegistry;
use crate::world::{UncompressedChunk, CHUNK_DIM};
use lru::LruCache;
use noise::{NoiseFn, Seedable, SuperSimplex};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;
use std::cell::RefCell;
use std::mem::MaybeUninit;
use thread_local::ThreadLocal;

const GLOBAL_SCALE_MOD: f64 = 10.0;
const GLOBAL_BIOME_SCALE: f64 = 256.0;
const SUPERGRID_SIZE: i32 = 4 * CHUNK_DIM as i32;
type InCellRng = Xoshiro256StarStar;
type CellPointsT = [CellPoint; 4];

fn distance2(a: Vector2<i32>, b: Vector2<i32>) -> i32 {
    (a - b).iter().map(|c| c * c).sum()
}

#[derive(Clone, Copy, Debug)]
struct CellPoint {
    pos: Vector2<i32>,
    /// 0..1
    elevation_class: f64,
}

impl CellPoint {
    fn calc(&mut self, cg: &mut CellGen) {
        self.elevation_class = cg.elevation_noise(self.pos);
    }
}

impl Default for CellPoint {
    fn default() -> Self {
        Self {
            pos: vec2(0, 0),
            elevation_class: 0.0,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VPElevation {
    LowLand,
    Hill,
    Mountain,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum VPMoisture {
    Deadland,
    Desert,
    LowMoist,
    MedMoist,
    HiMoist,
}

impl Default for VPElevation {
    fn default() -> Self {
        VPElevation::LowLand
    }
}

impl Default for VPMoisture {
    fn default() -> Self {
        VPMoisture::MedMoist
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct VoxelParams {
    height: i32,
    elevation: VPElevation,
    moisture: VPMoisture,
}

struct CellGen {
    seed: u64,
    height_map_gen: [SuperSimplex; 5],
    elevation_map_gen: SuperSimplex,
    moisture_map_gen: SuperSimplex,
    cell_points: LruCache<Vector2<i32>, CellPointsT>,
}

impl CellGen {
    fn new(seed: u64) -> Self {
        let mut s = Self {
            seed: 0,
            height_map_gen: [SuperSimplex::new(); 5],
            elevation_map_gen: SuperSimplex::new(),
            moisture_map_gen: SuperSimplex::new(),
            cell_points: LruCache::new(64),
        };
        s.set_seed(seed);
        s
    }

    fn set_seed(&mut self, seed: u64) {
        use rand::prelude::*;
        use rand_xoshiro::SplitMix64;
        let mut sdgen = SplitMix64::seed_from_u64(seed);
        self.seed = seed;
        for hmg in self.height_map_gen.iter_mut() {
            *hmg = hmg.set_seed(sdgen.next_u32());
        }
        self.elevation_map_gen = self.elevation_map_gen.set_seed(sdgen.next_u32());
        self.moisture_map_gen = self.moisture_map_gen.set_seed(sdgen.next_u32());
        self.cell_points.clear();
    }

    #[inline(always)]
    fn get_seed(&self, cell: Vector2<i32>) -> u64 {
        self.seed ^ (((cell.x as u64) << 32) | (cell.y as u64 & 0xFFFF_FFFF))
    }

    fn get_cell_points(&mut self, cell: Vector2<i32>) -> CellPointsT {
        if let Some(cp) = self.cell_points.get(&cell) {
            return *cp;
        }
        let mut pts: CellPointsT = Default::default();
        let mut r = InCellRng::seed_from_u64(self.get_seed(cell));
        for (i, (x, y)) in [
            (SUPERGRID_SIZE / 4, SUPERGRID_SIZE / 4),
            (3 * SUPERGRID_SIZE / 4, SUPERGRID_SIZE / 4),
            (0, 3 * SUPERGRID_SIZE / 4),
            (SUPERGRID_SIZE / 2, 3 * SUPERGRID_SIZE / 4),
        ]
        .iter()
        .enumerate()
        {
            const MOD: i32 = SUPERGRID_SIZE / 4;
            let xoff = (r.next_u32() % MOD as u32) as i32 - MOD / 2;
            let yoff = (r.next_u32() % MOD as u32) as i32 - MOD / 2;
            pts[i].pos = vec2(
                cell.x * SUPERGRID_SIZE + *x + xoff,
                cell.y * SUPERGRID_SIZE + *y + yoff,
            );
            pts[i].calc(self);
        }
        self.cell_points.put(cell, pts);
        pts
    }

    fn find_nearest_cell_points(&mut self, pos: Vector2<i32>, num: usize) -> Vec<(i32, CellPoint)> {
        let cell = pos / SUPERGRID_SIZE;

        let mut pts = Vec::with_capacity(6);
        for cdx in -1..=1 {
            for cdy in -1..=1 {
                for p in self.get_cell_points(cell + vec2(cdx, cdy)).iter() {
                    let dist = distance2(p.pos, pos);
                    pts.push((dist, *p));
                }
            }
        }
        pts.sort_by(|a, b| a.0.cmp(&b.0));
        pts.resize_with(num, || (0, CellPoint::default()));
        pts
    }

    fn elevation_noise(&self, pos: Vector2<i32>) -> f64 {
        let nf = |p: Vector2<f64>| (self.elevation_map_gen.get([p.x, p.y]) + 1.0) / 2.0;
        let scale_factor = GLOBAL_BIOME_SCALE * GLOBAL_SCALE_MOD;
        nf(pos.map(f64::from) / scale_factor)
    }

    /// 0..=1 height noise
    fn h_n(&self, i: usize, p: Vector2<f64>) -> f64 {
        (self.height_map_gen[i].get([p.x, p.y]) + 1.0) / 2.0
    }

    /// 0..=1 ridge noise
    fn h_rn(&self, i: usize, p: Vector2<f64>) -> f64 {
        2.0 * (0.5 - (0.5 - self.h_n(i, p)).abs())
    }

    fn plains_height_noise(&self, pos: Vector2<i32>) -> f64 {
        let scale_factor = GLOBAL_SCALE_MOD * 120.0;
        let p = pos.map(f64::from) / scale_factor;

        (0.75 * self.h_n(0, p) + 0.25 * self.h_n(1, p * 2.0)) * 5.0 + 10.0
    }

    fn hills_height_noise(&self, pos: Vector2<i32>) -> f64 {
        let scale_factor = GLOBAL_SCALE_MOD * 160.0;
        let p = pos.map(f64::from) / scale_factor;

        (0.60 * self.h_n(0, p) + 0.25 * self.h_n(1, p * 1.5) + 0.15 * self.h_n(2, p * 3.0)) * 30.0
            + 15.0
    }

    fn mountains_height_noise(&self, pos: Vector2<i32>) -> f64 {
        let scale_factor = GLOBAL_SCALE_MOD * 200.0;
        let p = pos.map(f64::from) / scale_factor;

        let h0 = 0.50 * self.h_rn(0, p);
        let h01 = 0.25 * self.h_rn(1, p * 2.0) + h0;

        (h01 + (h01 / 0.75) * 0.15 * self.h_n(2, p * 5.0)
            + (h01 / 0.75) * 0.05 * self.h_rn(3, p * 9.0))
            * 750.0
            + 40.0
    }

    fn calc_voxel_params(&mut self, pos: Vector2<i32>) -> VoxelParams {
        let cp = self.find_nearest_cell_points(pos, 1)[0].1;
        let cec = cp.elevation_class;
        let ec = self.elevation_noise(pos);

        let height: f64;
        if ec < 0.4 {
            let ph = self.plains_height_noise(pos);
            height = ph;
        } else if ec < 0.5 {
            let nlin = (ec - 0.4) / 0.1;
            let olin = 1.0 - nlin;
            let ph = self.plains_height_noise(pos);
            let hh = self.hills_height_noise(pos);
            height = olin * ph + nlin * hh;
        } else if ec < 0.7 {
            let hh = self.hills_height_noise(pos);
            height = hh;
        } else if ec < 0.8 {
            let nlin = (ec - 0.7) / 0.1;
            let olin = 1.0 - nlin;
            let hh = self.hills_height_noise(pos);
            let mh = self.mountains_height_noise(pos);
            height = olin * hh + nlin * mh;
        } else {
            let mh = self.mountains_height_noise(pos);
            height = mh;
        }
        let elevation: VPElevation;
        if cec < 0.4 {
            elevation = VPElevation::LowLand;
        } else if cec < 0.5 {
            let p = pos.map(f64::from);
            let bnoise = self.height_map_gen.last().unwrap().get([p.x, p.y]);
            if bnoise < 0.0 {
                elevation = VPElevation::LowLand;
            } else {
                elevation = VPElevation::Hill;
            }
        } else if cec < 0.7 {
            elevation = VPElevation::Hill;
        } else if cec < 0.8 {
            let p = pos.map(f64::from);
            let bnoise = self.height_map_gen.last().unwrap().get([p.x, p.y]);
            if bnoise < 0.0 {
                elevation = VPElevation::Hill;
            } else {
                elevation = VPElevation::Mountain;
            }
        } else {
            elevation = VPElevation::Mountain;
        }

        let moisture = VPMoisture::MedMoist;

        VoxelParams {
            height: height.round() as i32,
            elevation,
            moisture,
        }
    }
}

impl Default for CellGen {
    fn default() -> Self {
        Self::new(0)
    }
}

pub struct StdGenerator {
    seed: u64,
    cell_gen: ThreadLocal<RefCell<CellGen>>,
}

impl Default for StdGenerator {
    fn default() -> Self {
        Self::new(0)
    }
}

impl StdGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            cell_gen: ThreadLocal::default(),
        }
    }

    pub fn generate_chunk(&self, chunk: &mut UncompressedChunk, registry: &VoxelRegistry) {
        let i_air = registry
            .get_definition_from_name("core:void")
            .expect("No standard air block definition found")
            .id;
        let i_grass = registry
            .get_definition_from_name("core:grass")
            .expect("No standard grass block definition found")
            .id;
        let i_dirt = registry
            .get_definition_from_name("core:dirt")
            .expect("No standard dirt block definition found")
            .id;
        let i_stone = registry
            .get_definition_from_name("core:stone")
            .expect("No standard stone block definition found")
            .id;
        let i_snow_grass = registry
            .get_definition_from_name("core:snow_grass")
            .expect("No standard snow grass block definition found")
            .id;

        let mut cellgen = self
            .cell_gen
            .get_or(|| Box::new(RefCell::new(CellGen::new(self.seed))))
            .borrow_mut();

        const VCD: i32 = CHUNK_DIM as i32;

        let vparams: [VoxelParams; CHUNK_DIM * CHUNK_DIM] = {
            let mut vparams: [MaybeUninit<VoxelParams>; CHUNK_DIM * CHUNK_DIM] =
                unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            for (i, v) in vparams[..].iter_mut().enumerate() {
                let x = (i % CHUNK_DIM) as i32 + (chunk.position.x * VCD);
                let z = ((i / CHUNK_DIM) % CHUNK_DIM) as i32 + (chunk.position.z * VCD);
                let p = cellgen.calc_voxel_params(vec2(x, z));
                unsafe {
                    std::ptr::write(v.as_mut_ptr(), p);
                }
            }
            unsafe { std::mem::transmute(vparams) }
        };

        for (vidx, vox) in chunk.blocks_yzx.iter_mut().enumerate() {
            let xc = (vidx % CHUNK_DIM) as i32;
            let zc = ((vidx / CHUNK_DIM) % CHUNK_DIM) as i32;
            let yc = ((vidx / CHUNK_DIM / CHUNK_DIM) % CHUNK_DIM) as i32;
            //let x = (chunk.position.x * VCD) as i32 + xc;
            let y = (chunk.position.y * VCD) as i32 + yc;
            //let z = (chunk.position.z * VCD) as i32 + zc;
            let vp = vparams[(xc + zc * (CHUNK_DIM as i32)) as usize];

            let h = vp.height;
            //
            if y == h {
                if vp.elevation == VPElevation::Mountain && y > 80 {
                    vox.id = i_snow_grass;
                } else {
                    vox.id = i_grass;
                }
            } else if y < h - 5 {
                vox.id = i_stone;
            } else if y < h {
                vox.id = i_dirt;
            } else {
                vox.id = i_air;
            }
        }
    }
}
