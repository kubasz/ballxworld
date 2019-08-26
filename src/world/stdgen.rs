use crate::world::generation::WorldGenerator;
use crate::world::registry::VoxelRegistry;
use crate::world::{VoxelChunkRef, VOXEL_CHUNK_DIM};
use cgmath::prelude::*;
use cgmath::{Vector2, vec2};
use noise::{NoiseFn, Seedable, OpenSimplex};
use std::cell::RefCell;
use thread_local::ThreadLocal;
use rand_xoshiro::Xoshiro256StarStar;
use rand::prelude::*;
use lru::LruCache;
use cgmath::conv::array2;

const SUPERGRID_SIZE: i32 = VOXEL_CHUNK_DIM as i32;
type InCellRng = Xoshiro256StarStar;
type CellPointsT = [CellPoint; 4];

#[derive(Clone, Copy, Debug)]
struct CellPoint {
    pos: Vector2<i32>,
    height: i32,
}

impl CellPoint {
    fn calc(&mut self, cg: &mut CellGen) {
        let pf: Vector2<f64> = self.pos.map(f64::from) / (SUPERGRID_SIZE as f64);
        let h0 = cg.height_map_gen.get(array2(pf/2.0));
        let h1 = cg.height_map_gen.get(array2(pf/8.0));
        let h2 = cg.height_map_gen.get(array2(pf/32.0));
        self.height = ((h0*0.8 + h1*0.15 + h2*0.05 + 0.5).powi(3) * 50.0) as i32;
    }
}

impl Default for CellPoint {
    fn default() -> Self {
        Self {
            pos: vec2(0,0),
            height: 0,
        }
    }
}

struct CellGen {
    seed: u64,
    height_map_gen: OpenSimplex,
    cell_points: LruCache<Vector2<i32>, CellPointsT>,
}

impl CellGen {
    fn new(seed: u64) -> Self {
        let mut s = Self {
            seed: 0,
            height_map_gen: OpenSimplex::new(),
            cell_points: LruCache::new(24),
        };
        s.set_seed(seed);
        s
    }

    fn set_seed(&mut self, seed: u64) {
        let sd32: u32 = (seed ^ (seed >> 32)) as u32;
        self.seed = seed;
        self.height_map_gen = self.height_map_gen.set_seed(sd32);
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
                (SUPERGRID_SIZE/4, SUPERGRID_SIZE/4),
                (3*SUPERGRID_SIZE/4, SUPERGRID_SIZE/4),
                (0, 3*SUPERGRID_SIZE/4),
                (SUPERGRID_SIZE/2, 3*SUPERGRID_SIZE/4)
            ].iter().enumerate() {
            const MOD: i32 = SUPERGRID_SIZE/4;
            let xoff = (r.next_u32() % MOD as u32) as i32 - MOD/2;
            let yoff = (r.next_u32() % MOD as u32) as i32 - MOD/2;
            pts[i].pos = vec2(cell.x * SUPERGRID_SIZE + *x + xoff, cell.y * SUPERGRID_SIZE + *y + yoff);
            pts[i].calc(self);
        }
        self.cell_points.put(cell, pts);
        pts
    }

    fn find_nearest_cell_point(&mut self, pos: Vector2<i32>) -> CellPoint {
        let cell = pos / SUPERGRID_SIZE;
        let (mut nearest, mut ndist) = (CellPoint::default(), SUPERGRID_SIZE*20);
        for cdx in -1..=1 {
            for cdy in -1..=1 {
                for p in self.get_cell_points(cell + vec2(cdx, cdy)).iter() {
                    let dp = p.pos - pos;
                    let dist = dp.map(|c| c*c).sum();
                    if dist < ndist {
                        nearest = *p;
                        ndist = dist;
                    }
                }
            }
        }
        nearest
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

impl StdGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            cell_gen: ThreadLocal::default(),
        }
    }
}

impl Default for StdGenerator {
    fn default() -> Self {
        Self::new(0)
    }
}

impl WorldGenerator for StdGenerator {
    fn generate_chunk(&self, cref: VoxelChunkRef, registry: &VoxelRegistry) {
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
        let i_diamond = registry
            .get_definition_from_name("core:diamond_ore")
            .expect("No standard diamond ore block definition found")
            .id;

        let chunkarc = cref.chunk.upgrade();
        if chunkarc.is_none() {
            return;
        }
        let chunkarc = chunkarc.unwrap();
        let mut chunk = chunkarc.write().unwrap();

        let mut cellgen = self.cell_gen.get_or(|| Box::new(RefCell::new(CellGen::new(self.seed)))).borrow_mut();

        for (vidx, vox) in chunk.data.iter_mut().enumerate() {
            let vcd = VOXEL_CHUNK_DIM as i32;
            let xc = (vidx % VOXEL_CHUNK_DIM) as i32;
            let yc = ((vidx / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as i32;
            let zc = ((vidx / VOXEL_CHUNK_DIM / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as i32;
            let x = (cref.position.x * vcd) as i32 + xc;
            let y = (cref.position.y * vcd) as i32 + yc;
            let z = (cref.position.z * vcd) as i32 + zc;
            let ncell = cellgen.find_nearest_cell_point(vec2(x,z));
            if x == ncell.pos.x && z == ncell.pos.y {
                vox.id = i_air;
            } else if y == ncell.height {
                vox.id = i_grass;
            } else if y < ncell.height - 5 {
                vox.id = i_stone;
            } else if y < ncell.height {
                vox.id = i_dirt;
            } else {
                vox.id = i_air;
            }
        }
    }
}
