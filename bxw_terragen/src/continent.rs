use crate::math::*;
use bxw_util::rand;
use bxw_util::rand::prelude::*;
use bxw_util::rand_distr;
use bxw_util::rand_xoshiro::{Seed512, Xoshiro512StarStar};
use bxw_util::rstar;
use bxw_util::rstar::RTree;
use bxw_world::*;

#[derive(Clone, Debug)]
pub struct ContinentGenSettings {
    pub seed: u64,
    pub biome_avg_distance: f64,
    pub continent_size: u32,
}

impl Default for ContinentGenSettings {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinentGenSettings {
    pub fn new() -> Self {
        ContinentGenSettings {
            seed: 0x13372137CAFE,
            biome_avg_distance: 768.0,
            continent_size: 32000,
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            seed,
            ..Self::new()
        }
    }

    pub fn base_hasher(&self) -> Hasher {
        seeded_hasher(self.seed, "bxw_terragen::continent")
    }
}

pub type ContinentTilePosition = Vector2<i32>;
pub type ContinentBlockInnerPosition = Vector2<i32>;

#[derive(Clone)]
pub struct ContinentTile {
    pub position: ContinentTilePosition,
    pub biome_points: OptRTree<BiomeCentroid>,
}

pub fn continent_tilepos_from_blockpos(
    settings: &ContinentGenSettings,
    bpos: BlockPosition,
) -> ContinentTilePosition {
    let sz = settings.continent_size as i32;
    bpos.xz().map(|p| p.div_floor(&sz))
}

pub fn continent_tile_inner_pos(
    settings: &ContinentGenSettings,
    bpos: BlockPosition,
) -> ContinentBlockInnerPosition {
    let sz = settings.continent_size as i32;
    bpos.xz().map(|p| p.rem_floor(&sz))
}

#[derive(Clone, Debug, Default)]
pub struct BiomeCentroid {
    pub position: Vector2<f64>,
    pub random_shade: u8,
}

impl rstar::RTreeObject for BiomeCentroid {
    type Envelope = rstar::AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_point([self.position.x, self.position.y])
    }
}

impl rstar::PointDistance for BiomeCentroid {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let p = vec2(point[0], point[1]);
        (p - self.position).magnitude_squared()
    }
}

#[derive(Copy, Clone)]
pub struct OptRTreeParams;

impl rstar::RTreeParams for OptRTreeParams {
    const MIN_SIZE: usize = 8;
    const MAX_SIZE: usize = 16;
    const REINSERTION_COUNT: usize = 5;
    type DefaultInsertionStrategy = rstar::RStarInsertionStrategy;
}

pub type OptRTree<T> = rstar::RTree<T, OptRTreeParams>;
type RTreePoint2 = rstar::primitives::PointWithData<(), [f64; 2]>;

/// Based on https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf
fn poisson_disc_sampling<Dist: Distribution<f64> + Copy>(
    continent_size: f64,
    min_r: f64,
    r_dist: Dist,
    rng: &mut Xoshiro512StarStar,
) -> RTree<RTreePoint2> {
    let min_r2 = min_r * min_r;
    let a_dist = rand::distributions::Uniform::new(0.0, 2.0 * std::f64::consts::PI);
    let start_ox = rng.sample(r_dist);
    let start_oy = rng.sample(r_dist);
    let start_p = vec2(continent_size / 2.0, continent_size / 2.0) + vec2(start_ox, start_oy);
    let start_p = start_p.map(|p| p.max(1.0).min(continent_size - 1.0));
    let mut points: RTree<RTreePoint2> = RTree::new();
    let mut active_set: Vec<Vector2<f64>> = Vec::with_capacity(32);
    points.insert(RTreePoint2::new((), start_p.into()));
    active_set.push(start_p);
    while !active_set.is_empty() {
        let active_dist = rand::distributions::Uniform::new(0, active_set.len());
        let active_idx = rng.sample(active_dist);
        let active_point = active_set[active_idx];
        let mut found = false;
        for _k in 0..20 {
            let off_r = rng.sample(r_dist);
            let off_a = rng.sample(a_dist);
            let (sin_a, cos_a) = off_a.sin_cos();
            let off = vec2(off_r * cos_a, off_r * sin_a);
            let newpoint = active_point + off;
            if points
                .locate_within_distance(newpoint.into(), min_r2)
                .next()
                .is_some()
                || newpoint
                    .iter()
                    .any(|&c| c < 1.0 || c >= continent_size - 1.0)
            {
                continue;
            } else {
                found = true;
                active_set.push(newpoint);
                points.insert(RTreePoint2::new((), newpoint.into()));
            }
        }
        if !found {
            active_set.swap_remove(active_idx);
        }
    }
    points
}

pub fn generate_continent_tile(
    settings: &ContinentGenSettings,
    tile_position: ContinentTilePosition,
) -> ContinentTile {
    let hasher = {
        let mut h = settings.base_hasher();
        h.update(bytes_of(&tile_position.x));
        h.update(bytes_of(&tile_position.y));
        h
    };
    let continent_size = settings.continent_size as f64;
    // generate biome points by poisson disc sampling
    let mut biome_points: OptRTree<BiomeCentroid> = {
        let rngseed = {
            let mut h = hasher.clone();
            h.update(b"biomepts");
            let mut x = h.finalize_xof();
            let mut s = Seed512::default();
            x.fill(&mut s.0);
            s
        };
        let mut rng = Xoshiro512StarStar::from_seed(rngseed);
        let biome_r_dist = rand_distr::Normal::new(
            settings.biome_avg_distance,
            settings.biome_avg_distance / 4.0,
        )
        .expect("Invalid average biome distance");
        let disc_points: Vec<BiomeCentroid> = poisson_disc_sampling(
            continent_size,
            0.5 * settings.biome_avg_distance,
            biome_r_dist,
            &mut rng,
        )
        .iter()
        .map(|p| BiomeCentroid {
            position: Vector2::from(*p.position()),
            ..Default::default()
        })
        .collect();
        OptRTree::bulk_load_with_params(disc_points)
    };

    {
        let rngseed = {
            let mut h = hasher.clone();
            h.update(b"biomedbg");
            let mut x = h.finalize_xof();
            let mut s = Seed512::default();
            x.fill(&mut s.0);
            s
        };
        let mut rng = Xoshiro512StarStar::from_seed(rngseed);
        for biome in biome_points.iter_mut() {
            biome.random_shade = (rng.next_u32() & 0xFF) as u8;
        }
    }

    ContinentTile {
        position: tile_position,
        biome_points,
    }
}
