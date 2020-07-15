use crate::math::*;
use crate::supersimplex::SuperSimplex;
use bxw_util::rand;
use bxw_util::rand::prelude::*;
use bxw_util::rand_distr;
use bxw_util::rand_xoshiro::{Seed512, Xoshiro512StarStar};
use bxw_util::rstar;
use bxw_util::rstar::RTree;
use bxw_util::smallvec::SmallVec;
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
    pub biome_points: Vec<BiomeCentroid>,
    pub biome_point_tree: OptRTree<RTreeIdxPoint2>,
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

pub type ValueAndGradient2<T> = (T, Vector2<T>);

#[derive(Clone, Debug, Default)]
pub struct BiomeCentroid {
    pub position: Vector2<f64>,
    pub adjacent: SmallVec<[usize; 16]>,
    pub adjacent_distance: SmallVec<[f64; 16]>,
    pub debug_shade: Vector3<u8>,
    pub on_edge: bool,
    pub height: ValueAndGradient2<f64>,
    pub air_moisture: ValueAndGradient2<f64>,
    pub ground_moisture: ValueAndGradient2<f64>,
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
type RTreeIdxPoint2 = rstar::primitives::PointWithData<usize, [f64; 2]>;
type WipBiomeData<'a> = (
    ContinentTilePosition,
    &'a mut [BiomeCentroid],
    &'a OptRTree<RTreeIdxPoint2>,
);

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

    // generate biome points by poisson disc sampling
    let (biome_point_tree, mut biome_points) = generate_biome_points(&hasher, settings);
    // generate initial heights
    seed_biome_heights(
        settings,
        &hasher,
        (tile_position, &mut biome_points, &biome_point_tree),
    );

    // assign debug shades
    {
        for biome in biome_points.iter_mut() {
            let h = biome.height.0;
            if h < 0.0 {
                let hnorm = (h / SEABED).min(1.0);
                biome.debug_shade = vec3(0, 0, 255 - (hnorm * 255.0) as u8);
            } else {
                let hnorm = (h / TALLESTMOUNT).min(1.0);
                let h8 = (hnorm * 255.0) as u8;
                biome.debug_shade = vec3(h8, 220, h8);
            }
        }
    }

    ContinentTile {
        position: tile_position,
        biome_points,
        biome_point_tree,
    }
}

fn generate_biome_points(
    hasher: &Hasher,
    settings: &ContinentGenSettings,
) -> (OptRTree<RTreeIdxPoint2>, Vec<BiomeCentroid>) {
    let continent_size = settings.continent_size as f64;
    let biome_avg_distance = settings.biome_avg_distance;
    let rngseed = {
        let mut h = hasher.clone();
        h.update(b"biomepts");
        let mut x = h.finalize_xof();
        let mut s = Seed512::default();
        x.fill(&mut s.0);
        s
    };
    let mut rng = Xoshiro512StarStar::from_seed(rngseed);
    let biome_r_dist = rand_distr::Normal::new(biome_avg_distance, biome_avg_distance / 4.0)
        .expect("Invalid average biome distance");
    let mut disc_points: Vec<BiomeCentroid> = poisson_disc_sampling(
        continent_size,
        0.5 * biome_avg_distance,
        biome_r_dist,
        &mut rng,
    )
    .iter()
    .map(|p| BiomeCentroid {
        position: Vector2::from(*p.position()),
        ..Default::default()
    })
    .collect();
    let tree_load_vec: Vec<_> = disc_points
        .iter()
        .enumerate()
        .map(|(i, p)| RTreeIdxPoint2::new(i, p.position.into()))
        .collect();
    let point_tree = OptRTree::bulk_load_with_params(tree_load_vec);
    let neighbor_max_distance = 1.1 * biome_avg_distance;
    for i in 0..disc_points.len() {
        if disc_points[i]
            .position
            .iter()
            .any(|&c| c < biome_avg_distance * 2.0 || c > continent_size - biome_avg_distance * 2.0)
        {
            disc_points[i].on_edge = true;
        }
        for nn in point_tree.locate_within_distance(
            disc_points[i].position.into(),
            neighbor_max_distance * neighbor_max_distance,
        ) {
            if nn.data != i {
                disc_points[i].adjacent.push(nn.data);
                let dpos = disc_points[nn.data].position - disc_points[i].position;
                let dist2: f64 = dpos.dot(&dpos);
                disc_points[i].adjacent_distance.push(dist2.sqrt());
            }
        }
    }
    (point_tree, disc_points)
}

const SEABED: f64 = -512.0;
const LANDBED: f64 = 10.0;
const TALLESTMOUNT: f64 = 1024.0;
const MOUNT_PROBABILITY: f64 = 0.25;
const DEPRESSED_FRACTION: f64 = 0.05;
const FLATTEN_ITERS: u32 = 2;
const FLATTEN_SELF_WEIGHT: f64 = 24.0;
const FLATTEN_MOUNTAIN_WEIGHT: f64 = 3.0;
const COAST_DISPLACEMENT: f64 = 8.0;

fn update_gradients<F>(data: &mut [BiomeCentroid], fieldmap: F)
where
    F: Fn(&mut BiomeCentroid) -> &mut ValueAndGradient2<f64>,
{
    for i in 0..data.len() {
        let v0 = fieldmap(&mut data[i]).0;
        let mut total_grad = vec2(0.0, 0.0);
        let mut weights = 0.0;
        let adjacent = data[i].adjacent.clone();
        for (nidx, &n) in adjacent.iter().enumerate() {
            let r = data[i].adjacent_distance[nidx];
            let v1 = fieldmap(&mut data[n]).0;
            let rvec = data[n].position - data[i].position;
            let grad_weight = 1.0 / r;
            let grad = (rvec / r) * (v1 - v0);
            total_grad += grad * grad_weight;
            weights += grad_weight;
        }
        weights = weights.max(1.0e-9);
        total_grad /= weights;
        fieldmap(&mut data[i]).1 = total_grad;
    }
}

fn seed_biome_heights(settings: &ContinentGenSettings, hasher: &Hasher, data: WipBiomeData) {
    let continent_size = settings.continent_size as f64;
    let rngseed = {
        let mut h = hasher.clone();
        h.update(b"biomeheights");
        let mut x = h.finalize_xof();
        let mut s = Seed512::default();
        x.fill(&mut s.0);
        s
    };
    let mut rng = Xoshiro512StarStar::from_seed(rngseed);
    let noise_seed = rng.next_u64();
    let displacement_noise = SuperSimplex::new(noise_seed);
    let u01distr = rand_distr::Uniform::new_inclusive(0.0, 1.0);
    let underwater_distr = rand_distr::Normal::new(SEABED / 3.0, SEABED.abs() / 4.0).unwrap();
    let abovewater_distr = rand_distr::Exp::new(MOUNT_PROBABILITY).unwrap();
    let (_tile_pos, biome_points, _biome_tree) = data;
    // initial random distribution
    for ipoints in (0..biome_points.len()).step_by(8) {
        let ipoints8 = usize::min(ipoints + 8, biome_points.len());
        let mut points_x = WideF64::splat(0.0);
        let mut points_y = WideF64::splat(0.0);
        for (lane, point) in biome_points[ipoints..ipoints8].iter().enumerate() {
            points_x = points_x.replace(lane, point.position.x);
            points_y = points_y.replace(lane, point.position.y);
        }
        let points = vec2(points_x, points_y);
        let npoints = points / WideF64::splat(settings.biome_avg_distance * 12.0);
        let noff = WideF64::splat(2.7 * continent_size);
        let npoints2 = npoints + vec2(noff, noff);
        let points_displace_xw = displacement_noise.vnoise2_wide(npoints);
        let points_displace_yw = displacement_noise.vnoise2_wide(npoints2);
        let points_displacew = vec2(points_displace_xw, points_displace_yw)
            * WideF64::splat(settings.biome_avg_distance * COAST_DISPLACEMENT);
        let mut points_displace_x = [0.0; 8];
        let mut points_displace_y = [0.0; 8];
        points_displacew
            .x
            .write_to_slice_unaligned(&mut points_displace_x);
        points_displacew
            .y
            .write_to_slice_unaligned(&mut points_displace_y);
        for (lane, point) in biome_points[ipoints..ipoints8].iter_mut().enumerate() {
            let dispos = point.position + vec2(points_displace_x[lane], points_displace_y[lane]);
            let edge_distance_x = f64::min(dispos.x, continent_size - dispos.x);
            let edge_distance_y = f64::min(dispos.y, continent_size - dispos.y);
            let realedge_distance_x = f64::min(point.position.x, continent_size - point.position.x);
            let realedge_distance_y = f64::min(point.position.y, continent_size - point.position.y);
            let edge_distance = f64::min(edge_distance_x, edge_distance_y);
            let realedge_distance = f64::min(realedge_distance_x, realedge_distance_y);
            let norm_edge_distance = edge_distance / (15.0 * settings.biome_avg_distance);
            let norm_realedge_distance = realedge_distance / (10.0 * settings.biome_avg_distance);
            let rfactor = smoothstep(norm_edge_distance); // close to 0 for edge cells

            let mut under_sea_level = u01distr.sample(&mut rng) >= (rfactor - DEPRESSED_FRACTION);
            if norm_realedge_distance < 0.2 {
                under_sea_level = true;
            }
            // vaguely based on https://ngdc.noaa.gov/mgg/global/etopo1_surface_histogram.html
            let distributed_height = if under_sea_level {
                let sample = underwater_distr.sample(&mut rng);
                if norm_realedge_distance < 0.2 {
                    sample.min(SEABED / 4.0)
                } else {
                    sample
                }
            } else {
                (1.0 - abovewater_distr.sample(&mut rng).min(1.0)) * TALLESTMOUNT + LANDBED
            };
            point.height.0 = distributed_height;
        }
    }
    // averaging
    {
        let mut new_heights: Vec<f64> = Vec::new();
        new_heights.resize(biome_points.len(), 0.0);
        for _iter in 0..FLATTEN_ITERS {
            for (i, point) in biome_points.iter().enumerate() {
                let mut avg = point.height.0 * FLATTEN_SELF_WEIGHT;
                let mut avgn = FLATTEN_SELF_WEIGHT + point.adjacent.len() as f64;
                for &ineighbor in point.adjacent.iter() {
                    let oh = biome_points[ineighbor].height.0;
                    if oh >= TALLESTMOUNT / 2.0 {
                        avg += oh * FLATTEN_MOUNTAIN_WEIGHT;
                        avgn += FLATTEN_MOUNTAIN_WEIGHT - 1.0;
                    } else {
                        avg += oh;
                    }
                }
                new_heights[i] = avg / avgn;
            }
            for (i, &h) in new_heights.iter().enumerate() {
                biome_points[i].height.0 = h;
            }
        }
    }
    // update gradients
    update_gradients(biome_points, |b| &mut b.height);
}
