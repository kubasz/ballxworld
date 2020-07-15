use bxw_terragen::continent;
use bxw_terragen::math::*;
use bxw_util::itertools::*;
use std::time::Instant;

fn main() {
    let seed: u64 = if let Some(arg) = std::env::args().nth(1) {
        eprintln!("Using seed {}", &arg);
        arg.parse().unwrap()
    } else {
        0
    };
    //let ss = SuperSimplex::new(seed);
    let mut out_texture = vec![0u8; 3 * 1024 * 1024];
    let beforegen = Instant::now();
    let cont_set = continent::ContinentGenSettings::with_seed(seed);
    let cont = continent::generate_continent_tile(&cont_set, vec2(0, 0));
    let scale: f64 = cont_set.continent_size as f64 / 1024.0;
    let before = Instant::now();
    for (ytile, xtile) in iproduct!(
        (0..1024).into_iter().step_by(16),
        (0..1024).into_iter().step_by(16)
    ) {
        for (y, x) in iproduct!(ytile..ytile + 16, xtile..xtile + 16) {
            let xf = x as f64 * scale;
            let yf = y as f64 * scale;
            let nearest_idx = cont
                .biome_point_tree
                .nearest_neighbor(&[xf, yf])
                .unwrap()
                .data;
            out_texture[3 * (y * 1024 + x)] = cont.biome_points[nearest_idx].debug_shade.x;
            out_texture[3 * (y * 1024 + x) + 1] = cont.biome_points[nearest_idx].debug_shade.y;
            out_texture[3 * (y * 1024 + x) + 2] = cont.biome_points[nearest_idx].debug_shade.z;
        }
    }
    let after = Instant::now();
    eprintln!("Gen time: {} us", (before - beforegen).as_micros());
    eprintln!("Draw time: {} us", (after - before).as_micros());
    image::save_buffer_with_format(
        "noise_image.png",
        &out_texture,
        1024,
        1024,
        image::ColorType::Rgb8,
        image::ImageFormat::Png,
    )
    .unwrap();
}
