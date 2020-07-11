use bxw_terragen::math::*;
use bxw_terragen::continent;

fn main() {
    let seed: u64 = if let Some(arg) = std::env::args().nth(1) {
        eprintln!("Using seed {}", &arg);
        arg.parse().unwrap()
    } else {
        0
    };
    //let ss = SuperSimplex::new(seed);
    //let sswarp = SuperSimplex::new(seed + 7);
    let mut out_texture = vec![0u8; 1024 * 1024];
    let cont_set = continent::ContinentGenSettings::with_seed(seed);
    let cont = continent::generate_continent_tile(&cont_set, vec2(0, 0));
    let scale: f64 = 16.0;
    for y in 0..1024 {
        for x in 0..1024 {
            let xf = x as f64 * scale;
            let yf = y as f64 * scale;
            out_texture[y*1024 + x] = cont.biome_points.nearest_neighbor(&[xf, yf]).unwrap().random_shade;
            /*let xf = x as f64;
            let pos_x = WideF64::new(
                xf,
                xf + 1.0,
                xf + 2.0,
                xf + 3.0,
                xf + 4.0,
                xf + 5.0,
                xf + 6.0,
                xf + 7.0,
            ) * scale;
            let pos_y = WideF64::splat(y as f64) * scale;
            let pos = vec2(pos_x, pos_y);
            let hnoise = sswarp.vnoise2_wide(pos * WideF64::splat(1.0 / 8.0)) * 0.5 + 0.5;
            let noise = ss.vnoise2_wide(pos) * 0.5 + 0.5;
            let cnoise = noise.powf(hnoise + 1.0);
            let noise_int: WideU8 = (cnoise * 256.0).cast();
            noise_int.write_to_slice_unaligned(&mut out_texture[y * 1024 + x..y * 1024 + x + 8]);*/
        }
    }
    image::save_buffer_with_format(
        "noise_image.bmp",
        &out_texture,
        1024,
        1024,
        image::ColorType::L8,
        image::ImageFormat::Bmp,
    )
    .unwrap();
}
