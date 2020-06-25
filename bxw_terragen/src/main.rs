use bxw_terragen::math::*;
use bxw_terragen::supersimplex::SuperSimplex;

fn main() {
    let seed: u64 = if let Some(arg) = std::env::args().nth(1) {
        eprintln!("Using seed {}", &arg);
        arg.parse().unwrap()
    } else {
        0
    };
    let ss = SuperSimplex::new(seed);
    let mut noise_texture = vec![0u8; 256 * 256];
    let scale: f64 = 1.0 / 8.0;
    for y in 0..256 {
        for x in (0..256).step_by(8) {
            let xf = x as f64;
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
            let noise = ss.noise2_wide(pos_x, pos_y);
            let noise_int: WideU8 = ((noise + 1.0) / 2.0 * 256.0).cast();
            noise_int.write_to_slice_unaligned(&mut noise_texture[y * 256 + x..y * 256 + x + 8]);
        }
    }
    image::save_buffer_with_format(
        "noise_image.bmp",
        &noise_texture,
        256,
        256,
        image::ColorType::L8,
        image::ImageFormat::Bmp,
    )
    .unwrap();
}
