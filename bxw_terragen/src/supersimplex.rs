//! Based on K.jpg's OpenSimplex 2, smooth variant ("SuperSimplex")
//! From: https://github.com/KdotJPG/OpenSimplex2/blob/master/java/OpenSimplex2S.java
//! Versioned: https://github.com/KdotJPG/OpenSimplex2/blob/946fa5cc4b0f6c6f88d90db47becae8b8b79dbc0/java/OpenSimplex2S.java

use crate::math::*;

pub const SIMPLEX2_NATURAL_POINT_DISTANCE: f64 = 0.33;
const PSIZE: usize = 2048;
const PMASK: usize = 2047;

#[derive(Clone)]
pub struct SuperSimplex {
    seed: u64,
    perm: Vec<u16>,
    perm_gx: Vec<f64>,
    perm_gy: Vec<f64>,
}

impl SuperSimplex {
    pub fn new(mut seed: u64) -> Self {
        let orig_seed = seed;
        let mut perm = vec![0; PSIZE];
        let mut perm_gx = vec![0.0; PSIZE];
        let mut perm_gy = vec![0.0; PSIZE];
        let mut source: Vec<u16> = (0..PSIZE as u16).collect();
        let lut = lookup_2d::tables();
        for i in (0..PSIZE).rev() {
            seed = seed
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            let r = (seed.wrapping_add(31) % (i as u64 + 1)) as usize;
            perm[i] = source[r];
            perm_gx[i] = lut.grads_dx[perm[i] as usize];
            perm_gy[i] = lut.grads_dy[perm[i] as usize];
            source[r] = source[i];
        }
        Self {
            seed: orig_seed,
            perm,
            perm_gx,
            perm_gy,
        }
    }

    pub fn vnoise2_wide(&self, p: Vector2<WideF64>) -> WideF64 {
        self.noise2_wide(p.x, p.y)
    }

    pub fn noise2_wide(&self, x: WideF64, y: WideF64) -> WideF64 {
        let s = (x + y) * 0.366025403784439f64;
        let xs = x + s;
        let ys = y + s;
        self.noise2_base_wide(xs, ys)
    }

    pub fn noise2_base_wide(&self, xs: WideF64, ys: WideF64) -> WideF64 {
        let lut = lookup_2d::tables();
        let permptr = WideCptr::splat(self.perm.as_ptr());
        let permxptr = WideCptr::splat(self.perm_gx.as_ptr());
        let permyptr = WideCptr::splat(self.perm_gy.as_ptr());
        // base points and offsets
        let xsb: WideI32 = widefloor_f64_i32(xs);
        let ysb: WideI32 = widefloor_f64_i32(ys);
        let xsi: WideF64 = xs - Cast::<WideF64>::cast(xsb);
        let ysi: WideF64 = ys - Cast::<WideF64>::cast(ysb);
        // index to point list
        let a: WideI32 = (xsi + ysi).cast();
        let af: WideF64 = a.cast();
        let index: WideI32 = (a << 2)
            | Cast::<WideI32>::cast(xsi - ysi / 2.0 + 1.0 - af / 2.0) << 3
            | Cast::<WideI32>::cast(ysi - xsi / 2.0 + 1.0 - af / 2.0) << 4;

        let ssi: WideF64 = (xsi + ysi) * -0.211324865405187f64;
        let xi = xsi + ssi;
        let yi = ysi + ssi;

        let index: WideUsize = index.cast();
        let mut value = WideF64::splat(0.0);
        for i in 0..4usize {
            let (cx, cy, cdx, cdy) = unsafe { lut.read_points(index + i) };

            let dx: WideF64 = xi + cdx;
            let dy: WideF64 = yi + cdy;
            let attn: WideF64 = 2.0 / 3.0 - dx * dx - dy * dy;
            let modify: WideM64 = attn.gt(WideF64::splat(0.0));
            if modify.none() {
                continue;
            }

            let pxm: WideUsize = ((xsb + cx) & PMASK as i32).cast();
            let pym: WideUsize = ((ysb + cy) & PMASK as i32).cast();
            let permdat: WideUsize =
                unsafe { permptr.add(pxm).read(modify, WideU16::splat(0)) }.cast();
            let gradidx: WideUsize = permdat ^ pym;
            let gdx: WideF64 = unsafe { permxptr.add(gradidx).read(modify, WideF64::splat(0.0)) };
            let gdy: WideF64 = unsafe { permyptr.add(gradidx).read(modify, WideF64::splat(0.0)) };
            let extrapolation = gdx * dx + gdy * dy;
            let attn2 = attn * attn;
            let attn4 = attn2 * attn2;
            value = modify.select(value + attn4 * extrapolation, value);
        }
        value
    }
}

#[test]
fn check_working_noise() {
    let coords = [0.1, 0.5, 0.9, 1.0, 2.0, 3.0, 4.9, 5.753];
    let coordw = WideF64::from_slice_unaligned(&coords);
    for seed in 0..32 {
        let ss = SuperSimplex::new(seed);
        ss.noise2_wide(coordw, coordw);
    }
}

mod lookup_2d {
    use crate::math::*;
    use crate::supersimplex::PSIZE;
    use bxw_util::lazy_static::lazy_static;

    pub const LATTICE_POINTS: usize = 8 * 4;

    fn lattice_point_derivatives(xsv: i32, ysv: i32) -> (f64, f64) {
        let ssv = (xsv + ysv) as f64 * -0.211324865405187;
        (-xsv as f64 - ssv, -ysv as f64 - ssv)
    }

    #[derive(Copy, Clone)]
    pub struct Luts {
        pub points_x: &'static [i32; LATTICE_POINTS],
        pub points_y: &'static [i32; LATTICE_POINTS],
        pub points_dx: &'static [f64; LATTICE_POINTS],
        pub points_dy: &'static [f64; LATTICE_POINTS],
        pub grads_dx: &'static [f64; PSIZE],
        pub grads_dy: &'static [f64; PSIZE],
    }

    impl Luts {
        pub unsafe fn read_points(&self, index: WideUsize) -> (WideI32, WideI32, WideF64, WideF64) {
            debug_assert!(index.lt(WideUsize::splat(LATTICE_POINTS)).all());
            let xptr = WideCptr::splat(self.points_x.as_ptr()).add(index);
            let yptr = WideCptr::splat(self.points_y.as_ptr()).add(index);
            let dxptr = WideCptr::splat(self.points_dx.as_ptr()).add(index);
            let dyptr = WideCptr::splat(self.points_dy.as_ptr()).add(index);
            let mall = WideM32::splat(true);
            let zeroi = WideI32::splat(0);
            let zerof = WideF64::splat(0.0);
            (
                xptr.read(mall, zeroi),
                yptr.read(mall, zeroi),
                dxptr.read(mall, zerof),
                dyptr.read(mall, zerof),
            )
        }
    }

    pub fn tables() -> Luts {
        let lp: &'static LutType = &*LUT;
        let gs: &'static GradType = &*GRADS;
        Luts {
            points_x: &lp.0,
            points_y: &lp.1,
            points_dx: &lp.2,
            points_dy: &lp.3,
            grads_dx: &gs.0,
            grads_dy: &gs.1,
        }
    }

    type LutType = (
        [i32; LATTICE_POINTS],
        [i32; LATTICE_POINTS],
        [f64; LATTICE_POINTS],
        [f64; LATTICE_POINTS],
    );
    type GradType = ([f64; PSIZE], [f64; PSIZE]);

    lazy_static! {
        static ref LUT: LutType = init_lut();
        static ref GRADS: GradType = init_grads();
    }

    fn init_lut() -> LutType {
        let mut xs = [0i32; LATTICE_POINTS];
        let mut ys = [0i32; LATTICE_POINTS];
        let mut dxs = [0f64; LATTICE_POINTS];
        let mut dys = [0f64; LATTICE_POINTS];
        let mut set_point = |i: usize, x: i32, y: i32| {
            xs[i] = x;
            ys[i] = y;
            let (dx, dy) = lattice_point_derivatives(x, y);
            dxs[i] = dx;
            dys[i] = dy;
        };
        for i in 0..8 {
            let (i1, j1, i2, j2);
            if (i & 1) == 0 {
                if (i & 2) == 0 {
                    i1 = -1;
                    j1 = 0;
                } else {
                    i1 = 1;
                    j1 = 0;
                }
                if (i & 4) == 0 {
                    i2 = 0;
                    j2 = -1;
                } else {
                    i2 = 0;
                    j2 = 1;
                }
            } else {
                if (i & 2) != 0 {
                    i1 = 2;
                    j1 = 1;
                } else {
                    i1 = 0;
                    j1 = 1;
                }
                if (i & 4) != 0 {
                    i2 = 1;
                    j2 = 2;
                } else {
                    i2 = 1;
                    j2 = 0;
                }
            }
            set_point(i * 4, 0, 0);
            set_point(i * 4 + 1, 1, 1);
            set_point(i * 4 + 2, i1, j1);
            set_point(i * 4 + 3, i2, j2);
        }
        (xs, ys, dxs, dys)
    }

    const N2: f64 = 0.05481866495625118f64;

    fn init_grads() -> GradType {
        let mut xs = [0.0f64; PSIZE];
        let mut ys = [0.0f64; PSIZE];
        let mut grad2: [(f64, f64); 24] = [
            (0.130526192220052, 0.99144486137381),
            (0.38268343236509, 0.923879532511287),
            (0.608761429008721, 0.793353340291235),
            (0.793353340291235, 0.608761429008721),
            (0.923879532511287, 0.38268343236509),
            (0.99144486137381, 0.130526192220051),
            (0.99144486137381, -0.130526192220051),
            (0.923879532511287, -0.38268343236509),
            (0.793353340291235, -0.60876142900872),
            (0.608761429008721, -0.793353340291235),
            (0.38268343236509, -0.923879532511287),
            (0.130526192220052, -0.99144486137381),
            (-0.130526192220052, -0.99144486137381),
            (-0.38268343236509, -0.923879532511287),
            (-0.608761429008721, -0.793353340291235),
            (-0.793353340291235, -0.608761429008721),
            (-0.923879532511287, -0.38268343236509),
            (-0.99144486137381, -0.130526192220052),
            (-0.99144486137381, 0.130526192220051),
            (-0.923879532511287, 0.38268343236509),
            (-0.793353340291235, 0.608761429008721),
            (-0.608761429008721, 0.793353340291235),
            (-0.38268343236509, 0.923879532511287),
            (-0.130526192220052, 0.99144486137381),
        ];
        grad2.iter_mut().for_each(|x| *x = (x.0 / N2, x.1 / N2));
        for i in 0..PSIZE {
            let (x, y) = grad2[i % grad2.len()];
            xs[i] = x;
            ys[i] = y;
        }
        (xs, ys)
    }
}
