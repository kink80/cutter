//! Crop + perspective bake. The GUI's crop rectangle and four perspective
//! corner handles are turned here into a rectified working image on disk; the
//! existing pipeline (cmyk::load, stencil::stencil_masks) then loads it by path,
//! so nothing downstream needs to know a transform happened.
//!
//! Only the selected crop region is emitted, and the four `src_quad` points (in
//! original-image pixels) are mapped onto the output rectangle's corners — a
//! straight keystone/lens dewarp when the user pulls the corners in.

use image::{Rgb, RgbImage};

/// Solve the 3x3 homography H with H * dst ~ src (row-major, h8 == 1), mapping
/// the four `dst` points to the four `src` points. `None` if degenerate.
pub fn homography(dst: [[f64; 2]; 4], src: [[f64; 2]; 4]) -> Option<[f64; 9]> {
    // 8 unknowns (h0..h7, h8 fixed to 1): two rows per correspondence.
    let mut a = [[0.0f64; 8]; 8];
    let mut b = [0.0f64; 8];
    for i in 0..4 {
        let (x, y) = (dst[i][0], dst[i][1]);
        let (u, v) = (src[i][0], src[i][1]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
        b[2 * i] = u;
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
        b[2 * i + 1] = v;
    }
    let h = solve8(a, b)?;
    Some([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0])
}

/// Gaussian elimination with partial pivoting for an 8x8 system. `None` if singular.
fn solve8(mut a: [[f64; 8]; 8], mut b: [f64; 8]) -> Option<[f64; 8]> {
    for col in 0..8 {
        // Partial pivot: largest magnitude in this column at or below the diagonal.
        let mut piv = col;
        for r in (col + 1)..8 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        let inv = 1.0 / a[col][col];
        for r in 0..8 {
            if r == col {
                continue;
            }
            let f = a[r][col] * inv;
            if f != 0.0 {
                for c in col..8 {
                    a[r][c] -= f * a[col][c];
                }
                b[r] -= f * b[col];
            }
        }
    }
    let mut x = [0.0f64; 8];
    for i in 0..8 {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
}

/// Project a point through a 3x3 homography (perspective divide).
pub fn project(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    apply(h, x, y)
}

fn apply(h: &[f64; 9], x: f64, y: f64) -> (f64, f64) {
    let w = h[6] * x + h[7] * y + h[8];
    let iw = if w.abs() < 1e-12 { 1e-12 } else { w };
    ((h[0] * x + h[1] * y + h[2]) / iw, (h[3] * x + h[4] * y + h[5]) / iw)
}

/// Bilinear sample with clamp-to-edge. `sx,sy` are pixel-centre coordinates.
fn sample(img: &RgbImage, sx: f64, sy: f64) -> Rgb<u8> {
    let (iw, ih) = img.dimensions();
    let x = sx.clamp(0.0, iw as f64 - 1.0);
    let y = sy.clamp(0.0, ih as f64 - 1.0);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(iw - 1);
    let y1 = (y0 + 1).min(ih - 1);
    let fx = x - x0 as f64;
    let fy = y - y0 as f64;
    let p = |xx: u32, yy: u32| img.get_pixel(xx, yy).0;
    let (a, b, c, d) = (p(x0, y0), p(x1, y0), p(x0, y1), p(x1, y1));
    let mut out = [0u8; 3];
    for k in 0..3 {
        let top = a[k] as f64 * (1.0 - fx) + b[k] as f64 * fx;
        let bot = c[k] as f64 * (1.0 - fx) + d[k] as f64 * fx;
        out[k] = (top * (1.0 - fy) + bot * fy).round().clamp(0.0, 255.0) as u8;
    }
    Rgb(out)
}

/// Render the `out_w`x`out_h` rectangle whose corners (TL,TR,BR,BL) map to
/// `src_quad` in the original image, and write it to `dest` as PNG.
pub fn bake(
    orig_path: &str,
    out_w: usize,
    out_h: usize,
    src_quad: [[f64; 2]; 4],
    dest: &str,
) -> Result<(), String> {
    if out_w == 0 || out_h == 0 {
        return Err("warp: empty output region".into());
    }
    let img = image::open(orig_path)
        .map_err(|e| format!("warp: open {orig_path}: {e}"))?
        .to_rgb8();
    let dst = [
        [0.0, 0.0],
        [out_w as f64, 0.0],
        [out_w as f64, out_h as f64],
        [0.0, out_h as f64],
    ];
    let h = homography(dst, src_quad).ok_or("warp: degenerate perspective quad")?;
    let mut out = RgbImage::new(out_w as u32, out_h as u32);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let (sx, sy) = apply(&h, ox as f64 + 0.5, oy as f64 + 0.5);
            out.put_pixel(ox as u32, oy as u32, sample(&img, sx - 0.5, sy - 0.5));
        }
    }
    out.save(dest).map_err(|e| format!("warp: save {dest}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rect_maps_through() {
        // dst == src (a plain rectangle) -> homography is the identity map.
        let q = [[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]];
        let h = homography(q, q).unwrap();
        let (x, y) = apply(&h, 3.0, 7.0);
        assert!((x - 3.0).abs() < 1e-6 && (y - 7.0).abs() < 1e-6);
    }

    #[test]
    fn maps_corners_to_src_quad() {
        let dst = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]];
        let src = [[2.0, 1.0], [9.0, 0.0], [8.0, 6.0], [1.0, 5.0]];
        let h = homography(dst, src).unwrap();
        for i in 0..4 {
            let (x, y) = apply(&h, dst[i][0], dst[i][1]);
            assert!((x - src[i][0]).abs() < 1e-6 && (y - src[i][1]).abs() < 1e-6);
        }
    }
}
