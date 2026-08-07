//! Load + resize image, split into 4 CMYK grayscale maps (0 = no ink, 1 = full ink).

use image::imageops::FilterType;

/// Four row-major intensity maps, one per CMYK channel, plus canvas dimensions.
pub struct Layers {
    pub c: Vec<f32>,
    pub m: Vec<f32>,
    pub y: Vec<f32>,
    pub k: Vec<f32>,
}

/// One screened ink: its separation map + everything the pipeline needs to screen,
/// preview, and export it. Generalizes the fixed C/M/Y/K to N inks. `tamed` marks an
/// opaque ink (black, or an opaque spot) that goes through the taming path (deep-shadow
/// clip / UCR / width cap) instead of the translucent screen. `display_rgb` is the
/// ink's transmission colour, used for both the multiplicative preview and the SVG fill.
#[derive(Clone)]
pub struct Channel {
    pub density: Vec<f32>,
    pub angle: f32,
    pub display_rgb: [f32; 3],
    pub name: &'static str,
    pub suffix: &'static str,
    pub tamed: bool,
}

/// Which set of process inks to separate into.
#[derive(Clone, Copy, PartialEq)]
pub enum Inks {
    Cmyk,
    /// Extended gamut: CMYK plus orange + green (hi-fi "CMYKOG" printing).
    Cmykog,
}

impl Inks {
    /// Ink metadata in spray order (index 0 sprayed first). `angle` filled from the
    /// caller's list; `display_rgb` is the ink's transmission colour.
    /// (name, suffix, display_rgb, tamed)
    fn specs(self) -> &'static [(&'static str, &'static str, [f32; 3], bool)] {
        match self {
            Inks::Cmyk => &[
                ("Cyan", "c", [0.0, 1.0, 1.0], false),
                ("Magenta", "m", [1.0, 0.0, 1.0], false),
                ("Yellow", "y", [1.0, 1.0, 0.0], false),
                ("Black", "k", [0.0, 0.0, 0.0], true),
            ],
            Inks::Cmykog => &[
                ("Cyan", "c", [0.0, 1.0, 1.0], false),
                ("Magenta", "m", [1.0, 0.0, 1.0], false),
                ("Yellow", "y", [1.0, 1.0, 0.0], false),
                ("Orange", "o", [1.0, 0.5, 0.0], false),
                ("Green", "g", [0.0, 0.7, 0.3], false),
                ("Black", "k", [0.0, 0.0, 0.0], true),
            ],
        }
    }

    /// Default screen angles (degrees) for each ink, in spec order. CMYK matches the
    /// classic C15/M75/Y0/K45; O/G reuse spread angles to minimise moiré.
    pub fn default_angles(self) -> Vec<f32> {
        match self {
            Inks::Cmyk => vec![15.0, 75.0, 0.0, 45.0],
            Inks::Cmykog => vec![15.0, 75.0, 0.0, 30.0, 60.0, 45.0],
        }
    }

    pub fn count(self) -> usize {
        self.specs().len()
    }

    /// Short display names per ink, in spec order (for GUI angle sliders).
    pub fn names(self) -> Vec<&'static str> {
        self.specs().iter().map(|s| s.0).collect()
    }
}

/// Build the screened-channel list for an ink set from CMYK layers + per-ink angles.
/// For `Cmyk` this reproduces the historical C/M/Y/K channels exactly. For `Cmykog`
/// it splits orange/green out of the CMY (see `split_extended`). `angles.len()` must
/// match `inks.count()`.
pub fn channels(layers: &Layers, inks: Inks, angles: &[f32]) -> Vec<Channel> {
    let specs = inks.specs();
    let maps: Vec<Vec<f32>> = match inks {
        Inks::Cmyk => vec![layers.c.clone(), layers.m.clone(), layers.y.clone(), layers.k.clone()],
        Inks::Cmykog => split_extended(layers),
    };
    specs
        .iter()
        .enumerate()
        .map(|(i, &(name, suffix, rgb, tamed))| Channel {
            density: maps[i].clone(),
            angle: angles.get(i).copied().unwrap_or(0.0),
            display_rgb: rgb,
            name,
            suffix,
            tamed,
        })
        .collect()
}

/// First-pass RGB→CMYKOG separation heuristic. ponytail: this is an approximation, not
/// a real ICC/Neugebauer model — it pulls orange and green out of the CMY where those
/// spot inks can reproduce the colour more purely, reducing the muddiness of mixing
/// C+M+Y. Upgrade path: a measured spectral/Neugebauer separation.
/// Returns 6 maps in spec order: C, M, Y, O, G, K.
fn split_extended(layers: &Layers) -> Vec<Vec<f32>> {
    let n = layers.c.len();
    let (mut c, mut m, mut y) = (layers.c.clone(), layers.m.clone(), layers.y.clone());
    let mut o = vec![0.0f32; n];
    let mut g = vec![0.0f32; n];
    for i in 0..n {
        // Orange ≈ where magenta+yellow overlap with little cyan (warm reds/oranges).
        // Move the shared M∧Y ink (that isn't cancelled by C) into the orange channel.
        let orange = (m[i].min(y[i]) - c[i]).max(0.0);
        // Green ≈ where cyan+yellow overlap with little magenta.
        let green = (c[i].min(y[i]) - m[i]).max(0.0);
        // GCR-style: remove the extracted amount from the CMY it came from.
        o[i] = orange;
        m[i] = (m[i] - orange).max(0.0);
        y[i] = (y[i] - orange - green).max(0.0);
        g[i] = green;
        c[i] = (c[i] - green).max(0.0);
    }
    vec![c, m, y, o, g, layers.k.clone()]
}

/// Naive textbook RGB->CMYK. Inputs r,g,b in [0,1]. Returns (c,m,y,k) in [0,1].
fn rgb_to_cmyk(r: f32, g: f32, b: f32) -> (f32, f32, f32, f32) {
    let k = 1.0 - r.max(g).max(b);
    if k >= 1.0 {
        return (0.0, 0.0, 0.0, 1.0); // pure black: all ink is K
    }
    let inv = 1.0 - k;
    (
        (1.0 - r - k) / inv,
        (1.0 - g - k) / inv,
        (1.0 - b - k) / inv,
        k,
    )
}

/// CMYK back to RGB (all in [0,1]). Inverse of `rgb_to_cmyk`.
pub fn cmyk_to_rgb(c: f32, m: f32, y: f32, k: f32) -> (f32, f32, f32) {
    let inv = 1.0 - k;
    ((1.0 - c) * inv, (1.0 - m) * inv, (1.0 - y) * inv)
}

/// sRGB (0..1 per channel) -> CIELAB. L in ~[0,100], a/b in ~[-128,128].
/// Standard sRGB->linear->XYZ (D65)->Lab. This is the perceptual space
/// BayStencil quantizes in; Euclidean distance in Lab ~ human colour distance.
pub fn rgb_to_lab(r: f32, g: f32, b: f32) -> [f32; 3] {
    let lin = |c: f32| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
    let (r, g, b) = (lin(r), lin(g), lin(b));
    // linear sRGB -> XYZ (D65), then normalise by the D65 white point.
    let x = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
    let f = |t: f32| if t > 0.008856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// K-means clustering in Lab space -> `n` cluster centers, returned as 8-bit RGB.
/// Deterministic: seeded by luminance percentiles (no RNG), fixed iteration cap.
/// This replaces median-cut; k-means in Lab gives the "composed" palette because
/// splits follow perceptual clusters, not axis-aligned RGB boxes.
pub fn kmeans_lab(pixels: &[[u8; 3]], n: usize) -> Vec<[u8; 3]> {
    if pixels.is_empty() {
        return vec![[0, 0, 0]];
    }
    let n = n.min(pixels.len()).max(1);
    let labs: Vec<[f32; 3]> = pixels
        .iter()
        .map(|p| rgb_to_lab(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0))
        .collect();

    // Deterministic seeding: sort pixel indices by L, pick n evenly-spaced percentiles.
    let mut order: Vec<usize> = (0..labs.len()).collect();
    order.sort_by(|&a, &b| labs[a][0].partial_cmp(&labs[b][0]).unwrap());
    let mut centers: Vec<[f32; 3]> = (0..n)
        .map(|k| labs[order[(k * (order.len() - 1)) / n.max(1)]])
        .collect();

    let dist2 = |a: &[f32; 3], b: &[f32; 3]| {
        (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
    };

    // Lloyd iterations, fixed cap; converges well before 20 for photo palettes.
    for _ in 0..20 {
        let mut sums = vec![[0.0f64; 3]; n];
        let mut counts = vec![0u64; n];
        for lab in &labs {
            let mut best = 0;
            let mut bd = f32::INFINITY;
            for (i, c) in centers.iter().enumerate() {
                let d = dist2(lab, c);
                if d < bd { bd = d; best = i; }
            }
            for ch in 0..3 { sums[best][ch] += lab[ch] as f64; }
            counts[best] += 1;
        }
        let mut moved = false;
        for i in 0..n {
            if counts[i] == 0 { continue; } // keep empty center where it is
            let nc = [
                (sums[i][0] / counts[i] as f64) as f32,
                (sums[i][1] / counts[i] as f64) as f32,
                (sums[i][2] / counts[i] as f64) as f32,
            ];
            if dist2(&nc, &centers[i]) > 1e-4 { moved = true; }
            centers[i] = nc;
        }
        if !moved { break; }
    }

    // Map each center back to the mean *RGB* of the pixels assigned to it (Lab->RGB
    // round-trip is lossy; averaging real pixels is simpler and exact).
    let mut sums = vec![[0u64; 3]; n];
    let mut counts = vec![0u64; n];
    for (lab, px) in labs.iter().zip(pixels.iter()) {
        let mut best = 0;
        let mut bd = f32::INFINITY;
        for (i, c) in centers.iter().enumerate() {
            let d = dist2(lab, c);
            if d < bd { bd = d; best = i; }
        }
        for ch in 0..3 { sums[best][ch] += px[ch] as u64; }
        counts[best] += 1;
    }
    (0..n)
        .map(|i| {
            let c = counts[i].max(1);
            [(sums[i][0] / c) as u8, (sums[i][1] / c) as u8, (sums[i][2] / c) as u8]
        })
        .collect()
}

/// Snap an 8-bit RGB color to the nearest printable CMYK color.
/// Returns (r,g,b snapped, cmyk components in [0,1]) — the round-trip is what a
/// CMYK printer/sprayer actually reproduces.
pub fn snap_to_cmyk(r: u8, g: u8, b: u8) -> ([u8; 3], [f32; 4]) {
    let (c, m, y, k) = rgb_to_cmyk(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
    let (rr, gg, bb) = cmyk_to_rgb(c, m, y, k);
    let px = |v: f32| (v * 255.0).round().clamp(0.0, 255.0) as u8;
    ([px(rr), px(gg), px(bb)], [c, m, y, k])
}

/// Edge-preserving bilateral filter on an RGB image (PDF "secret sauce" §1).
/// Averages a pixel with neighbours weighted by BOTH spatial distance and colour
/// similarity, so it smooths sensor noise / JPEG blocks / skin texture in flat
/// areas while keeping primary edges razor-sharp — unlike a plain Gaussian, which
/// blurs edges into fuzzy mid-tones that turn into ugly line stubs after screening.
/// `sigma_s` = spatial radius (px), `sigma_r` = colour sigma (0..255). radius=0 off.
pub fn bilateral(img: &image::RgbImage, sigma_s: f32, sigma_r: f32) -> image::RgbImage {
    if sigma_s <= 0.0 {
        return img.clone();
    }
    let (w, h) = (img.width() as i32, img.height() as i32);
    let rad = (2.0 * sigma_s).ceil() as i32; // 2-sigma window
    let inv_s2 = 1.0 / (2.0 * sigma_s * sigma_s);
    let inv_r2 = 1.0 / (2.0 * sigma_r * sigma_r);
    let mut out = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let center = img.get_pixel(x as u32, y as u32).0;
            let mut acc = [0.0f32; 3];
            let mut wsum = 0.0f32;
            for dy in -rad..=rad {
                for dx in -rad..=rad {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w || ny >= h {
                        continue;
                    }
                    let p = img.get_pixel(nx as u32, ny as u32).0;
                    let sp = (dx * dx + dy * dy) as f32 * inv_s2;
                    let dc: f32 = (0..3)
                        .map(|c| {
                            let d = p[c] as f32 - center[c] as f32;
                            d * d
                        })
                        .sum();
                    let weight = (-(sp + dc * inv_r2)).exp();
                    for c in 0..3 {
                        acc[c] += p[c] as f32 * weight;
                    }
                    wsum += weight;
                }
            }
            let px = [
                (acc[0] / wsum).round().clamp(0.0, 255.0) as u8,
                (acc[1] / wsum).round().clamp(0.0, 255.0) as u8,
                (acc[2] / wsum).round().clamp(0.0, 255.0) as u8,
            ];
            out.put_pixel(x as u32, y as u32, image::Rgb(px));
        }
    }
    out
}

/// Load `path`, resize to `w`x`h` px (Lanczos3), separate into CMYK maps.
/// `bilateral_px` > 0 applies an edge-preserving pre-smooth before separation.
pub fn load(path: &str, w: usize, h: usize) -> Result<Layers, String> {
    load_filtered(path, w, h, 0.0)
}

/// Like `load` but with an optional bilateral pre-filter radius (px).
pub fn load_filtered(path: &str, w: usize, h: usize, bilateral_px: f32) -> Result<Layers, String> {
    let img = image::open(path).map_err(|e| format!("failed to open image: {e}"))?;
    let mut img = img
        .resize_exact(w as u32, h as u32, FilterType::Lanczos3)
        .to_rgb8();
    if bilateral_px > 0.0 {
        // sigma_r ~ 30/255: preserve edges with a colour jump above ~12%.
        img = bilateral(&img, bilateral_px, 30.0);
    }

    let n = w * h;
    let mut layers = Layers {
        c: vec![0.0; n],
        m: vec![0.0; n],
        y: vec![0.0; n],
        k: vec![0.0; n],
    };
    for (i, px) in img.pixels().enumerate() {
        let (r, g, b) = (
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        );
        let (c, m, y, k) = rgb_to_cmyk(r, g, b);
        layers.c[i] = c;
        layers.m[i] = m;
        layers.y[i] = y;
        layers.k[i] = k;
    }
    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn cmyk_identities() {
        let (c, m, y, k) = rgb_to_cmyk(1.0, 0.0, 0.0); // red
        assert!(approx(c, 0.0) && approx(m, 1.0) && approx(y, 1.0) && approx(k, 0.0));
        let (c, m, y, k) = rgb_to_cmyk(1.0, 1.0, 1.0); // white
        assert!(approx(c, 0.0) && approx(m, 0.0) && approx(y, 0.0) && approx(k, 0.0));
        let (c, m, y, k) = rgb_to_cmyk(0.0, 0.0, 0.0); // black
        assert!(approx(c, 0.0) && approx(m, 0.0) && approx(y, 0.0) && approx(k, 1.0));
    }

    #[test]
    fn kmeans_recovers_two_clusters() {
        // Two well-separated colors -> k-means must return one near each, deterministically.
        let mut px = vec![[10u8, 10, 10]; 100];
        px.extend(vec![[240u8, 240, 240]; 100]);
        let c = kmeans_lab(&px, 2);
        assert_eq!(c.len(), 2);
        let dark = c.iter().min_by_key(|p| p[0] as i32).unwrap();
        let light = c.iter().max_by_key(|p| p[0] as i32).unwrap();
        assert!(dark[0] < 40 && light[0] > 200, "clusters {c:?}");
        // Determinism: same input -> same output.
        assert_eq!(c, kmeans_lab(&px, 2));
    }

    #[test]
    fn lab_black_white() {
        assert!(rgb_to_lab(0.0, 0.0, 0.0)[0].abs() < 1.0);       // L~0
        assert!((rgb_to_lab(1.0, 1.0, 1.0)[0] - 100.0).abs() < 1.0); // L~100
    }

    #[test]
    fn cmyk_roundtrip_is_stable() {
        // Snapping to CMYK and back should be near-lossless for these primaries.
        for rgb in [[255, 0, 0], [0, 128, 64], [200, 200, 200], [0, 0, 0], [255, 255, 255]] {
            let (snapped, _) = snap_to_cmyk(rgb[0], rgb[1], rgb[2]);
            for ch in 0..3 {
                assert!((snapped[ch] as i32 - rgb[ch] as i32).abs() <= 1, "{rgb:?} -> {snapped:?}");
            }
        }
    }
}
