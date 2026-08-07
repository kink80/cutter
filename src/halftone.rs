//! Tone controls shared by the analytic line screen (`lines.rs`) and the stencil
//! path: standard screen angles, Photoshop-style "Levels", and per-channel
//! auto-levels. (The old raster line-screen + trace path was replaced by the
//! analytic quad generator in `lines.rs`, which matches the PDF's "no image
//! tracing" Step 2-4; only the tone helpers survive here.)

// Default screen angles now live per ink-set in `cmyk::Inks::default_angles()`.

/// Photoshop/Inkscape "Levels" on a single ink value, before screening.
/// White-point subsumes the old hard ink-eps cliff: raise it and off-white
/// backgrounds collapse to 0 => no stroke.
pub fn levels(i: f32, wp: f32, bp: f32, gamma: f32) -> f32 {
    // Degenerate range (bp <= wp) would divide by <= 0. At i == wp that's 0.0/0.0 =
    // NaN, which then poisons every downstream ribbon coordinate and panics the
    // rasterizer's float sort. Collapse it to a hard threshold at wp: below -> 0,
    // at/above -> 1. The GUI lets the two level sliders coincide; the CLI rejects it.
    let span = bp - wp;
    if span <= 0.0 {
        return if i >= wp { 1.0 } else { 0.0 };
    }
    ((i - wp) / span).clamp(0.0, 1.0).powf(1.0 / gamma)
}

/// Auto-levels: pick (white_point, black_point) for one ink map from its
/// histogram. wp = ink at the low percentile, bp = ink at the high percentile.
/// Percentiles (not min/max) so a few noise pixels don't peg the range.
/// Returns identity (0,1) if the map is degenerate (wp >= bp).
pub fn auto_levels(map: &[f32], lo_pct: f32, hi_pct: f32) -> (f32, f32) {
    let mut v: Vec<f32> = map.iter().copied().collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |pct: f32| v[((pct * (v.len() - 1) as f32).round() as usize).min(v.len() - 1)];
    let (wp, bp) = (at(lo_pct), at(hi_pct));
    if wp >= bp {
        (0.0, 1.0)
    } else {
        (wp, bp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_maps_endpoints() {
        // Below white-point clamps to 0, above black-point clamps to 1.
        assert_eq!(levels(0.1, 0.2, 0.8, 1.0), 0.0);
        assert_eq!(levels(0.9, 0.2, 0.8, 1.0), 1.0);
        // Midpoint of the range, gamma 1, is ~0.5.
        assert!((levels(0.5, 0.2, 0.8, 1.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn levels_degenerate_range_is_a_threshold_not_nan() {
        // wp == bp (the GUI lets both sliders sit at 0.0) must not divide by zero.
        // The dangerous input is i == wp: old code did 0.0/0.0 = NaN.
        for &(i, wp, bp) in &[(0.0f32, 0.0, 0.0), (0.3, 0.3, 0.3), (0.5, 0.8, 0.2)] {
            assert!(levels(i, wp, bp, 1.0).is_finite(), "no NaN at wp==bp, i==wp");
        }
        assert_eq!(levels(0.0, 0.0, 0.0, 1.0), 1.0, "at/above wp -> 1");
        assert_eq!(levels(0.2, 0.5, 0.5, 1.0), 0.0, "below threshold -> 0");
    }

    #[test]
    fn auto_levels_stretches_to_percentiles() {
        // Ink clustered in 0.2..0.8 with a couple of outliers at 0 and 1.
        // Percentile clip points must land inside the cluster, not on the outliers.
        let mut map = vec![0.0f32; 5];
        map.extend(std::iter::repeat(0.2).take(45));
        map.extend(std::iter::repeat(0.8).take(45));
        map.extend(std::iter::repeat(1.0).take(5));
        let (wp, bp) = auto_levels(&map, 0.06, 0.94); // clip past the 5% outlier tails
        assert!(wp > 0.0 && wp <= 0.2, "wp ignores low outliers: {wp}");
        assert!(bp >= 0.8 && bp < 1.0, "bp ignores high outliers: {bp}");
        // Degenerate uniform map => identity, never a zero-width range.
        assert_eq!(auto_levels(&vec![0.5; 10], 0.005, 0.995), (0.0, 1.0));
    }
}
