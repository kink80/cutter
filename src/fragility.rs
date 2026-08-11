//! Will this sheet survive being sprayed? A heavily-cut layer turns into a floppy
//! lattice that lifts off the work and lets paint bleed under it. The preview shows
//! tone, not stiffness — a gorgeous preview and an unsprayable sheet look identical —
//! so this measures the physical property directly from the layer's keep-mask.

/// Physical-stiffness readout for one layer.
pub struct Fragility {
    /// Fraction of the sheet cut away. Context for `neck_px`, not a stiffness proxy
    /// on its own: 60% removed as wide ribbons is stiff, as a fine lattice is floppy.
    pub removed: f32,
    /// 5th-percentile material thickness in px — "the thinnest strands holding this
    /// sheet together are about this wide". The number that predicts tearing.
    pub neck_px: f32,
}

/// Measure one layer. `keep[i] == true` means material STAYS there (not cut away).
///
/// `neck_px` comes from a chamfer distance transform (distance from each material
/// pixel to the nearest cut), lifted to each strand's RIDGE by a local-max filter,
/// then the 5th percentile of that — doubled to turn a half-width into a full width.
/// The ridge step matters: without it, every strand has edge pixels one step from a
/// cut, so a low percentile just reports "an edge exists" for hairlines and slabs
/// alike.
///
/// ponytail: p05 rather than min, because a single-pixel rasterization artifact
/// would pin a `min` to the floor forever and the readout would never move. p05 is a
/// robust "the thin 5% is this thin". Upgrade path: if a real tear ever traces back
/// to a strand thinner than p05 reported, drop to p01.
pub fn measure(keep: &[bool], w: usize, h: usize) -> Fragility {
    let n = w * h;
    if n == 0 || keep.len() < n {
        return Fragility { removed: 0.0, neck_px: 0.0 };
    }
    let kept = keep[..n].iter().filter(|&&k| k).count();
    let removed = 1.0 - kept as f32 / n as f32;
    if kept == 0 {
        // Everything cut away: no material, so no strand holds anything.
        return Fragility { removed, neck_px: 0.0 };
    }

    // Two-pass chamfer distance transform with the classic 3-4 integer weights (3 per
    // orthogonal step, 4 per diagonal, so one px == 3 units). Cut pixels are the zero
    // set; material pixels accumulate distance outward from them.
    const ORTH: u16 = 3;
    const DIAG: u16 = 4;
    const FAR: u16 = u16::MAX - DIAG; // headroom so `+ DIAG` can't wrap
    let mut d: Vec<u16> = keep[..n].iter().map(|&k| if k { FAR } else { 0 }).collect();

    // Forward pass: top-to-bottom, left-to-right — looks at N, NW, NE, W.
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if d[i] == 0 {
                continue;
            }
            let mut best = d[i];
            if y > 0 {
                best = best.min(d[i - w] + ORTH);
                if x > 0 {
                    best = best.min(d[i - w - 1] + DIAG);
                }
                if x + 1 < w {
                    best = best.min(d[i - w + 1] + DIAG);
                }
            }
            if x > 0 {
                best = best.min(d[i - 1] + ORTH);
            }
            d[i] = best;
        }
    }
    // Backward pass: bottom-to-top, right-to-left — looks at S, SE, SW, E.
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let i = y * w + x;
            if d[i] == 0 {
                continue;
            }
            let mut best = d[i];
            if y + 1 < h {
                best = best.min(d[i + w] + ORTH);
                if x + 1 < w {
                    best = best.min(d[i + w + 1] + DIAG);
                }
                if x > 0 {
                    best = best.min(d[i + w - 1] + DIAG);
                }
            }
            if x + 1 < w {
                best = best.min(d[i + 1] + ORTH);
            }
            d[i] = best;
        }
    }

    // Every strand has edge pixels one step from a cut, so a low percentile over ALL
    // material pixels just measures "does an edge exist" — always 1px, for a hairline
    // and a slab alike. The strand's half-width lives at its RIDGE (the local maximum
    // across its cross-section), so lift each pixel to the max over a small
    // neighbourhood first; thin strands have a low ridge, thick ones a high one.
    // Radius 3px: wide enough to reach the ridge of any strand up to ~6px, which is
    // well past the point where a sheet stops being fragile.
    const R: isize = 3;
    let mut ridge: Vec<u16> = vec![0; n];
    // Separable max filter: horizontal pass into `ridge`, then vertical in place.
    let mut row: Vec<u16> = vec![0; w];
    for y in 0..h {
        for x in 0..w {
            let lo = (x as isize - R).max(0) as usize;
            let hi = ((x as isize + R) as usize).min(w - 1);
            row[x] = d[y * w + lo..=y * w + hi].iter().copied().max().unwrap_or(0);
        }
        ridge[y * w..(y + 1) * w].copy_from_slice(&row);
    }
    let mut col: Vec<u16> = vec![0; h];
    for x in 0..w {
        for y in 0..h {
            let lo = (y as isize - R).max(0) as usize;
            let hi = ((y as isize + R) as usize).min(h - 1);
            col[y] = (lo..=hi).map(|yy| ridge[yy * w + x]).max().unwrap_or(0);
        }
        for y in 0..h {
            ridge[y * w + x] = col[y];
        }
    }

    // 5th percentile of the ridge over material pixels, by histogram — a sort here
    // would be the only real cost. Bucket = chamfer units (px * 3); the cap saturates
    // around 170px of neck, far past any sheet that could be called fragile.
    const BUCKETS: usize = 512;
    let mut hist = [0u32; BUCKETS];
    for (i, &k) in keep[..n].iter().enumerate() {
        if k {
            hist[(ridge[i] as usize).min(BUCKETS - 1)] += 1;
        }
    }
    let target = ((kept as f32) * 0.05).ceil().max(1.0) as u32;
    let mut acc = 0u32;
    let mut p05 = (BUCKETS - 1) as f32;
    for (bucket, &count) in hist.iter().enumerate() {
        acc += count;
        if acc >= target {
            p05 = bucket as f32;
            break;
        }
    }
    // Chamfer units -> px, then ridge half-width -> full strand width. The -0.5px is
    // the pixel-center offset: the DT measures center-to-center distance to the nearest
    // CUT pixel, but the material actually ends half a pixel short of that center.
    //
    // ponytail: resolution is ~2px, because a ridge sits on whole-pixel centers — a
    // 3px and a 4px strand both report 3px. Good enough for its one job (telling a
    // 1px lattice from a 4px web while a slider moves) and deliberately not more:
    // finer would need the mask rasterized at 2x, doubling the preview's cost. Upgrade
    // path if the 3-vs-4px distinction ever matters: supersample the mask, not the
    // weights — scaling the weights alone does NOT help, the ridge quantum is the
    // pixel grid itself.
    let half_px = (p05 / ORTH as f32 - 0.5).max(0.0);
    Fragility { removed, neck_px: half_px * 2.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole justification for measuring neck width instead of cut area: two
    /// sheets with the SAME removed-area, one stiff and one floppy. Removed-area
    /// cannot tell them apart; neck width must.
    #[test]
    fn neck_tracks_strand_width_not_removed_area() {
        // 32x32 vertical stripes: A = 4px material / 4px cut (stiff),
        // B = 1px material / 1px cut (floppy lattice). Both ~50% cut.
        let strp = |m: usize, c: usize| -> Vec<bool> {
            (0..32 * 32).map(|i| (i % 32) % (m + c) < m).collect()
        };
        let a = measure(&strp(4, 4), 32, 32);
        let b = measure(&strp(1, 1), 32, 32);
        assert!(
            (a.removed - b.removed).abs() < 0.02,
            "removed-area is blind to strand width: {} vs {}",
            a.removed,
            b.removed
        );
        // The load-bearing property: neck separates them by a wide margin, and each
        // lands within the ~2px resolution of its true strand width.
        assert!(
            a.neck_px >= b.neck_px * 2.0,
            "neck must separate stiff from floppy: {} vs {}",
            a.neck_px,
            b.neck_px
        );
        assert!((a.neck_px - 4.0).abs() <= 1.0, "4px strand reads ~4px: {}", a.neck_px);
        assert!((b.neck_px - 1.0).abs() <= 1.0, "1px lattice reads ~1px: {}", b.neck_px);
    }

    /// Resolution where it counts: at real screen pitches the standing material is
    /// 1-4px, so the readout must track that band within its ~2px quantum. This is
    /// what makes the number usable rather than merely well-ordered.
    #[test]
    fn neck_tracks_the_one_to_four_px_band() {
        let (w, h) = (48usize, 48usize);
        for mat in [1usize, 2, 3, 4] {
            // Vertical stripes at pitch 8 with `mat` px of standing material.
            let keep: Vec<bool> = (0..w * h).map(|i| (i % w) % 8 < mat).collect();
            let f = measure(&keep, w, h);
            assert!(
                (f.neck_px - mat as f32).abs() <= 1.0,
                "standing material {mat}px must read within 1px, got {}",
                f.neck_px
            );
        }
    }

    /// Monotonicity is what makes the GUI readout tunable: widening the strands must
    /// never lower the reported neck.
    #[test]
    fn neck_is_monotone_in_strand_width() {
        let strp = |m: usize| -> Vec<bool> {
            (0..48 * 48).map(|i| (i % 48) % (m + 4) < m).collect()
        };
        let mut prev = 0.0f32;
        for m in [1usize, 2, 3, 5, 8] {
            let f = measure(&strp(m), 48, 48);
            assert!(f.neck_px >= prev, "neck dropped at strand width {m}: {} < {prev}", f.neck_px);
            prev = f.neck_px;
        }
    }

    #[test]
    fn solid_sheet_is_not_fragile() {
        let f = measure(&vec![true; 16 * 16], 16, 16);
        assert_eq!(f.removed, 0.0);
        // No cut anywhere -> every pixel is far from a cut -> a thick "strand".
        assert!(f.neck_px > 4.0, "solid sheet read as thin: {}", f.neck_px);
    }

    #[test]
    fn fully_cut_sheet_has_no_material() {
        let f = measure(&vec![false; 16 * 16], 16, 16);
        assert_eq!(f.removed, 1.0);
        assert_eq!(f.neck_px, 0.0);
    }
}

