//! Analytic line-screen halftone (the PDF's Step 2-4 "no image tracing" path).
//!
//! Instead of thresholding a raster ramp and tracing the mask (see `halftone.rs`),
//! this steps ALONG each rotated screen line, samples channel density, and emits a
//! pure 4-point polygon quad per line-segment directly in vector space. This is the
//! "analytical geometry" the PDF calls the secret sauce: the laser moves in smooth
//! linear passes, no micro-bezier stutter, minimal heat buildup.
//!
//! Width per segment obeys the PDF's physical laser constraints:
//!   - dead-zone:  W_raw < w_min  =>  W_final = 0   (don't cut fragile hairlines)
//!   - max cut:    W_final = min(W_raw, spacing - min_material)
//!   - kerf:       the laser removes K_laser of material; we shrink the commanded
//!                 slot by K_laser so the *resulting* gap matches the target width.
//!
//! Continuous cut lines are broken by physical bridges every `bridge_interval`,
//! and bridges on adjacent lines are staggered by half the interval so they form a
//! hexagonal lattice (PDF "Secret Sauce #1") that keeps the sheet flat.


/// How the K (black) channel is generated. Black spray paint is 100% opaque, so
/// treating K like a translucent CMY ink lets it obliterate the colour work under
/// it. Both modes tame that; pick per image.
#[derive(Clone, Copy, PartialEq)]
pub enum KMode {
    /// Trimmed line-screen halftone: deep-shadow clip + steep gamma + width cap +
    /// UCR suppression, so K fires only in true blacks as thin accents.
    Tonal,
    /// Difference-of-Gaussians edge lines: K becomes crisp comic-book contours over
    /// the colour fills instead of a tonal screen.
    Contour,
}

/// The MARK shape a channel's density map is screened into. Every variant emits
/// `Vec<Ribbon>` (closed polygons), so the preview rasterizer and SVG writer are
/// unchanged. Mirrors the `KMode` enum pattern — an enum selecting a geometry
/// strategy inside the one halftone technique.
#[derive(Clone, Copy, PartialEq)]
pub enum Shape {
    /// Straight rotated screen lines; width encodes tone. The original path.
    Lines,
    /// Line screen with a sinusoidally displaced centerline — a wavy cut.
    Wavy,
    /// Classic AM halftone: one variable-radius circle per screen-grid cell.
    Dots,
    /// Frequency-modulated screen: fixed-size dots at variable frequency, placed by
    /// error-diffusion (blue-noise-like). No grid, no moiré; registration-forgiving.
    BlueNoise,
    /// Line screen whose local angle follows the image's dominant edge orientation,
    /// quantized into bins so each region stays a parallel (cut-safe) screen.
    Hatch,
}

/// Extra knobs only some shapes use, kept off `LineParams` so shapes that don't
/// need them don't carry loose fields. All have sensible defaults derived from or
/// independent of `spacing_px`; the GUI/CLI expose the few that are worth tuning.
#[derive(Clone, Copy)]
pub struct ShapeParams {
    pub wave_amp_frac: f32,   // Wavy: amplitude as a fraction of spacing
    pub wave_len_frac: f32,   // Wavy: wavelength as a multiple of spacing
    pub hatch_bins: u32,      // Hatch: number of orientation bins over 180°
    // BlueNoise: dot DIAMETER range (px). Size scales with local tone between these,
    // so darkness drives size (AM) on top of frequency (FM). Both clamped cut-safe
    // in `generate_bluenoise`. Default = INFINITY => both collapse to the cut-safe
    // radius, i.e. the original fixed-size FM screen (byte-for-byte identical).
    pub dot_min_px: f32,      // smallest dot (lightest placed tone)
    pub dot_max_px: f32,      // largest dot (darkest tone)
}

impl Default for ShapeParams {
    fn default() -> Self {
        ShapeParams {
            wave_amp_frac: 0.35,
            wave_len_frac: 4.0,
            hatch_bins: 6,
            dot_min_px: f32::INFINITY,
            dot_max_px: f32::INFINITY,
        }
    }
}

/// Physical/laser constraints for the analytic screen. All lengths in px.
#[derive(Clone, Copy)]
pub struct LineParams {
    pub shape: Shape,          // which mark shape to screen into
    pub shape_params: ShapeParams, // per-shape extra knobs (defaults for unused)
    pub spacing_px: f32,       // center-to-center distance between lines (screen period)
    pub w_min_px: f32,         // dead-zone: raw width below this cuts nothing
    pub min_material_px: f32,  // min standing material -> max cut = spacing - this
    pub kerf_px: f32,          // laser beam width removed; slot shrunk to compensate
    pub bridge_interval_px: f32, // distance between bridges along a line (0 = off)
    pub bridge_px: f32,        // bridge (uncut gap) length along the line
    pub white_point: f32,      // levels, applied before the S-curve
    pub black_point: f32,
    pub gamma: f32,
    pub scurve: f32,           // spray S-curve strength (0 = off); see `scurve`
    /// Per-ink cut-width scale in (0,1], set from `Channel::load` by `generate_all`.
    /// Scales the commanded width AND the max-cut budget together, so the whole
    /// density->width ramp compresses instead of clipping. 1.0 = unchanged.
    pub load: f32,

    // --- K-channel taming (black is opaque; keep it thin and deep) ---
    pub k_mode: KMode,         // Tonal (trimmed screen) or Contour (DoG edges)
    pub k_deep_clip: f32,      // Tk: K density below this clips to 0 (deep-shadow only)
    pub k_gamma: f32,          // steep gamma on the clipped K (thin until absolute black)
    pub k_width_frac: f32,     // K max cut width as a fraction of pitch (e.g. 0.40)
    pub ucr: f32,              // under-colour removal: K -= min(C,M,Y)*ucr, 0 = off
    pub dog_sigma1: f32,       // DoG small blur (fine detail) — Contour mode
    pub dog_sigma2: f32,       // DoG large blur (structure) — Contour mode
    pub dog_threshold: f32,    // DoG edge threshold in [0,1] — Contour mode
}

/// One emitted cut ribbon: a single closed polygon (canvas px) covering a maximal
/// run of consecutive same-line segments. The laser cuts it in ONE continuous
/// contour — one pierce, one pass — instead of re-piercing every ~4px segment.
/// Points wind up the left edge (centerline − normal·halfwidth) then back down the
/// right edge, so a variable-width strip stays a valid outline.
pub struct Ribbon {
    pub pts: Vec<(f32, f32)>,
}

/// Spray-paint S-curve (PDF page 4 §2). Offset print ink is linear; spray builds
/// opacity fast, so a 40% line already reads as ~80%. This aggressively shrinks
/// shadows (low density stays low) and lifts mid-tones, capping toward 1. `k`=0 is
/// the identity; larger `k` = stronger S. Uses the standard normalized logistic so
/// endpoints stay pinned at 0 and 1.
pub fn scurve(d: f32, k: f32) -> f32 {
    if k <= 0.0 {
        return d;
    }
    // Centered logistic on [0,1], renormalized so f(0)=0, f(1)=1.
    let s = |x: f32| 1.0 / (1.0 + (-k * (x - 0.5)).exp());
    let (lo, hi) = (s(0.0), s(1.0));
    ((s(d) - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Levels then S-curve: the full tone transfer for one density sample.
fn tone(d: f32, p: &LineParams) -> f32 {
    let l = crate::halftone::levels(d, p.white_point, p.black_point, p.gamma);
    scurve(l, p.scurve)
}

/// The shared physical-constraint law: a toned density -> the commanded cut extent
/// in px (a slot width, or a dot diameter), or 0 in the dead-zone. Factored out so
/// every shape applies the SAME dead-zone / kerf / max-cut rules (see the width rule
/// inline in `generate_quads_capped`). `d` is already tone-mapped.
fn cut_extent(d: f32, p: &LineParams, max_cut: f32) -> f32 {
    // `load` scales the ramp; `generate_all` scales `max_cut` by the same factor, so
    // lowering it compresses this ink's whole tonal range rather than clipping its
    // dark end (which is what a ceiling-only cap would do).
    let raw = d * p.spacing_px * p.load; // PDF: W_raw = D * P
    if raw < p.w_min_px {
        0.0 // dead-zone: fragile hairlines are not cut at all
    } else {
        // Kerf: the beam removes kerf_px, so command a narrower slot; the resulting
        // gap widens back to ~raw. Clamp to [0, max_cut].
        (raw - p.kerf_px).clamp(0.0, max_cut)
    }
}

/// Deep-shadow clip for the K channel: density below `tk` clips to 0, the rest is
/// remapped to [0,1] and raised to `gamma_k` so K stays vanishingly thin until near
/// absolute black. Keeps K out of mid-tones where opaque black would muddy the CMY.
fn deep_clip(d: f32, tk: f32, gamma_k: f32) -> f32 {
    if d < tk || tk >= 1.0 {
        0.0
    } else {
        ((d - tk) / (1.0 - tk)).clamp(0.0, 1.0).powf(gamma_k)
    }
}

/// Build the effective K density map: under-colour removal (suppress K where CMY
/// already overlap into dark) then the deep-shadow clip + steep gamma. Returns a
/// new row-major map ready to screen. `k_map` is the raw K, `c/m/y` the raw CMY.
/// `under` are the density maps of the translucent inks beneath this opaque one.
/// UCR subtracts `min(under)·ucr` — where those inks already overlap into dark, the
/// opaque ink is pulled back so it doesn't fire on top and muddy the colour. For CMYK
/// (under = C,M,Y) this is exactly the historical `min(C,M,Y)` behaviour.
fn effective_k(k_map: &[f32], under: &[&[f32]], p: &LineParams) -> Vec<f32> {
    k_map
        .iter()
        .enumerate()
        .map(|(i, &k)| {
            let under_min = under.iter().map(|u| u[i]).fold(f32::INFINITY, f32::min);
            let under_min = if under_min.is_finite() { under_min } else { 0.0 };
            let k_ucr = (k - under_min * p.ucr).max(0.0);
            deep_clip(k_ucr, p.k_deep_clip, p.k_gamma)
        })
        .collect()
}

/// Bilinear sample of a row-major density map at canvas coord (x,y) in px.
fn sample(map: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let x = x.clamp(0.0, w as f32 - 1.0);
    let y = y.clamp(0.0, h as f32 - 1.0);
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let v = |xi: usize, yi: usize| map[yi * w + xi];
    let top = v(x0, y0) * (1.0 - fx) + v(x1, y0) * fx;
    let bot = v(x0, y1) * (1.0 - fx) + v(x1, y1) * fx;
    top * (1.0 - fy) + bot * fy
}

/// Generate analytic cut quads for one channel at a screen angle.
///
/// Grid space: `u` runs across lines (steps of `spacing`), `v` runs along a line.
/// Map (u,v) back to canvas with the rotation, sample density, compute the slot
/// width under the dead-zone/max-cut/kerf rules, and emit a widened quad per step
/// — unless a staggered bridge falls on this v.
// Default-cap convenience over `generate_quads_capped`. The shape dispatcher calls
// the capped form directly, so this now only serves the unit tests.
#[allow(dead_code)]
pub fn generate_quads(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
) -> Vec<Ribbon> {
    generate_quads_capped(map, w, h, angle_deg, p, max_cut_for(p))
}

/// As `generate_quads`, but with an explicit maximum cut width — the K channel
/// passes a tighter cap (a fraction of pitch) so opaque black lines can't spread
/// enough to cover the CMY colour underneath.
pub fn generate_quads_capped(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
    max_cut: f32,
) -> Vec<Ribbon> {
    generate_quads_prox(map, w, h, angle_deg, p, max_cut, None)
}

/// As `generate_quads_capped`, but with an optional per-pixel PROXIMITY field (row-
/// major, px = distance to the nearest already-placed cut). When present, each knot's
/// half-width is additionally capped to its local proximity, so a line NARROWS as it
/// nears another cut and pinches to zero only at exact overlap — used by hatch to keep
/// min-material between cuts across bin seams without dropping whole lines. `None`
/// leaves the screen byte-for-byte identical (Lines/Wavy/Dots path).
fn generate_quads_prox(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
    max_cut: f32,
    proximity: Option<&[f32]>,
) -> Vec<Ribbon> {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let max_cut = max_cut.max(0.0);
    let diag = ((w * w + h * h) as f32).sqrt();
    let v_step = (p.spacing_px * 0.5).clamp(0.5, 4.0); // segment length along the line
    let half_span = diag / 2.0 + p.spacing_px;

    // Line index range so lines cover the whole rotated canvas.
    let n_lines = (half_span / p.spacing_px).ceil() as i32;
    let mut ribbons = Vec::new();

    for li in -n_lines..=n_lines {
        let u = li as f32 * p.spacing_px;
        // Staggered hex lattice: shift every other line's bridge phase by half the
        // interval, so bridges never line up across adjacent lines.
        let bridge_phase = if li.rem_euclid(2) == 0 {
            0.0
        } else {
            p.bridge_interval_px * 0.5
        };

        // Accumulate a maximal RUN of consecutive cut segments on this line, then
        // flush the whole run as one ribbon. `run` holds (v_position, half_width)
        // centerline knots; a run has n+1 knots for n segments. Any break —
        // out-of-bounds, bridge, or dead-zone — flushes and starts fresh.
        let mut run: Vec<(f32, f32)> = Vec::new();
        let mut v = -half_span;
        let flush = |run: &mut Vec<(f32, f32)>, ribbons: &mut Vec<Ribbon>| {
            if run.len() >= 2 {
                ribbons.push(build_ribbon(run, u, cx, cy, sin, cos));
            }
            run.clear();
        };

        while v < half_span {
            let v_next = (v + v_step).min(half_span);
            let vm = (v + v_next) * 0.5; // sample at the segment midpoint

            // Rotate grid (u,v) back to canvas space to sample. Line direction is
            // (cos,sin); the across-line normal is (-sin,cos).
            let mid_x = cx + u * -sin + vm * cos;
            let mid_y = cy + u * cos + vm * sin;

            let out_of_bounds = mid_x < 0.0 || mid_x >= w as f32 || mid_y < 0.0 || mid_y >= h as f32;
            let is_bridge = p.bridge_interval_px > 0.0
                && p.bridge_px > 0.0
                && (v - bridge_phase).rem_euclid(p.bridge_interval_px) < p.bridge_px;

            let width = if out_of_bounds || is_bridge {
                0.0
            } else {
                cut_extent(tone(sample(map, w, h, mid_x, mid_y), p), p, max_cut)
            };

            // Proximity cap: shrink the half-width so this cut's EDGE stays >= min-material
            // from any already-placed cut. `prox` is the distance (px) from this centerline
            // point to the nearest placed cut; the new edge reaches `hw` toward it, so we
            // need hw <= prox - min_material. The line narrows smoothly as it nears a seam
            // and pinches to zero once there's no room left.
            let mut hw = width * 0.5;
            if let Some(prox) = proximity {
                if !out_of_bounds {
                    // Margin covers the sampling/rasterization error: proximity is read
                    // at the segment MIDPOINT and stamped from rounded pixels, so leave
                    // a v_step's slack (plus 1px) so the true edge gap never dips below
                    // min-material anywhere along the segment.
                    let margin = v_step + 1.0;
                    let room = sample(prox, w, h, mid_x, mid_y) - p.min_material_px - margin;
                    hw = hw.min(room.max(0.0));
                }
            }

            if hw <= 0.0 {
                flush(&mut run, &mut ribbons); // break the run at any gap (or full collision)
            } else {
                // Start the run with the segment's leading knot, then always append
                // the trailing knot; consecutive segments share a knot so the run is
                // a continuous centerline with its per-knot half-width.
                if run.is_empty() {
                    run.push((v, hw));
                }
                run.push((v_next, hw));
            }

            v = v_next;
        }
        flush(&mut run, &mut ribbons);
    }
    ribbons
}

/// Build one closed ribbon polygon from a run of centerline knots `(v, half_width)`
/// on line `u`. Walks the left edge (centerline − normal·hw) forward through every
/// knot, then the right edge (+normal·hw) backward — a single outline the laser
/// cuts in one pass. The last knot of a segment and the first of the next share the
/// same `v`, so the shared point uses the max of the two half-widths (a step in
/// width becomes a short bevel, not a self-intersection).
fn build_ribbon(run: &[(f32, f32)], u: f32, cx: f32, cy: f32, sin: f32, cos: f32) -> Ribbon {
    // Straight centerline: knot v -> canvas point, constant across-line normal.
    let center = |v: f32| (cx + u * -sin + v * cos, cy + u * cos + v * sin);
    let knots: Vec<(f32, f32, f32)> = run.iter().map(|&(v, hw)| {
        let c = center(v);
        (c.0, c.1, hw)
    }).collect();
    // Constant normal (-sin,cos) for a straight line — pass it as a fixed override so
    // the Lines path stays byte-for-byte identical to the pre-refactor emit.
    stroke_path(&knots, Some((-sin, cos)))
}

/// Stroke an open centerline (points with a per-knot half-width) into one closed
/// ribbon: walk the left edge (center − normal·hw) forward through every knot, then
/// the right edge (+normal·hw) backward. Used by every non-straight shape (Wavy,
/// SpaceFilling, Flow, Tsp). `fixed_normal` forces a constant normal (straight
/// lines); when `None` each knot's normal is perpendicular to the local tangent so a
/// curving path stays a valid outline.
fn stroke_path(knots: &[(f32, f32, f32)], fixed_normal: Option<(f32, f32)>) -> Ribbon {
    let n = knots.len();
    // Per-knot unit normal: perpendicular to the tangent (prev->next), unless fixed.
    let normal = |i: usize| -> (f32, f32) {
        if let Some(nn) = fixed_normal {
            return nn;
        }
        let prev = knots[i.saturating_sub(1)];
        let next = knots[(i + 1).min(n - 1)];
        let (tx, ty) = (next.0 - prev.0, next.1 - prev.1); // tangent (prev->next)
        let len = (tx * tx + ty * ty).sqrt().max(1e-6);
        (-ty / len, tx / len) // rotate tangent +90°
    };
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(n * 2);
    for i in 0..n {
        let (x, y, hw) = knots[i];
        let (nx, ny) = normal(i);
        pts.push((x - nx * hw, y - ny * hw));
    }
    for i in (0..n).rev() {
        let (x, y, hw) = knots[i];
        let (nx, ny) = normal(i);
        pts.push((x + nx * hw, y + ny * hw));
    }
    Ribbon { pts }
}

/// The max cut width for a channel: spacing minus the standing min-material.
fn max_cut_for(p: &LineParams) -> f32 {
    (p.spacing_px - p.min_material_px).max(0.0)
}

/// Screen one density map into ribbons using the selected shape. This is the single
/// funnel every channel goes through (CMY and the tamed K), so a shape swap changes
/// only HOW the map becomes geometry — never how the K map was computed, nor
/// anything downstream (all shapes emit closed `Ribbon` polygons).
fn generate_shape(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
    max_cut: f32,
) -> Vec<Ribbon> {
    match p.shape {
        Shape::Lines => generate_quads_capped(map, w, h, angle_deg, p, max_cut),
        Shape::Wavy => generate_wavy(map, w, h, angle_deg, p, max_cut),
        Shape::Dots => generate_dots(map, w, h, angle_deg, p, max_cut),
        Shape::BlueNoise => generate_bluenoise(map, w, h, p, max_cut),
        Shape::Hatch => generate_hatch(map, w, h, angle_deg, p, max_cut),
    }
}

/// Wavy line screen: identical to `generate_quads_capped`, but the centerline is
/// displaced across-line by `A·sin(2π·v/λ)`. Width still encodes tone (the wave is
/// aesthetic); dead-zone/kerf/max-cut/bridges all apply per the line walker. Amplitude
/// is clamped so adjacent wavy lines can't collide (`A ≤ (spacing − max_cut)/2`).
fn generate_wavy(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
    max_cut: f32,
) -> Vec<Ribbon> {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let max_cut = max_cut.max(0.0);
    let diag = ((w * w + h * h) as f32).sqrt();
    let v_step = (p.spacing_px * 0.5).clamp(0.5, 4.0);
    let half_span = diag / 2.0 + p.spacing_px;
    let n_lines = (half_span / p.spacing_px).ceil() as i32;
    // Amplitude capped so neighbouring lines keep min-material between them.
    let amp = (p.shape_params.wave_amp_frac * p.spacing_px)
        .min((p.spacing_px - max_cut) * 0.5)
        .max(0.0);
    let lambda = (p.shape_params.wave_len_frac * p.spacing_px).max(1.0);
    let tau = std::f32::consts::TAU;

    let mut ribbons = Vec::new();
    for li in -n_lines..=n_lines {
        let u = li as f32 * p.spacing_px;
        let bridge_phase = if li.rem_euclid(2) == 0 { 0.0 } else { p.bridge_interval_px * 0.5 };
        // A run of (canvas_x, canvas_y, half_width) knots along the wavy centerline.
        let mut run: Vec<(f32, f32, f32)> = Vec::new();
        let mut v = -half_span;
        let flush = |run: &mut Vec<(f32, f32, f32)>, ribbons: &mut Vec<Ribbon>| {
            if run.len() >= 2 {
                ribbons.push(stroke_path(run, None));
            }
            run.clear();
        };
        while v < half_span {
            let v_next = (v + v_step).min(half_span);
            let vm = (v + v_next) * 0.5;
            let u_disp = u + amp * (vm * tau / lambda).sin(); // wavy: shift across-line
            let mid_x = cx + u_disp * -sin + vm * cos;
            let mid_y = cy + u_disp * cos + vm * sin;
            let oob = mid_x < 0.0 || mid_x >= w as f32 || mid_y < 0.0 || mid_y >= h as f32;
            let is_bridge = p.bridge_interval_px > 0.0
                && p.bridge_px > 0.0
                && (v - bridge_phase).rem_euclid(p.bridge_interval_px) < p.bridge_px;
            let width = if oob || is_bridge {
                0.0
            } else {
                cut_extent(tone(sample(map, w, h, mid_x, mid_y), p), p, max_cut)
            };
            if width <= 0.0 {
                flush(&mut run, &mut ribbons);
            } else {
                // Emit canvas knots for both segment endpoints (displaced centerline).
                let hw = width * 0.5;
                let knot = |vv: f32| {
                    let ud = u + amp * (vv * tau / lambda).sin();
                    (cx + ud * -sin + vv * cos, cy + ud * cos + vv * sin, hw)
                };
                if run.is_empty() {
                    run.push(knot(v));
                }
                run.push(knot(v_next));
            }
            v = v_next;
        }
        flush(&mut run, &mut ribbons);
    }
    ribbons
}

/// AM-halftone dots: one variable-radius circle per cell of the rotated screen grid
/// (period `spacing_px` in both grid axes). Radius from the shared constraint law
/// (`cut_extent` gives a diameter; radius is half), so the dead-zone drops faint dots
/// and `max_cut` keeps standing material between neighbours. Bridges are inert (a dot
/// is already an isolated island). Each dot is an N-gon closed `Ribbon`.
fn generate_dots(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
    max_cut: f32,
) -> Vec<Ribbon> {
    let (sin, cos) = angle_deg.to_radians().sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let max_cut = max_cut.max(0.0);
    let diag = ((w * w + h * h) as f32).sqrt();
    let half_span = diag / 2.0 + p.spacing_px;
    let n = (half_span / p.spacing_px).ceil() as i32;
    let mut ribbons = Vec::new();
    for gi in -n..=n {
        for gj in -n..=n {
            let (u, vv) = (gi as f32 * p.spacing_px, gj as f32 * p.spacing_px);
            let ccx = cx + u * -sin + vv * cos;
            let ccy = cy + u * cos + vv * sin;
            if ccx < 0.0 || ccx >= w as f32 || ccy < 0.0 || ccy >= h as f32 {
                continue;
            }
            // cut_extent is a diameter; halve for the radius. Dead-zone -> no dot.
            let d = cut_extent(tone(sample(map, w, h, ccx, ccy), p), p, max_cut);
            let r = d * 0.5;
            if r <= 0.0 {
                continue;
            }
            ribbons.push(emit_dot(ccx, ccy, r));
        }
    }
    ribbons
}

/// A circle as a closed N-gon `Ribbon`: ~1.5px chord target, clamped to [8,48] points.
/// Shared by `generate_dots` (variable radius) and `generate_bluenoise` (fixed radius).
fn emit_dot(cx: f32, cy: f32, r: f32) -> Ribbon {
    let tau = std::f32::consts::TAU;
    let seg = ((tau * r / 1.5).ceil() as usize).clamp(8, 48);
    let pts = (0..seg)
        .map(|k| {
            let a = k as f32 / seg as f32 * tau;
            (cx + r * a.cos(), cy + r * a.sin())
        })
        .collect();
    Ribbon { pts }
}

/// Blue-noise FM screen: fixed-size dots at variable FREQUENCY (denser = darker),
/// placed by Floyd–Steinberg error diffusion on a `spacing`-resolution grid — the
/// serpentine diffusion gives a blue-noise-like distribution with no grid, no moiré,
/// no rosettes, and (isolated fixed dots) forgiving of layer registration. Cut-safe
/// like `Dots`: a fixed radius <= max_cut/2 so dots never touch. Deterministic.
fn generate_bluenoise(map: &[f32], w: usize, h: usize, p: &LineParams, max_cut: f32) -> Vec<Ribbon> {
    let max_cut = max_cut.max(0.0);
    let sp = p.spacing_px.max(1.0);
    // Coarse grid at one cell per `spacing`. Each retained cell => one dot at its center.
    let gw = (w as f32 / sp).ceil().max(1.0) as usize;
    let gh = (h as f32 / sp).ceil().max(1.0) as usize;
    // Toned density per coarse cell (sample at the cell center). Dead-zone: below the
    // w_min fraction of pitch, force 0 so faint areas emit no sub-threshold dots.
    let dead = (p.w_min_px / sp).clamp(0.0, 1.0);
    let mut buf: Vec<f32> = (0..gw * gh)
        .map(|i| {
            let (gx, gy) = (i % gw, i / gw);
            let cxp = (gx as f32 + 0.5) * sp;
            let cyp = (gy as f32 + 0.5) * sp;
            let d = tone(sample(map, w, h, cxp, cyp), p);
            if d < dead { 0.0 } else { d }
        })
        .collect();
    // Snapshot the CLEAN per-cell tone before error diffusion mutates `buf`. Dot SIZE
    // keys off this (how dark the cell really is); `buf` becomes the placement signal.
    let tone_at = buf.clone();
    // Cut-safe radius ceiling: as large as fits without touching a neighbour (dots are
    // >= spacing apart on the grid), kerf-shrunk. All sizes clamp to this so the slider
    // can never produce an unsafe cut.
    let safe_r = ((max_cut * 0.5) - p.kerf_px * 0.5).max(0.0);
    if safe_r <= 0.0 {
        return Vec::new();
    }
    // AM range: max clamps down to the cut-safe ceiling; min clamps into
    // [dead-zone radius, max_r]. Default dot_{min,max}_px = INFINITY => both == safe_r
    // => the original fixed-size FM screen.
    let max_r = (p.shape_params.dot_max_px * 0.5).min(safe_r);
    // Lower bound is the dead-zone radius, but never above max_r (a big min-cut on a
    // tiny cut-safe ceiling would make lo > hi and panic clamp()). min_r <= max_r always.
    let lo = (p.w_min_px * 0.5).min(max_r);
    let min_r = (p.shape_params.dot_min_px * 0.5).clamp(lo, max_r);
    let mut ribbons = Vec::new();
    // Serpentine Floyd–Steinberg over the coarse grid.
    for gy in 0..gh {
        let l2r = gy % 2 == 0;
        for step in 0..gw {
            let gx = if l2r { step } else { gw - 1 - step };
            let idx = gy * gw + gx;
            let old = buf[idx];
            let new = if old >= 0.5 { 1.0 } else { 0.0 };
            let err = old - new;
            if new >= 0.5 {
                let cxp = (gx as f32 + 0.5) * sp;
                let cyp = (gy as f32 + 0.5) * sp;
                // Size by the cell's clean tone: lighter -> min_r, darker -> max_r.
                let t = tone_at[idx].clamp(0.0, 1.0);
                let r = min_r + (max_r - min_r) * t;
                ribbons.push(emit_dot(cxp, cyp, r));
            }
            // Diffuse error to not-yet-visited neighbours (mirrored for R->L rows).
            let dir: i32 = if l2r { 1 } else { -1 };
            let mut push = |dx: i32, dy: usize, frac: f32| {
                let nx = gx as i32 + dx * dir;
                let ny = gy + dy;
                if nx >= 0 && (nx as usize) < gw && ny < gh {
                    buf[ny * gw + nx as usize] += err * frac;
                }
            };
            push(1, 0, 7.0 / 16.0);
            push(-1, 1, 3.0 / 16.0);
            push(0, 1, 5.0 / 16.0);
            push(1, 1, 1.0 / 16.0);
        }
    }
    ribbons
}

/// Orientation hatch: a line screen whose local angle follows the image's dominant
/// edge orientation (structure tensor), QUANTIZED into `hatch_bins` angle bins. Each
/// bin's pixels are screened as an ordinary straight line-screen at the bin angle.
/// Within a bin, lines share one phase grid so they're always >= spacing apart; but
/// adjacent bins run on independent grids, so at a bin BOUNDARY two cuts could crowd
/// below min-material. A shared occupancy guard (dilated by min-material) drops any
/// boundary cut that would violate the gap — so material always connects, unlike free
/// streamlines which could enclose a patch. `angle_deg` offsets all bins so the
/// per-channel screen angles still separate the inks.
fn generate_hatch(
    map: &[f32],
    w: usize,
    h: usize,
    angle_deg: f32,
    p: &LineParams,
    max_cut: f32,
) -> Vec<Ribbon> {
    if w < 3 || h < 3 {
        return Vec::new();
    }
    let bins = p.shape_params.hatch_bins.clamp(1, 24) as usize;
    // Structure tensor: gradients, then blur the products to get a stable local
    // orientation. Reuse `gaussian`. Sigma ~ spacing keeps orientation coherent over a
    // screen cell.
    let sigma = p.spacing_px.max(1.0);
    let mut jxx = vec![0.0f32; w * h];
    let mut jyy = vec![0.0f32; w * h];
    let mut jxy = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let xi = x.clamp(1, w - 2);
            let yi = y.clamp(1, h - 2);
            let gx = map[yi * w + xi + 1] - map[yi * w + xi - 1];
            let gy = map[(yi + 1) * w + xi] - map[(yi - 1) * w + xi];
            jxx[y * w + x] = gx * gx;
            jyy[y * w + x] = gy * gy;
            jxy[y * w + x] = gx * gy;
        }
    }
    let jxx = gaussian(&jxx, w, h, sigma);
    let jyy = gaussian(&jyy, w, h, sigma);
    let jxy = gaussian(&jxy, w, h, sigma);
    // Per-pixel bin index from the dominant-orientation angle in [0,180).
    let pi = std::f32::consts::PI;
    let bin_of = |i: usize| -> usize {
        // Orientation of the tensor's principal axis; lines run ALONG the edge.
        let theta = 0.5 * (2.0 * jxy[i]).atan2(jxx[i] - jyy[i]); // [-pi/2, pi/2]
        let mut a = theta.rem_euclid(pi) / pi; // [0,1)
        if !a.is_finite() {
            a = 0.0;
        }
        ((a * bins as f32) as usize).min(bins - 1)
    };
    // Screen each bin's masked density as straight lines. `bin_of` gives the GRADIENT
    // orientation; lines should run ALONG the isophote (perpendicular to the gradient),
    // so rotate the screen angle by 90°. (+ channel `angle_deg` offset for ink separation.)
    //
    // Within a bin, lines share one phase grid so they're >= spacing apart. But adjacent
    // bins run at different angles on independent grids, so at a bin BOUNDARY two cuts
    // could land closer than min-material. Keep every line, but NARROW it near a seam:
    // a running proximity field holds each pixel's distance to the nearest already-placed
    // cut (capped at min-material = "no constraint"), and the screen caps each knot's
    // half-width to that distance. A line thus thins smoothly as it approaches another
    // cut and pinches to zero only on direct overlap — the min gap is preserved without
    // dropping whole lines (which thinned the seams and looked worse).
    let mut ribbons = Vec::new();
    // A pixel farther than (min-material + margin + max half-width) from any cut can
    // carry a full-width line: saturate the field there so it imposes no cap. The margin
    // matches the sampling slack applied in generate_quads_prox (v_step + 1).
    let v_step = (p.spacing_px * 0.5).clamp(0.5, 4.0);
    let sat = p.min_material_px.max(0.0) + v_step + 1.0 + max_cut.max(0.0) * 0.5;
    // proximity[i] = distance (px) to nearest placed cut, saturated at `sat`.
    let mut proximity = vec![sat; w * h];
    for b in 0..bins {
        let bin_angle = angle_deg + (b as f32 / bins as f32) * 180.0 + 90.0;
        let masked: Vec<f32> = (0..w * h)
            .map(|i| if bin_of(i) == b { map[i] } else { 0.0 })
            .collect();
        let bin_ribbons = generate_quads_prox(&masked, w, h, bin_angle, p, max_cut, Some(&proximity));
        // Stamp the cuts we just placed into the proximity field so later bins narrow
        // around them.
        for r in &bin_ribbons {
            stamp_proximity(r, &mut proximity, w, h, sat);
        }
        ribbons.extend(bin_ribbons);
    }
    ribbons
}

/// Lower `proximity` around ribbon `r`'s footprint: every pixel within `radius` px of
/// a cut pixel records its (rounded) distance to that cut, clamped so the field never
/// exceeds `radius` (beyond `radius` there's no width constraint). Row-major w×h.
fn stamp_proximity(r: &Ribbon, proximity: &mut [f32], w: usize, h: usize, radius: f32) {
    let rad = radius.ceil() as isize;
    let r2 = radius * radius;
    ribbon_pixels(r, w, h, |px, py| {
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                let d2 = (dx * dx + dy * dy) as f32;
                if d2 > r2 {
                    continue; // round disk
                }
                let (nx, ny) = (px as isize + dx, py as isize + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    let idx = ny as usize * w + nx as usize;
                    let d = d2.sqrt();
                    if d < proximity[idx] {
                        proximity[idx] = d;
                    }
                }
            }
        }
    });
}

/// Call `f(px, py)` for every interior pixel of a closed ribbon polygon (same
/// scanline even-odd fill as `rasterize`, but streaming to a callback so callers can
/// test/stamp without materialising a full mask).
fn ribbon_pixels(r: &Ribbon, w: usize, h: usize, mut f: impl FnMut(usize, usize)) {
    let n = r.pts.len();
    if n < 3 {
        return;
    }
    let miny = r.pts.iter().map(|p| p.1).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
    let maxy = (r.pts.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil() as isize)
        .clamp(0, h as isize - 1) as usize;
    let mut xs: Vec<f32> = Vec::new();
    for py in miny..=maxy {
        let yc = py as f32 + 0.5;
        xs.clear();
        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = r.pts[i];
            let (xj, yj) = r.pts[j];
            if (yi <= yc) != (yj <= yc) {
                xs.push(xi + (yc - yi) / (yj - yi) * (xj - xi));
            }
            j = i;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut k = 0;
        while k + 1 < xs.len() {
            let a = xs[k].ceil().max(0.0) as usize;
            let b = (xs[k + 1].floor() as isize).min(w as isize - 1);
            for px in a..=(b.max(a as isize) as usize).min(w - 1) {
                if (px as f32) >= xs[k] && (px as f32) <= xs[k + 1] {
                    f(px, py);
                }
            }
            k += 2;
        }
    }
}

/// Separable Gaussian blur of a row-major f32 map (reflect at edges). `sigma` in px.
fn gaussian(map: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return map.to_vec();
    }
    let rad = (3.0 * sigma).ceil() as i32;
    let kernel: Vec<f32> = (-rad..=rad)
        .map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp())
        .collect();
    let ksum: f32 = kernel.iter().sum();
    let refl = |i: i32, n: i32| {
        let mut i = i;
        if i < 0 { i = -i - 1; }
        if i >= n { i = 2 * n - i - 1; }
        i.clamp(0, n - 1) as usize
    };
    // Horizontal pass.
    let mut tmp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let xi = refl(x as i32 + k as i32 - rad, w as i32);
                acc += map[y * w + xi] * kv;
            }
            tmp[y * w + x] = acc / ksum;
        }
    }
    // Vertical pass.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                let yi = refl(y as i32 + k as i32 - rad, h as i32);
                acc += tmp[yi * w + x] * kv;
            }
            out[y * w + x] = acc / ksum;
        }
    }
    out
}

/// Difference-of-Gaussians edge map for the Contour-K mode. Works on the luminance
/// implied by the K density (K = 1 - max(r,g,b), so high K = dark = structural).
/// Returns a density map in [0,1] that's ~1 on sharp contrast boundaries and 0 on
/// flat tonal areas — the crisp comic-book line art the CMY fills sit under.
pub fn dog_edges(k_map: &[f32], w: usize, h: usize, sigma1: f32, sigma2: f32, threshold: f32) -> Vec<f32> {
    let g1 = gaussian(k_map, w, h, sigma1.min(sigma2));
    let g2 = gaussian(k_map, w, h, sigma1.max(sigma2));
    g1.iter()
        .zip(g2.iter())
        .map(|(&a, &b)| {
            // Positive DoG = a dark-side edge (fine blur darker than coarse blur).
            let diff = a - b;
            if diff > threshold { 1.0 } else { 0.0 }
        })
        .collect()
}

/// One ribbon set per channel, applying auto-levels if requested. Translucent
/// (untamed) inks screen straight; tamed (opaque) inks — black, or an opaque spot —
/// go through the taming path (UCR + deep-shadow clip + tight width cap in Tonal mode,
/// or DoG edge contours in Contour mode). N-general: for the CMYK channel list this
/// reproduces the historical C/M/Y-then-K behaviour exactly.
pub fn generate_all(
    channels: &[crate::cmyk::Channel],
    w: usize,
    h: usize,
    p: &LineParams,
    auto: bool,
) -> Vec<Vec<Ribbon>> {
    // Under-inks for UCR: every untamed channel's density (what a tamed ink sits over).
    let under: Vec<&[f32]> = channels
        .iter()
        .filter(|c| !c.tamed)
        .map(|c| c.density.as_slice())
        .collect();
    channels
        .iter()
        .map(|ch| {
            if ch.tamed {
                // ponytail: tamed inks ignore `load` — K already has its own ceiling
                // (`k_width_frac`), which is a cap PAIRED with a tonal rescale
                // (`deep_clip` + `k_gamma`), not a linear load scale. Upgrade path:
                // express `k_width_frac` as a `load` once `deep_clip` is proven
                // equivalent to that rescale.
                generate_tamed(&ch.density, &under, w, h, p, ch.angle)
            } else {
                let cp = if auto {
                    // ponytail: with auto on, `auto_levels` maps this channel's p99.5
                    // to full cut before `load` scales it, so the mid-tone effect of a
                    // lowered load reads weaker than expected. Stiffness is still
                    // bounded, because `load` also scales the ceiling below.
                    let (wp, bp) = crate::halftone::auto_levels(&ch.density, 0.005, 0.995);
                    LineParams { white_point: wp, black_point: bp, load: ch.load, ..*p }
                } else {
                    LineParams { load: ch.load, ..*p }
                };
                // Scale the ceiling by the same factor as the ramp in `cut_extent`.
                generate_shape(&ch.density, w, h, ch.angle, &cp, max_cut_for(&cp) * ch.load)
            }
        })
        .collect()
}

/// Screen a tamed (opaque) ink under whichever taming mode is selected. `under` are the
/// translucent inks beneath it (for UCR).
fn generate_tamed(map: &[f32], under: &[&[f32]], w: usize, h: usize, p: &LineParams, angle: f32) -> Vec<Ribbon> {
    let k_cap = (p.k_width_frac * p.spacing_px).min(p.spacing_px - p.min_material_px);
    // Tone is baked into the effective map (clip+gamma or DoG), so screen it with
    // identity levels/gamma and no S-curve — don't double-apply.
    let kp = LineParams { white_point: 0.0, black_point: 1.0, gamma: 1.0, scurve: 0.0, ..*p };
    let eff = match p.k_mode {
        KMode::Tonal => effective_k(map, under, p),
        KMode::Contour => dog_edges(map, w, h, p.dog_sigma1, p.dog_sigma2, p.dog_threshold),
    };
    generate_shape(&eff, w, h, angle, &kp, k_cap)
}

/// Rasterize ribbons onto a channel keep-mask (for the composite preview) by
/// SCANLINE fill: for each row a ribbon spans, find its edge crossings and fill the
/// spans between them. This is O(edges + filled_px) per ribbon, versus the old
/// per-pixel point-in-poly over the whole (huge, mostly-empty) bounding box of a
/// long diagonal strip — which was the preview's dominant cost.
fn rasterize(ribbons: &[Ribbon], w: usize, h: usize) -> Vec<bool> {
    let mut mask = vec![false; w * h];
    let mut xs: Vec<f32> = Vec::new();
    for r in ribbons {
        let n = r.pts.len();
        if n < 3 {
            continue;
        }
        let miny = r.pts.iter().map(|p| p.1).fold(f32::MAX, f32::min).floor().max(0.0) as usize;
        let maxy = (r.pts.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil() as isize)
            .clamp(0, h as isize - 1) as usize;
        for py in miny..=maxy {
            let yc = py as f32 + 0.5;
            xs.clear();
            // Edge crossings at this scanline (half-open to avoid double-counting vertices).
            let mut j = n - 1;
            for i in 0..n {
                let (xi, yi) = r.pts[i];
                let (xj, yj) = r.pts[j];
                if (yi <= yc) != (yj <= yc) {
                    xs.push(xi + (yc - yi) / (yj - yi) * (xj - xi));
                }
                j = i;
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Fill between consecutive crossing pairs (even-odd).
            let mut k = 0;
            while k + 1 < xs.len() {
                let a = xs[k].ceil().max(0.0) as usize;
                let b = (xs[k + 1].floor() as isize).min(w as isize - 1);
                for px in a..=(b.max(a as isize) as usize).min(w - 1) {
                    if (px as f32) >= xs[k] && (px as f32) <= xs[k + 1] {
                        mask[py * w + px] = true;
                    }
                }
                k += 2;
            }
        }
    }
    mask
}

/// Composite the N channel ribbon sets into an RGB preview by MULTIPLICATIVE ink
/// stacking: start white, and for each ink covering a pixel multiply the running RGB
/// by that ink's transmission colour (`display_rgb`). This generalizes to any N inks
/// and any colours; for the four CMYK inks (C=(0,1,1), M=(1,0,1), Y=(1,1,0), K=0) it is
/// exactly the old `cmyk_to_rgb` subtractive mix for binary coverage.
/// Also returns the per-layer stiffness readout, measured off the masks this render
/// already built — so it costs one extra linear pass, not a second screening.
pub fn render_preview(
    channels: &[crate::cmyk::Channel],
    w: usize,
    h: usize,
    p: &LineParams,
    auto: bool,
) -> (image::RgbImage, Vec<crate::fragility::Fragility>) {
    let quads = generate_all(channels, w, h, p, auto);
    let masks: Vec<Vec<bool>> = quads.iter().map(|q| rasterize(q, w, h)).collect();
    // NOTE polarity: `rasterize` marks where the CUT is (it drives ink painting
    // below), so the keep-mask `fragility::measure` wants is the inverse. Stencil
    // masks are already keep-polarity — don't invert those.
    let frag = masks
        .iter()
        .map(|m| {
            let keep: Vec<bool> = m.iter().map(|&cut| !cut).collect();
            crate::fragility::measure(&keep, w, h)
        })
        .collect();
    let mut buf = image::RgbImage::new(w as u32, h as u32);
    for (idx, px) in buf.pixels_mut().enumerate() {
        let mut rgb = [1.0f32; 3]; // start white
        for (ci, ch) in channels.iter().enumerate() {
            if masks[ci][idx] {
                for k in 0..3 {
                    rgb[k] *= ch.display_rgb[k];
                }
            }
        }
        *px = image::Rgb([
            (rgb[0] * 255.0) as u8,
            (rgb[1] * 255.0) as u8,
            (rgb[2] * 255.0) as u8,
        ]);
    }
    (buf, frag)
}

/// Write the 4 analytic cut SVGs + composite preview PNG. Each sheet carries corner
/// punch holes + crosshairs + a layer label so the four physical layers pin
/// together. `paper` (with `margin_mm`) sizes the SVG to a real A-size sheet with
/// the artwork centered inside the margin; `None` keeps the raw px viewBox.
pub fn export(
    channels: &[crate::cmyk::Channel],
    w: usize,
    h: usize,
    p: &LineParams,
    auto: bool,
    paper: Option<crate::svg::Paper>,
    margin_mm: f32,
    prefix: &str,
) -> Result<(), String> {
    let ribbons = generate_all(channels, w, h, p, auto);
    let layout = paper.map(|pp| crate::svg::PaperLayout::fit(pp, margin_mm, w, h));
    // Punch-hole radius: at least the min-material cut width (converted px->mm via
    // the sheet scale) so the hole is a real, visible, pinnable circle — never a
    // hairline. A 2mm floor keeps it usable even when min-material is tiny.
    let hole_r_mm = layout
        .map(|l| (p.min_material_px * l.scale).max(2.0))
        .unwrap_or(2.0);
    let n = channels.len();
    for (i, ch) in channels.iter().enumerate() {
        // SVG fill: the ink's display colour as a hex string.
        let fill = format!(
            "#{:02x}{:02x}{:02x}",
            (ch.display_rgb[0] * 255.0) as u8,
            (ch.display_rgb[1] * 255.0) as u8,
            (ch.display_rgb[2] * 255.0) as u8
        );
        let label = format!("{} {}/{}", ch.name.to_uppercase(), i + 1, n);
        let doc = crate::svg::ribbons_to_svg(&ribbons[i], w, h, &fill, layout, &label, hole_r_mm);
        let path = format!("{prefix}_{}.svg", ch.suffix);
        std::fs::write(&path, doc).map_err(|e| format!("write {path}: {e}"))?;
        eprintln!("wrote {path} ({} cut ribbons)", ribbons[i].len());
    }
    let (buf, frag) = render_preview(channels, w, h, p, auto);
    let ppath = format!("{prefix}_preview.png");
    buf.save(&ppath).map_err(|e| format!("write {ppath}: {e}"))?;
    eprintln!("wrote {ppath} (analytic composite preview)");
    // Stiffness per sheet: `neck` is the thinnest standing material. A sheet under
    // ~1px of neck is lace and will lift or tear when sprayed.
    for (ch, f) in channels.iter().zip(&frag) {
        eprintln!(
            "layer {}: cut {:.0}%, neck {:.1}px{}",
            ch.suffix,
            f.removed * 100.0,
            f.neck_px,
            if f.neck_px < 1.0 { "  <- fragile" } else { "" }
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> LineParams {
        LineParams {
            shape: Shape::Lines,
            shape_params: ShapeParams::default(),
            spacing_px: 10.0,
            w_min_px: 1.0,
            min_material_px: 2.0,
            kerf_px: 0.0,
            bridge_interval_px: 0.0,
            bridge_px: 0.0,
            white_point: 0.0,
            black_point: 1.0,
            gamma: 1.0,
            scurve: 0.0,
            load: 1.0,
            k_mode: KMode::Tonal,
            k_deep_clip: 0.75,
            k_gamma: 2.0,
            k_width_frac: 0.40,
            ucr: 0.8,
            dog_sigma1: 1.0,
            dog_sigma2: 2.0,
            dog_threshold: 0.05,
        }
    }

    #[test]
    fn scurve_pins_endpoints_and_is_monotone() {
        assert!((scurve(0.0, 8.0) - 0.0).abs() < 1e-4);
        assert!((scurve(1.0, 8.0) - 1.0).abs() < 1e-4);
        assert_eq!(scurve(0.37, 0.0), 0.37); // k=0 identity
        // Monotone increasing.
        let mut prev = -1.0;
        for i in 0..=10 {
            let y = scurve(i as f32 / 10.0, 6.0);
            assert!(y >= prev, "monotone at {i}");
            prev = y;
        }
        // S shrinks shadows: mid-low input maps below identity.
        assert!(scurve(0.3, 8.0) < 0.3, "shadows compressed");
    }

    /// Cross-line extent (width) of a ribbon at angle 0: max-min of its y coords.
    fn ribbon_width(r: &Ribbon) -> f32 {
        let ys = r.pts.iter().map(|p| p.1);
        ys.clone().fold(f32::MIN, f32::max) - ys.fold(f32::MAX, f32::min)
    }

    #[test]
    fn dead_zone_zeros_faint_ink() {
        // Uniform faint density whose raw width < w_min => no ribbons at all.
        // raw = d*spacing = 0.05*10 = 0.5 < w_min(1.0).
        let map = vec![0.05f32; 40 * 40];
        let r = generate_quads(&map, 40, 40, 0.0, &base());
        assert!(r.is_empty(), "faint ink below dead-zone cuts nothing");
        // Stronger ink (raw 5.0 >= 1.0) does cut.
        let map2 = vec![0.5f32; 40 * 40];
        let r2 = generate_quads(&map2, 40, 40, 0.0, &base());
        assert!(!r2.is_empty(), "above dead-zone cuts");
    }

    #[test]
    fn coalesces_into_one_ribbon_per_line() {
        // A uniform field with no bridges: each screen line is ONE continuous cut,
        // so #ribbons == #lines that fall on the canvas — far fewer than the ~w/v_step
        // separate segments the old per-quad emit produced. This is the speed win.
        let map = vec![0.5f32; 40 * 40];
        let r = generate_quads(&map, 40, 40, 0.0, &base());
        // 40px tall / 10px spacing => ~4-5 visible lines. Each is a single ribbon
        // with many knots. Assert we're well under the old segment count (~4/line).
        assert!(r.len() <= 6, "one ribbon per line, got {}", r.len());
        assert!(r[0].pts.len() > 4, "ribbon spans many knots (a run), got {}", r[0].pts.len());
    }

    #[test]
    fn width_clamped_to_max_cut() {
        // Full ink (raw = 10) must clamp to max_cut = spacing - min_material = 8,
        // leaving a material bridge. Check emitted ribbon cross-line width.
        let map = vec![1.0f32; 40 * 40];
        let r = generate_quads(&map, 40, 40, 0.0, &base());
        assert!(!r.is_empty());
        let width = ribbon_width(&r[0]);
        assert!(width <= 8.01, "width clamped to max_cut, got {width}");
        assert!(width >= 7.99, "full ink reaches max_cut, got {width}");
    }

    #[test]
    fn kerf_shrinks_slot() {
        // With kerf, the commanded width is raw - kerf. d=0.5 -> raw=5, kerf=1 -> 4.
        let mut p = base();
        p.kerf_px = 1.0;
        let map = vec![0.5f32; 40 * 40];
        let r = generate_quads(&map, 40, 40, 0.0, &p);
        assert!((ribbon_width(&r[0]) - 4.0).abs() < 0.01, "kerf-shrunk width");
    }

    #[test]
    fn bridges_split_lines_into_multiple_ribbons() {
        // With bridges on, a single line's continuous ribbon is broken into several
        // shorter ribbons (one per uncut run between tabs) — so bridged output has
        // MORE ribbons than the one-per-line no-bridge case, but each is still a
        // coalesced run, not per-segment quads.
        let mut p = base();
        p.bridge_interval_px = 20.0;
        p.bridge_px = 4.0;
        p.min_material_px = 0.0; // let full width through
        let map = vec![1.0f32; 200 * 200];
        let r = generate_quads(&map, 200, 200, 0.0, &p);
        assert!(!r.is_empty(), "bridged lines still emit cut runs");
        let mut p0 = p;
        p0.bridge_interval_px = 0.0;
        let r0 = generate_quads(&map, 200, 200, 0.0, &p0);
        assert!(r.len() > r0.len(), "bridges split lines into more runs: {} > {}", r.len(), r0.len());
        // But still coalesced: far fewer than segment-per-quad (v_step=4 over 200px
        // ~= 50 segments/line * ~20 lines = ~1000). Bridged run count is a fraction.
        assert!(r.len() < 400, "runs stay coalesced, got {}", r.len());
    }

    #[test]
    fn deep_clip_kills_midtones_keeps_blacks() {
        // Below Tk=0.75 -> 0 (mid-tones excluded). Above -> steep gamma, so even a
        // fairly dark 0.85 stays thin (well under linear).
        assert_eq!(deep_clip(0.6, 0.75, 2.0), 0.0, "mid-tone clipped");
        assert_eq!(deep_clip(0.75, 0.75, 2.0), 0.0, "at threshold clips");
        let d = deep_clip(0.85, 0.75, 2.0); // (0.10/0.25)^2 = 0.16
        assert!((d - 0.16).abs() < 1e-4, "steep gamma keeps K thin: {d}");
        assert_eq!(deep_clip(1.0, 0.75, 2.0), 1.0, "absolute black fires full");
    }

    #[test]
    fn ucr_suppresses_k_under_cmy_overlap() {
        // Where C,M,Y all overlap dark, K is subtracted. k=0.9, cmy_min=0.5, ucr=0.8
        // -> k_ucr = 0.9 - 0.4 = 0.5, which is below Tk(0.75) => clipped to 0.
        let mut p = base();
        p.ucr = 0.8;
        let k = vec![0.9f32; 4];
        let c = vec![0.5f32; 4];
        let m = vec![0.5f32; 4];
        let y = vec![0.5f32; 4];
        let eff = effective_k(&k, &[&c, &m, &y], &p);
        assert!(eff.iter().all(|&v| v == 0.0), "K suppressed under CMY overlap: {eff:?}");
        // With no CMY (under_min=0), the same K survives the clip.
        let zero = vec![0.0f32; 4];
        let eff2 = effective_k(&k, &[&zero, &zero, &zero], &p);
        assert!(eff2.iter().all(|&v| v > 0.0), "K fires where CMY misses: {eff2:?}");
    }

    #[test]
    fn k_width_capped_below_cmy() {
        // Full black K, tonal mode: the K ribbon width must be <= k_width_frac*pitch,
        // which is far below the CMY max_cut. Uses generate_all so the K path runs.
        let n = 60 * 60;
        let layers = crate::cmyk::Layers {
            c: vec![0.0; n], m: vec![0.0; n], y: vec![0.0; n], k: vec![1.0; n],
        };
        let mut p = base();
        p.ucr = 0.0; // don't suppress; we want K to fire
        p.k_width_frac = 0.40; // cap = 4px at pitch 10
        let chans = crate::cmyk::channels(
            &layers,
            crate::cmyk::Inks::Cmyk,
            &[15.0, 75.0, 0.0, 0.0],
            &crate::cmyk::Inks::Cmyk.default_loads(),
        );
        let out = generate_all(&chans, 60, 60, &p, false);
        let krib = &out[3];
        assert!(!krib.is_empty(), "full black produces K cut");
        // angle 0 for K here: width = y-extent. Must be <= 4px cap (not the 8px CMY cap).
        let maxw = krib.iter().map(|r| {
            let ys = r.pts.iter().map(|pp| pp.1);
            ys.clone().fold(f32::MIN, f32::max) - ys.fold(f32::MAX, f32::min)
        }).fold(0.0f32, f32::max);
        assert!(maxw <= 4.01, "K width capped to 40% pitch, got {maxw}");
    }

    #[test]
    fn degenerate_levels_do_not_panic_the_rasterizer() {
        // Repro: white-point == black-point (both 0.0 in the GUI). A map pixel equal
        // to the coincident point gave `levels` 0.0/0.0 = NaN -> NaN ribbon coords ->
        // partial_cmp().unwrap() panic in rasterize. The map MUST MIX pixels AT the
        // coincident point (density 0.0 -> old 0/0 NaN) with pixels above it (finite),
        // so one ribbon gets BOTH finite and NaN coords: the finite ones give it a
        // valid scanline range, the NaN crossing inside it panicked the float sort.
        // A uniform map would NOT reproduce (all-NaN bbox collapses to empty). We build
        // a synthetic density channel with that exact mix — no image file needed.
        let (w, h) = (64usize, 64usize);
        // Horizontal gradient: left column density 0.0 (the coincident point), rising
        // to the right. Every screen ribbon spans the 0.0 band and the finite band.
        let density: Vec<f32> = (0..w * h)
            .map(|i| (i % w) as f32 / (w - 1) as f32)
            .collect();
        let ch = crate::cmyk::Channel {
            density,
            angle: 15.0,
            load: 1.0,
            display_rgb: [0.0, 1.0, 1.0],
            name: "Cyan",
            suffix: "c",
            tamed: false,
        };
        let p = LineParams { white_point: 0.0, black_point: 0.0, ..base() };
        // render_preview runs generate_all + rasterize — the exact GUI path.
        let (img, _) = render_preview(&[ch], w, h, &p, false);
        assert_eq!(img.dimensions(), (w as u32, h as u32), "renders without panicking");
    }

    /// Total filled pixels for a shape over a uniform density field of value `d`.
    fn coverage(shape: Shape, d: f32) -> usize {
        let (w, h) = (80usize, 80usize);
        let map = vec![d; w * h];
        let p = LineParams { shape, ..base() };
        let ribbons = generate_shape(&map, w, h, 0.0, &p, max_cut_for(&p));
        rasterize(&ribbons, w, h).iter().filter(|&&b| b).count()
    }

    /// Every emitted ribbon is a valid closed polygon (>=3 points).
    fn all_valid(shape: Shape, d: f32) -> bool {
        let (w, h) = (80usize, 80usize);
        let map = vec![d; w * h];
        let p = LineParams { shape, ..base() };
        generate_shape(&map, w, h, 0.0, &p, max_cut_for(&p))
            .iter()
            .all(|r| r.pts.len() >= 3)
    }

    #[test]
    fn every_shape_darker_means_more_ink() {
        // The core halftone invariant: more density -> more filled area, for each shape.
        for shape in [Shape::Lines, Shape::Wavy, Shape::Dots, Shape::BlueNoise, Shape::Hatch] {
            let light = coverage(shape, 0.25);
            let dark = coverage(shape, 0.85);
            assert!(dark > light, "{:?}: dark {dark} should exceed light {light}", shape as u8);
            assert!(all_valid(shape, 0.6), "{:?}: all ribbons closed polygons", shape as u8);
        }
    }

    #[test]
    fn every_shape_respects_dead_zone() {
        // Faint ink (raw = 0.05*10 = 0.5 < w_min 1.0) cuts nothing, for every shape.
        for shape in [Shape::Lines, Shape::Wavy, Shape::Dots, Shape::BlueNoise, Shape::Hatch] {
            assert_eq!(coverage(shape, 0.05), 0, "{:?}: dead-zone suppresses faint ink", shape as u8);
        }
    }

    #[test]
    fn bluenoise_is_deterministic_and_dots_never_merge() {
        // Deterministic: same input -> identical ribbons (no RNG, error diffusion).
        let (w, h) = (80usize, 80usize);
        let map = vec![0.6f32; w * h];
        let p = LineParams { shape: Shape::BlueNoise, ..base() };
        let a = generate_bluenoise(&map, w, h, &p, max_cut_for(&p));
        let b = generate_bluenoise(&map, w, h, &p, max_cut_for(&p));
        assert_eq!(a.len(), b.len(), "deterministic dot count");
        assert!(!a.is_empty(), "mid-tone produces dots");
        // Fixed radius: every dot's diameter <= max_cut (never touch a neighbour).
        for r in &a {
            let xs = r.pts.iter().map(|q| q.0);
            let dia = xs.clone().fold(f32::MIN, f32::max) - xs.fold(f32::MAX, f32::min);
            assert!(dia <= 8.01, "blue-noise dot diameter {dia} within max_cut");
        }
    }

    // Diameter (x-extent) and center-x of a dot ribbon.
    fn dot_dia_cx(r: &Ribbon) -> (f32, f32) {
        let mut lo = f32::MAX;
        let mut hi = f32::MIN;
        for q in &r.pts {
            lo = lo.min(q.0);
            hi = hi.max(q.0);
        }
        (hi - lo, (lo + hi) * 0.5)
    }

    #[test]
    fn bluenoise_dot_size_tracks_tone() {
        // Horizontal tone ramp: dark on the LEFT (x small) -> light on the RIGHT.
        // With an open size range, dots on the dark side must be bigger than on the
        // light side. base(): spacing 10, min_material 2, kerf 0 -> safe_r = 4.
        let (w, h) = (200usize, 40usize);
        let map: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / (w - 1) as f32;
                1.0 - x // left dark (1.0), right light (0.0)
            })
            .collect();
        let mut p = LineParams { shape: Shape::BlueNoise, ..base() };
        p.shape_params.dot_min_px = 1.0; // min_r = 0.5
        p.shape_params.dot_max_px = 8.0; // max_r = 4.0 (clamped to safe_r)
        let dots = generate_bluenoise(&map, w, h, &p, max_cut_for(&p));
        assert!(!dots.is_empty(), "ramp produces dots");

        // Mean radius of dots in the dark-left third vs light-right third.
        let (mut ldia, mut ln, mut rdia, mut rn) = (0.0f32, 0u32, 0.0f32, 0u32);
        for d in &dots {
            let (dia, cx) = dot_dia_cx(d);
            if cx < w as f32 / 3.0 { ldia += dia; ln += 1; }
            else if cx > 2.0 * w as f32 / 3.0 { rdia += dia; rn += 1; }
        }
        assert!(ln > 0 && rn > 0, "dots on both sides ({ln} left, {rn} right)");
        let (lmean, rmean) = (ldia / ln as f32, rdia / rn as f32);
        assert!(lmean > rmean + 1.0, "dark dots bigger: left {lmean} vs right {rmean}");
    }

    #[test]
    fn bluenoise_no_panic_when_min_cut_exceeds_safe_radius() {
        // Regression: a large min-cut (dead-zone radius) on a small cut-safe ceiling
        // used to make clamp(lo > hi) panic. Must just produce dots at max_r.
        let (w, h) = (40usize, 40usize);
        let map = vec![1.0f32; w * h];
        let mut p = LineParams { shape: Shape::BlueNoise, ..base() };
        p.w_min_px = 6.0;                 // dead-zone radius 3.0 ...
        p.min_material_px = 6.0;          // ... but safe_r = (10-6)/2 = 2.0 < 3.0
        p.shape_params.dot_min_px = 0.0;
        p.shape_params.dot_max_px = 1000.0;
        let dots = generate_bluenoise(&map, w, h, &p, max_cut_for(&p)); // must not panic
        for d in &dots {
            let (dia, _) = dot_dia_cx(d);
            assert!(dia <= 4.01, "diameter {dia} clamped to max_cut even when min-cut is large");
        }
    }

    #[test]
    fn bluenoise_dots_never_merge_with_max_size() {
        // Full black + huge max_dot: every dot must still clamp to <= max_cut, and no
        // two placed dots overlap (centres are >= spacing apart on the grid).
        let (w, h) = (80usize, 80usize);
        let map = vec![1.0f32; w * h];
        let mut p = LineParams { shape: Shape::BlueNoise, ..base() };
        p.shape_params.dot_min_px = 0.0;
        p.shape_params.dot_max_px = 1000.0; // clamps to safe_r
        let dots = generate_bluenoise(&map, w, h, &p, max_cut_for(&p));
        assert!(!dots.is_empty(), "black produces dots");
        let mut centers = Vec::new();
        for d in &dots {
            let (dia, cx) = dot_dia_cx(d);
            assert!(dia <= 8.01, "diameter {dia} clamped to max_cut");
            // center y from the ribbon too
            let cy = d.pts.iter().map(|q| q.1).sum::<f32>() / d.pts.len() as f32;
            centers.push((cx, cy));
        }
        // No two dots closer than spacing (10) -> radii (<=4 each) can't overlap.
        for i in 0..centers.len() {
            for j in (i + 1)..centers.len() {
                let dx = centers[i].0 - centers[j].0;
                let dy = centers[i].1 - centers[j].1;
                assert!((dx * dx + dy * dy).sqrt() >= 9.99, "dots >= spacing apart");
            }
        }
    }

    #[test]
    fn hatch_lines_follow_a_diagonal_edge() {
        // A field split by a diagonal edge -> in the textured region the hatch lines
        // should run roughly ALONG the edge (~45°), not across it. Check the dominant
        // emitted segment direction is closer to 45° than to 135°.
        let (w, h) = (80usize, 80usize);
        // Diagonal ramp: value depends on (x - y) -> gradient points along (1,-1),
        // edge orientation is the (1,1) diagonal (45°).
        let map: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                (((x - y) / w as f32) * 0.5 + 0.5).clamp(0.0, 1.0)
            })
            .collect();
        let p = LineParams { shape: Shape::Hatch, ..base() };
        let ribbons = generate_hatch(&map, w, h, 0.0, &p, max_cut_for(&p));
        assert!(!ribbons.is_empty(), "hatch produces ribbons over a ramp");
        assert!(ribbons.iter().all(|r| r.pts.len() >= 3), "all ribbons closed");
        // Dominant direction: sum unit segment vectors folded onto [0,180) via doubling.
        let (mut sx, mut sy) = (0.0f32, 0.0f32);
        for r in &ribbons {
            for w2 in r.pts.windows(2) {
                let (dx, dy) = (w2[1].0 - w2[0].0, w2[1].1 - w2[0].1);
                let a = dy.atan2(dx) * 2.0; // double-angle: direction, not sense
                let len = (dx * dx + dy * dy).sqrt();
                sx += len * a.cos();
                sy += len * a.sin();
            }
        }
        let dom = 0.5 * sy.atan2(sx); // back to [−90,90)
        let deg = dom.to_degrees().rem_euclid(180.0);
        // Edge orientation is 45°; accept within a bin's worth (~30°) of it.
        let d45 = (deg - 45.0).abs().min(180.0 - (deg - 45.0).abs());
        assert!(d45 < 30.0, "hatch aligns near the 45° edge, got {deg:.1}°");
    }

    #[test]
    fn hatch_respects_min_spacing_across_bins() {
        // A swirl activates many orientation bins, so bin boundaries abound — the exact
        // place two cuts from different bins could crowd. Verify (independently of the
        // guard's own mask) that no two DISTINCT ribbons have pixels closer than
        // min_material: label each ribbon's pixels, then scan a min_material disk around
        // every cut pixel for a different label.
        let (w, h) = (96usize, 96usize);
        let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
        let map: Vec<f32> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32 - cx, (i / w) as f32 - cy);
                // Swirl: brightness follows the polar angle -> gradient direction sweeps
                // through all orientations, exercising every bin and their boundaries.
                (y.atan2(x).sin() * 0.5 + 0.5).clamp(0.0, 1.0)
            })
            .collect();
        let p = LineParams { shape: Shape::Hatch, min_material_px: 2.0, ..base() };
        let ribbons = generate_hatch(&map, w, h, 0.0, &p, max_cut_for(&p));
        assert!(ribbons.len() > 5, "swirl activates several bins: {} ribbons", ribbons.len());

        // Label grid: which ribbon (1-based) owns each pixel; 0 = uncut.
        let mut label = vec![0u32; w * h];
        for (id, r) in ribbons.iter().enumerate() {
            ribbon_pixels(r, w, h, |px, py| label[py * w + px] = id as u32 + 1);
        }
        // The guard enforces min_material between DISTINCT cuts. Scan just inside that
        // radius (leave a 1px slack for rasterization rounding at shared edges).
        let excl = p.min_material_px;
        let rad = (excl - 1.0).floor().max(0.0) as isize;
        let r2 = (excl - 1.0).max(0.0).powi(2);
        for py in 0..h as isize {
            for px in 0..w as isize {
                let me = label[py as usize * w + px as usize];
                if me == 0 { continue; }
                for dy in -rad..=rad {
                    for dx in -rad..=rad {
                        if (dx * dx + dy * dy) as f32 > r2 { continue; }
                        let (nx, ny) = (px + dx, py + dy);
                        if nx < 0 || ny < 0 || nx >= w as isize || ny >= h as isize { continue; }
                        let other = label[ny as usize * w + nx as usize];
                        assert!(
                            other == 0 || other == me,
                            "cut {me} within {excl}px of cut {other} at ({px},{py}) — min-spacing violated",
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dots_never_merge() {
        // Full black: each dot's diameter must stay <= max_cut (spacing-min_material=8),
        // so adjacent dots keep standing material between them (no merged blob).
        let (w, h) = (60usize, 60usize);
        let map = vec![1.0f32; w * h];
        let p = LineParams { shape: Shape::Dots, ..base() };
        let dots = generate_dots(&map, w, h, 0.0, &p, max_cut_for(&p));
        assert!(!dots.is_empty(), "full black produces dots");
        for r in &dots {
            let xs = r.pts.iter().map(|q| q.0);
            let dia = xs.clone().fold(f32::MIN, f32::max) - xs.fold(f32::MAX, f32::min);
            assert!(dia <= 8.01, "dot diameter {dia} within max_cut");
        }
    }

    #[test]
    fn dog_finds_edges_not_flats() {
        // A flat field yields no DoG edges; a sharp step yields edges at the boundary.
        let (w, h) = (20usize, 20usize);
        let flat = vec![0.5f32; w * h];
        let e_flat = dog_edges(&flat, w, h, 1.0, 2.0, 0.02);
        assert!(e_flat.iter().all(|&v| v == 0.0), "flat area has no edges");
        // Left half 0, right half 1 -> edge near the middle column.
        let mut step = vec![0.0f32; w * h];
        for y in 0..h { for x in w/2..w { step[y * w + x] = 1.0; } }
        let e_step = dog_edges(&step, w, h, 1.0, 2.0, 0.02);
        assert!(e_step.iter().any(|&v| v > 0.0), "sharp step produces edges");
    }

    /// The magenta-heavy case: pulling ONE ink's load back must shrink that ink's
    /// widest cut (so its sheet stays stiff) and leave every other ink bit-for-bit
    /// untouched. The exact-equality check on C also pins the backwards-compat
    /// promise: load 1.0 is an identity, not an approximation.
    #[test]
    fn per_ink_load_shrinks_only_that_ink() {
        let (w, h) = (40usize, 40usize);
        let mk = |d: f32, load: f32, name: &'static str| crate::cmyk::Channel {
            density: vec![d; w * h],
            angle: 0.0,
            load,
            display_rgb: [1.0, 0.0, 1.0],
            name,
            suffix: "x",
            tamed: false,
        };
        let p = base(); // spacing 10, min_material 2 => full budget 8
        let full = generate_all(&[mk(1.0, 1.0, "M"), mk(0.5, 1.0, "C")], w, h, &p, false);
        let cut = generate_all(&[mk(1.0, 0.5, "M"), mk(0.5, 1.0, "C")], w, h, &p, false);
        let widest = |rs: &[Ribbon]| rs.iter().map(ribbon_width).fold(0.0f32, f32::max);

        // Saturated M clips at the budget; half load halves it.
        assert!((widest(&full[0]) - 8.0).abs() < 0.01, "M full: {}", widest(&full[0]));
        assert!((widest(&cut[0]) - 4.0).abs() < 0.01, "M loaded: {}", widest(&cut[0]));
        // Standing material — the neck the stiffness readout measures — improved.
        assert!(
            p.spacing_px - widest(&cut[0]) > p.spacing_px - widest(&full[0]),
            "lowering M's load must widen M's standing material"
        );
        // C is untouched, bit-for-bit: per-ink isolation AND load-1.0 identity.
        assert_eq!(full[1].len(), cut[1].len(), "C ribbon count unchanged");
        for (a, b) in full[1].iter().zip(cut[1].iter()) {
            assert_eq!(a.pts, b.pts, "C geometry untouched by M's load");
        }
    }

    /// The tune loop the GUI relies on: neck width must be monotone non-decreasing
    /// as load drops, so dragging the slider can never make a sheet more fragile.
    #[test]
    fn lower_load_never_worsens_neck() {
        let (w, h) = (48usize, 48usize);
        let p = base();
        let mut prev = 0.0f32;
        for load in [0.4f32, 0.6, 0.8, 1.0] {
            let ch = crate::cmyk::Channel {
                density: vec![0.9; w * h],
                angle: 0.0,
                load,
                display_rgb: [1.0, 0.0, 1.0],
                name: "M",
                suffix: "m",
                tamed: false,
            };
            let (_, frag) = render_preview(&[ch], w, h, &p, false);
            let neck = frag[0].neck_px;
            if prev > 0.0 {
                assert!(neck <= prev + 0.01, "neck must not grow with load: {neck} > {prev}");
            }
            prev = neck;
        }
    }
}
