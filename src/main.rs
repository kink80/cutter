//! Laser halftone vectorizer: raster image -> 4 rotated CMYK halftone SVGs.

mod cmyk;
mod fragility;
mod gui;
mod halftone;
mod lines;
mod smooth;
mod stencil;
mod svg;
mod warp;

use std::collections::HashMap;

/// Format CMYK components (0..1) as "C0 M0 Y0 K100" integer percentages.
fn cmyk_str(cmyk: &[f32; 4]) -> String {
    let p = |v: f32| (v * 100.0).round() as i32;
    format!("C{} M{} Y{} K{}", p(cmyk[0]), p(cmyk[1]), p(cmyk[2]), p(cmyk[3]))
}

/// Parse `--flag value` pairs into a map, skipping the leading subcommand token.
fn parse_flags(skip: usize) -> Result<HashMap<String, String>, String> {
    let mut m = HashMap::new();
    let mut it = std::env::args().skip(skip);
    while let Some(flag) = it.next() {
        let key = flag.trim_start_matches("--").to_string();
        let val = it.next().ok_or(format!("missing value for --{key}"))?;
        m.insert(key, val);
    }
    Ok(m)
}

/// Image's native pixel dimensions.
fn image_size(path: &str) -> Result<(usize, usize), String> {
    let d = image::image_dimensions(path).map_err(|e| format!("read {path}: {e}"))?;
    Ok((d.0 as usize, d.1 as usize))
}

/// Common canvas: validate input exists, return its native px size + out prefix.
fn canvas(m: &HashMap<String, String>) -> Result<(usize, usize, String), String> {
    let input = m.get("input").cloned().ok_or("missing required --input")?;
    if !std::path::Path::new(&input).exists() {
        return Err(format!("input file not found: {input}"));
    }
    let (w_px, h_px) = image_size(&input)?;
    let out_prefix = m.get("out-prefix").cloned().unwrap_or_else(|| "out".into());
    Ok((w_px, h_px, out_prefix))
}

struct Args {
    input: String,
    shape: lines::Shape,
    shape_params: lines::ShapeParams,
    spacing_px: f32,
    min_material_px: f32,
    min_cut_px: f32,
    smooth_px: f32,
    out_prefix: String,
    inks: cmyk::Inks,  // CMYK or extended CMYKOG
    angles: Vec<f32>,  // one screen angle per ink (len == inks.count())
    loads: Vec<f32>,   // one cut-width scale per ink; 1.0 = full budget (unchanged)
    white_point: f32,  // levels: ink <= this -> white (no stroke)
    black_point: f32,  // levels: ink >= this -> full stroke
    gamma: f32,        // levels: midtone curve
    auto_levels: bool, // per-channel percentile wp/bp, overrides white/black-point
    kerf_px: f32,          // laser beam width; slot shrunk to compensate
    bridge_interval_px: f32, // distance between bridges along a line (0 = off)
    bridge_px: f32,        // bridge (uncut tab) length along the line
    scurve: f32,           // spray S-curve strength (0 = off)
    bilateral_px: f32,     // edge-preserving pre-filter radius (0 = off)
    k_contour: bool,       // K mode: true = DoG edge contours, false = tonal screen
    k_deep_clip: f32,      // K deep-shadow clip Tk
    k_gamma: f32,          // K steep gamma
    k_width_frac: f32,     // K max width as fraction of pitch
    ucr: f32,              // under-colour removal strength
    dog_sigma1: f32,       // DoG small blur (contour mode)
    dog_sigma2: f32,       // DoG large blur (contour mode)
    dog_threshold: f32,    // DoG edge threshold (contour mode)
    paper: Option<svg::Paper>, // A2..A5 sheet size (None = raw px viewBox)
    margin_mm: f32,        // blank border on all sides (squeeze)
}

fn parse_args(skip: usize) -> Result<Args, String> {
    let m = parse_flags(skip)?;
    let get = |k: &str| m.get(k).cloned().ok_or(format!("missing required --{k}"));
    let num = |k: &str| -> Result<f32, String> {
        get(k)?.parse::<f32>().map_err(|_| format!("--{k} must be a number"))
    };

    Ok(Args {
        input: get("input")?,
        shape: match m.get("shape").map(String::as_str) {
            Some("wavy") => lines::Shape::Wavy,
            Some("dots") => lines::Shape::Dots,
            Some("blue-noise") => lines::Shape::BlueNoise,
            Some("hatch") => lines::Shape::Hatch,
            _ => lines::Shape::Lines,
        },
        shape_params: lines::ShapeParams {
            wave_amp_frac: m.get("wave-amp-frac").and_then(|s| s.parse().ok()).unwrap_or(0.35),
            wave_len_frac: m.get("wave-len-frac").and_then(|s| s.parse().ok()).unwrap_or(4.0),
            wave_width_frac: m.get("wave-width-frac").and_then(|s| s.parse().ok()).unwrap_or(1.0),
            hatch_bins: m.get("hatch-bins").and_then(|s| s.parse().ok()).unwrap_or(6),
            // BlueNoise dot size range; default INFINITY => fixed cut-safe size (FM).
            dot_min_px: m.get("dot-min-px").and_then(|s| s.parse().ok()).unwrap_or(f32::INFINITY),
            dot_max_px: m.get("dot-max-px").and_then(|s| s.parse().ok()).unwrap_or(f32::INFINITY),
        },
        spacing_px: num("spacing-px")?,
        min_material_px: num("min-material-px")?,
        min_cut_px: num("min-cut-px")?,
        smooth_px: m.get("smooth-px").and_then(|s| s.parse().ok()).unwrap_or(1.5),
        out_prefix: m.get("out-prefix").cloned().unwrap_or_else(|| "out".into()),
        inks: match m.get("inks").map(String::as_str) {
            Some("cmykog") => cmyk::Inks::Cmykog,
            _ => cmyk::Inks::Cmyk,
        },
        angles: {
            let inks = match m.get("inks").map(String::as_str) {
                Some("cmykog") => cmyk::Inks::Cmykog,
                _ => cmyk::Inks::Cmyk,
            };
            let mut a = inks.default_angles();
            // --angles 15,75,0,45[,...] overrides the whole list.
            if let Some(csv) = m.get("angles") {
                let parsed: Result<Vec<f32>, _> =
                    csv.split(',').map(|s| s.trim().parse::<f32>()).collect();
                let parsed = parsed.map_err(|_| "--angles must be comma-separated numbers".to_string())?;
                if parsed.len() != inks.count() {
                    return Err(format!("--angles needs {} values for these inks (got {})", inks.count(), parsed.len()));
                }
                a = parsed;
            } else {
                // Legacy per-ink flags (CMYK back-compat): override individual entries.
                for (i, k) in ["angle-c", "angle-m", "angle-y", "angle-k"].iter().enumerate() {
                    if let Some(v) = m.get(*k) {
                        a[i] = v.parse::<f32>().map_err(|_| format!("--{k} must be a number"))?;
                    }
                }
            }
            a
        },
        loads: {
            let inks = match m.get("inks").map(String::as_str) {
                Some("cmykog") => cmyk::Inks::Cmykog,
                _ => cmyk::Inks::Cmyk,
            };
            let mut v = inks.default_loads();
            // --loads 1,0.55,1,1 pulls one heavy ink back so its sheet stays stiff.
            if let Some(csv) = m.get("loads") {
                let parsed: Result<Vec<f32>, _> =
                    csv.split(',').map(|s| s.trim().parse::<f32>()).collect();
                let parsed = parsed.map_err(|_| "--loads must be comma-separated numbers".to_string())?;
                if parsed.len() != inks.count() {
                    return Err(format!("--loads needs {} values for these inks (got {})", inks.count(), parsed.len()));
                }
                v = parsed;
            }
            v
        },
        white_point: m.get("white-point").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        black_point: m.get("black-point").and_then(|s| s.parse().ok()).unwrap_or(1.0),
        gamma: m.get("gamma").and_then(|s| s.parse().ok()).unwrap_or(1.0),
        auto_levels: m.get("auto-levels").map(|v| v == "on").unwrap_or(false),
        kerf_px: m.get("kerf-px").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        bridge_interval_px: m.get("bridge-interval-px").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        bridge_px: m.get("bridge-px").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        scurve: m.get("scurve").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        bilateral_px: m.get("bilateral-px").and_then(|s| s.parse().ok()).unwrap_or(0.0),
        k_contour: matches!(m.get("k-mode").map(String::as_str), Some("contour")),
        k_deep_clip: m.get("k-deep-clip").and_then(|s| s.parse().ok()).unwrap_or(0.75),
        k_gamma: m.get("k-gamma").and_then(|s| s.parse().ok()).unwrap_or(2.0),
        k_width_frac: m.get("k-width-frac").and_then(|s| s.parse().ok()).unwrap_or(0.40),
        ucr: m.get("ucr").and_then(|s| s.parse().ok()).unwrap_or(0.8),
        dog_sigma1: m.get("dog-sigma1").and_then(|s| s.parse().ok()).unwrap_or(1.0),
        dog_sigma2: m.get("dog-sigma2").and_then(|s| s.parse().ok()).unwrap_or(2.0),
        dog_threshold: m.get("dog-threshold").and_then(|s| s.parse().ok()).unwrap_or(0.05),
        paper: match m.get("paper") {
            Some(v) => Some(svg::Paper::parse(v).ok_or(format!("--paper must be A2|A3|A4|A5 (got {v})"))?),
            None => None,
        },
        margin_mm: m.get("margin-mm").and_then(|s| s.parse().ok()).unwrap_or(10.0),
    })
}

fn run_halftone(skip: usize) -> Result<(), String> {
    let a = parse_args(skip)?;

    // Validation at the trust boundary.
    for (name, v) in [
        ("spacing-px", a.spacing_px),
        ("min-material-px", a.min_material_px),
        ("min-cut-px", a.min_cut_px),
    ] {
        if v <= 0.0 {
            return Err(format!("--{name} must be > 0 (got {v})"));
        }
    }
    // Two-sided, so it can't fold into the loop above. Reject > 1 rather than clamping:
    // a load above 1 would silently exceed spacing - min_material and break the
    // min-material guarantee every shape depends on.
    for (i, l) in a.loads.iter().enumerate() {
        if !(0.0..=1.0).contains(l) || *l <= 0.0 {
            return Err(format!("--loads[{i}] must be in (0,1] (got {l})"));
        }
    }
    if !std::path::Path::new(&a.input).exists() {
        return Err(format!("input file not found: {}", a.input));
    }

    // Levels validation at the trust boundary: 0 <= wp < bp <= 1, gamma > 0.
    // Skipped for --auto-levels: wp/bp are computed per channel, not user-supplied.
    if !a.auto_levels
        && (!(0.0..=1.0).contains(&a.white_point)
        || !(0.0..=1.0).contains(&a.black_point)
        || a.white_point >= a.black_point)
    {
        return Err(format!(
            "levels require 0 <= --white-point ({}) < --black-point ({}) <= 1",
            a.white_point, a.black_point
        ));
    }
    if a.gamma <= 0.0 {
        return Err(format!("--gamma must be > 0 (got {})", a.gamma));
    }

    let w_max_px = a.spacing_px - a.min_material_px;
    if a.min_cut_px >= w_max_px {
        return Err(format!(
            "min-cut-px ({}) must be < spacing-px ({}) - min-material-px ({}); i.e. < {}",
            a.min_cut_px, a.spacing_px, a.min_material_px, w_max_px
        ));
    }
    for (name, v) in [("kerf-px", a.kerf_px), ("bridge-interval-px", a.bridge_interval_px), ("bridge-px", a.bridge_px), ("scurve", a.scurve), ("bilateral-px", a.bilateral_px)] {
        if v < 0.0 {
            return Err(format!("--{name} must be >= 0 (got {v})"));
        }
    }
    if a.kerf_px >= a.spacing_px {
        return Err(format!("--kerf-px ({}) must be < spacing-px ({})", a.kerf_px, a.spacing_px));
    }

    let (w_px, h_px) = image_size(&a.input)?;
    let layers = cmyk::load_filtered(&a.input, w_px, h_px, a.bilateral_px)?;

    if !(0.0..1.0).contains(&a.k_deep_clip) {
        return Err(format!("--k-deep-clip must be in [0,1) (got {})", a.k_deep_clip));
    }
    if a.k_gamma <= 0.0 || a.k_width_frac <= 0.0 || a.k_width_frac > 1.0 || !(0.0..=1.0).contains(&a.ucr) {
        return Err("--k-gamma > 0, 0 < --k-width-frac <= 1, 0 <= --ucr <= 1 required".into());
    }

    let p = lines::LineParams {
        shape: a.shape,
        shape_params: a.shape_params,
        spacing_px: a.spacing_px,
        w_min_px: a.min_cut_px,
        min_material_px: a.min_material_px,
        kerf_px: a.kerf_px,
        bridge_interval_px: a.bridge_interval_px,
        bridge_px: a.bridge_px,
        white_point: a.white_point,
        black_point: a.black_point,
        gamma: a.gamma,
        scurve: a.scurve,
        // Placeholder: `generate_all` overwrites this per channel from `Channel::load`.
        load: 1.0,
        k_mode: if a.k_contour { lines::KMode::Contour } else { lines::KMode::Tonal },
        k_deep_clip: a.k_deep_clip,
        k_gamma: a.k_gamma,
        k_width_frac: a.k_width_frac,
        ucr: a.ucr,
        dog_sigma1: a.dog_sigma1,
        dog_sigma2: a.dog_sigma2,
        dog_threshold: a.dog_threshold,
    };
    // smooth_px no longer used by the analytic path (no tracing); kept as a flag
    // for backward-compat so old invocations don't error.
    let _ = a.smooth_px;

    if a.margin_mm < 0.0 {
        return Err(format!("--margin-mm must be >= 0 (got {})", a.margin_mm));
    }

    let chans = cmyk::channels(&layers, a.inks, &a.angles, &a.loads);
    lines::export(&chans, w_px, h_px, &p, a.auto_levels, a.paper, a.margin_mm, &a.out_prefix)
}

fn run_stencil(skip: usize) -> Result<(), String> {
    let m = parse_flags(skip)?;
    let (w_px, h_px, out_prefix) = canvas(&m)?;

    let input = m.get("input").cloned().unwrap(); // canvas() already validated presence
    let colors: usize = m
        .get("colors")
        .ok_or("missing required --colors")?
        .parse()
        .map_err(|_| "--colors must be an integer".to_string())?;
    if colors < 2 {
        return Err(format!("--colors must be >= 2 (got {colors})"));
    }
    // Bridges default OFF: cleaner outlines, but enclosed islands fall out on the
    // laser. Turn on with `--bridges on` for cut-safe stencils.
    let bridges = matches!(m.get("bridges").map(String::as_str), Some("on" | "true" | "1"));
    // min-material-px only matters when bridging; required only then.
    let min_material_px: f32 = if bridges {
        let v: f32 = m
            .get("min-material-px")
            .ok_or("--bridges on requires --min-material-px")?
            .parse()
            .map_err(|_| "--min-material-px must be a number".to_string())?;
        if v <= 0.0 {
            return Err("--min-material-px must be > 0".into());
        }
        v
    } else {
        0.0
    };
    // Bridge tab width, independent of min-material. Defaults to min-material-px
    // when omitted so existing invocations are unchanged.
    let bridge_px: f32 = m
        .get("bridge-px")
        .and_then(|s| s.parse().ok())
        .filter(|&v: &f32| v > 0.0)
        .unwrap_or(min_material_px);
    let min_feature_px: f32 = m
        .get("min-feature-px")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    // DP smoothing tolerance in px: how far the curve may stray from the pixel
    // staircase. ~1.5px kills the stair steps like Potrace; bump for smoother/
    // fewer nodes, drop toward 0 for a tighter (blockier) trace.
    let smooth_px: f32 = m
        .get("smooth-px")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5);

    // Coarsen detail: blur radius in px before quantizing. Bigger = fewer, chunkier
    // areas (less granular). 0 = off, quantize the sharp image.
    let coarsen_px: f32 = m
        .get("coarsen-px")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);

    let white_point: f32 = m.get("white-point").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let black_point: f32 = m.get("black-point").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let gamma: f32 = m.get("gamma").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    if gamma <= 0.0 || !(0.0..=1.0).contains(&white_point) || !(0.0..=1.0).contains(&black_point) || white_point >= black_point {
        eprintln!(
            "error: levels require 0 <= --white-point ({white_point}) < --black-point ({black_point}) <= 1 and --gamma > 0"
        );
        std::process::exit(1);
    }

    let p = stencil::Params {
        colors,
        bridge_px,
        min_feature_px: min_feature_px.powi(2), // area floor in px^2
        bridges,
        blur_px: coarsen_px,
        white_point,
        black_point,
        gamma,
    };

    let format = m.get("format").map(String::as_str).unwrap_or("svg");
    let mut warn = |m: String| eprintln!("warning: {m}");

    let mut palette = String::from("layer\trgb_hex\n");
    use std::fmt::Write;

    if format == "png" {
        // Raster path: skip trace/smooth/SVG entirely. Each layer is a PNG where
        // kept material = the layer color (opaque), cut area = transparent.
        let (pal, masks) = stencil::stencil_masks(&input, w_px, h_px, &p, &mut warn)?;
        for (i, (color, mask)) in pal.iter().zip(masks.iter()).enumerate() {
            let fill = format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b);
            let _ = writeln!(palette, "{i}\t{fill}\t{}", cmyk_str(&color.cmyk));
            let mut buf = image::RgbaImage::new(w_px as u32, h_px as u32);
            for (px, &keep) in buf.pixels_mut().zip(mask.iter()) {
                *px = if keep {
                    image::Rgba([color.r, color.g, color.b, 255])
                } else {
                    image::Rgba([0, 0, 0, 0])
                };
            }
            let path = format!("{out_prefix}_{i}.png");
            buf.save(&path).map_err(|e| format!("write {path}: {e}"))?;
            let kept = mask.iter().filter(|&&b| b).count();
            eprintln!("wrote {path} (color {fill}, {kept}/{} px kept)", mask.len());
        }
    } else if format == "svg" {
        return stencil::export(&input, w_px, h_px, &p, smooth_px, &out_prefix, &mut warn);
    } else {
        return Err(format!("--format must be svg or png (got {format})"));
    }
    let ppath = format!("{out_prefix}_palette.txt");
    std::fs::write(&ppath, palette).map_err(|e| format!("write {ppath}: {e}"))?;
    eprintln!("wrote {ppath} (spray order: index 0 = darkest, sprayed first)");
    Ok(())
}

const USAGE: &str = "usage (all sizes in px; output matches the input image's pixel dimensions):\n  \
    gui\n  \
    halftone --input <img> --spacing-px <f> --min-material-px <f> --min-cut-px <f> \
[--shape lines|wavy|dots|blue-noise|hatch] [--wave-amp-frac <f>] [--wave-len-frac <f>] [--wave-width-frac <f>] [--hatch-bins <n>] \
  [--dot-min-px <f>] [--dot-max-px <f>] \
[--inks cmyk|cmykog] [--angles 15,75,0,45[,...]] [--loads 1,0.55,1,1] \
[--kerf-px <f>] [--bridge-interval-px <f>] [--bridge-px <f>] [--scurve <f>] [--bilateral-px <f>] [--auto-levels on] \
[--k-mode tonal|contour] [--k-deep-clip <f>] [--k-gamma <f>] [--k-width-frac <f>] [--ucr <f>] [--dog-sigma1 <f>] [--dog-sigma2 <f>] [--dog-threshold <f>] [--paper A2|A3|A4|A5] [--margin-mm <f>] [--out-prefix <s>]\n  \
    stencil  --input <img> --colors <N> \
[--format svg|png] [--bridges on] [--min-material-px <f>] [--bridge-px <f>] [--min-feature-px <f>] [--coarsen-px <f>] [--smooth-px <f>] [--out-prefix <s>]";

fn run() -> Result<(), String> {
    // ponytail: dispatch on argv[1]; no clap. Default to halftone when the first
    // arg is a flag (backward-compatible with the Op 1 CLI).
    match std::env::args().nth(1).as_deref() {
        Some("gui") => gui::run(),
        Some("halftone") => run_halftone(2),
        Some("stencil") => run_stencil(2),
        Some(s) if s.starts_with("--") => run_halftone(1), // legacy: no subcommand
        _ => Err(USAGE.into()),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
