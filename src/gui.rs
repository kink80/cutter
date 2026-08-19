//! Native GUI: pick an image, tweak params, see the composite preview update
//! live, and generate all layer files. Two modes — Halftone (CMYK line screens)
//! and Stencil (Lab-quantized spray masks). The pipeline runs on a background
//! thread (A1): the UI never blocks and fast slider drags coalesce because only
//! one render is in flight at a time.

use crate::cmyk;
use crate::lines::{self, LineParams};
use crate::stencil;
use eframe::egui;
use std::sync::mpsc;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Halftone,
    Stencil,
}

/// A snapshot of every knob, sent to the worker to render one preview frame.
#[derive(Clone)]
enum Job {
    Halftone {
        path: String,
        layers: Arc<cmyk::Layers>,
        w: usize,
        h: usize,
        p: LineParams,
        inks: cmyk::Inks,
        angles: Vec<f32>,
        auto: bool,
        bilateral_px: f32,
        loads: Vec<f32>,
    },
    Stencil {
        path: String,
        w: usize,
        h: usize,
        p: stencil::Params,
    },
}

struct Rendered {
    w: usize,
    h: usize,
    rgb: Vec<u8>, // w*h*3
    /// Per-layer stiffness: (layer name, readout). Drives the STIFFNESS panel.
    frag: Vec<(String, crate::fragility::Fragility)>,
}

pub fn run() -> Result<(), String> {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1040.0, 820.0])
            // Small min so the window fits short screens; the controls scroll and the
            // generate button stays pinned, so nothing is unreachable when shrunk.
            .with_min_inner_size([560.0, 320.0]),
        ..Default::default()
    };
    eframe::run_native(
        "Laser Halftone",
        opts,
        Box::new(|cc| {
            theme(&cc.egui_ctx);
            Ok(Box::new(App::new()))
        }),
    )
    .map_err(|e| format!("gui: {e}"))
}

/// Dark theme with a laser-red accent, roomier spacing, rounded widgets.
fn theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let v = &mut style.visuals;
    v.dark_mode = true;
    let accent = egui::Color32::from_rgb(0xE0, 0x3A, 0x2F); // laser red
    v.override_text_color = Some(egui::Color32::from_gray(0xDE));
    v.panel_fill = egui::Color32::from_gray(0x1A);
    v.window_fill = egui::Color32::from_gray(0x1A);
    v.extreme_bg_color = egui::Color32::from_gray(0x0E); // slider troughs / preview bg
    v.widgets.inactive.bg_fill = egui::Color32::from_gray(0x2A);
    v.widgets.hovered.bg_fill = egui::Color32::from_gray(0x38);
    v.widgets.active.bg_fill = accent;
    v.selection.bg_fill = accent.linear_multiply(0.5);
    v.selection.stroke = egui::Stroke::new(1.0, accent);
    for w in [
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.noninteractive,
        &mut v.widgets.open,
    ] {
        w.rounding = egui::Rounding::same(4.0);
    }
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.slider_width = 120.0;
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    ctx.set_style(style);
}

struct App {
    mode: Mode,

    // halftone knobs
    shape: crate::lines::Shape,
    wave_amp_frac: f32,
    wave_len_frac: f32,
    wave_width_frac: f32,
    hatch_bins: u32,
    bn_dot_min_px: f32,
    bn_dot_max_px: f32,
    spacing_px: f32,
    min_material_px: f32,
    min_cut_px: f32,
    kerf_px: f32,
    bridge_interval_px: f32,
    ht_bridge_px: f32,
    scurve: f32,
    bilateral_px: f32,
    k_contour: bool,   // K mode toggle: true = DoG contour, false = tonal screen
    k_deep_clip: f32,
    k_gamma: f32,
    k_width_frac: f32,
    ucr: f32,
    dog_sigma1: f32,
    dog_sigma2: f32,
    dog_threshold: f32,
    paper: Option<crate::svg::Paper>, // None = raw px (no physical sizing)
    margin_mm: f32,
    inks: cmyk::Inks,       // CMYK or extended CMYKOG
    angles: Vec<f32>,       // one screen angle per ink (len == inks.count())
    angles_locked: bool,    // drag one angle, all rotate by the same delta
    loads: Vec<f32>,        // one cut-width scale per ink; 1.0 = full budget
    white_point: f32,
    black_point: f32,
    gamma: f32,
    auto_levels: bool,

    // stencil knobs
    colors: usize,
    coarsen_px: f32,
    min_feature_px: f32,
    bridges: bool,
    bridge_px: f32,
    smooth_px: f32,

    // loaded image
    path: Option<String>,
    layers: Option<Arc<cmyk::Layers>>,
    dims: (usize, usize),
    status: String,

    // preview / worker
    texture: Option<egui::TextureHandle>,
    /// Latest per-layer stiffness readout, from the most recent render.
    frag: Vec<(String, crate::fragility::Fragility)>,
    dirty: bool,
    in_flight: bool,
    to_worker: mpsc::Sender<Job>,
    from_worker: mpsc::Receiver<Rendered>,

    // generate (runs off the UI thread so the window never freezes)
    generating: bool,
    gen_tx: mpsc::Sender<String>,
    gen_rx: mpsc::Receiver<String>,
}

impl App {
    fn new() -> Self {
        let (to_worker, job_rx) = mpsc::channel::<Job>();
        let (result_tx, from_worker) = mpsc::channel::<Rendered>();
        std::thread::spawn(move || {
            // ponytail: drain to the LATEST job before rendering, so a burst of
            // slider updates renders once, not once-per-tick.
            while let Ok(mut job) = job_rx.recv() {
                while let Ok(newer) = job_rx.try_recv() {
                    job = newer;
                }
                let rendered = match job {
                    Job::Halftone { path, layers, w, h, p, inks, angles, auto, bilateral_px, loads } => {
                        // Reload with the edge-preserving pre-filter only when it's on
                        // (it's slow); otherwise reuse the cached, unfiltered layers.
                        let names = inks.names();
                        let (buf, frag) = if bilateral_px > 0.0 {
                            match cmyk::load_filtered(&path, w, h, bilateral_px) {
                                Ok(l) => {
                                    let chans = cmyk::channels(&l, inks, &angles, &loads);
                                    lines::render_preview(&chans, w, h, &p, auto)
                                }
                                Err(_) => continue,
                            }
                        } else {
                            let chans = cmyk::channels(&layers, inks, &angles, &loads);
                            lines::render_preview(&chans, w, h, &p, auto)
                        };
                        let frag = names.iter().map(|n| n.to_string()).zip(frag).collect();
                        Rendered { w, h, rgb: buf.into_raw(), frag }
                    }
                    Job::Stencil { path, w, h, p } => {
                        // ponytail: stencil reloads from disk each frame (quantize is
                        // the cost, not the read). Fine for preview; the file dialog
                        // guaranteed the path exists.
                        match stencil::stencil_masks(&path, w, h, &p, &mut |_| {}) {
                            Ok((palette, masks)) => {
                                // Stencil masks are already keep-polarity (true = the
                                // material this layer keeps), unlike halftone's cut
                                // masks — measure them directly, no inversion.
                                let frag = masks
                                    .iter()
                                    .enumerate()
                                    .map(|(i, m)| {
                                        (format!("{i}"), crate::fragility::measure(m, w, h))
                                    })
                                    .collect();
                                let buf = stencil::preview(&palette, &masks, w, h);
                                Rendered { w, h, rgb: buf.into_raw(), frag }
                            }
                            // On error just skip the frame; status stays as-is.
                            Err(_) => continue,
                        }
                    }
                };
                let _ = result_tx.send(rendered);
            }
        });

        let (gen_tx, gen_rx) = mpsc::channel::<String>();
        App {
            mode: Mode::Halftone,
            shape: crate::lines::Shape::Lines,
            wave_amp_frac: 0.35,
            wave_len_frac: 4.0,
            wave_width_frac: 0.6,
            hatch_bins: 6,
            // Blue-noise dot size range. Default both = spacing -> both clamp to the
            // cut-safe radius -> fixed-size FM (original look). Drop min for AM detail.
            bn_dot_min_px: 8.0,
            bn_dot_max_px: 8.0,
            spacing_px: 8.0,
            min_material_px: 1.0,
            min_cut_px: 0.5,
            kerf_px: 0.0,
            bridge_interval_px: 0.0,
            ht_bridge_px: 2.0,
            scurve: 0.0,
            bilateral_px: 0.0,
            k_contour: false,
            k_deep_clip: 0.75,
            k_gamma: 2.0,
            k_width_frac: 0.40,
            ucr: 0.8,
            dog_sigma1: 1.0,
            dog_sigma2: 2.0,
            dog_threshold: 0.05,
            paper: Some(crate::svg::Paper::A4),
            margin_mm: 10.0,
            inks: cmyk::Inks::Cmyk,
            angles: cmyk::Inks::Cmyk.default_angles(),
            angles_locked: false,
            loads: cmyk::Inks::Cmyk.default_loads(),
            white_point: 0.0,
            black_point: 1.0,
            gamma: 1.0,
            auto_levels: false,
            colors: 4,
            coarsen_px: 0.0,
            min_feature_px: 3.0,
            bridges: false,
            bridge_px: 1.0,
            smooth_px: 1.5,
            path: None,
            layers: None,
            dims: (0, 0),
            status: "Open an image to begin".into(),
            texture: None,
            frag: Vec::new(),
            dirty: false,
            in_flight: false,
            to_worker,
            from_worker,
            generating: false,
            gen_tx,
            gen_rx,
        }
    }

    /// Serialize every knob to a simple `key = value` text preset (dependency-free,
    /// same plain-text spirit as the palette files). Runtime/worker state and the
    /// loaded image are intentionally excluded — a preset is just the settings.
    fn preset_string(&self) -> String {
        use std::fmt::Write as _;
        let shape = match self.shape {
            crate::lines::Shape::Lines => "lines",
            crate::lines::Shape::Wavy => "wavy",
            crate::lines::Shape::Dots => "dots",
            crate::lines::Shape::BlueNoise => "blue-noise",
            crate::lines::Shape::Hatch => "hatch",
        };
        let inks = match self.inks {
            cmyk::Inks::Cmyk => "cmyk",
            cmyk::Inks::Cmykog => "cmykog",
        };
        let paper = match self.paper {
            None => "none".to_string(),
            Some(crate::svg::Paper::A2) => "a2".into(),
            Some(crate::svg::Paper::A3) => "a3".into(),
            Some(crate::svg::Paper::A4) => "a4".into(),
            Some(crate::svg::Paper::A5) => "a5".into(),
        };
        let mode = if self.mode == Mode::Stencil { "stencil" } else { "halftone" };
        let csv = |v: &[f32]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
        let mut s = String::new();
        let _ = writeln!(s, "# laser_halftone preset v1");
        let _ = writeln!(s, "mode = {mode}");
        let _ = writeln!(s, "# --- halftone ---");
        let _ = writeln!(s, "shape = {shape}");
        let _ = writeln!(s, "wave_amp_frac = {}", self.wave_amp_frac);
        let _ = writeln!(s, "wave_len_frac = {}", self.wave_len_frac);
        let _ = writeln!(s, "wave_width_frac = {}", self.wave_width_frac);
        let _ = writeln!(s, "hatch_bins = {}", self.hatch_bins);
        let _ = writeln!(s, "bn_dot_min_px = {}", self.bn_dot_min_px);
        let _ = writeln!(s, "bn_dot_max_px = {}", self.bn_dot_max_px);
        let _ = writeln!(s, "spacing_px = {}", self.spacing_px);
        let _ = writeln!(s, "min_material_px = {}", self.min_material_px);
        let _ = writeln!(s, "min_cut_px = {}", self.min_cut_px);
        let _ = writeln!(s, "kerf_px = {}", self.kerf_px);
        let _ = writeln!(s, "bridge_interval_px = {}", self.bridge_interval_px);
        let _ = writeln!(s, "ht_bridge_px = {}", self.ht_bridge_px);
        let _ = writeln!(s, "scurve = {}", self.scurve);
        let _ = writeln!(s, "bilateral_px = {}", self.bilateral_px);
        let _ = writeln!(s, "k_contour = {}", self.k_contour);
        let _ = writeln!(s, "k_deep_clip = {}", self.k_deep_clip);
        let _ = writeln!(s, "k_gamma = {}", self.k_gamma);
        let _ = writeln!(s, "k_width_frac = {}", self.k_width_frac);
        let _ = writeln!(s, "ucr = {}", self.ucr);
        let _ = writeln!(s, "dog_sigma1 = {}", self.dog_sigma1);
        let _ = writeln!(s, "dog_sigma2 = {}", self.dog_sigma2);
        let _ = writeln!(s, "dog_threshold = {}", self.dog_threshold);
        let _ = writeln!(s, "paper = {paper}");
        let _ = writeln!(s, "margin_mm = {}", self.margin_mm);
        let _ = writeln!(s, "inks = {inks}");
        let _ = writeln!(s, "angles = {}", csv(&self.angles));
        let _ = writeln!(s, "angles_locked = {}", self.angles_locked);
        let _ = writeln!(s, "loads = {}", csv(&self.loads));
        let _ = writeln!(s, "white_point = {}", self.white_point);
        let _ = writeln!(s, "black_point = {}", self.black_point);
        let _ = writeln!(s, "gamma = {}", self.gamma);
        let _ = writeln!(s, "auto_levels = {}", self.auto_levels);
        let _ = writeln!(s, "# --- stencil ---");
        let _ = writeln!(s, "colors = {}", self.colors);
        let _ = writeln!(s, "coarsen_px = {}", self.coarsen_px);
        let _ = writeln!(s, "min_feature_px = {}", self.min_feature_px);
        let _ = writeln!(s, "bridges = {}", self.bridges);
        let _ = writeln!(s, "bridge_px = {}", self.bridge_px);
        let _ = writeln!(s, "smooth_px = {}", self.smooth_px);
        s
    }

    /// Apply a preset text over the current knobs. Unknown/missing keys keep their
    /// current value (forward/backward compatible), and out-of-range vectors fall
    /// back to the ink defaults so a mismatched `inks`/`angles` pair can't panic.
    fn apply_preset(&mut self, text: &str) {
        let map: std::collections::HashMap<String, String> = text
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                if l.is_empty() || l.starts_with('#') {
                    return None;
                }
                let (k, v) = l.split_once('=')?;
                Some((k.trim().to_string(), v.trim().to_string()))
            })
            .collect();
        let f = |k: &str, cur: f32| map.get(k).and_then(|s| s.parse().ok()).unwrap_or(cur);
        let u = |k: &str, cur: u32| map.get(k).and_then(|s| s.parse().ok()).unwrap_or(cur);
        let b = |k: &str, cur: bool| map.get(k).and_then(|s| s.parse().ok()).unwrap_or(cur);

        if let Some(v) = map.get("mode") {
            self.mode = if v == "stencil" { Mode::Stencil } else { Mode::Halftone };
        }
        if let Some(v) = map.get("shape") {
            use crate::lines::Shape::*;
            self.shape = match v.as_str() {
                "lines" => Lines,
                "wavy" => Wavy,
                "dots" => Dots,
                "blue-noise" => BlueNoise,
                "hatch" => Hatch,
                _ => self.shape,
            };
        }
        self.wave_amp_frac = f("wave_amp_frac", self.wave_amp_frac);
        self.wave_len_frac = f("wave_len_frac", self.wave_len_frac);
        self.wave_width_frac = f("wave_width_frac", self.wave_width_frac);
        self.hatch_bins = u("hatch_bins", self.hatch_bins);
        self.bn_dot_min_px = f("bn_dot_min_px", self.bn_dot_min_px);
        self.bn_dot_max_px = f("bn_dot_max_px", self.bn_dot_max_px);
        self.spacing_px = f("spacing_px", self.spacing_px);
        self.min_material_px = f("min_material_px", self.min_material_px);
        self.min_cut_px = f("min_cut_px", self.min_cut_px);
        self.kerf_px = f("kerf_px", self.kerf_px);
        self.bridge_interval_px = f("bridge_interval_px", self.bridge_interval_px);
        self.ht_bridge_px = f("ht_bridge_px", self.ht_bridge_px);
        self.scurve = f("scurve", self.scurve);
        self.bilateral_px = f("bilateral_px", self.bilateral_px);
        self.k_contour = b("k_contour", self.k_contour);
        self.k_deep_clip = f("k_deep_clip", self.k_deep_clip);
        self.k_gamma = f("k_gamma", self.k_gamma);
        self.k_width_frac = f("k_width_frac", self.k_width_frac);
        self.ucr = f("ucr", self.ucr);
        self.dog_sigma1 = f("dog_sigma1", self.dog_sigma1);
        self.dog_sigma2 = f("dog_sigma2", self.dog_sigma2);
        self.dog_threshold = f("dog_threshold", self.dog_threshold);
        if let Some(v) = map.get("paper") {
            use crate::svg::Paper::*;
            self.paper = match v.as_str() {
                "a2" => Some(A2),
                "a3" => Some(A3),
                "a4" => Some(A4),
                "a5" => Some(A5),
                _ => None,
            };
        }
        self.margin_mm = f("margin_mm", self.margin_mm);
        // Inks first, then angles/loads (their length must match the ink count).
        if let Some(v) = map.get("inks") {
            self.inks = match v.as_str() {
                "cmykog" => cmyk::Inks::Cmykog,
                _ => cmyk::Inks::Cmyk,
            };
        }
        let parse_vec = |s: &str| -> Vec<f32> {
            s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
        };
        if let Some(v) = map.get("angles") {
            let a = parse_vec(v);
            if a.len() == self.inks.count() {
                self.angles = a;
            }
        }
        if let Some(v) = map.get("loads") {
            let l = parse_vec(v);
            if l.len() == self.inks.count() {
                self.loads = l;
            }
        }
        // Guarantee the vectors match the ink count even if the preset omitted them
        // or switched inks without listing new angles/loads.
        if self.angles.len() != self.inks.count() {
            self.angles = self.inks.default_angles();
        }
        if self.loads.len() != self.inks.count() {
            self.loads = self.inks.default_loads();
        }
        self.angles_locked = b("angles_locked", self.angles_locked);
        self.white_point = f("white_point", self.white_point);
        self.black_point = f("black_point", self.black_point);
        self.gamma = f("gamma", self.gamma);
        self.auto_levels = b("auto_levels", self.auto_levels);
        self.colors = map.get("colors").and_then(|s| s.parse().ok()).unwrap_or(self.colors);
        self.coarsen_px = f("coarsen_px", self.coarsen_px);
        self.min_feature_px = f("min_feature_px", self.min_feature_px);
        self.bridges = b("bridges", self.bridges);
        self.bridge_px = f("bridge_px", self.bridge_px);
        self.smooth_px = f("smooth_px", self.smooth_px);
        self.dirty = true;
    }

    /// Prompt for a `.preset` file and write the current settings to it.
    fn save_preset(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("preset", &["preset", "txt"])
            .set_file_name("laser_halftone.preset")
            .save_file()
        else {
            return;
        };
        let path = path.display().to_string();
        match std::fs::write(&path, self.preset_string()) {
            Ok(()) => self.status = format!("saved preset {path}"),
            Err(e) => self.status = format!("save preset {path}: {e}"),
        }
    }

    /// Prompt for a `.preset` file and apply it over the current settings.
    fn load_preset(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("preset", &["preset", "txt"])
            .pick_file()
        else {
            return;
        };
        let path = path.display().to_string();
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                self.apply_preset(&text);
                self.status = format!("loaded preset {path}");
            }
            Err(e) => self.status = format!("load preset {path}: {e}"),
        }
    }

    fn open_image(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("image", &["png", "jpg", "jpeg", "webp", "bmp"])
            .pick_file()
        else {
            return;
        };
        let path = path.display().to_string();
        let (w, h) = match image::image_dimensions(&path) {
            Ok((w, h)) => (w as usize, h as usize),
            Err(e) => {
                self.status = format!("read {path}: {e}");
                return;
            }
        };
        match cmyk::load(&path, w, h) {
            Ok(layers) => {
                self.layers = Some(Arc::new(layers));
                self.path = Some(path.clone());
                self.dims = (w, h);
                self.status = format!("{path} ({w}x{h})");
                self.dirty = true;
            }
            Err(e) => self.status = e,
        }
    }

    fn halftone_params(&self) -> LineParams {
        LineParams {
            shape: self.shape,
            shape_params: crate::lines::ShapeParams {
                wave_amp_frac: self.wave_amp_frac,
                wave_len_frac: self.wave_len_frac,
                wave_width_frac: self.wave_width_frac,
                hatch_bins: self.hatch_bins,
                dot_min_px: self.bn_dot_min_px,
                dot_max_px: self.bn_dot_max_px,
            },
            spacing_px: self.spacing_px,
            w_min_px: self.min_cut_px,
            min_material_px: self.min_material_px.min(self.spacing_px - 0.01).max(0.0),
            kerf_px: self.kerf_px.min(self.spacing_px - 0.01).max(0.0),
            bridge_interval_px: self.bridge_interval_px,
            bridge_px: self.ht_bridge_px,
            white_point: self.white_point,
            black_point: self.black_point,
            gamma: self.gamma,
            scurve: self.scurve,
            // Placeholder: `generate_all` overwrites this per channel from `Channel::load`.
            load: 1.0,
            k_mode: if self.k_contour { crate::lines::KMode::Contour } else { crate::lines::KMode::Tonal },
            k_deep_clip: self.k_deep_clip,
            k_gamma: self.k_gamma,
            k_width_frac: self.k_width_frac,
            ucr: self.ucr,
            dog_sigma1: self.dog_sigma1,
            dog_sigma2: self.dog_sigma2,
            dog_threshold: self.dog_threshold,
        }
    }

    fn stencil_params(&self) -> stencil::Params {
        stencil::Params {
            colors: self.colors,
            bridge_px: self.bridge_px,
            min_feature_px: self.min_feature_px.powi(2),
            bridges: self.bridges,
            blur_px: self.coarsen_px,
            white_point: self.white_point,
            black_point: self.black_point,
            gamma: self.gamma,
        }
    }

    /// Build a preview Job from current knobs. None if no image is loaded.
    fn make_job(&self) -> Option<Job> {
        let (w, h) = self.dims;
        match self.mode {
            Mode::Halftone => Some(Job::Halftone {
                path: self.path.clone()?,
                layers: self.layers.clone()?,
                w,
                h,
                p: self.halftone_params(),
                inks: self.inks,
                angles: self.angles.clone(),
                auto: self.auto_levels,
                bilateral_px: self.bilateral_px,
                loads: self.loads.clone(),
            }),
            Mode::Stencil => Some(Job::Stencil {
                path: self.path.clone()?,
                w,
                h,
                p: self.stencil_params(),
            }),
        }
    }

    /// "Generate all layers": prompt for an output prefix, then write every file.
    fn generate(&mut self) {
        let (Some(path), (w, h)) = (self.path.clone(), self.dims) else {
            self.status = "Open an image first".into();
            return;
        };
        let Some(out) = rfd::FileDialog::new()
            .set_file_name("out")
            .save_file()
        else {
            return;
        };
        let prefix = out.display().to_string();
        // ponytail: export traces + writes files (~1s); run it off the UI thread
        // so the window doesn't freeze, report the outcome back over a channel.
        let mode = self.mode;
        let layers = self.layers.clone();
        let hp = self.halftone_params();
        let sp = self.stencil_params();
        let inks = self.inks;
        let angles = self.angles.clone();
        let loads = self.loads.clone();
        let auto = self.auto_levels;
        let smooth = self.smooth_px;
        let bilateral_px = self.bilateral_px;
        let paper = self.paper;
        let margin_mm = self.margin_mm;
        let tx = self.gen_tx.clone();
        self.generating = true;
        self.status = format!("generating {prefix}_* …");
        std::thread::spawn(move || {
            let mut warn = |m: String| eprintln!("warning: {m}");
            let res = match mode {
                Mode::Halftone => {
                    // Reload with the bilateral pre-filter for the final output if on.
                    let loaded;
                    let l = if bilateral_px > 0.0 {
                        match cmyk::load_filtered(&path, w, h, bilateral_px) {
                            Ok(x) => { loaded = std::sync::Arc::new(x); loaded.as_ref() }
                            Err(e) => { let _ = tx.send(format!("generate failed: {e}")); return; }
                        }
                    } else {
                        layers.as_ref().unwrap().as_ref()
                    };
                    let chans = cmyk::channels(l, inks, &angles, &loads);
                    lines::export(&chans, w, h, &hp, auto, paper, margin_mm, &prefix)
                }
                Mode::Stencil => stencil::export(
                    &path, w, h, &sp, smooth, &prefix, &mut warn,
                ),
            };
            let _ = tx.send(match res {
                Ok(()) => format!("wrote layers to {prefix}_*"),
                Err(e) => format!("generate failed: {e}"),
            });
        });
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Pull any finished render, upload as a texture.
        if let Ok(r) = self.from_worker.try_recv() {
            let img = egui::ColorImage::from_rgb([r.w, r.h], &r.rgb);
            self.texture = Some(ctx.load_texture("preview", img, egui::TextureOptions::LINEAR));
            self.frag = r.frag;
            self.in_flight = false;
        }
        // Pick up a finished generate.
        if let Ok(msg) = self.gen_rx.try_recv() {
            self.status = msg;
            self.generating = false;
        }
        // Kick off a render if dirty and nothing is running.
        if self.dirty && !self.in_flight {
            if let Some(job) = self.make_job() {
                if self.to_worker.send(job).is_ok() {
                    self.in_flight = true;
                    self.dirty = false;
                }
            }
        }

        egui::SidePanel::left("controls")
            .resizable(false)
            .exact_width(320.0)
            .show(ctx, |ui| {
            // Pin generate + status to the bottom FIRST so egui reserves its space;
            // the scrolling controls below then fill only what's left. Without this
            // the sliders overflow a short window and push the button off-screen.
            egui::TopBottomPanel::bottom("gen")
                .frame(egui::Frame::none().inner_margin(egui::Margin::symmetric(0.0, 6.0)))
                .show_inside(ui, |ui| {
                    let can_gen = self.path.is_some() && !self.generating;
                    let btn = egui::Button::new(egui::RichText::new("Generate all layers").strong());
                    if ui.add_enabled(can_gen, btn.min_size(egui::vec2(ui.available_width(), 32.0))).clicked() {
                        self.generate();
                    }
                    ui.add_space(4.0);
                    if self.in_flight || self.generating {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(if self.generating { "generating…" } else { "rendering…" });
                        });
                    } else {
                        ui.label(egui::RichText::new(&self.status).small().color(egui::Color32::from_gray(0x9A)));
                    }
                });

            // Everything above the pinned button scrolls when the window is short.
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            ui.add_space(4.0);
            ui.heading("Laser Halftone");
            ui.add_space(6.0);
            if ui.add_sized([ui.available_width(), 30.0], egui::Button::new("Open image…")).clicked() {
                self.open_image();
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                if ui.add_sized([w, 24.0], egui::Button::new("Save preset…")).clicked() {
                    self.save_preset();
                }
                if ui.add_sized([w, 24.0], egui::Button::new("Load preset…")).clicked() {
                    self.load_preset();
                }
            });
            ui.add_space(8.0);

            // Mode toggle.
            ui.horizontal(|ui| {
                let w = (ui.available_width() - ui.spacing().item_spacing.x) / 2.0;
                let ht = ui.add_sized([w, 26.0], egui::SelectableLabel::new(self.mode == Mode::Halftone, "Halftone")).clicked();
                let st = ui.add_sized([w, 26.0], egui::SelectableLabel::new(self.mode == Mode::Stencil, "Stencil")).clicked();
                if ht { self.mode = Mode::Halftone; self.dirty = true; }
                if st { self.mode = Mode::Stencil; self.dirty = true; }
            });
            ui.add_space(8.0);

            let mut ch = false;
            let s = |ui: &mut egui::Ui, ch: &mut bool, v: &mut f32, lo, hi, name| {
                *ch |= ui.add(egui::Slider::new(v, lo..=hi).text(name)).changed();
            };

            // ponytail: tiny helper so each section is one titled, framed group.
            let section = |ui: &mut egui::Ui, title: &str, body: &mut dyn FnMut(&mut egui::Ui)| {
                ui.add_space(2.0);
                ui.label(egui::RichText::new(title).small().color(egui::Color32::from_gray(0x8A)));
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    body(ui);
                });
            };

            // STIFFNESS first: a heavily-cut sheet turns into a floppy lattice that
            // lifts and bleeds when sprayed, and the preview can't show that. Sits
            // above the sliders so it stays visible while they're being dragged.
            if !self.frag.is_empty() {
                // Snapshot: the section closure needs &mut ui while self.frag is borrowed.
                let rows: Vec<(String, f32, f32)> = self
                    .frag
                    .iter()
                    .map(|(n, f)| (n.clone(), f.removed, f.neck_px))
                    .collect();
                section(ui, "STIFFNESS", &mut |ui| {
                    for (name, removed, neck) in &rows {
                        // ponytail: 2px/1px are placeholders — the honest threshold
                        // depends on material and kerf. They rank layers against each
                        // other correctly, which is what tuning needs. Upgrade path:
                        // derive from min_material_px and the sheet scale.
                        let col = if *neck >= 2.0 {
                            egui::Color32::from_gray(0x8A)
                        } else if *neck >= 1.0 {
                            egui::Color32::from_rgb(0xE0, 0xA0, 0x30)
                        } else {
                            egui::Color32::from_rgb(0xE0, 0x3A, 0x2F)
                        };
                        ui.label(
                            egui::RichText::new(format!(
                                "{:<8} cut {:>3.0}%   neck {:>4.1}px",
                                name,
                                removed * 100.0,
                                neck
                            ))
                            .monospace()
                            .color(col),
                        );
                    }
                    ui.label(
                        egui::RichText::new("neck = thinnest standing material; under 1px is lace")
                            .small()
                            .weak(),
                    );
                });
            }

            match self.mode {
                Mode::Halftone => {
                    use crate::lines::Shape;
                    section(ui, "SHAPE", &mut |ui| {
                        let label = match self.shape {
                            Shape::Lines => "Lines",
                            Shape::Wavy => "Wavy",
                            Shape::Dots => "Dots",
                            Shape::BlueNoise => "Blue-noise",
                            Shape::Hatch => "Orientation hatch",
                        };
                        egui::ComboBox::from_label("shape").selected_text(label).show_ui(ui, |ui| {
                            for (v, name) in [
                                (Shape::Lines, "Lines"),
                                (Shape::Wavy, "Wavy"),
                                (Shape::Dots, "Dots"),
                                (Shape::BlueNoise, "Blue-noise"),
                                (Shape::Hatch, "Orientation hatch"),
                            ] {
                                ch |= ui.selectable_value(&mut self.shape, v, name).changed();
                            }
                        });
                        // Shape-specific knobs; shared SCREEN/LEVELS/BLACK follow below.
                        if self.shape == Shape::Wavy {
                            s(ui, &mut ch, &mut self.wave_amp_frac, 0.0, 0.5, "wave amp (× pitch)");
                            s(ui, &mut ch, &mut self.wave_len_frac, 1.0, 12.0, "wave length (× pitch)");
                            s(ui, &mut ch, &mut self.wave_width_frac, 0.1, 1.0, "colour line width (× pitch)");
                        }
                        if self.shape == Shape::Hatch {
                            let mut bins = self.hatch_bins as f32;
                            if ui.add(egui::Slider::new(&mut bins, 1.0..=24.0).integer().text("orientation bins")).changed() {
                                self.hatch_bins = bins as u32;
                                ch = true;
                            }
                        }
                        if self.shape == Shape::BlueNoise {
                            // Dot size range (clamped cut-safe). min==max => fixed size.
                            // Whole-px steps: sub-px dot sizes aren't worth tuning.
                            let hi = self.spacing_px;
                            ch |= ui.add(egui::Slider::new(&mut self.bn_dot_min_px, 0.0..=hi).integer().text("min dot px")).changed();
                            ch |= ui.add(egui::Slider::new(&mut self.bn_dot_max_px, 0.0..=hi).integer().text("max dot px")).changed();
                        }
                    });
                    section(ui, "SCREEN", &mut |ui| {
                        s(ui, &mut ch, &mut self.spacing_px, 2.0, 64.0, "spacing px");
                        s(ui, &mut ch, &mut self.min_material_px, 0.0, self.spacing_px, "min material px");
                        s(ui, &mut ch, &mut self.min_cut_px, 0.0, self.spacing_px, "min cut px");
                        s(ui, &mut ch, &mut self.kerf_px, 0.0, self.spacing_px, "kerf px");
                    });
                    // Dots are isolated islands — bridges do nothing; grey them out.
                    let bridges_live = !matches!(self.shape, Shape::Dots | Shape::BlueNoise);
                    section(ui, "BRIDGES", &mut |ui| {
                        ui.add_enabled_ui(bridges_live, |ui| {
                            s(ui, &mut ch, &mut self.bridge_interval_px, 0.0, 200.0, "interval px (0=off)");
                            s(ui, &mut ch, &mut self.ht_bridge_px, 0.0, 16.0, "bridge px");
                            if !bridges_live {
                                ui.label(egui::RichText::new("(n/a for dots)").small().weak());
                            }
                        });
                    });
                    section(ui, "INKS", &mut |ui| {
                        let label = match self.inks {
                            cmyk::Inks::Cmyk => "CMYK",
                            cmyk::Inks::Cmykog => "CMYKOG (extended)",
                        };
                        let mut new_inks = self.inks;
                        egui::ComboBox::from_label("inks").selected_text(label).show_ui(ui, |ui| {
                            ui.selectable_value(&mut new_inks, cmyk::Inks::Cmyk, "CMYK");
                            ui.selectable_value(&mut new_inks, cmyk::Inks::Cmykog, "CMYKOG (extended)");
                        });
                        if new_inks != self.inks {
                            // Reset angles to the new preset's defaults (resizes the vec).
                            // Loads reset too: CMYKOG's magenta has already had orange
                            // pulled out of it, so a load tuned for CMYK's M no longer
                            // applies to the same ink.
                            self.inks = new_inks;
                            self.angles = new_inks.default_angles();
                            self.loads = new_inks.default_loads();
                            ch = true;
                        }
                    });
                    section(ui, "ANGLES", &mut |ui| {
                        let names = self.inks.names();
                        ui.checkbox(&mut self.angles_locked, "lock (move together)");
                        for i in 0..self.angles.len() {
                            let before = self.angles[i];
                            s(ui, &mut ch, &mut self.angles[i], 0.0, 180.0, names[i]);
                            if self.angles_locked {
                                // Screens repeat every 180 deg, so shift all inks by the
                                // same delta and wrap - relative spacing is what matters.
                                let d = self.angles[i] - before;
                                if d != 0.0 {
                                    for (j, a) in self.angles.iter_mut().enumerate() {
                                        if j != i {
                                            *a = (*a + d).rem_euclid(180.0);
                                        }
                                    }
                                }
                            }
                        }
                    });
                    section(ui, "INK LOAD", &mut |ui| {
                        // A magenta-heavy photo overloads one sheet while the rest stay
                        // sparse. Pull that ink back until its STIFFNESS row goes grey.
                        // Capped at 1.0: above it, cuts would exceed spacing - min
                        // material and break the guarantee every shape relies on.
                        let names = self.inks.names();
                        for i in 0..self.loads.len() {
                            s(ui, &mut ch, &mut self.loads[i], 0.2, 1.0, names[i]);
                        }
                        ui.label(
                            egui::RichText::new("1.0 = full cut budget; lower = stiffer sheet")
                                .small()
                                .weak(),
                        );
                        if self.inks.names().last() == Some(&"Black") {
                            ui.label(
                                egui::RichText::new("(black uses K width, below)").small().weak(),
                            );
                        }
                    });
                    let auto_cp = if self.auto_levels {
                        self.layers.as_ref().map(|l| {
                            // Per-channel clip points for the current ink set (tamed inks
                            // bake their own tone, so only untamed ones auto-level).
                            cmyk::channels(l, self.inks, &self.angles, &self.loads).into_iter()
                                .filter(|c| !c.tamed)
                                .map(|c| {
                                    let (wp, bp) = crate::halftone::auto_levels(&c.density, 0.005, 0.995);
                                    format!("{}  wp {wp:.3}  bp {bp:.3}", c.name)
                                }).collect::<Vec<_>>()
                        })
                    } else {
                        None
                    };
                    section(ui, "LEVELS", &mut |ui| {
                        ch |= ui.checkbox(&mut self.auto_levels, "auto levels").changed();
                        if let Some(lines) = &auto_cp {
                            // auto is per-channel: show the four computed clip points
                            // instead of the (ignored) manual sliders.
                            for line in lines {
                                ui.label(egui::RichText::new(line).monospace().weak());
                            }
                        } else {
                            s(ui, &mut ch, &mut self.white_point, 0.0, 1.0, "white point");
                            s(ui, &mut ch, &mut self.black_point, 0.0, 1.0, "black point");
                        }
                        s(ui, &mut ch, &mut self.gamma, 0.2, 3.0, "gamma");
                        s(ui, &mut ch, &mut self.scurve, 0.0, 12.0, "spray s-curve");
                    });
                    section(ui, "BLACK (K)", &mut |ui| {
                        // Black spray is opaque: keep K thin and deep so it can't
                        // bully the CMY colour underneath.
                        ch |= ui.checkbox(&mut self.k_contour, "contour mode (DoG edges)").changed();
                        s(ui, &mut ch, &mut self.k_width_frac, 0.1, 0.8, "K width (× pitch)");
                        if self.k_contour {
                            s(ui, &mut ch, &mut self.dog_sigma1, 0.5, 4.0, "DoG fine σ");
                            s(ui, &mut ch, &mut self.dog_sigma2, 1.0, 8.0, "DoG coarse σ");
                            s(ui, &mut ch, &mut self.dog_threshold, 0.0, 0.3, "DoG threshold");
                        } else {
                            s(ui, &mut ch, &mut self.k_deep_clip, 0.0, 0.95, "deep-shadow clip");
                            s(ui, &mut ch, &mut self.k_gamma, 1.0, 3.0, "K gamma");
                            s(ui, &mut ch, &mut self.ucr, 0.0, 1.0, "under-colour removal");
                        }
                    });
                    section(ui, "PRE-FILTER", &mut |ui| {
                        s(ui, &mut ch, &mut self.bilateral_px, 0.0, 6.0, "bilateral px");
                    });
                    section(ui, "SHEET", &mut |ui| {
                        // Export-only: physical paper size + margin (squeeze). Doesn't
                        // change the preview raster, so no `ch`/dirty here.
                        let label = match self.paper {
                            None => "raw px",
                            Some(crate::svg::Paper::A2) => "A2",
                            Some(crate::svg::Paper::A3) => "A3",
                            Some(crate::svg::Paper::A4) => "A4",
                            Some(crate::svg::Paper::A5) => "A5",
                        };
                        egui::ComboBox::from_label("paper").selected_text(label).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.paper, None, "raw px");
                            ui.selectable_value(&mut self.paper, Some(crate::svg::Paper::A2), "A2");
                            ui.selectable_value(&mut self.paper, Some(crate::svg::Paper::A3), "A3");
                            ui.selectable_value(&mut self.paper, Some(crate::svg::Paper::A4), "A4");
                            ui.selectable_value(&mut self.paper, Some(crate::svg::Paper::A5), "A5");
                        });
                        ui.add_enabled_ui(self.paper.is_some(), |ui| {
                            ui.add(egui::Slider::new(&mut self.margin_mm, 0.0..=40.0).text("margin mm"));
                        });
                    });
                }
                Mode::Stencil => {
                    section(ui, "QUANTIZE", &mut |ui| {
                        let mut colors_f = self.colors as f32;
                        if ui.add(egui::Slider::new(&mut colors_f, 2.0..=16.0).integer().text("colors")).changed() {
                            self.colors = colors_f as usize;
                            ch = true;
                        }
                        s(ui, &mut ch, &mut self.coarsen_px, 0.0, 16.0, "coarsen px");
                        s(ui, &mut ch, &mut self.min_feature_px, 0.0, 16.0, "min feature px");
                    });
                    section(ui, "BRIDGES", &mut |ui| {
                        ch |= ui.checkbox(&mut self.bridges, "bridges (cut-safe)").changed();
                        ui.add_enabled_ui(self.bridges, |ui| {
                            s(ui, &mut ch, &mut self.min_material_px, 0.5, 16.0, "min material px");
                            s(ui, &mut ch, &mut self.bridge_px, 1.0, 32.0, "bridge width px");
                        });
                    });
                    section(ui, "LEVELS", &mut |ui| {
                        s(ui, &mut ch, &mut self.white_point, 0.0, 1.0, "white point");
                        s(ui, &mut ch, &mut self.black_point, 0.0, 1.0, "black point");
                        s(ui, &mut ch, &mut self.gamma, 0.2, 3.0, "gamma");
                    });
                }
            }
            // OUTPUT (smooth) only affects the traced stencil path, not the analytic
            // halftone quads — show it only in Stencil mode.
            if self.mode == Mode::Stencil {
                section(ui, "OUTPUT", &mut |ui| {
                    s(ui, &mut ch, &mut self.smooth_px, 0.0, 4.0, "smooth px");
                });
            }

            if ch {
                self.dirty = true;
            }
            }); // ScrollArea
        });

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(egui::Color32::from_gray(0x0E)).inner_margin(12.0))
            .show(ctx, |ui| match &self.texture {
            Some(tex) => {
                let avail = ui.available_size();
                let [tw, th] = tex.size();
                let scale = (avail.x / tw as f32).min(avail.y / th as f32).min(1.0);
                ui.centered_and_justified(|ui| {
                    ui.image((tex.id(), egui::vec2(tw as f32 * scale, th as f32 * scale)));
                });
            }
            None => {
                ui.centered_and_justified(|ui| {
                    ui.label(egui::RichText::new("Open an image to begin").size(16.0).color(egui::Color32::from_gray(0x66)));
                });
            }
        });

        // While a render or generate is in flight, keep repainting so the
        // finished frame / status lands.
        if self.in_flight || self.generating {
            ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_round_trips_every_knob() {
        // Mutate a representative slice of knobs (scalars, bools, an enum, the ink
        // set + its angle/load vectors, paper, mode), serialize, then apply onto a
        // fresh App and confirm every value survived the text round-trip.
        let mut a = App::new();
        a.mode = Mode::Stencil;
        a.shape = crate::lines::Shape::Wavy;
        a.wave_amp_frac = 0.42;
        a.wave_width_frac = 0.37;
        a.spacing_px = 12.5;
        a.min_material_px = 1.75;
        a.k_contour = true;
        a.auto_levels = true;
        a.paper = Some(crate::svg::Paper::A3);
        a.inks = cmyk::Inks::Cmykog;
        a.angles = a.inks.default_angles();
        a.loads = a.inks.default_loads();
        a.angles[0] = 33.0;
        a.loads[1] = 0.5;
        a.colors = 7;

        let text = a.preset_string();
        let mut b = App::new();
        assert_ne!(b.spacing_px, a.spacing_px, "fresh App differs before apply");
        b.apply_preset(&text);

        assert!(b.mode == a.mode);
        assert!(b.shape == a.shape);
        assert_eq!(b.wave_amp_frac, a.wave_amp_frac);
        assert_eq!(b.wave_width_frac, a.wave_width_frac);
        assert_eq!(b.spacing_px, a.spacing_px);
        assert_eq!(b.min_material_px, a.min_material_px);
        assert_eq!(b.k_contour, a.k_contour);
        assert_eq!(b.auto_levels, a.auto_levels);
        assert!(b.paper == a.paper);
        assert!(b.inks == a.inks);
        assert_eq!(b.angles, a.angles);
        assert_eq!(b.loads, a.loads);
        assert_eq!(b.colors, a.colors);
        assert!(b.dirty, "loading a preset marks the preview dirty");
    }

    #[test]
    fn apply_preset_keeps_current_on_missing_or_bad_keys() {
        let mut a = App::new();
        let before = a.spacing_px;
        // Unknown key ignored; bad value falls back to the current field.
        a.apply_preset("nonsense_key = 5
spacing_px = not_a_number
");
        assert_eq!(a.spacing_px, before);
    }

    #[test]
    fn apply_preset_fixes_mismatched_ink_vectors() {
        // Switching to cmykog but giving 4-length angles must not stick a bad-length
        // vector on the App — it falls back to the ink defaults (len == count).
        let mut a = App::new();
        a.apply_preset("inks = cmykog
angles = 15,75,0,45
");
        assert!(a.inks == cmyk::Inks::Cmykog);
        assert_eq!(a.angles.len(), a.inks.count());
        assert_eq!(a.loads.len(), a.inks.count());
    }
}
