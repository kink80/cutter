# Novel halftone techniques: blue-noise FM, orientation hatch, extended inks

## Context

The Halftone mode now has three cut-safe mark shapes (Lines, Wavy, Dots) selected via a
`Shape` enum through one dispatcher `generate_shape(map, w, h, angle, params, max_cut)
-> Vec<Ribbon>` (`src/lines.rs:352`). This round adds three genuinely different
techniques, all constrained by the project's north star: **laser-cuttable stencils are
the marking primitive** — cuts are slots/dots the material is sprayed through and must
hold together — and, per the user, favor techniques **forgiving of hand-registration
error** between layers.

Chosen in brainstorming (skipping the ones dropped for cut-safety earlier — flow
streamlines, Hilbert):
1. **Blue-noise FM screen** — a new `Shape`. Fixed-size dots at variable *frequency*.
2. **Orientation hatch** — a new `Shape`. The cut-safe version of the dropped Flow idea.
3. **Extended process inks (CMYKOG)** — generalize the pipeline from fixed 4 channels to
   N screened inks for a wider gamut.

Ship order: **shapes first** (small, self-contained, independent of the refactor), then
extended inks (bigger, has a genuinely hard color-science core).

## Key leverage point

`generate_shape` already funnels every channel (CMY loop + tamed K) through one
channel-agnostic path returning `Vec<Ribbon>`. New shapes are new `match` arms; nothing
downstream changes as long as they emit closed ribbons. The dot-polygon emit already
exists in `generate_dots` (`src/lines.rs`), and `gaussian` (`src/lines.rs:~505`) is
reusable for the orientation field. Physical constraints live in `cut_extent`.

---

## Phase 1 — Blue-noise FM screen (`Shape::BlueNoise`)

**Idea:** frequency-modulated screen — dots of *fixed* size placed at *variable*
frequency (denser = darker), blue-noise distributed so there's no grid, no moiré, no
rosettes. Registration-forgiving: isolated fixed dots need no layer alignment to read
correctly. Cut-safe by construction (isolated islands, like `Dots`).

**Algorithm (no lookup tables):** Floyd–Steinberg error diffusion on the density map
**downsampled to a `spacing_px`-resolution grid**, so each retained cell becomes one
real, cuttable circle. FS produces blue-noise-like point sets for free.
- Downsample: average the toned density (`tone(sample(...))`) over each `spacing`×`spacing`
  cell into a coarse grid of dimensions `⌈w/spacing⌉ × ⌈h/spacing⌉`.
- Serpentine FS over the coarse grid; threshold 0.5; push quantization error to
  neighbours (7/16, 3/16, 5/16, 1/16).
- Each cell that quantizes to 1 → one dot at the cell center, fixed radius
  `r = (max_cut/2)` shrunk by `kerf_px/2`. Dead-zone: if the coarse cell's density is
  below the `w_min` fraction, force it to 0 before diffusion (don't emit sub-threshold
  hairline dots).
- Emit the N-gon circle exactly as `generate_dots` does (extract a shared
  `emit_dot(cx, cy, r) -> Ribbon` helper; `generate_dots` and `BlueNoise` both call it).

**Constraints:** fixed radius ≤ `max_cut/2` guarantees dots never touch → `min_material`
preserved. Bridges inert (isolated dots), like `Dots` — greyed out in GUI.
**Determinism:** FS is deterministic; no RNG (preview == export).

**Files:** `src/lines.rs` (new `Shape::BlueNoise`, `generate_bluenoise`, `emit_dot`
helper), GUI `Shape` ComboBox + label, CLI `--shape blue-noise`.

**Tests (mirror existing `lines.rs` style):** darker uniform field → more dots; faint
field (below dead-zone) → zero dots; every dot ribbon closed (`pts.len() >= 3`); dot
diameter ≤ `max_cut` (never merge). Determinism: same input → identical ribbons.

---

## Phase 2 — Orientation hatch (`Shape::Hatch`)

**Idea:** the feature-following look we wanted from Flow, made cut-safe. Keep a regular
parallel line-screen, but rotate the *local* screen angle toward the image's dominant
edge orientation. Because each neighbourhood stays a parallel screen with guaranteed
`spacing`, material always connects — **no enclosure** (the failure that killed Flow).

**Safe formulation (ship the conservative version first):**
- Build a smoothed orientation field from the structure tensor: gradients `gx,gy` (central
  differences on the K/luminance map), then blur `gx², gy², gx·gy` with `gaussian`
  (reuse `src/lines.rs` gaussian). Dominant orientation per region =
  `0.5·atan2(2·⟨gx·gy⟩, ⟨gx²⟩−⟨gy²⟩)`.
- **Quantize orientation into a few bins** (e.g. 6 bins over 180°). Partition the canvas
  into bin-regions; screen each region with the existing straight line-screen
  (`generate_quads_capped`) at that bin's angle, masking the density map to the region.
  Parallel-within-region ⇒ trivially cut-safe; region boundaries are the only seams, and
  they leave standing material (adjacent regions are independent screens).
- Width encodes tone as usual (`cut_extent`); dead-zone/kerf/max-cut/bridges all apply
  unchanged (it *is* the line walker per region).

**Why not the smooth continuously-varying field:** a smoothly-curving line screen risks
converging lines (spacing collapses → material pinches). Binned regions keep spacing
exactly `spacing` everywhere. Note the smooth-field variant as a deferred follow-up
(`ponytail:`), gated on a spacing-preservation guarantee.

**Files:** `src/lines.rs` (`Shape::Hatch`, `generate_hatch`, orientation-field helper),
GUI (ComboBox entry + a "orientation bins" slider in the shape-specific knobs), CLI
`--shape hatch [--hatch-bins N]`. Add `hatch_bins` to `ShapeParams`.

**Tests:** an image with a strong directional feature (a synthetic diagonal edge) →
lines in that region align near the feature orientation (assert the dominant emitted
segment direction is within a bin of the true edge angle); uniform field → falls back to
a single bin (behaves like `Lines`); darker → more ink; all ribbons closed; cut-safety:
assert min gap between adjacent ribbons within a region ≥ `min_material` (spacing
preserved).

---

## Phase 3 — Extended process inks (N screened channels, CMYKOG preset)

**Idea:** unlock the halftone pipeline from fixed C/M/Y/K to **N** screened inks. Ship a
**CMYKOG** preset (add orange + green) for a wider gamut; CMYK and duotone fall out as
other channel lists. This is the big one and has a genuinely hard color-science core
(RGB→N-ink separation + generalized under-color-removal) — the separation is shipped as a
**tunable first-pass heuristic**, explicitly marked for later refinement.

**Data model** (replaces the fixed 4-field `Layers`, `src/cmyk.rs:6`):
```rust
pub struct Channel {
    pub density: Vec<f32>,      // row-major separation map
    pub angle: f32,             // screen angle
    pub display_rgb: [f32; 3],  // transmission colour: preview compositing + SVG fill
    pub name: &'static str,     // "Orange"
    pub suffix: &'static str,   // "o" (SVG filename)
    pub taming: Option<Taming>, // Some only for opaque inks (K, and optionally opaque spots)
}
pub struct Taming { mode: KMode, deep_clip, gamma, width_frac, ucr, dog_sigma1/2, dog_threshold }
```
`Layers` → `Vec<Channel>`. The `k_*` fields move from `LineParams` into `Taming`.

**Generalized generate_all** (`src/lines.rs:555`): one loop over channels — if
`taming.is_none()` run the current CMY branch (auto-levels + `generate_shape`); if
`Some`, run the `generate_k` taming against that channel's map. Returns `Vec<Vec<Ribbon>>`.
UCR generalizes to "subtract the overlap of inks *beneath* this one in a defined order"
(needs a channel ordering; dark→light). Ship generalized UCR as the heuristic core.

**Separation (RGB→CMYKOG), first-pass heuristic:** start from textbook RGB→CMYK
(`rgb_to_cmyk`, `src/cmyk.rs:14`), then pull **orange** from regions where C is low and
M/Y high, **green** where C+Y high and M low (GCR-style: move ink from the CMY that the
spot ink can reproduce into the spot channel, reducing muddiness). Tunable split
strength. Marked `ponytail:` as an approximation — a real ICC/Neugebauer model is the
upgrade path.

**Preview compositing** (`render_preview`, `src/lines.rs:653`): replace
`cmyk_to_rgb` with **multiplicative ink stacking** — start white, for each on-channel
multiply `rgb *= (1 − ink·(1 − display_rgb))`. Generalizes to any N and any ink colours.

**Export** (`src/lines.rs:681`): iterate channels using `name`/`display_rgb`/`suffix`,
replacing the hardcoded `chans` table and `"{}/4"`. `svg::ribbons_to_svg` is unchanged
(already takes a `fill: &str`).

**CLI/GUI:**
- CLI: `--inks cmyk|cmykog` preset; angles become `--angles 15,75,0,45[,...]` (replace the
  four `--angle-c/m/y/k` flags; keep the old flags working for CMYK back-compat).
- GUI: an INKS preset ComboBox (CMYK / CMYKOG); the ANGLES section iterates the channel
  vec (N sliders, labelled by `channel.name`); the BLACK(K) section becomes a per-tamed-
  channel panel (only channels with `Some(taming)`).

**Tests:** CMYK preset reproduces current output byte-for-byte (regression guard on the
existing tests — the refactor must be behaviour-preserving for 4 channels); CMYKOG preset
emits 6 channel ribbon-sets; multiplicative preview of a single full-orange channel over
white yields ~orange; generalized UCR suppresses an under-ink where a darker ink already
covers.

---

## Branch

Do all this work on a **fresh branch** off the current `stencil-shapes` (e.g.
`novel-halftone`) so it can be reverted easily and independently of the shipped shapes.

## Files to change (all phases)

- `src/lines.rs` — two new `Shape` arms + generators (P1, P2); `emit_dot`, orientation-
  field, `hatch_bins` in `ShapeParams` (P1/P2); `Channel`/`Taming`, N-general
  `generate_all`, multiplicative `render_preview`, N-general `export` (P3).
- `src/cmyk.rs` — `Layers` → `Vec<Channel>`; RGB→CMYKOG separation heuristic; keep
  `rgb_to_cmyk`/`cmyk_to_rgb` (still used by CMYK path + stencil) (P3).
- `src/gui.rs` — ComboBox entries + shape knobs (P1/P2); INKS preset, N angle sliders,
  per-tamed-channel panel (P3).
- `src/main.rs` — `--shape blue-noise|hatch` (+ `--hatch-bins`) (P1/P2); `--inks`,
  `--angles` (P3); USAGE.
- `src/halftone.rs` — the four `ANGLE_*` constants generalize to a per-preset default
  angle list (P3). `svg.rs` unchanged.

## Verification (per phase)

- `cargo test` green after each phase; **Phase 3's first gate is that CMYK output is
  unchanged** (behaviour-preserving refactor).
- CLI smoke per new shape on `cata.jpg`: `halftone --input cata.jpg --spacing-px 8
  --min-material-px 1 --min-cut-px 0.5 --shape blue-noise` (then `hatch`, then
  `--inks cmykog`); confirm channel SVGs + preview PNG write without error.
- Eyeball each `*_preview.png`: blue-noise = grainless stochastic dots dark-where-dark;
  hatch = lines bending to follow features but never converging; CMYKOG = visibly wider
  gamut / cleaner oranges & greens than CMYK on a colourful image.
- Validate one SVG per phase is well-formed XML (raw-px and A4 paper layouts).
- GUI launches without panic; cycle the new shapes and the INKS preset.

## Deliberately deferred (`ponytail:` comments)

- Blue-noise via a precomputed void-and-cluster mask (FS is the lazy stand-in).
- Orientation hatch smooth continuously-varying field (ship binned regions first).
- Real ICC/Neugebauer N-ink separation (ship the GCR-style heuristic first).

## Spec doc location

On exit from plan mode, also write this spec to
`docs/superpowers/specs/2026-08-04-novel-halftone-techniques-design.md` and commit
(brainstorming-skill convention); this plan file is the working copy.
