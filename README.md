# laser_halftone

[![CI](https://github.com/kink80/cutter/actions/workflows/ci.yml/badge.svg)](https://github.com/kink80/cutter/actions/workflows/ci.yml)

Prebuilt binaries for Linux/macOS/Windows are attached to every [release](https://github.com/kink80/cutter/releases/latest).

Turn a photo into layered, laser-cuttable **stencils** you spray paint through to
reproduce the image.

Instead of printing ink, this vectorizes an image into cut geometry: the laser cuts
slots/dots out of a sheet, you lay each sheet over your surface and spray one colour
through it, and the stacked layers rebuild the picture. Every cut respects the physical
constraints of a real stencil — a minimum standing material so the sheet doesn't fall
apart, kerf compensation, and optional bridges — so the output is ready to cut, not just
pretty on screen.

There are two independent pipelines:

- **Halftone** — analytic CMYK (or extended-gamut) line/dot screens. Steps *along* each
  screen line sampling density and emits vector cut geometry directly (no raster
  tracing), so the laser moves in smooth continuous passes.
- **Stencil** — multicolour spray masks. Quantizes the image to N flat colours
  (k-means in Lab space) and traces each colour into a smoothed cut layer.

## Install

Download the binary for your OS from the [latest release](https://github.com/kink80/cutter/releases/latest):

| OS | file |
|----|------|
| Linux | `laser_halftone-linux-x86_64` |
| macOS (Apple Silicon) | `laser_halftone-macos-arm64` |
| Windows | `laser_halftone-windows-x86_64.exe` |

On Linux/macOS mark it executable and (optionally) put it on your `PATH`:

```sh
chmod +x laser_halftone-linux-x86_64
mv laser_halftone-linux-x86_64 ~/.local/bin/laser_halftone   # now just `laser_halftone`
```

macOS Gatekeeper may block an unsigned binary the first time — right-click → Open, or
`xattr -d com.apple.quarantine laser_halftone-macos-arm64`. On Windows, run the `.exe`
directly (the examples below use `laser_halftone` — substitute the full filename).

> Prefer building from source? It's a plain Rust (edition 2024) project — clone it and run
> `cargo run --release -- <subcommand> …`. Every example below works from source too: just
> replace `laser_halftone` with `cargo run --release --`.

## Run

```sh
laser_halftone gui                                                              # interactive app
laser_halftone halftone --input photo.jpg --spacing-px 8 --min-material-px 1 --min-cut-px 0.5
laser_halftone stencil  --input photo.jpg --colors 4
```

The GUI lets you open an image, tweak every parameter with a live preview, and
"Generate all layers". The CLI does the same headless.

Output for an N-ink halftone is one SVG per ink (`out_c.svg`, `out_m.svg`, …), each with
corner **punch holes + registration crosshairs** at identical coordinates so the sheets
pin onto alignment pins and line up, plus a composite `out_preview.png`.

## Snapshots

A photo run through each pipeline (`--format png` renders these composite previews;
production runs emit the per-ink / per-colour **SVGs** you actually cut):

| input photo | halftone (blue-noise AM+FM) | stencil layer (bridged) |
|:---:|:---:|:---:|
| ![source](docs/img/source.jpg) | ![halftone blue-noise preview](docs/img/halftone-bluenoise.png) | ![stencil layer with bridges](docs/img/stencil-layer.png) |

```sh
# left -> middle: variable-size FM screen, composite PNG preview
laser_halftone halftone --input photo.jpg --shape blue-noise \
  --spacing-px 8 --min-material-px 1 --min-cut-px 0.5 --dot-min-px 1 --dot-max-px 6 \
  --format png --out-prefix preview

# left -> right: one bridged colour layer of a 6-colour split
laser_halftone stencil --input photo.jpg --colors 6 \
  --bridges on --min-material-px 2 --min-feature-px 3 --format png --out-prefix layer
```

The interactive GUI shows the same preview live while you drag any parameter:

```sh
laser_halftone gui
```

## Halftone mode

Each ink channel is screened into a chosen **mark shape**. All shapes honour the same
physical laser constraints (dead-zone below `min-cut-px`, max cut = `spacing − min-material`,
kerf shrink) and are cut-safe — material always holds together.

| `--shape` | what it is |
|-----------|------------|
| `lines` | straight rotated screen lines; **width** encodes tone (default) |
| `wavy` | line screen with a sinusoidally displaced centreline |
| `dots` | classic AM halftone — one variable-radius circle per grid cell |
| `blue-noise` | frequency-modulated screen: dots placed at variable frequency (error diffusion). No grid, no moiré; forgiving of hand-registration. `--dot-min-px`/`--dot-max-px` also scale dot *size* with tone (AM+FM hybrid) for finer detail; default is fixed size. Sizes clamp cut-safe |
| `hatch` | line screen whose local angle follows the image's dominant edge orientation, so cuts run *along* features. Bin seams narrow rather than collide, preserving the minimum gap |

**Inks.** `--inks cmyk` (default) or `--inks cmykog` for an extended gamut (adds orange +
green for cleaner oranges/greens than muddy C+M+Y overlap). Screen angles: `--angles
15,75,0,45[,…]` (one per ink), or the legacy per-channel `--angle-c/m/y/k`.

**Black is special.** Spray black is opaque, so K is tamed to stay a thin accent rather
than obliterating the colour underneath: `--k-mode tonal` (deep-shadow clip + gamma +
under-colour removal + width cap) or `--k-mode contour` (Difference-of-Gaussians edge
lines — comic-book contours).

**Other knobs:** `--kerf-px`, `--bridge-interval-px`/`--bridge-px` (physical tabs, laid
out as a staggered hex lattice to keep the sheet flat), `--scurve` (spray-paint opacity
curve), `--bilateral-px` (edge-preserving pre-filter), `--auto-levels on`, and
`--paper A2|A3|A4|A5` + `--margin-mm` to emit at true physical size on a sheet.

```sh
laser_halftone halftone --input photo.jpg \
  --spacing-px 8 --min-material-px 1 --min-cut-px 0.5 \
  --shape hatch --inks cmykog --paper A4 --margin-mm 10 --out-prefix art
```

## Stencil mode

Splits the image into N solid-colour layers (Inkscape "Trace Bitmap → Multicolor" style),
darkest first (spray order). Outputs one traced SVG per colour plus a palette.

```sh
laser_halftone stencil --input photo.jpg --colors 6 \
  --bridges on --min-material-px 2 --min-feature-px 3 --out-prefix poster
```

- `--colors N` — palette size (k-means in perceptual Lab space).
- `--bridges on` + `--min-material-px` — tab enclosed islands so they don't fall out.
- `--min-feature-px` — de-speckle: absorb regions smaller than this.
- `--coarsen-px` — blur before quantizing for chunkier areas.
- `--smooth-px` — how closely the traced curve hugs the pixel staircase.
- `--format svg` (default) or `png` (per-layer raster masks).

## How to use the output

1. Cut each layer SVG from your stencil material on a laser cutter.
2. Pin all sheets through the corner punch holes so they register to one grid.
3. Spray each colour through its layer in order (darkest/black last for halftone,
   darkest first for stencil — see the generated `*_palette.txt`).

## Project layout

| file | role |
|------|------|
| `src/lines.rs` | halftone: density → cut ribbons, all mark shapes, preview + export |
| `src/cmyk.rs` | image load/resize, RGB↔CMYK, Lab k-means, ink separation (CMYK/CMYKOG) |
| `src/stencil.rs` | multicolour quantize → island bridging → marching-squares trace |
| `src/svg.rs` | geometry → SVG, paper sizing, punch holes, registration marks |
| `src/smooth.rs` | Potrace-style ring smoothing (DP simplify + Bézier) for stencil edges |
| `src/halftone.rs` | shared tone helpers (levels, auto-levels) |
| `src/gui.rs` | native egui app with live preview |
| `docs/superpowers/specs/` | design specs |
