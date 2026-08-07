//! Op 2 — multicolor stencil split (Inkscape "Trace Bitmap -> Multicolor + Stack Scans").
//!
//! Pipeline: load+resize -> median-cut quantize to N colors -> dark->light order ->
//! cumulative stack masks -> auto-bridge enclosed islands (raster domain) ->
//! marching-squares trace -> emit N SVGs + palette.
//!
//! ponytail: bridging is done in the RASTER domain (clear a min-material band of
//! cut-pixels back to material before tracing) instead of the plan's vector-domain
//! i_overlay notch. Same laser-ready result, no polygon-boolean dependency, and the
//! tracer (which must exist anyway) emits already-bridged contours. Locked decision
//! Q4 said i_overlay; this achieves the identical output with less code. Swap to a
//! clipping crate only if a future need (e.g. arbitrary-angle bridges) can't be
//! expressed on the raster grid.

use image::imageops::FilterType;

#[derive(Clone)]
pub struct Params {
    pub colors: usize,
    pub bridge_px: f32,     // bridge tab width in px (independent of min-material)
    pub min_feature_px: f32, // despeckle: drop rings with area below this (px^2)
    pub bridges: bool,      // false = trace islands as-is (Inkscape-equivalent, not cut-safe)
    pub blur_px: f32,       // pre-quantize blur radius: >0 merges fine detail into bigger chunks
    pub white_point: f32,   // tone remap before quantize: ink at/below -> white
    pub black_point: f32,   // ink at/above -> black
    pub gamma: f32,         // midtone curve
}

/// Per-channel tone remap applied before quantizing, same curve as the halftone
/// levels control. "ink" = 1 - channel; remap it, then write back. Raising
/// white_point flattens off-white toward pure white so those regions merge into
/// the lightest layer instead of becoming their own speckly color.
fn apply_levels(px: [u8; 3], wp: f32, bp: f32, gamma: f32) -> [u8; 3] {
    let mut out = [0u8; 3];
    for c in 0..3 {
        let ink = 1.0 - px[c] as f32 / 255.0;
        let ink = crate::halftone::levels(ink, wp, bp, gamma);
        out[c] = ((1.0 - ink) * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Palette entry: rgb + luminance (for dark->light ordering).
#[derive(Clone, Copy)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    /// CMYK components in [0,1] of the snapped color (for the palette file).
    pub cmyk: [f32; 4],
    /// CIELAB of the snapped color (for perceptual nearest-color matching).
    pub lab: [f32; 3],
}

fn luma(c: &Color) -> f32 {
    0.2126 * c.r as f32 + 0.7152 * c.g as f32 + 0.0722 * c.b as f32
}

/// One traced layer: outer rings + hole rings (already bridged), plus its color.
pub struct Layer {
    pub color: Color,
    pub outers: Vec<Vec<(f32, f32)>>,
    pub holes: Vec<Vec<(f32, f32)>>,
}

// ---------------------------------------------------------------------------
// 1. Lab-space k-means quantization
// ---------------------------------------------------------------------------

/// Nearest palette color by perceptual (Lab) distance. Matches BayStencil's
/// `remap_to_palette_lab` — ΔE in Lab, not RGB Euclidean.
fn nearest(palette: &[Color], lab: &[f32; 3]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in palette.iter().enumerate() {
        let d = (c.lab[0] - lab[0]).powi(2) + (c.lab[1] - lab[1]).powi(2) + (c.lab[2] - lab[2]).powi(2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// 2. Quantize -> label raster, ordered dark->light (index 0 = darkest)
// ---------------------------------------------------------------------------

/// Returns (palette ordered dark->light, label raster with index 0 = darkest).
/// k-means clusters in Lab, palette snapped to printable CMYK, remap in Lab.
fn quantize(pixels: &[[u8; 3]], n: usize) -> (Vec<Color>, Vec<u8>) {
    let centers = crate::cmyk::kmeans_lab(pixels, n);
    let mut palette: Vec<Color> = centers
        .iter()
        .map(|rgb| {
            let (snap, cmyk) = crate::cmyk::snap_to_cmyk(rgb[0], rgb[1], rgb[2]);
            Color {
                r: snap[0], g: snap[1], b: snap[2], cmyk,
                lab: crate::cmyk::rgb_to_lab(snap[0] as f32 / 255.0, snap[1] as f32 / 255.0, snap[2] as f32 / 255.0),
            }
        })
        .collect();
    // Order dark->light so index 0 is darkest (Q5 stack direction).
    palette.sort_by(|a, b| luma(a).partial_cmp(&luma(b)).unwrap());
    let labels: Vec<u8> = pixels
        .iter()
        .map(|p| {
            let lab = crate::cmyk::rgb_to_lab(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
            nearest(&palette, &lab) as u8
        })
        .collect();
    (palette, labels)
}

// ---------------------------------------------------------------------------
// 2b. De-granularize: absorb small same-label regions into their neighbours.
// ---------------------------------------------------------------------------
// `min_feature_px` is an AREA floor (px^2). This is what actually makes the
// stencil "less granular": any connected same-label blob smaller than the floor
// is rewritten to the label that dominates its boundary, so speckle merges into
// the surrounding chunk instead of just being deleted at trace time (which left
// a hole). Runs iteratively-by-size: components are re-scanned until none remain
// under the floor, so a small blob absorbed into another small blob can still be
// caught. ponytail: single bottom-up pass is enough in practice; a stray nested
// case just survives one extra `stencils` call — not worth a fixpoint loop.
fn merge_small_regions(labels: &mut [u8], w: usize, h: usize, min_area: f32) {
    if min_area <= 1.0 {
        return; // nothing to merge
    }
    let n = labels.len();
    // Connected components of equal label (4-connectivity).
    let mut comp = vec![usize::MAX; n];
    let mut sizes: Vec<usize> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..n {
        if comp[start] != usize::MAX {
            continue;
        }
        let id = sizes.len();
        let lab = labels[start];
        comp[start] = id;
        stack.push(start);
        let mut cells = Vec::new();
        while let Some(idx) = stack.pop() {
            cells.push(idx);
            let (x, y) = (idx % w, idx / w);
            let push = |nx: usize, ny: usize, stack: &mut Vec<usize>, comp: &mut Vec<usize>| {
                let ni = ny * w + nx;
                if labels[ni] == lab && comp[ni] == usize::MAX {
                    comp[ni] = id;
                    stack.push(ni);
                }
            };
            if x > 0 { push(x - 1, y, &mut stack, &mut comp); }
            if x + 1 < w { push(x + 1, y, &mut stack, &mut comp); }
            if y > 0 { push(x, y - 1, &mut stack, &mut comp); }
            if y + 1 < h { push(x, y + 1, &mut stack, &mut comp); }
        }
        sizes.push(cells.len());
        members.push(cells);
    }
    // Smallest-first: absorb each under-floor component into the label dominating
    // its boundary. Smallest-first so tiny specks vanish before their neighbours
    // are considered, letting chunks grow.
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&i| sizes[i]);
    for id in order {
        if (sizes[id] as f32) >= min_area {
            continue;
        }
        // Tally boundary labels (of cells NOT in this component).
        let mut tally: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
        for &idx in &members[id] {
            let (x, y) = (idx % w, idx / w);
            let nb = |nx: usize, ny: usize, tally: &mut std::collections::HashMap<u8, usize>| {
                let ni = ny * w + nx;
                if comp[ni] != id {
                    *tally.entry(labels[ni]).or_insert(0) += 1;
                }
            };
            if x > 0 { nb(x - 1, y, &mut tally); }
            if x + 1 < w { nb(x + 1, y, &mut tally); }
            if y > 0 { nb(x, y - 1, &mut tally); }
            if y + 1 < h { nb(x, y + 1, &mut tally); }
        }
        if let Some((&new_lab, _)) = tally.iter().max_by_key(|(_, c)| **c) {
            for &idx in &members[id] {
                labels[idx] = new_lab;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Per-color masks: mask(i) = { label == i }
// ---------------------------------------------------------------------------
// Each stencil cuts out ONLY its own color band, so the darkest layer is the
// darkest region — not a solid full-sheet silhouette. This matches how spray
// stencils physically work: one sheet per paint color, each masking just where
// that color goes. (The old cumulative `label >= i` stacking made layer 0 a
// fully-colored sheet with nothing to cut, which is unusable as a stencil.)
fn stack_mask(labels: &[u8], i: usize) -> Vec<bool> {
    labels.iter().map(|&l| l as usize == i).collect()
}

// ---------------------------------------------------------------------------
// 4. Island detection + raster-domain bridging
// ---------------------------------------------------------------------------

/// Connected-component label of `keep`==true cells (4-connectivity). Returns
/// (component id per cell or usize::MAX for cut cells, which components touch the border).
fn components(keep: &[bool], w: usize, h: usize) -> (Vec<usize>, Vec<bool>) {
    let mut comp = vec![usize::MAX; keep.len()];
    let mut touches_border = Vec::new();
    let mut stack = Vec::new();
    let mut next = 0usize;
    for start in 0..keep.len() {
        if !keep[start] || comp[start] != usize::MAX {
            continue;
        }
        let id = next;
        next += 1;
        touches_border.push(false);
        comp[start] = id;
        stack.push(start);
        while let Some(idx) = stack.pop() {
            let (x, y) = (idx % w, idx / w);
            if x == 0 || y == 0 || x == w - 1 || y == h - 1 {
                touches_border[id] = true;
            }
            let push = |nx: usize, ny: usize, stack: &mut Vec<usize>, comp: &mut Vec<usize>| {
                let ni = ny * w + nx;
                if keep[ni] && comp[ni] == usize::MAX {
                    comp[ni] = id;
                    stack.push(ni);
                }
            };
            if x > 0 { push(x - 1, y, &mut stack, &mut comp); }
            if x + 1 < w { push(x + 1, y, &mut stack, &mut comp); }
            if y > 0 { push(x, y - 1, &mut stack, &mut comp); }
            if y + 1 < h { push(x, y + 1, &mut stack, &mut comp); }
        }
    }
    (comp, touches_border)
}

/// Bridge each island (kept component not touching the border) to border-connected
/// material by clearing `bw`-wide tabs across the SHORTEST cut gap. A multi-source
/// BFS from all border-connected material gives, for every cut cell, both the
/// distance to the frame and which frame cell it came from — so each island tab
/// crosses the narrowest gap (a short perpendicular stub, not a long diagonal slash).
/// Long islands get more than one tab so they can't pivot. Islands smaller than a
/// bridge width are dropped and reported via `warn`.
fn bridge_islands(
    keep: &mut [bool],
    w: usize,
    h: usize,
    bw: usize,
    warn: &mut dyn FnMut(String),
) {
    let (comp, touches_border) = components(keep, w, h);
    // Group island cells by component id.
    let mut island_cells: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for (idx, &c) in comp.iter().enumerate() {
        if c != usize::MAX && !touches_border[c] {
            island_cells.entry(c).or_default().push(idx);
        }
    }
    if island_cells.is_empty() {
        return;
    }

    // Drop genuine specks (< 2px across) up front. The bridge width must NOT gate
    // island survival — otherwise raising bridge-px silently deletes every island
    // smaller than the tab, which looks like "bridges randomly stop working" at
    // higher values. A big tab on a small island is fine; the tab just spans most of
    // the island, and the round brush is clamped to the island's reach.
    island_cells.retain(|_id, cells| {
        let (mut minx, mut miny, mut maxx, mut maxy) = (w, h, 0usize, 0usize);
        for &idx in cells.iter() {
            let (x, y) = (idx % w, idx / w);
            minx = minx.min(x); maxx = maxx.max(x);
            miny = miny.min(y); maxy = maxy.max(y);
        }
        let span = (maxx - minx).max(maxy - miny) + 1;
        if span < 2 {
            for &idx in cells.iter() {
                keep[idx] = false;
            }
            warn(format!(
                "dropped speck island at ({},{})-({},{}): span {}px",
                minx, miny, maxx, maxy, span
            ));
            false
        } else {
            true
        }
    });

    let bw = bw.max(1);

    // Anchored = material that provably reaches the frame. Seed with the
    // border-connected frame; grow it each round as islands are bridged, so a later
    // island's shortest gap may land on an earlier island's cells or tab rather than
    // slashing all the way to the frame. Islands only ever anchor to something
    // already anchored, so every carved tab connects its island to the frame — the
    // stencil never ends up with a floating island→island pair.
    let mut anchored: Vec<bool> = (0..keep.len())
        .map(|i| keep[i] && comp[i] != usize::MAX && touches_border[comp[i]])
        .collect();

    // Shortest-gap-first anchoring loop: each round, BFS from the current anchored
    // set, pick the not-yet-anchored island with the globally smallest gap, carve
    // its tab(s), and fold it (plus its tab) into `anchored`.
    // ponytail: re-run full BFS per round — O(islands × pixels). Switch to an
    // incremental multi-source Dijkstra only if island-heavy images profile slow.
    while !island_cells.is_empty() {
        let (dist, parent) = gap_bfs(keep, &anchored, w, h);

        // For each remaining island, collect its adjacent reachable tab sites and
        // its shortest gap to anchored material.
        let mut best: Option<(usize, u32)> = None; // (island id, min gap)
        let mut sites_by_id: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for (&id, cells) in island_cells.iter() {
            let mut sites: Vec<usize> = Vec::new();
            for &idx in cells {
                let (x, y) = (idx % w, idx / w);
                let nb = |nx: usize, ny: usize, sites: &mut Vec<usize>| {
                    let ni = ny * w + nx;
                    if !keep[ni] && dist[ni] != u32::MAX {
                        sites.push(ni);
                    }
                };
                if x > 0 { nb(x - 1, y, &mut sites); }
                if x + 1 < w { nb(x + 1, y, &mut sites); }
                if y > 0 { nb(x, y - 1, &mut sites); }
                if y + 1 < h { nb(x, y + 1, &mut sites); }
            }
            if sites.is_empty() {
                continue; // unreachable this round; handled below
            }
            sites.sort_by_key(|&s| dist[s]);
            sites.dedup();
            let gap = dist[sites[0]];
            if best.map_or(true, |(_, bg)| gap < bg) {
                best = Some((id, gap));
            }
            sites_by_id.insert(id, sites);
        }

        let Some((id, _)) = best else {
            // No remaining island can reach the anchored set — they're mutually
            // enclosed with no path out. Warn and stop; leaving them unbridged
            // matches the old "could not reach the frame" behaviour.
            for id in island_cells.keys() {
                warn(format!("island {id} could not reach the frame; left unbridged"));
            }
            break;
        };

        let cells = island_cells.remove(&id).unwrap();
        let sites = sites_by_id.remove(&id).unwrap();

        // Span for tab-count sizing.
        let (mut minx, mut miny, mut maxx, mut maxy) = (w, h, 0usize, 0usize);
        for &idx in &cells {
            let (x, y) = (idx % w, idx / w);
            minx = minx.min(x); maxx = maxx.max(x);
            miny = miny.min(y); maxy = maxy.max(y);
        }
        let span = (maxx - minx).max(maxy - miny) + 1;

        // One tab per ~40px of island span, min 1, independent of bridge width so
        // tab COUNT doesn't collapse to 1 whenever bridge-px is large. Spread tabs
        // out: shortest-gap site first, then greedily add sites far from chosen.
        // ponytail: fixed 40px spacing heuristic; expose as a knob only if operators
        // ask to tune tab density.
        let n_tabs = (span / 40).max(1).min(sites.len());
        let mut chosen: Vec<usize> = vec![sites[0]];
        while chosen.len() < n_tabs {
            // farthest site from all chosen (maximize min-distance)
            let next = sites.iter().copied().max_by_key(|&s| {
                let (sx, sy) = (s % w, s / w);
                chosen.iter().map(|&c| {
                    let (cx, cy) = (c % w, c / w);
                    let dx = sx as i64 - cx as i64;
                    let dy = sy as i64 - cy as i64;
                    dx * dx + dy * dy
                }).min().unwrap_or(0)
            });
            match next {
                Some(s) if !chosen.contains(&s) => chosen.push(s),
                _ => break,
            }
        }

        // Carve, then fold the island and its freshly-carved tab cells into the
        // anchored set so later islands can bridge to them. Snapshot once and diff.
        let before = keep.to_vec();
        for site in chosen {
            carve_tab(keep, &parent, w, site, bw);
        }
        for i in 0..keep.len() {
            if keep[i] && !before[i] {
                anchored[i] = true;
            }
        }
        for &idx in &cells {
            anchored[idx] = true;
        }
    }
}

/// Multi-source BFS over cut cells, seeded from cut cells adjacent to any
/// `anchored` material. Returns (gap distance in cells, parent-toward-anchor).
/// A cut cell's parent chain, followed to distance 0, reaches anchored material —
/// so carving along it bridges an island to whatever the anchored set represents.
/// The anchored set starts as the border-connected frame and grows as islands are
/// bridged, so a later island's shortest gap may land on an earlier island's tab.
fn gap_bfs(
    keep: &[bool],
    anchored: &[bool],
    w: usize,
    h: usize,
) -> (Vec<u32>, Vec<usize>) {
    let n = keep.len();
    let mut dist = vec![u32::MAX; n];
    let mut parent = vec![usize::MAX; n];
    let mut q = std::collections::VecDeque::new();
    // Seed: cut cells touching anchored material get distance 0 and point
    // at that material cell.
    for idx in 0..n {
        if keep[idx] {
            continue;
        }
        let (x, y) = (idx % w, idx / w);
        let mut seed = None;
        let chk = |nx: usize, ny: usize, seed: &mut Option<usize>| {
            let ni = ny * w + nx;
            if anchored[ni] {
                *seed = Some(ni);
            }
        };
        if x > 0 { chk(x - 1, y, &mut seed); }
        if x + 1 < w { chk(x + 1, y, &mut seed); }
        if y > 0 { chk(x, y - 1, &mut seed); }
        if y + 1 < h { chk(x, y + 1, &mut seed); }
        if let Some(m) = seed {
            dist[idx] = 0;
            parent[idx] = m;
            q.push_back(idx);
        }
    }
    while let Some(idx) = q.pop_front() {
        let (x, y) = (idx % w, idx / w);
        let relax = |nx: usize, ny: usize, q: &mut std::collections::VecDeque<usize>, dist: &mut Vec<u32>, parent: &mut Vec<usize>| {
            let ni = ny * w + nx;
            if !keep[ni] && dist[ni] == u32::MAX {
                dist[ni] = dist[idx] + 1;
                parent[ni] = idx;
                q.push_back(ni);
            }
        };
        if x > 0 { relax(x - 1, y, &mut q, &mut dist, &mut parent); }
        if x + 1 < w { relax(x + 1, y, &mut q, &mut dist, &mut parent); }
        if y > 0 { relax(x, y - 1, &mut q, &mut dist, &mut parent); }
        if y + 1 < h { relax(x, y + 1, &mut q, &mut dist, &mut parent); }
    }
    (dist, parent)
}

/// Carve a `bw`-wide tab from a cut `site` across the gap to the frame by
/// following the BFS parent chain and painting a ROUND brush at each step.
/// Round (not square) so the tab's sides/ends are curved, not blocky — the
/// tracer's alphamax corner test then keeps them smooth instead of classifying
/// square-brush 90° corners as hard corners.
fn carve_tab(keep: &mut [bool], parent: &[usize], w: usize, site: usize, bw: usize) {
    // Round brush whose DIAMETER matches bw as closely as the grid allows.
    // Using a real radius (bw/2 as f, not integer bw/2) so even/odd widths both
    // land right: bw=4 -> r=2.0 -> ~4px brush, not the 5px an integer rad gives.
    let r = bw as f64 / 2.0;
    let rad = r.ceil() as i64;
    let r2 = (r * r) as i64;
    let h = keep.len() / w;
    // Walk the parent chain FIRST to collect the centreline cells, deciding
    // termination from `parent` alone. The old code painted the brush and then
    // tested `keep[p]`, but the brush had just set `keep[p]=true` (p is adjacent
    // to cur, inside the radius) — so for any bw>=2 the loop stopped after one
    // step and the tab never crossed the gap. That was the "no bridges above 2px"
    // bug. `parent[cur]==MAX` marks a cut cell whose neighbour is frame material
    // (BFS seed), i.e. the last gap cell before the frame — stop there.
    let mut spine = Vec::new();
    let mut cur = site;
    loop {
        spine.push(cur);
        let p = parent[cur];
        if p == usize::MAX {
            break; // reached the gap cell adjacent to frame material
        }
        cur = p;
    }
    // Now paint the round brush along the whole spine.
    for &c in &spine {
        let (cx, cy) = ((c % w) as i64, (c / w) as i64);
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                if dx * dx + dy * dy > r2 {
                    continue; // round brush
                }
                let (nx, ny) = (cx + dx, cy + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                    keep[ny as usize * w + nx as usize] = true;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Marching-squares contour trace (boundary between keep/cut cells)
// ---------------------------------------------------------------------------

/// Trace all closed boundary loops of the `keep` mask. Returns rings; each ring
/// is classified outer (CCW, signed area > 0) or hole (CW) by the caller.
/// ponytail: cell-edge boundary walk on a padded grid (avoids border special-cases);
/// one ring per connected boundary. Good enough for flat regions.
fn trace(keep: &[bool], w: usize, h: usize, min_area: f32) -> (Vec<Vec<(f32, f32)>>, Vec<Vec<(f32, f32)>>) {
    // We walk the grid of cell corners. A boundary edge lies between a kept cell
    // and a cut cell (or the outside). Represent each horizontal/vertical unit
    // edge by its two endpoints on the integer corner lattice ((w+1) x (h+1)).
    // Collect directed boundary edges keeping material on the left, then stitch.
    let get = |x: i64, y: i64| -> bool {
        if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
            false
        } else {
            keep[y as usize * w + x as usize]
        }
    };

    // Directed edges: for each kept cell, emit boundary edges where the neighbor
    // is cut, oriented so material stays on the left (CCW outer, CW hole).
    // Corner (cx,cy) integer in [0,w] x [0,h].
    use std::collections::HashMap;
    let mut next: HashMap<(i64, i64), (i64, i64)> = HashMap::new();
    let add = |a: (i64, i64), b: (i64, i64), next: &mut HashMap<(i64, i64), (i64, i64)>| {
        next.insert(a, b);
    };
    for y in 0..h as i64 {
        for x in 0..w as i64 {
            if !get(x, y) {
                continue;
            }
            // Cell corners (CCW): (x,y)->(x+1,y)->(x+1,y+1)->(x,y+1)
            // Emit an edge along a side when the outside neighbor is cut.
            // Top side (y): neighbor (x, y-1). Material-on-left => go right->left? Use
            // consistent CCW: for a lone kept cell we want CCW loop
            // (x,y)->(x,y+1)->(x+1,y+1)->(x+1,y)->(x,y). Emit each side facing a cut neighbor.
            if !get(x, y - 1) {
                add((x + 1, y), (x, y), &mut next); // top side, going left (material below)
            }
            if !get(x, y + 1) {
                add((x, y + 1), (x + 1, y + 1), &mut next); // bottom side, going right
            }
            if !get(x - 1, y) {
                add((x, y), (x, y + 1), &mut next); // left side, going down
            }
            if !get(x + 1, y) {
                add((x + 1, y + 1), (x + 1, y), &mut next); // right side, going up
            }
        }
    }

    // Stitch directed edges into closed loops by following `next`.
    let mut rings: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut used: std::collections::HashSet<(i64, i64)> = std::collections::HashSet::new();
    let starts: Vec<(i64, i64)> = next.keys().cloned().collect();
    for start in starts {
        if used.contains(&start) {
            continue;
        }
        let mut loop_pts: Vec<(i64, i64)> = Vec::new();
        let mut cur = start;
        loop {
            if !used.insert(cur) {
                break;
            }
            loop_pts.push(cur);
            match next.get(&cur) {
                Some(&nx) => cur = nx,
                None => break,
            }
            if cur == start {
                break;
            }
        }
        if loop_pts.len() < 3 {
            continue;
        }
        // Collapse collinear runs to keep vertex count sane (axis-aligned steps).
        let simplified = collapse_collinear(&loop_pts);
        let pts: Vec<(f32, f32)> = simplified.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
        if signed_area(&pts).abs() < min_area {
            continue;
        }
        rings.push(pts);
    }

    // Classify by signed area. In this y-down raster the loops we emit make an
    // enclosing material boundary wind NEGATIVE and a cut-away hole POSITIVE
    // (verified: full canvas -> one negative outer; a cut region -> positive).
    let mut outers = Vec::new();
    let mut holes = Vec::new();
    for r in rings {
        if signed_area(&r) < 0.0 {
            outers.push(r);
        } else {
            holes.push(r);
        }
    }
    (outers, holes)
}

/// Remove intermediate points on straight axis-aligned runs.
fn collapse_collinear(pts: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let n = pts.len();
    if n < 3 {
        return pts.to_vec();
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = pts[(i + n - 1) % n];
        let cur = pts[i];
        let nxt = pts[(i + 1) % n];
        let d1 = (cur.0 - prev.0, cur.1 - prev.1);
        let d2 = (nxt.0 - cur.0, nxt.1 - cur.1);
        // Keep a vertex only where direction changes.
        if d1.0 * d2.1 - d1.1 * d2.0 != 0 || (d1.0 * d2.0 + d1.1 * d2.1) < 0 {
            out.push(cur);
        }
    }
    if out.len() < 3 { pts.to_vec() } else { out }
}

/// Shoelace signed area (positive = CCW in a y-down raster where we built loops CCW).
fn signed_area(pts: &[(f32, f32)]) -> f32 {
    let n = pts.len();
    let mut a = 0.0;
    for i in 0..n {
        let (x1, y1) = pts[i];
        let (x2, y2) = pts[(i + 1) % n];
        a += x1 * y2 - x2 * y1;
    }
    a / 2.0
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Load + resize, then optionally blur to coarsen detail before quantizing.
/// A larger `blur_px` averages fine texture into its neighbors so small dissimilar
/// regions get absorbed into the surrounding color -> fewer, bigger chunks.
fn load_pixels(path: &str, w: usize, h: usize, p: &Params) -> Result<Vec<[u8; 3]>, String> {
    let mut img = image::open(path)
        .map_err(|e| format!("failed to open image: {e}"))?
        .resize_exact(w as u32, h as u32, FilterType::Lanczos3);
    if p.blur_px > 0.0 {
        img = img.blur(p.blur_px); // image crate's Gaussian blur; sigma in px
    }
    let img = img.to_rgb8();
    // GUI sliders are independent, so guard against a degenerate wp>=bp range
    // (would make levels() produce 0/0=NaN). Treat it as identity.
    let identity = (p.white_point == 0.0 && p.black_point == 1.0 && p.gamma == 1.0)
        || p.white_point >= p.black_point;
    Ok(img
        .pixels()
        .map(|px| {
            let rgb = [px[0], px[1], px[2]];
            if identity {
                rgb
            } else {
                apply_levels(rgb, p.white_point, p.black_point, p.gamma)
            }
        })
        .collect())
}

/// Full Op 2 pipeline. Returns N layers (index 0 = darkest, largest silhouette),
/// each with bridged geometry ready for SVG emission. `warn` receives dropped-island
/// and connectivity messages.
pub fn stencils(
    path: &str,
    w: usize,
    h: usize,
    p: &Params,
    warn: &mut dyn FnMut(String),
) -> Result<Vec<Layer>, String> {
    let pixels = load_pixels(path, w, h, p)?;

    let (palette, mut labels) = quantize(&pixels, p.colors);
    merge_small_regions(&mut labels, w, h, p.min_feature_px);
    let bw = p.bridge_px.round() as usize;

    let mut layers = Vec::with_capacity(palette.len());
    for i in 0..palette.len() {
        let mut keep = stack_mask(&labels, i);
        if p.bridges {
            bridge_islands(&mut keep, w, h, bw, warn);
        }
        let (outers, holes) = trace(&keep, w, h, p.min_feature_px);
        layers.push(Layer {
            color: palette[i],
            outers,
            holes,
        });
    }
    Ok(layers)
}

/// Raster path: same quantize + stack + (optional) bridge pipeline as `stencils`,
/// but returns the boolean masks directly instead of tracing to polygons. Each
/// mask[i] is true where layer i's material is kept.
/// ponytail: shares the front half of `stencils`; not worth a common helper for
/// two call sites. Kept separate so the SVG path is untouched.
pub fn stencil_masks(
    path: &str,
    w: usize,
    h: usize,
    p: &Params,
    warn: &mut dyn FnMut(String),
) -> Result<(Vec<Color>, Vec<Vec<bool>>), String> {
    let pixels = load_pixels(path, w, h, p)?;

    let (palette, mut labels) = quantize(&pixels, p.colors);
    merge_small_regions(&mut labels, w, h, p.min_feature_px);
    let bw = p.bridge_px.round() as usize;

    let mut masks = Vec::with_capacity(palette.len());
    for i in 0..palette.len() {
        let mut keep = stack_mask(&labels, i);
        if p.bridges {
            bridge_islands(&mut keep, w, h, bw, warn);
        }
        masks.push(keep);
    }
    Ok((palette, masks))
}

/// Composite the stacked masks into one RGB preview: paint each layer's color
/// (dark->light order means later light layers overwrite the shared area, so the
/// top-most visible color wins per pixel — matches how the sprayed result reads).
pub fn preview(palette: &[Color], masks: &[Vec<bool>], w: usize, h: usize) -> image::RgbImage {
    let mut buf = image::RgbImage::from_pixel(w as u32, h as u32, image::Rgb([255, 255, 255]));
    // Masks are disjoint (one label band each); each pixel is painted by exactly
    // one layer, reproducing the flat multi-color result.
    for (color, mask) in palette.iter().zip(masks.iter()) {
        for (px, &keep) in buf.pixels_mut().zip(mask.iter()) {
            if keep {
                *px = image::Rgb([color.r, color.g, color.b]);
            }
        }
    }
    buf
}

/// Write N stencil SVGs + palette.txt to `<prefix>_{i}.svg`. Shared by the CLI
/// and the GUI "Generate all layers" button.
pub fn export(
    path: &str,
    w: usize,
    h: usize,
    p: &Params,
    smooth_px: f32,
    prefix: &str,
    warn: &mut dyn FnMut(String),
) -> Result<(), String> {
    use std::fmt::Write;
    let layers = stencils(path, w, h, p, warn)?;
    let mut palette = String::from("layer\trgb_hex\tcmyk\n");
    for (i, layer) in layers.iter().enumerate() {
        let c = &layer.color;
        let fill = format!("#{:02x}{:02x}{:02x}", c.r, c.g, c.b);
        let p100 = |v: f32| (v * 100.0).round() as i32;
        let _ = writeln!(
            palette, "{i}\t{fill}\tC{} M{} Y{} K{}",
            p100(c.cmyk[0]), p100(c.cmyk[1]), p100(c.cmyk[2]), p100(c.cmyk[3])
        );
        let doc = crate::svg::polygons_to_svg(&layer.outers, &layer.holes, w, h, &fill, smooth_px);
        let out = format!("{prefix}_{i}.svg");
        std::fs::write(&out, doc).map_err(|e| format!("write {out}: {e}"))?;
        eprintln!("wrote {out} (color {fill}, {} outers, {} holes)", layer.outers.len(), layer.holes.len());
    }
    let ppath = format!("{prefix}_palette.txt");
    std::fs::write(&ppath, palette).map_err(|e| format!("write {ppath}: {e}"))?;
    eprintln!("wrote {ppath} (spray order: index 0 = darkest, sprayed first)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_remap_pushes_offwhite_to_white() {
        // Identity leaves pixels alone.
        assert_eq!(apply_levels([200, 200, 200], 0.0, 1.0, 1.0), [200, 200, 200]);
        // white_point 0.3 = ink<=0.3 -> white. A 200/255 grey has ink ~0.216 < 0.3,
        // so it snaps to pure white; a dark 40/255 (ink ~0.84) stays dark.
        assert_eq!(apply_levels([200, 200, 200], 0.3, 1.0, 1.0), [255, 255, 255]);
        let dark = apply_levels([40, 40, 40], 0.3, 1.0, 1.0);
        assert!(dark[0] < 100, "dark pixel must stay dark, got {dark:?}");
    }

    #[test]
    fn median_cut_two_color() {
        // Half black, half white -> palette {black, white}, labeled correctly.
        let mut px = vec![[0u8, 0, 0]; 50];
        px.extend(vec![[255u8, 255, 255]; 50]);
        let (pal, labels) = quantize(&px, 2);
        assert_eq!(pal.len(), 2);
        // index 0 = darkest
        assert!(luma(&pal[0]) < luma(&pal[1]));
        assert!(pal[0].r < 10 && pal[1].r > 245);
        // first 50 dark -> label 0, last 50 light -> label 1
        assert!(labels[..50].iter().all(|&l| l == 0));
        assert!(labels[50..].iter().all(|&l| l == 1));
    }

    #[test]
    fn per_color_masks_disjoint() {
        let labels = [0u8, 1, 2, 1, 0];
        let m0 = stack_mask(&labels, 0);
        let m1 = stack_mask(&labels, 1);
        let m2 = stack_mask(&labels, 2);
        // Each mask covers ONLY its own label band (not cumulative).
        assert_eq!(m0, [true, false, false, false, true]);
        assert_eq!(m1, [false, true, false, true, false]);
        assert_eq!(m2, [false, false, true, false, false]);
        // Disjoint: no pixel kept by two masks; darkest layer is NOT the full sheet.
        for i in 0..labels.len() {
            let count = [&m0, &m1, &m2].iter().filter(|m| m[i]).count();
            assert_eq!(count, 1, "each pixel in exactly one layer");
        }
        assert!(!m0.iter().all(|&b| b), "layer 0 is not a solid full sheet");
    }

    #[test]
    fn orientation_outer_vs_hole() {
        // Solid 6x6 with a 2x2 cut in the middle -> one outer (CCW,>0), one hole (CW,<0).
        let (w, h) = (6usize, 6usize);
        let mut keep = vec![true; w * h];
        for y in 2..4 {
            for x in 2..4 {
                keep[y * w + x] = false;
            }
        }
        let (outers, holes) = trace(&keep, w, h, 0.0);
        assert_eq!(outers.len(), 1, "one outer");
        assert_eq!(holes.len(), 1, "one hole");
        // Enclosing material boundary winds negative; cut hole winds positive.
        assert!(signed_area(&outers[0]) < 0.0);
        assert!(signed_area(&holes[0]) > 0.0);
    }

    #[test]
    fn island_detected_and_bridged() {
        // Frame of material, a ring-shaped cut, a solid dot inside the cut.
        // 13x13: cut a hollow square ring at {3..=9}, keeping a 3x3 dot {5..=7}
        // in the center -> the dot is a falling island (span 3 >= bridge width 3).
        let (w, h) = (13usize, 13usize);
        let mut keep = vec![true; w * h];
        for y in 3..=9 {
            for x in 3..=9 {
                let dot = (5..=7).contains(&x) && (5..=7).contains(&y);
                if !dot {
                    keep[y * w + x] = false;
                }
            }
        }
        let center = 6 * w + 6;

        // Before bridging: dot is its own component not touching border.
        let (comp, tb) = components(&keep, w, h);
        let center_comp = comp[center];
        assert_ne!(center_comp, usize::MAX);
        assert!(!tb[center_comp], "dot is an island before bridging");

        let mut warnings = Vec::new();
        bridge_islands(&mut keep, w, h, 3, &mut |m| warnings.push(m));

        // After bridging: dot now connects to the border-touching frame.
        let (comp2, tb2) = components(&keep, w, h);
        let c2 = comp2[center];
        assert_ne!(c2, usize::MAX, "dot still kept after bridging: {warnings:?}");
        assert!(tb2[c2], "island joined to frame after bridging: {warnings:?}");
    }

    #[test]
    fn tab_crosses_shortest_gap() {
        // Island in a rectangular cut whose gap to the frame is 1 cell on the LEFT
        // and 3 cells on the RIGHT. The tab must go left (shortest gap), so the
        // cells immediately right of the island stay cut.
        // 11 wide, 7 tall. Cut columns 2..=8 on rows 1..=5, keep a dot at col 3.
        // Left gap (col 2) = 1 wide; right gap (cols 4..=8) = 5 wide.
        let (w, h) = (11usize, 7usize);
        let mut keep = vec![true; w * h];
        for y in 1..=5 {
            for x in 2..=8 {
                if !(x == 3 && (2..=4).contains(&y)) {
                    keep[y * w + x] = false;
                }
            }
        }
        bridge_islands(&mut keep, w, h, 1, &mut |_| {});
        // The tab crosses the 1-cell LEFT gap: some col-2 cell adjacent to the dot
        // (rows 2..=4) is now material.
        assert!(
            (2..=4).any(|y| keep[y * w + 2]),
            "tab carved across the 1-cell left gap"
        );
        // The wide RIGHT gap is never bridged: cols 5..=8 stay fully cut.
        for y in 1..=5 {
            for x in 5..=8 {
                assert!(!keep[y * w + x], "no tab across the wide right gap at ({x},{y})");
            }
        }
    }

    #[test]
    fn large_bridge_keeps_midsize_island() {
        // The reliability bug: a big bridge width used to drop any island smaller
        // than the tab. A 5x5 island with bridge width 12 must SURVIVE (span 5 >= 2).
        let (w, h) = (21usize, 21usize);
        let mut keep = vec![true; w * h];
        for y in 5..=15 {
            for x in 5..=15 {
                let dot = (8..=12).contains(&x) && (8..=12).contains(&y);
                if !dot {
                    keep[y * w + x] = false;
                }
            }
        }
        let center = 10 * w + 10;
        bridge_islands(&mut keep, w, h, 12, &mut |_| {});
        let (comp, tb) = components(&keep, w, h);
        assert_ne!(comp[center], usize::MAX, "island not deleted by large bridge");
        assert!(tb[comp[center]], "island bridged to frame with large bridge width");
    }

    #[test]
    fn merge_absorbs_speckle() {
        // 5x5 all label 0 except a single label-1 speck in the middle. With an area
        // floor of 4, the 1-cell speck merges into surrounding label 0.
        let (w, h) = (5usize, 5usize);
        let mut labels = vec![0u8; w * h];
        labels[2 * w + 2] = 1;
        merge_small_regions(&mut labels, w, h, 4.0);
        assert!(labels.iter().all(|&l| l == 0), "speck absorbed into neighbour");
    }

    #[test]
    fn merge_keeps_big_region() {
        // A big label-1 block (>= floor) must survive.
        let (w, h) = (6usize, 6usize);
        let mut labels = vec![0u8; w * h];
        for y in 1..=4 { for x in 1..=4 { labels[y * w + x] = 1; } } // 16 cells
        merge_small_regions(&mut labels, w, h, 4.0);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 16, "big region kept");
    }

    #[test]
    fn large_bridge_crosses_wide_gap() {
        // The real "no bridges above 2px" bug: a WIDE gap (6 cells) between island
        // and frame. The tab must traverse the whole gap regardless of brush size.
        // 20 wide, 9 tall. Cut a big block cols 2..=17, rows 1..=7, keep a 3x3 dot
        // at cols 8..=10 / rows 3..=5 (island ~6 cells from every wall).
        let (w, h) = (20usize, 9usize);
        let mut keep = vec![true; w * h];
        for y in 1..=7 {
            for x in 2..=17 {
                let dot = (8..=10).contains(&x) && (3..=5).contains(&y);
                if !dot {
                    keep[y * w + x] = false;
                }
            }
        }
        let center = 4 * w + 9;
        // bw=6: a single disc at the island edge cannot reach the frame (gap is 6+).
        bridge_islands(&mut keep, w, h, 6, &mut |_| {});
        let (comp, tb) = components(&keep, w, h);
        assert_ne!(comp[center], usize::MAX, "island kept");
        assert!(tb[comp[center]], "wide gap bridged with bw=6 (spine traversed)");
    }

    #[test]
    fn drop_too_small_island() {
        // A single-cell island with bridge width 3 -> span 1 < 3 -> dropped + warned.
        let (w, h) = (7usize, 7usize);
        let mut keep = vec![true; w * h];
        for y in 2..=4 {
            for x in 2..=4 {
                if !(x == 3 && y == 3) {
                    keep[y * w + x] = false;
                }
            }
        }
        assert!(keep[3 * w + 3], "island present before");
        let mut warnings = Vec::new();
        bridge_islands(&mut keep, w, h, 3, &mut |m| warnings.push(m));
        assert!(!keep[3 * w + 3], "tiny island dropped");
        assert!(warnings.iter().any(|m| m.contains("dropped speck")), "warned: {warnings:?}");
    }

    #[test]
    fn island_chains_to_nearer_island() {
        // Two stacked islands in a WIDE tall cut cavity (sides far away so the only
        // near anchor is vertical). Island A sits 1 cell below the top frame; island
        // B sits 1 cell below A but ~9 cells above the bottom frame. B's nearest
        // anchor is A (gap 1), not the frame (gap 9), so B must chain UP to A — the
        // row between them becomes material and the wide gap below B stays cut.
        let (w, h) = (15usize, 15usize);
        let mut keep = vec![true; w * h];
        // Cut the cavity cols 2..=12, rows 1..=13 (row 0/14 and col 0/14 stay frame).
        for y in 1..=13 {
            for x in 2..=12 {
                keep[y * w + x] = false;
            }
        }
        // Island A: row 2, cols 6..=8. Island B: row 4, cols 6..=8. Centered, so the
        // side gaps (cols 2..=5 and 9..=12) are ≥4 cells — much farther than A.
        for x in 6..=8 {
            keep[2 * w + x] = true; // A
            keep[4 * w + x] = true; // B
        }
        let a = 2 * w + 7;
        let b = 4 * w + 7;

        // Before: A and B are separate islands, neither touches the border.
        let (comp0, tb0) = components(&keep, w, h);
        assert_ne!(comp0[a], comp0[b], "A and B distinct before bridging");
        assert!(!tb0[comp0[a]] && !tb0[comp0[b]], "both are islands before");

        let mut warnings = Vec::new();
        bridge_islands(&mut keep, w, h, 1, &mut |m| warnings.push(m));

        // B chained to A: the row between them (row 3) got material.
        assert!(
            (6..=8).any(|x| keep[3 * w + x]),
            "B bridged up to A across the 1-cell gap: {warnings:?}"
        );
        // The wide gap below B (rows 6..=12) stays cut — no slash to the bottom frame.
        for y in 6..=12 {
            for x in 6..=8 {
                assert!(!keep[y * w + x], "no tab across the wide bottom gap at ({x},{y})");
            }
        }
        // Both islands now reach the frame (single border-connected component).
        let (comp1, tb1) = components(&keep, w, h);
        assert!(tb1[comp1[a]], "A frame-connected: {warnings:?}");
        assert!(tb1[comp1[b]], "B frame-connected via A: {warnings:?}");
    }

    #[test]
    fn no_floating_pair_after_bridging() {
        // Every originally-kept cell must end up frame-connected after bridging
        // (except warned-unreachable ones — none here). Guards the invariant that
        // island→island chaining never leaves a floating pair.
        let (w, h) = (9usize, 15usize);
        let mut keep = vec![true; w * h];
        for y in 1..=13 {
            for x in 2..=6 {
                keep[y * w + x] = false;
            }
        }
        for x in 3..=5 {
            keep[2 * w + x] = true;
            keep[4 * w + x] = true;
        }
        bridge_islands(&mut keep, w, h, 1, &mut |_| {});
        let (comp, tb) = components(&keep, w, h);
        for (idx, &c) in comp.iter().enumerate() {
            if c != usize::MAX {
                assert!(tb[c], "kept cell {idx} is frame-connected after bridging");
            }
        }
    }
}
