//! Traced rings -> smoothed cubic-Bezier SVG document.

use std::fmt::Write;

/// ISO A paper sizes in mm (portrait W x H).
#[derive(Clone, Copy, PartialEq)]
pub enum Paper {
    A2,
    A3,
    A4,
    A5,
}

impl Paper {
    /// Portrait dimensions (w_mm, h_mm).
    pub fn dims(self) -> (f32, f32) {
        match self {
            Paper::A2 => (420.0, 594.0),
            Paper::A3 => (297.0, 420.0),
            Paper::A4 => (210.0, 297.0),
            Paper::A5 => (148.0, 210.0),
        }
    }

    pub fn parse(s: &str) -> Option<Paper> {
        match s.to_ascii_uppercase().as_str() {
            "A2" => Some(Paper::A2),
            "A3" => Some(Paper::A3),
            "A4" => Some(Paper::A4),
            "A5" => Some(Paper::A5),
            _ => None,
        }
    }
}

/// How the px artwork is placed on a physical sheet: paper size, a margin (space on
/// all sides), and the resulting fit-scale + centering. Orientation follows the
/// image (landscape image -> landscape paper) so the artwork fills as much of the
/// inner box as possible while keeping its aspect ratio and leaving the margin.
#[derive(Clone, Copy)]
pub struct PaperLayout {
    pub paper_w: f32, // mm
    pub paper_h: f32, // mm
    pub scale: f32,   // px -> mm
    pub off_x: f32,   // mm, left offset of the artwork
    pub off_y: f32,   // mm, top offset
    pub art_w: f32,   // mm, scaled artwork width
    pub art_h: f32,   // mm, scaled artwork height
}

impl PaperLayout {
    /// Fit a `w_px` x `h_px` image onto `paper` with `margin_mm` of blank space on
    /// every side, centered. `margin_mm` also controls feature #3 (squeeze): a bigger
    /// margin shrinks the artwork away from the edges.
    pub fn fit(paper: Paper, margin_mm: f32, w_px: usize, h_px: usize) -> PaperLayout {
        let (pw, ph) = paper.dims();
        // Match paper orientation to the image so we don't waste the inner box.
        let (paper_w, paper_h) = if (w_px > h_px) == (pw > ph) { (pw, ph) } else { (ph, pw) };
        let m = margin_mm.max(0.0).min(paper_w.min(paper_h) / 2.0 - 1.0);
        let inner_w = (paper_w - 2.0 * m).max(1.0);
        let inner_h = (paper_h - 2.0 * m).max(1.0);
        let scale = (inner_w / w_px as f32).min(inner_h / h_px as f32);
        let art_w = w_px as f32 * scale;
        let art_h = h_px as f32 * scale;
        PaperLayout {
            paper_w,
            paper_h,
            scale,
            off_x: (paper_w - art_w) / 2.0,
            off_y: (paper_h - art_h) / 2.0,
            art_w,
            art_h,
        }
    }
}

/// Assemble an SVG document for one stencil layer: outer rings as filled paths,
/// holes as counter-wound subpaths in the same `<path>` (even-odd fill so cuts
/// render as holes). Post-bridge geometry already has notches, so bridged holes
/// trace correctly. Physical mm on the root; px coords in the viewBox.
///
/// Each staircase ring is Potrace-style smoothed (DP simplify + Catmull-Rom) and
/// emitted as cubic Beziers, so edges are smooth curves like Inkscape's Trace
/// Bitmap rather than blocky pixel steps. `smooth_eps` is the DP tolerance in px.
pub fn polygons_to_svg(
    outers: &[Vec<(f32, f32)>],
    holes: &[Vec<(f32, f32)>],
    w_px: usize,
    h_px: usize,
    fill: &str,
    smooth_eps: f32,
) -> String {
    let mut s = String::new();
    let _ = write!(
        s,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w_px}\" height=\"{h_px}\" viewBox=\"0 0 {w_px} {h_px}\">\n"
    );
    let emit_ring = |s: &mut String, ring: &[(f32, f32)]| {
        let Some(c) = crate::smooth::smooth_ring(ring, smooth_eps) else {
            return;
        };
        let _ = write!(s, "M{:.3} {:.3} ", c.start.0, c.start.1);
        for seg in &c.segs {
            let _ = write!(
                s,
                "C{:.3} {:.3} {:.3} {:.3} {:.3} {:.3} ",
                seg.c1.0, seg.c1.1, seg.c2.0, seg.c2.1, seg.p.0, seg.p.1
            );
        }
        s.push_str("Z ");
    };
    // One combined path: all outers + all holes, even-odd fill turns holes into cuts.
    let _ = write!(s, "<path fill-rule=\"evenodd\" fill=\"{fill}\" d=\"");
    for ring in outers {
        emit_ring(&mut s, ring);
    }
    for ring in holes {
        emit_ring(&mut s, ring);
    }
    s.push_str("\"/>\n");
    s.push_str("</svg>\n");
    s
}

/// Registration crosshairs at three canvas corners (PDF "Secret Sauce #4"): every
/// channel SVG embeds the SAME marks at the SAME coords, so all four sheets pin to
/// one physical grid. Three corners (not four) fixes both translation and rotation
/// while staying asymmetric enough to catch a flipped sheet. `mark` is the arm
/// length in px. Emitted as thin red stroked lines (cut/score, ignored by fill).
fn registration_marks(w: usize, h: usize, mark: f32) -> String {
    let mut s = String::new();
    let m = mark.max(1.0);
    let corners = [(m, m), (w as f32 - m, m), (m, h as f32 - m)];
    let _ = write!(s, "<g stroke=\"red\" stroke-width=\"0.2\" fill=\"none\">");
    for (cxp, cyp) in corners {
        // A plus sign: horizontal + vertical arm through the corner point.
        let _ = write!(
            s,
            "<path d=\"M{:.3} {:.3} L{:.3} {:.3} M{:.3} {:.3} L{:.3} {:.3}\"/>",
            cxp - m, cyp, cxp + m, cyp, cxp, cyp - m, cxp, cyp + m
        );
    }
    s.push_str("</g>\n");
    s
}

/// Alignment marks in PAPER (mm) space: a crosshair AND a punch hole (circle) at all
/// four sheet corners, `inset` mm in from the edge. Identical on every layer so the
/// four sheets slide onto physical alignment pins through the holes and line up. A
/// `label` (e.g. "CYAN 1/4") is printed in the top margin to identify the sheet.
/// `hole_r` is the punch-hole radius (mm).
fn paper_marks(pw: f32, ph: f32, inset: f32, hole_r: f32, label: &str) -> String {
    let mut s = String::new();
    let corners = [
        (inset, inset),
        (pw - inset, inset),
        (pw - inset, ph - inset),
        (inset, ph - inset),
    ];
    let arm = hole_r * 2.0;
    let _ = write!(s, "<g stroke=\"red\" stroke-width=\"0.2\" fill=\"none\">");
    for (cx, cy) in corners {
        // Crosshair + punch-hole circle at each corner.
        let _ = write!(
            s,
            "<path d=\"M{:.3} {:.3} L{:.3} {:.3} M{:.3} {:.3} L{:.3} {:.3}\"/>",
            cx - arm, cy, cx + arm, cy, cx, cy - arm, cx, cy + arm
        );
        let _ = write!(s, "<circle cx=\"{cx:.3}\" cy=\"{cy:.3}\" r=\"{hole_r:.3}\"/>");
    }
    s.push_str("</g>\n");
    // Sheet label in the top margin, centered.
    let _ = write!(
        s,
        "<text x=\"{:.3}\" y=\"{:.3}\" font-size=\"{:.2}\" fill=\"red\" text-anchor=\"middle\">{}</text>\n",
        pw / 2.0, (inset).max(3.0), (inset * 0.6).clamp(2.0, 6.0), label
    );
    s
}

/// Emit analytic cut ribbons (from `lines.rs`) as one SVG document for a channel.
/// Each ribbon is a filled closed polygon covering a whole run of same-line
/// segments — no smoothing, no tracing: one continuous linear laser pass per run.
///
/// With a `PaperLayout`, the SVG is emitted at true physical size (mm viewBox): the
/// artwork is scaled + centered inside the paper's margin box (leaving blank space
/// on all sides), and corner punch holes + crosshairs + a sheet `label` are drawn
/// in paper space so all four layers pin together. Without a layout, falls back to a
/// raw px viewBox with the old corner crosshairs.
pub fn ribbons_to_svg(
    ribbons: &[crate::lines::Ribbon],
    w_px: usize,
    h_px: usize,
    fill: &str,
    layout: Option<PaperLayout>,
    label: &str,
    hole_r_mm: f32,
) -> String {
    let mut s = String::new();
    let emit_ribbons = |s: &mut String| {
        let _ = write!(s, "<path fill=\"{fill}\" d=\"");
        for r in ribbons {
            let mut it = r.pts.iter();
            if let Some(p0) = it.next() {
                let _ = write!(s, "M{:.3} {:.3} ", p0.0, p0.1);
                for p in it {
                    let _ = write!(s, "L{:.3} {:.3} ", p.0, p.1);
                }
                s.push_str("Z ");
            }
        }
        s.push_str("\"/>\n");
    };

    match layout {
        Some(l) => {
            let _ = write!(
                s,
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{:.2}mm\" height=\"{:.2}mm\" viewBox=\"0 0 {:.2} {:.2}\">\n",
                l.paper_w, l.paper_h, l.paper_w, l.paper_h
            );
            // Place the px artwork into the margin-inset box: translate to the
            // centered offset, then scale px->mm. Ribbon coords stay untouched.
            let _ = write!(
                s,
                "<g transform=\"translate({:.3} {:.3}) scale({:.5})\">\n",
                l.off_x, l.off_y, l.scale
            );
            emit_ribbons(&mut s);
            s.push_str("</g>\n");
            // Marks live in paper space; inset by the hole radius plus a little so
            // the whole punched circle sits in the blank border, off the artwork.
            let hole_r = hole_r_mm.max(0.5);
            let inset = (hole_r + (l.paper_w.min(l.paper_h) - l.art_w.min(l.art_h)) / 4.0)
                .clamp(hole_r + 2.0, 12.0);
            s.push_str(&paper_marks(l.paper_w, l.paper_h, inset, hole_r, label));
        }
        None => {
            let _ = write!(
                s,
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w_px}\" height=\"{h_px}\" viewBox=\"0 0 {w_px} {h_px}\">\n"
            );
            emit_ribbons(&mut s);
            s.push_str(&registration_marks(w_px, h_px, 5.0));
        }
    }
    s.push_str("</svg>\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registration_marks_present_on_three_corners() {
        // Three crosshairs => three <path> children inside a red <g>.
        let g = registration_marks(100, 80, 5.0);
        assert!(g.contains("stroke=\"red\""));
        assert_eq!(g.matches("<path").count(), 3, "one crosshair per corner");
    }

    #[test]
    fn ribbons_svg_px_fallback_has_marks_and_polys() {
        let r = vec![crate::lines::Ribbon { pts: vec![(0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)] }];
        let doc = ribbons_to_svg(&r, 50, 40, "#00ffff", None, "", 2.0);
        assert!(doc.starts_with("<svg"));
        assert!(doc.contains("fill=\"#00ffff\""));
        assert!(doc.contains("M0.000 0.000"), "ribbon path emitted");
        assert!(doc.contains("stroke=\"red\""), "registration marks embedded");
        assert!(doc.trim_end().ends_with("</svg>"));
    }

    #[test]
    fn paper_layout_fits_with_margin_and_centering() {
        // 100x50 landscape image on A4 with 10mm margin. Paper rotates to landscape
        // (297x210). Artwork fits inside 277x190 keeping aspect, centered.
        let l = PaperLayout::fit(Paper::A4, 10.0, 100, 50);
        assert!((l.paper_w - 297.0).abs() < 0.1 && (l.paper_h - 210.0).abs() < 0.1, "A4 landscape");
        // Scale limited by width: 277/100 = 2.77; height 50*2.77=138.5 < 190. OK.
        assert!((l.scale - 2.77).abs() < 0.01, "fit scale, got {}", l.scale);
        // Centered: equal margins left/right and top/bottom, both >= 10mm.
        assert!(l.off_x >= 10.0 && l.off_y >= 10.0, "margin on all sides: {},{}", l.off_x, l.off_y);
        assert!((l.off_x * 2.0 + l.art_w - l.paper_w).abs() < 0.1, "horizontally centered");
        assert!((l.off_y * 2.0 + l.art_h - l.paper_h).abs() < 0.1, "vertically centered");
    }

    #[test]
    fn ribbons_svg_paper_has_mm_holes_and_label() {
        let r = vec![crate::lines::Ribbon { pts: vec![(0.0, 0.0), (5.0, 0.0), (5.0, 2.0), (0.0, 2.0)] }];
        let l = PaperLayout::fit(Paper::A5, 8.0, 50, 40);
        let doc = ribbons_to_svg(&r, 50, 40, "cyan", Some(l), "CYAN 1/4", 3.0);
        assert!(doc.contains("mm\""), "physical mm size on root");
        assert!(doc.contains("<circle"), "punch holes present");
        assert_eq!(doc.matches("<circle").count(), 4, "one hole per corner");
        assert!(doc.contains("r=\"3.000\""), "hole radius honoured (min-material floor)");
        assert!(doc.contains("CYAN 1/4"), "sheet label present");
        assert!(doc.contains("transform=\"translate"), "artwork placed in margin box");
    }

    #[test]
    fn emits_valid_svg_with_paths() {
        // One square outer ring -> a <path> with an M and Z, wrapped in <svg>.
        let outer = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let doc = polygons_to_svg(&[outer], &[], 10, 10, "#000000", 1.0);
        assert!(doc.starts_with("<svg"));
        assert!(doc.contains("<path"));
        assert!(doc.contains(" M") || doc.contains("\"M"));
        assert!(doc.contains("Z"));
        assert!(doc.trim_end().ends_with("</svg>"));
    }
}
