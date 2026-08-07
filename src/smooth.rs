//! Turn a raw pixel-staircase ring into a smooth curve, the way Inkscape's Trace
//! Bitmap (Potrace) does. Two steps: Douglas-Peucker to drop the staircase noise
//! into a sparse polyline, then Catmull-Rom through those points expressed as
//! cubic Beziers so corners round off instead of stepping.
//!
//! ponytail: this is the Potrace-equivalent, not Potrace. DP + Catmull-Rom is a
//! few lines and gives the smooth-curve look that "looks like Inkscape". Swap in a
//! real potrace crate only if curve *fidelity* (adaptive corner detection,
//! straight-run preservation) ever matters more than "not blocky".

/// One cubic Bezier segment: end anchor `p` reached from the previous anchor via
/// control points `c1`,`c2`. A ring is a start anchor plus a Vec of these.
#[derive(Clone, Copy)]
pub struct Cubic {
    pub c1: (f32, f32),
    pub c2: (f32, f32),
    pub p: (f32, f32),
}

pub struct Curve {
    pub start: (f32, f32),
    pub segs: Vec<Cubic>,
}

/// Perpendicular distance from `p` to the infinite line through `a`,`b`.
fn perp_dist(p: (f32, f32), a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-9 {
        let (ex, ey) = (p.0 - a.0, p.1 - a.1);
        return (ex * ex + ey * ey).sqrt();
    }
    ((p.0 - a.0) * dy - (p.1 - a.1) * dx).abs() / len
}

/// Douglas-Peucker on an OPEN polyline.
fn dp(pts: &[(f32, f32)], eps: f32, out: &mut Vec<(f32, f32)>) {
    if pts.len() < 2 {
        return;
    }
    let (a, b) = (pts[0], pts[pts.len() - 1]);
    let mut idx = 0;
    let mut max = 0.0;
    for (i, &p) in pts.iter().enumerate().take(pts.len() - 1).skip(1) {
        let d = perp_dist(p, a, b);
        if d > max {
            max = d;
            idx = i;
        }
    }
    if max > eps {
        dp(&pts[..=idx], eps, out);
        out.pop(); // avoid duplicating the split point
        dp(&pts[idx..], eps, out);
    } else {
        out.push(a);
        out.push(b);
    }
}

/// Simplify a CLOSED ring with Douglas-Peucker. Splits the ring at its two most
/// distant anchors, runs DP on each open half, and stitches the halves back into
/// a ring (shared split points dropped so they aren't duplicated).
fn simplify_ring(ring: &[(f32, f32)], eps: f32) -> Vec<(f32, f32)> {
    let n = ring.len();
    if n < 4 {
        return ring.to_vec();
    }
    // Split at ring[0] and the point farthest (euclidean) from it, so DP keeps the
    // gross shape rather than collapsing across the seam.
    let far = (1..n)
        .max_by(|&i, &j| {
            let di = (ring[i].0 - ring[0].0).powi(2) + (ring[i].1 - ring[0].1).powi(2);
            let dj = (ring[j].0 - ring[0].0).powi(2) + (ring[j].1 - ring[0].1).powi(2);
            di.partial_cmp(&dj).unwrap()
        })
        .unwrap();

    // Half A: ring[0..=far]. Half B: ring[far..] + ring[0].
    let mut a = Vec::new();
    dp(&ring[..=far], eps, &mut a);
    let mut chain_b: Vec<(f32, f32)> = ring[far..].to_vec();
    chain_b.push(ring[0]);
    let mut b = Vec::new();
    dp(&chain_b, eps, &mut b);

    // a = [ring[0] .. ring[far]], b = [ring[far] .. ring[0]]. Concatenate a with
    // b's interior (drop b's first=ring[far] and last=ring[0], both already in a).
    let mut merged = a;
    let b_interior = b.len().saturating_sub(1);
    merged.extend(b.into_iter().take(b_interior).skip(1));
    if merged.len() < 3 {
        return ring.to_vec();
    }
    merged
}

fn unit(v: (f32, f32)) -> (f32, f32) {
    let l = (v.0 * v.0 + v.1 * v.1).sqrt();
    if l < 1e-9 { (0.0, 0.0) } else { (v.0 / l, v.1 / l) }
}

/// Turn angle at a vertex, radians: 0 = straight through, PI = full hairpin.
/// `prev`->`v`->`next`.
fn turn_angle(prev: (f32, f32), v: (f32, f32), next: (f32, f32)) -> f32 {
    let a = unit((v.0 - prev.0, v.1 - prev.1));
    let b = unit((next.0 - v.0, next.1 - v.1));
    (a.0 * b.0 + a.1 * b.1).clamp(-1.0, 1.0).acos()
}

/// Catmull-Rom through a CLOSED set of anchor points, converted to cubic Beziers.
///
/// Potrace-style corner classification: a vertex whose turn is sharper than
/// `alphamax` (radians) is a HARD CORNER — its tangent is zeroed so the curve
/// passes straight through it (two lines meeting at a point) instead of rounding.
/// Gentler vertices keep the Catmull-Rom tangent. This is the one behavior the
/// plain scaled-tangent fit was missing: sharp-where-sharp, smooth-where-smooth.
///
/// Overshoot is still tamed the two old ways: (1) tangent scaled by straightness,
/// (2) control points clamped onto their p1->p2 segment.
fn catmull_rom_closed(pts: &[(f32, f32)], alphamax: f32) -> Curve {
    let n = pts.len();
    let mut segs = Vec::with_capacity(n);
    let clamp = |c: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (c.0.clamp(a.0.min(b.0), a.0.max(b.0)), c.1.clamp(a.1.min(b.1), a.1.max(b.1)))
    };
    // corner[i] = true when vertex i turns sharper than alphamax -> hard corner.
    let corner: Vec<bool> = (0..n)
        .map(|i| turn_angle(pts[(i + n - 1) % n], pts[i], pts[(i + 1) % n]) > alphamax)
        .collect();
    for i in 0..n {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];
        // Straightness at each endpoint: 1 for a straight pass-through, ->0 for a
        // hairpin. Scales the tangent so corners don't fling control points out.
        let s1 = if corner[i] {
            0.0 // hard corner at p1: no outgoing tangent -> c1 sits on p1 (line out)
        } else {
            (unit((p1.0 - p0.0, p1.1 - p0.1)).0 * unit((p2.0 - p1.0, p2.1 - p1.1)).0
                + unit((p1.0 - p0.0, p1.1 - p0.1)).1 * unit((p2.0 - p1.0, p2.1 - p1.1)).1)
                .max(0.0)
        };
        let s2 = if corner[(i + 1) % n] {
            0.0 // hard corner at p2: no incoming tangent -> c2 sits on p2 (line in)
        } else {
            (unit((p2.0 - p1.0, p2.1 - p1.1)).0 * unit((p3.0 - p2.0, p3.1 - p2.1)).0
                + unit((p2.0 - p1.0, p2.1 - p1.1)).1 * unit((p3.0 - p2.0, p3.1 - p2.1)).1)
                .max(0.0)
        };
        let c1 = (p1.0 + (p2.0 - p0.0) / 6.0 * s1, p1.1 + (p2.1 - p0.1) / 6.0 * s1);
        let c2 = (p2.0 - (p3.0 - p1.0) / 6.0 * s2, p2.1 - (p3.1 - p1.1) / 6.0 * s2);
        segs.push(Cubic {
            c1: clamp(c1, p1, p2),
            c2: clamp(c2, p1, p2),
            p: p2,
        });
    }
    Curve { start: pts[0], segs }
}

/// Vertices sharper than this turn angle (radians) are kept as hard corners.
/// ~1.0 rad (~57°) matches Potrace's default alphamax feel. ponytail: fixed, not a
/// CLI flag — expose one if an operator wants to tune corner sharpness.
const ALPHAMAX: f32 = 1.0;

/// Full smooth: DP-simplify the staircase, then fit a closed Catmull-Rom spline
/// with hard-corner classification (alphamax).
/// `eps` is the DP tolerance in px (bigger = smoother/fewer nodes).
pub fn smooth_ring(ring: &[(f32, f32)], eps: f32) -> Option<Curve> {
    let simplified = simplify_ring(ring, eps);
    if simplified.len() < 3 {
        return None;
    }
    Some(catmull_rom_closed(&simplified, ALPHAMAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dp_drops_staircase() {
        // A closed ring whose top edge is a pixel staircase (rising then a flat
        // return) must simplify to far fewer nodes than the raw step count.
        let mut ring = Vec::new();
        for i in 0..10 {
            ring.push((i as f32, i as f32));       // up one
            ring.push((i as f32 + 1.0, i as f32)); // over one
        }
        // Close the loop back along the bottom so it's a real ring, not an open line.
        ring.push((10.0, 20.0));
        ring.push((0.0, 20.0));
        let simp = simplify_ring(&ring, 1.5);
        assert!(
            simp.len() < ring.len() / 2,
            "staircase simplified: {} of {}",
            simp.len(),
            ring.len()
        );
    }

    #[test]
    fn smooth_preserves_gross_shape() {
        // A blocky square ring smooths to a curve whose points stay near the box.
        let ring = vec![
            (0.0, 0.0), (5.0, 0.0), (10.0, 0.0),
            (10.0, 5.0), (10.0, 10.0),
            (5.0, 10.0), (0.0, 10.0),
            (0.0, 5.0),
        ];
        let c = smooth_ring(&ring, 0.5).unwrap();
        for s in &c.segs {
            assert!(s.p.0 >= -2.0 && s.p.0 <= 12.0, "x in range: {}", s.p.0);
            assert!(s.p.1 >= -2.0 && s.p.1 <= 12.0, "y in range: {}", s.p.1);
        }
        assert!(c.segs.len() >= 3);
    }

    #[test]
    fn sharp_corner_stays_sharp() {
        // A right-angle corner (turn ~90° > alphamax) must be emitted as a hard
        // corner: the tangents on both sides of that vertex collapse onto it, so
        // the curve passes straight through instead of rounding it off.
        let ring = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let c = catmull_rom_closed(&ring, ALPHAMAX);
        // Every vertex here is a 90° corner => every seg is a straight line:
        // c1 sits on its start anchor, c2 sits on its end anchor.
        for (i, s) in c.segs.iter().enumerate() {
            let start = if i == 0 { c.start } else { c.segs[i - 1].p };
            assert!(
                (s.c1.0 - start.0).abs() < 1e-3 && (s.c1.1 - start.1).abs() < 1e-3,
                "seg {i} c1 should sit on its start anchor: {:?} vs {:?}",
                s.c1,
                start
            );
            assert!(
                (s.c2.0 - s.p.0).abs() < 1e-3 && (s.c2.1 - s.p.1).abs() < 1e-3,
                "seg {i} c2 should sit on its end anchor: {:?} vs {:?}",
                s.c2,
                s.p
            );
        }
    }

    #[test]
    fn gentle_bend_stays_curved() {
        // A shallow zig (turn well under alphamax) must keep a real curve: at least
        // one control point lifts off its anchors.
        let ring = vec![(0.0, 0.0), (10.0, 1.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)];
        let c = catmull_rom_closed(&ring, ALPHAMAX);
        let lifted = c.segs.iter().enumerate().any(|(i, s)| {
            let start = if i == 0 { c.start } else { c.segs[i - 1].p };
            (s.c1.0 - start.0).abs() > 1e-2 || (s.c1.1 - start.1).abs() > 1e-2
        });
        assert!(lifted, "a gentle bend must produce at least one curved segment");
    }

    #[test]
    fn no_control_point_overshoot() {
        // A tight square: clamped control points must stay within the box, i.e. the
        // curve can't bulge outside the sheet the way plain Catmull-Rom does.
        let ring = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let c = smooth_ring(&ring, 0.5).unwrap();
        for s in &c.segs {
            for pt in [s.c1, s.c2, s.p] {
                assert!(pt.0 >= -0.01 && pt.0 <= 10.01, "cx overshoot: {}", pt.0);
                assert!(pt.1 >= -0.01 && pt.1 <= 10.01, "cy overshoot: {}", pt.1);
            }
        }
    }
}
