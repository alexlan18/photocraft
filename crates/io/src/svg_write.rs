//! SVG output for traced images (File › Export › Image to SVG): each traced colour becomes one
//! `<path>` with an even-odd fill, drawn in stacking order. See
//! [`photocraft_vector::image_trace`] for how the shapes are made.

use std::fmt::Write as _;

use photocraft_doc::vector::{Knot, Path};
use photocraft_vector::image_trace::TracedRegion;

/// An SVG document of `width × height` px holding `regions` in order (later ones on top).
pub fn traced_svg(width: u32, height: u32, title: &str, regions: &[TracedRegion]) -> String {
    let mut s = String::new();
    let _ = writeln!(s, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    let _ = writeln!(s, r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">"#);
    if !title.is_empty() {
        let _ = writeln!(s, "<title>{}</title>", escape(title));
    }
    let _ = writeln!(s, "<desc>Traced by PhotoCraft</desc>");
    for r in regions {
        let d = path_data(&r.path);
        if d.is_empty() {
            continue;
        }
        let _ = writeln!(s, r#"<path fill="{}" fill-rule="evenodd" d="{d}"/>"#, hex(r.color));
    }
    s.push_str("</svg>\n");
    s
}

/// The `d` attribute of a path: one `M … C … Z` run per subpath, cubic segments throughout
/// (straight ones too, with their handles on the anchors), coordinates to 1/100 px.
pub fn path_data(path: &Path) -> String {
    let mut d = String::new();
    for sp in &path.subpaths {
        let (Some(first), Some(_)) = (sp.knots.first(), sp.knots.get(1)) else { continue };
        if !d.is_empty() {
            d.push(' ');
        }
        let _ = write!(d, "M{} {}", num(first.anchor.x), num(first.anchor.y));
        let seg = |d: &mut String, a: &Knot, b: &Knot| {
            let _ = write!(d, "C{} {} {} {} {} {}", num(a.out_ctrl.x), num(a.out_ctrl.y), num(b.in_ctrl.x), num(b.in_ctrl.y), num(b.anchor.x), num(b.anchor.y));
        };
        for w in sp.knots.windows(2) {
            if let [a, b] = w {
                seg(&mut d, a, b);
            }
        }
        if sp.closed {
            if let Some(last) = sp.knots.last() {
                seg(&mut d, last, first);
            }
            d.push('Z');
        }
    }
    d
}

/// `#rrggbb` from straight RGB in `0..=1`.
fn hex(c: [f32; 3]) -> String {
    let b = |v: f32| if v.is_finite() { (v.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 };
    format!("#{:02x}{:02x}{:02x}", b(c[0]), b(c[1]), b(c[2]))
}

/// A coordinate to two decimals without trailing zeros (`-0` as `0`, non-finite as `0`).
fn num(v: f64) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    let s = format!("{:.2}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() { "0".into() } else { s.to_string() }
}

fn escape(t: &str) -> String {
    t.chars()
        .filter(|c| !c.is_control())
        .map(|c| match c {
            '&' => "&amp;".to_string(),
            '<' => "&lt;".to_string(),
            '>' => "&gt;".to_string(),
            '"' => "&quot;".to_string(),
            c => c.to_string(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::vector::Subpath;

    #[test]
    fn numbers_are_short() {
        assert_eq!(num(1.0), "1");
        assert_eq!(num(1.5), "1.5");
        assert_eq!(num(-0.001), "0");
        assert_eq!(num(2.345), "2.35");
        assert_eq!(num(f64::NAN), "0");
        assert_eq!(hex([1.0, 0.5, f32::NAN]), "#ff8000");
    }

    #[test]
    fn a_traced_square_round_trips_through_the_svg_importer() {
        let path = Path::new(vec![Subpath::polygon(&[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0)])]);
        let regions = [TracedRegion { color: [1.0, 0.0, 0.0], pixels: 36, path }];
        let svg = traced_svg(10, 10, "A <b> & \"c\"", &regions);
        assert!(svg.contains(r##"fill="#ff0000""##), "{svg}");
        assert!(svg.contains("<title>A &lt;b&gt; &amp; &quot;c&quot;</title>"), "{svg}");
        assert!(svg.contains("M2 2C2 2 8 2 8 2"), "{svg}");
        let doc = crate::svg::import_svg("t.svg", svg.as_bytes()).unwrap().document;
        assert_eq!((doc.size.width, doc.size.height), (10, 10));
        assert!(!doc.layers.is_empty());
    }

    #[test]
    fn subpaths_with_fewer_than_two_knots_are_skipped() {
        let lone = Subpath { closed: true, knots: vec![Knot::corner(1.0, 1.0)], op: Default::default() };
        assert_eq!(path_data(&Path::new(vec![lone, Subpath { closed: false, knots: vec![], op: Default::default() }])), "");
    }
}
