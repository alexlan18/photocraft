//! Image Trace: a picture → flat-colour vector shapes ([`photocraft_vector::image_trace`]).
//!
//! - `file.export.svg` (File › Export › Image to SVG…) writes the trace as an SVG file, or
//!   returns its text, without changing the document.
//! - `image.trace` (Image › Image Trace…) adds the trace to the document as a group of shape
//!   layers, one per colour, editable with the path tools.
//!
//! Both trace the visible composite (or one layer's pixels with `"source":"layer"`) and run as
//! cancellable background jobs when started from the UI.

use photocraft_color::Color;
use photocraft_doc::{Document, Fill, Layer, LayerContent, ShapeLayer};
use photocraft_vector::image_trace::{self, MAX_COLORS, TraceOptions, TracedRegion};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::file_cmds::{native_doc, stem, write_file};
use crate::{EngineError, Result, Session};

const EXPORT: &str = "file.export.svg";
const TRACE: &str = "image.trace";

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

macro_rules! spec {
    ($id:expr, $label:literal, [$($m:literal),*], $params:expr, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            EXPORT,
            "Image to SVG…",
            ["File", "Export"],
            concat!(
                r#"{"path":str? (.svg; omit to return the text as "svg"),"#,
                r#""colors":2..64=16 (palette size),"tolerance":0.1..10=1 (curve fit in px; larger is smoother with fewer points),"despeckle":0..10000=8 (regions under this many pixels merge into their neighbours),"source":"document|layer"="document" (the visible composite, or the active layer's own pixels)"#,
                r#","title":str?=document name} → {path|svg, width, height, colors, paths, bytes} (traces the picture into flat-colour shapes; the document is unchanged)"#
            ),
            native_or_return,
            export_svg
        ),
        spec!(
            TRACE,
            "Image Trace…",
            ["Image"],
            concat!(
                r#"{"colors":2..64=16 (palette size),"tolerance":0.1..10=1 (curve fit in px; larger is smoother with fewer points),"despeckle":0..10000=8 (regions under this many pixels merge into their neighbours),"source":"document|layer"="document" (the visible composite, or the active layer's own pixels)"#,
                r#","hideSource":bool=false (hide the traced layer, or every other layer for the composite)} → {group, layers, colors} (adds a group of shape layers, one per colour, above the active layer)"#
            ),
            has_doc,
            trace_to_shapes
        ),
    ]
}

/// Export needs a document; writing a file needs the file system, returning the text does not.
fn native_or_return(s: &Session) -> std::result::Result<(), String> {
    if cfg!(target_arch = "wasm32") { has_doc(s) } else { native_doc(s) }
}

fn num(cmd: &str, p: &Value, key: &str, default: f64, lo: f64, hi: f64) -> Result<f64> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(v) => {
            let n = v.as_f64().ok_or_else(|| bad(cmd, format!("{key} must be a number")))?;
            if n.is_finite() && (lo..=hi).contains(&n) { Ok(n) } else { Err(bad(cmd, format!("{key} must be {lo}..={hi}"))) }
        }
    }
}

fn options(cmd: &str, p: &Value) -> Result<TraceOptions> {
    if !(p.is_object() || p.is_null()) {
        return Err(bad(cmd, "params must be an object"));
    }
    Ok(TraceOptions {
        colors: num(cmd, p, "colors", 16.0, 2.0, MAX_COLORS as f64)?.round() as usize,
        tolerance: num(cmd, p, "tolerance", 1.0, 0.1, 10.0)?,
        despeckle: num(cmd, p, "despeckle", 8.0, 0.0, 10_000.0)?.round() as u32,
        alpha_cutoff: 0.5,
    })
}

/// What is traced: the composite, or the active layer.
#[derive(Clone, Copy, PartialEq)]
enum Source {
    Document,
    Layer,
}

fn source(cmd: &str, p: &Value) -> Result<Source> {
    match p.get("source") {
        None | Some(Value::Null) => Ok(Source::Document),
        Some(Value::String(s)) if s == "document" => Ok(Source::Document),
        Some(Value::String(s)) if s == "layer" => Ok(Source::Layer),
        Some(_) => Err(bad(cmd, r#"source must be "document" or "layer""#)),
    }
}

/// The pixels to trace (straight RGBA over the canvas) and the layer they came from.
fn pixels(cmd: &str, doc: &Document, active: Option<photocraft_doc::LayerId>, src: Source) -> Result<(Vec<[f32; 4]>, Option<photocraft_doc::LayerId>)> {
    let (w, h) = (doc.size.width, doc.size.height);
    if u64::from(w) * u64::from(h) > image_trace::MAX_PIXELS {
        return Err(bad(cmd, format!("the canvas is larger than {} megapixels; reduce it before tracing", image_trace::MAX_PIXELS / 1_000_000)));
    }
    match src {
        Source::Document => Ok((photocraft_compose::flatten(doc).px, None)),
        Source::Layer => {
            let id = active.ok_or_else(|| bad(cmd, "no active layer to trace"))?;
            let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            if matches!(layer.content, LayerContent::Group(_)) {
                return Err(bad(cmd, "the active layer is a group; trace the document or one layer inside it"));
            }
            Ok((photocraft_compose::render_layer(layer, doc.bounds()).px, Some(id)))
        }
    }
}

fn trace(cmd: &str, px: &[[f32; 4]], w: u32, h: u32, opts: &TraceOptions, ctx: &crate::jobs::JobCtx) -> Result<Vec<TracedRegion>> {
    ctx.stage(0.0, 1.0, "Tracing", |ctl| image_trace::trace_image(px, w, h, opts, ctl)).map_err(|e| match e {
        image_trace::TraceError::Cancelled => EngineError::Cancelled,
        e => bad(cmd, e.to_string()),
    })
}

fn export_svg(s: &mut Session, p: &Value) -> Result<Value> {
    let opts = options(EXPORT, p)?;
    let src = source(EXPORT, p)?;
    let path = match p.get("path") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
        Some(_) => return Err(bad(EXPORT, "path must be a non-empty string")),
    };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let doc = st.doc.clone();
    let active = st.active_layer;
    let title = p.get("title").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stem(&doc.name));
    // Check the source before the job starts, so a bad request fails at once.
    if src == Source::Layer {
        let id = active.ok_or_else(|| bad(EXPORT, "no active layer to trace"))?;
        doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    }
    crate::jobs::run(
        s,
        "Image to SVG",
        false,
        move |ctx| {
            let (w, h) = (doc.size.width, doc.size.height);
            let (px, _) = pixels(EXPORT, &doc, active, src)?;
            let regions = trace(EXPORT, &px, w, h, &opts, ctx)?;
            let svg = photocraft_io::svg_write::traced_svg(w, h, &title, &regions);
            let paths = regions.len();
            if let Some(path) = &path {
                write_file(path, svg.as_bytes())?;
            }
            Ok((svg, w, h, paths, path))
        },
        move |_, (svg, w, h, paths, path)| {
            let mut out = json!({"width": w, "height": h, "colors": paths, "paths": paths, "bytes": svg.len()});
            match path {
                Some(path) => out["path"] = json!(path),
                None => out["svg"] = json!(svg),
            }
            Ok(out)
        },
    )
}

fn trace_to_shapes(s: &mut Session, p: &Value) -> Result<Value> {
    let opts = options(TRACE, p)?;
    let src = source(TRACE, p)?;
    let hide = match p.get("hideSource") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad(TRACE, "hideSource must be true or false"))?,
    };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if src == Source::Layer {
        let id = st.active_layer.ok_or_else(|| bad(TRACE, "no active layer to trace"))?;
        st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    }
    crate::jobs::edit_job(
        s,
        "Image Trace",
        move |doc, active, ctx| {
            let (w, h) = (doc.size.width, doc.size.height);
            let (px, traced) = pixels(TRACE, doc, *active, src)?;
            let regions = trace(TRACE, &px, w, h, &opts, ctx)?;
            if regions.is_empty() {
                return Err(bad(TRACE, "nothing to trace: the pixels are transparent"));
            }
            let mut children = Vec::with_capacity(regions.len());
            // A group's first child is the bottom of its stack: the background colour.
            for r in &regions {
                let [cr, cg, cb] = r.color;
                let mut sh = ShapeLayer { path: r.path.clone(), fill: Some(Fill::Solid(Color::rgba(cr, cg, cb, 1.0))), ..Default::default() };
                crate::vector_cmds::refresh_shape(doc, &mut sh);
                let hex = format!("#{:02x}{:02x}{:02x}", byte(cr), byte(cg), byte(cb));
                children.push(Layer::new(hex, LayerContent::Shape(sh)));
            }
            let layers: Vec<u64> = children.iter().map(|l| l.id.0).collect();
            if hide {
                match traced {
                    Some(id) => {
                        if let Some(l) = doc.layer_mut(id) {
                            l.visible = false;
                        }
                    }
                    None => doc.layers.iter_mut().for_each(|l| l.visible = false),
                }
            }
            let name = doc.next_layer_name("Image Trace");
            let group = Layer::group(name, children);
            let gid = doc.insert_above(*active, group);
            *active = Some(gid);
            Ok((gid.0, layers))
        },
        |(group, layers)| json!({"group": group, "colors": layers.len(), "layers": layers}),
    )
}

fn byte(v: f32) -> u8 {
    if v.is_finite() { (v.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::Size;

    /// A 40×30 document: white with a red square at (10,5)-(30,25).
    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 30, "name": "logo"})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::new();
            for y in 0..30 {
                for x in 0..40 {
                    let red = (10..30).contains(&x) && (5..25).contains(&y);
                    data.extend_from_slice(&(if red { [1.0, 0.0, 0.0] } else { [1.0, 1.0, 1.0] })[..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(photocraft_geom::Rect::new(0, 0, 40, 30), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    #[test]
    fn the_svg_comes_back_as_text_without_a_path() {
        let mut s = session();
        let r = s.execute(EXPORT, json!({"colors": 2})).unwrap();
        assert_eq!((r["width"].as_u64(), r["height"].as_u64(), r["paths"].as_u64()), (Some(40), Some(30), Some(2)));
        let svg = r["svg"].as_str().unwrap();
        assert!(svg.contains("<title>logo</title>") && svg.contains(r##"fill="#ff0000""##) && svg.contains(r##"fill="#ffffff""##), "{svg}");
        // The export leaves the document alone.
        assert_eq!(s.active().unwrap().doc.layers.len(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_svg_is_written_to_the_path() {
        let dir = std::env::temp_dir().join(format!("photocraft-trace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("logo.svg");
        let mut s = session();
        let r = s.execute(EXPORT, json!({"path": file.to_string_lossy(), "colors": 4, "tolerance": 2})).unwrap();
        assert!(r.get("svg").is_none());
        let text = std::fs::read_to_string(&file).unwrap();
        assert!(text.starts_with("<?xml") && text.contains("<path"), "{text}");
        // It opens again as shape layers.
        let back = photocraft_io::import("logo.svg", text.as_bytes()).unwrap().document;
        assert_eq!(back.size, Size::new(40, 30));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn image_trace_adds_a_group_of_shape_layers() {
        let mut s = session();
        let r = s.execute(TRACE, json!({"colors": 2, "hideSource": true})).unwrap();
        assert_eq!(r["colors"].as_u64(), Some(2));
        let st = s.active().unwrap();
        let group = st.doc.layer(photocraft_doc::LayerId(r["group"].as_u64().unwrap())).unwrap();
        let LayerContent::Group(g) = &group.content else { panic!("not a group") };
        assert_eq!(g.children.len(), 2);
        assert!(g.children.iter().all(|l| matches!(l.content, LayerContent::Shape(_))));
        assert_eq!(g.children[1].name, "#ff0000");
        assert!(!st.doc.layers[0].visible, "the traced layers are hidden");
        // One undo step takes it all back.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.len(), 1);
    }

    #[test]
    fn bad_params_are_errors() {
        let mut s = session();
        for p in [
            json!({"colors": 1}),
            json!({"colors": 65}),
            json!({"colors": "many"}),
            json!({"tolerance": 0}),
            json!({"tolerance": f64::MAX}),
            json!({"despeckle": -1}),
            json!({"source": "selection"}),
            json!({"path": ""}),
            json!({"path": 3}),
            json!([1, 2]),
        ] {
            assert!(s.execute(EXPORT, p.clone()).is_err(), "{p}");
        }
        assert!(s.execute(TRACE, json!({"hideSource": "yes"})).is_err());
        assert!(Session::new().execute(TRACE, json!({})).is_err(), "no document");
    }

    #[test]
    fn a_transparent_layer_has_nothing_to_trace() {
        let mut s = session();
        s.execute("layer.new.layer", json!({})).unwrap();
        let e = s.execute(TRACE, json!({"source": "layer"})).unwrap_err();
        assert!(e.to_string().contains("nothing to trace"), "{e}");
        let r = s.execute(EXPORT, json!({"source": "layer"})).unwrap();
        assert_eq!(r["paths"].as_u64(), Some(0));
    }
}
