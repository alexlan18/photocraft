//! Colour image → filled paths ("Image Trace", the inverse of rendering a picture of flat
//! colour shapes; what File › Export › Image to SVG writes).
//!
//! 1. **Palette.** Median cut over a sample of the opaque pixels (at most [`SAMPLE_LIMIT`]),
//!    refined by a few Lloyd (k-means) iterations, gives `colors` representative colours.
//! 2. **Labels.** Every opaque pixel takes its nearest palette colour; pixels with alpha below
//!    the cutoff stay transparent and are never covered.
//! 3. **Despeckle.** Four-connected regions smaller than `despeckle` pixels take the colour
//!    most of their neighbours have, so noise and anti-aliasing fringes do not each become a
//!    path.
//! 4. **Stacking.** Colours are ordered by area, largest first, and colour *k*'s shape covers
//!    its own pixels *and* every pixel of the colours drawn after it. Later shapes paint over
//!    earlier ones, so neighbouring shapes never leave a hairline gap where their outlines are
//!    fitted independently, and each mask has fewer holes to trace.
//! 5. **Curves.** Each mask is traced with [`crate::trace::trace_mask`] (marching squares and
//!    Schneider curve fitting within `tolerance` px).

use photocraft_doc::Path;
use photocraft_geom::Rect;
use photocraft_raster::Interrupt;

/// Most palette colours a trace may use.
pub const MAX_COLORS: usize = 64;
/// Largest image traced (pixels); labels, region ids and one mask are held at once.
pub const MAX_PIXELS: u64 = 36_000_000;
/// Pixels the palette is computed from; larger images are sampled on a regular stride.
pub const SAMPLE_LIMIT: usize = 120_000;
/// Lloyd iterations refining the median-cut palette.
const LLOYD_ITERATIONS: usize = 6;
/// Label of a transparent pixel.
const CLEAR: u8 = u8::MAX;

/// How an image is traced.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceOptions {
    /// Palette size, `2..=MAX_COLORS`.
    pub colors: usize,
    /// Curve-fitting tolerance in pixels (larger: fewer, smoother curves).
    pub tolerance: f64,
    /// Regions smaller than this many pixels merge into their neighbours (0 keeps them all).
    pub despeckle: u32,
    /// Pixels with alpha below this are left uncovered.
    pub alpha_cutoff: f32,
}

impl Default for TraceOptions {
    fn default() -> Self {
        Self { colors: 16, tolerance: 1.0, despeckle: 8, alpha_cutoff: 0.5 }
    }
}

/// One traced colour: draw them in order, each over the ones before.
#[derive(Clone, Debug, PartialEq)]
pub struct TracedRegion {
    /// Straight RGB in the image's own encoding, `0..=1`.
    pub color: [f32; 3],
    /// Pixels of this colour (not counting those of later regions its shape also covers).
    pub pixels: u64,
    /// The shape, in image pixel coordinates (even-odd: contours after the first exclude).
    pub path: Path,
}

/// Why a trace failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    /// The pixel slice does not hold `width × height` pixels, or the image is empty.
    BadSize,
    /// Past [`MAX_PIXELS`].
    TooLarge,
    /// The options are out of range.
    BadOptions(String),
    /// The interrupt asked to stop.
    Cancelled,
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TraceError::BadSize => write!(f, "the image is empty or its pixel data does not match its size"),
            TraceError::TooLarge => write!(f, "the image is larger than {} megapixels; reduce it before tracing", MAX_PIXELS / 1_000_000),
            TraceError::BadOptions(m) => write!(f, "{m}"),
            TraceError::Cancelled => write!(f, "cancelled"),
        }
    }
}

impl std::error::Error for TraceError {}

impl From<photocraft_raster::Cancelled> for TraceError {
    fn from(_: photocraft_raster::Cancelled) -> Self {
        TraceError::Cancelled
    }
}

/// Traces straight-alpha RGBA pixels (row-major, `width × height`) into stacked colour shapes,
/// largest area first. A fully transparent image gives no regions.
pub fn trace_image(px: &[[f32; 4]], width: u32, height: u32, opts: &TraceOptions, ctl: &Interrupt) -> Result<Vec<TracedRegion>, TraceError> {
    let n = u64::from(width) * u64::from(height);
    if n == 0 || px.len() as u64 != n {
        return Err(TraceError::BadSize);
    }
    if n > MAX_PIXELS {
        return Err(TraceError::TooLarge);
    }
    if !(2..=MAX_COLORS).contains(&opts.colors) {
        return Err(TraceError::BadOptions(format!("colors must be 2..={MAX_COLORS}")));
    }
    if !(opts.tolerance.is_finite() && opts.tolerance > 0.0 && opts.tolerance <= 100.0) {
        return Err(TraceError::BadOptions("tolerance must be above 0 and at most 100 px".into()));
    }
    let cutoff = if opts.alpha_cutoff.is_finite() { opts.alpha_cutoff.clamp(0.0, 1.0) } else { 0.5 };
    let opaque = |p: &[f32; 4]| p[3].is_finite() && p[3] >= cutoff && p[3] > 0.0;
    let rgb = |p: &[f32; 4]| [clean(p[0]), clean(p[1]), clean(p[2])];

    // 1. Palette from a strided sample.
    let stride = (px.len() / SAMPLE_LIMIT).max(1);
    let sample: Vec<[f32; 3]> = px.iter().step_by(stride).filter(|p| opaque(p)).map(rgb).collect();
    if sample.is_empty() && !px.iter().any(opaque) {
        return Ok(Vec::new());
    }
    let sample = if sample.is_empty() { px.iter().filter(|p| opaque(p)).map(rgb).take(SAMPLE_LIMIT).collect() } else { sample };
    let mut palette = median_cut(&sample, opts.colors);
    for _ in 0..LLOYD_ITERATIONS {
        palette = lloyd(&sample, &palette);
    }
    ctl.check()?;
    ctl.progress(0.05);

    // 2. Labels.
    let mut labels = label(px, &palette, &opaque, &rgb);
    ctl.check()?;
    ctl.progress(0.15);

    // 3. Despeckle.
    let (w, h) = (width as usize, height as usize);
    if opts.despeckle > 1 {
        despeckle(&mut labels, w, h, opts.despeckle as usize, palette.len());
    }
    ctl.check()?;
    ctl.progress(0.25);

    // 4. Stacking order, largest area first; empty colours are dropped.
    let mut counts = vec![0u64; palette.len()];
    let mut boxes = vec![Bounds::EMPTY; palette.len()];
    for (i, &l) in labels.iter().enumerate() {
        if let (Some(c), Some(b)) = (counts.get_mut(l as usize), boxes.get_mut(l as usize)) {
            *c += 1;
            b.add(i % w, i / w);
        }
    }
    let mut order: Vec<usize> = (0..palette.len()).filter(|&c| counts[c] > 0).collect();
    order.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    let mut rank = vec![usize::MAX; palette.len()];
    for (r, &c) in order.iter().enumerate() {
        rank[c] = r;
    }

    // 5. One mask per rank: the colour and everything stacked above it, cropped to its bounds.
    let mut out = Vec::with_capacity(order.len());
    let mut cover = Bounds::EMPTY;
    let mut covers = vec![Bounds::EMPTY; order.len()];
    for r in (0..order.len()).rev() {
        cover = cover.union(boxes[order[r]]);
        covers[r] = cover;
    }
    let total = order.len().max(1) as f32;
    for (r, &c) in order.iter().enumerate() {
        ctl.check()?;
        let b = covers[r];
        let (bw, bh) = (b.x1 - b.x0 + 1, b.y1 - b.y0 + 1);
        let mut mask = vec![0f32; bw * bh];
        for y in 0..bh {
            let row = &labels[(b.y0 + y) * w + b.x0..(b.y0 + y) * w + b.x0 + bw];
            for (x, &l) in row.iter().enumerate() {
                if rank.get(l as usize).is_some_and(|&lr| lr >= r) {
                    mask[y * bw + x] = 1.0;
                }
            }
        }
        let rect = Rect::new(b.x0 as i32, b.y0 as i32, (b.x0 + bw) as i32, (b.y0 + bh) as i32);
        let path = crate::trace::trace_mask(&mask, rect, 0.0, opts.tolerance);
        if !path.subpaths.is_empty() {
            out.push(TracedRegion { color: palette[c], pixels: counts[c], path });
        }
        ctl.progress(0.25 + 0.75 * (r + 1) as f32 / total);
    }
    Ok(out)
}

fn clean(v: f32) -> f32 {
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

fn dist2(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dr, dg, db) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    // Weighted towards green as the eye is (the common "redmean"-free approximation).
    0.3 * dr * dr + 0.59 * dg * dg + 0.11 * db * db
}

fn nearest(palette: &[[f32; 3]], c: [f32; 3]) -> usize {
    let mut best = (0, f32::INFINITY);
    for (i, p) in palette.iter().enumerate() {
        let d = dist2(*p, c);
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// Median cut: split the box with the widest channel range at its median until there are `k`
/// boxes (or nothing left to split); each box's mean is a palette colour.
fn median_cut(sample: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    let mut boxes: Vec<Vec<[f32; 3]>> = vec![sample.to_vec()];
    while boxes.len() < k {
        let widest = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.len() > 1)
            .map(|(i, b)| {
                let (ch, range) = widest_channel(b);
                (i, ch, range)
            })
            .filter(|&(_, _, range)| range > 0.0)
            .max_by(|a, b| a.2.total_cmp(&b.2));
        let Some((i, ch, _)) = widest else { break };
        let mut b = boxes.swap_remove(i);
        b.sort_by(|p, q| p[ch].total_cmp(&q[ch]));
        let hi = b.split_off(b.len() / 2);
        boxes.push(b);
        boxes.push(hi);
    }
    boxes.iter().filter(|b| !b.is_empty()).map(|b| mean(b)).collect()
}

fn widest_channel(b: &[[f32; 3]]) -> (usize, f32) {
    let mut best = (0, 0.0f32);
    for ch in 0..3 {
        let (lo, hi) = b.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p[ch]), hi.max(p[ch])));
        if hi - lo > best.1 {
            best = (ch, hi - lo);
        }
    }
    best
}

fn mean(b: &[[f32; 3]]) -> [f32; 3] {
    let n = b.len().max(1) as f64;
    let s = b.iter().fold([0f64; 3], |s, p| [s[0] + f64::from(p[0]), s[1] + f64::from(p[1]), s[2] + f64::from(p[2])]);
    [(s[0] / n) as f32, (s[1] / n) as f32, (s[2] / n) as f32]
}

/// One Lloyd step; a colour no sample is nearest to keeps its place.
fn lloyd(sample: &[[f32; 3]], palette: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let mut sums = vec![[0f64; 3]; palette.len()];
    let mut counts = vec![0u64; palette.len()];
    for &c in sample {
        let i = nearest(palette, c);
        if let (Some(s), Some(n)) = (sums.get_mut(i), counts.get_mut(i)) {
            for ch in 0..3 {
                s[ch] += f64::from(c[ch]);
            }
            *n += 1;
        }
    }
    palette
        .iter()
        .zip(sums.iter().zip(&counts))
        .map(|(p, (s, &n))| if n == 0 { *p } else { [(s[0] / n as f64) as f32, (s[1] / n as f64) as f32, (s[2] / n as f64) as f32] })
        .collect()
}

fn label(px: &[[f32; 4]], palette: &[[f32; 3]], opaque: &(dyn Fn(&[f32; 4]) -> bool + Sync), rgb: &(dyn Fn(&[f32; 4]) -> [f32; 3] + Sync)) -> Vec<u8> {
    let one = |p: &[f32; 4]| if opaque(p) { nearest(palette, rgb(p)) as u8 } else { CLEAR };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        px.par_iter().map(one).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        px.iter().map(one).collect()
    }
}

/// Merges four-connected regions under `min` pixels into the colour most of their neighbours
/// have. Transparent pixels are never filled and never spread.
fn despeckle(labels: &mut [u8], w: usize, h: usize, min: usize, colors: usize) {
    let mut seen = vec![false; labels.len()];
    let mut region: Vec<usize> = Vec::new();
    let mut queue: Vec<usize> = Vec::new();
    let mut votes = vec![0u32; colors];
    for start in 0..labels.len() {
        if seen[start] || labels[start] == CLEAR {
            continue;
        }
        let l = labels[start];
        region.clear();
        queue.clear();
        queue.push(start);
        seen[start] = true;
        let mut small = true;
        votes.iter_mut().for_each(|v| *v = 0);
        while let Some(i) = queue.pop() {
            if small {
                region.push(i);
                small = region.len() < min;
            }
            let (x, y) = (i % w, i / w);
            let mut visit = |j: usize, queue: &mut Vec<usize>| {
                let lj = labels[j];
                if lj == l {
                    if !seen[j] {
                        seen[j] = true;
                        queue.push(j);
                    }
                } else if let Some(v) = votes.get_mut(lj as usize) {
                    *v += 1;
                }
            };
            if x > 0 {
                visit(i - 1, &mut queue);
            }
            if x + 1 < w {
                visit(i + 1, &mut queue);
            }
            if y > 0 {
                visit(i - w, &mut queue);
            }
            if y + 1 < h {
                visit(i + w, &mut queue);
            }
        }
        if !small {
            continue;
        }
        let best = votes.iter().enumerate().filter(|&(_, &v)| v > 0).max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0))).map(|(c, _)| c as u8);
        if let Some(to) = best {
            for &i in &region {
                labels[i] = to;
            }
        }
    }
}

/// Inclusive pixel bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bounds {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl Bounds {
    const EMPTY: Bounds = Bounds { x0: usize::MAX, y0: usize::MAX, x1: 0, y1: 0 };

    fn add(&mut self, x: usize, y: usize) {
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x);
        self.y1 = self.y1.max(y);
    }

    fn union(self, o: Bounds) -> Bounds {
        if o.x0 == usize::MAX {
            return self;
        }
        if self.x0 == usize::MAX {
            return o;
        }
        Bounds { x0: self.x0.min(o.x0), y0: self.y0.min(o.y0), x1: self.x1.max(o.x1), y1: self.y1.max(o.y1) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    fn image(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Vec<[f32; 4]> {
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect()
    }

    fn opts(colors: usize) -> TraceOptions {
        TraceOptions { colors, ..Default::default() }
    }

    #[test]
    fn a_square_on_white_is_two_stacked_shapes() {
        let px = image(40, 30, |x, y| if (10..30).contains(&x) && (5..25).contains(&y) { RED } else { WHITE });
        let regions = trace_image(&px, 40, 30, &opts(2), &Interrupt::NONE).unwrap();
        assert_eq!(regions.len(), 2);
        // White has the larger area, so it is the background and covers the whole canvas.
        assert!(regions[0].color.iter().all(|&c| (c - 1.0).abs() < 1e-3), "{:?}", regions[0].color);
        assert_eq!(regions[0].pixels, 40 * 30 - 400);
        let (x0, y0, x1, y1) = regions[0].path.control_bounds().unwrap();
        assert!(x0 <= 0.01 && y0 <= 0.01 && x1 >= 39.99 && y1 >= 29.99, "{x0} {y0} {x1} {y1}");
        assert!((regions[1].color[0] - 1.0).abs() < 1e-3 && regions[1].color[1] < 1e-3);
        // Control bounds include the handles, which may overshoot a fitted corner slightly.
        let (x0, y0, x1, y1) = regions[1].path.control_bounds().unwrap();
        assert!((x0 - 10.0).abs() < 1.0 && (y0 - 5.0).abs() < 1.0 && (x1 - 30.0).abs() < 1.0 && (y1 - 25.0).abs() < 1.0, "{x0} {y0} {x1} {y1}");
    }

    #[test]
    fn transparent_pixels_stay_uncovered() {
        let clear = [0.0, 0.0, 0.0, 0.0];
        let px = image(20, 20, |x, _| if x < 10 { BLUE } else { clear });
        let regions = trace_image(&px, 20, 20, &opts(4), &Interrupt::NONE).unwrap();
        assert_eq!(regions.len(), 1);
        let (_, _, x1, _) = regions[0].path.control_bounds().unwrap();
        assert!(x1 <= 10.6, "{x1}");
        assert!(trace_image(&vec![clear; 16], 4, 4, &opts(4), &Interrupt::NONE).unwrap().is_empty());
    }

    #[test]
    fn specks_merge_into_their_surroundings() {
        // Isolated single blue pixels on red: noise, not shapes.
        let px = image(32, 32, |x, y| if x % 8 == 3 && y % 8 == 3 { BLUE } else { RED });
        let kept = trace_image(&px, 32, 32, &TraceOptions { despeckle: 0, ..opts(2) }, &Interrupt::NONE).unwrap();
        assert_eq!(kept.len(), 2);
        let merged = trace_image(&px, 32, 32, &TraceOptions { despeckle: 4, ..opts(2) }, &Interrupt::NONE).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].pixels, 32 * 32);
    }

    #[test]
    fn hostile_input_is_an_error_not_a_panic() {
        let nan = [f32::NAN, f32::INFINITY, -1.0, f32::NAN];
        assert_eq!(trace_image(&[], 0, 0, &opts(2), &Interrupt::NONE), Err(TraceError::BadSize));
        assert_eq!(trace_image(&[RED; 3], 2, 2, &opts(2), &Interrupt::NONE), Err(TraceError::BadSize));
        assert_eq!(trace_image(&[RED], u32::MAX, u32::MAX, &opts(2), &Interrupt::NONE), Err(TraceError::BadSize));
        assert!(matches!(trace_image(&[RED; 4], 2, 2, &opts(1), &Interrupt::NONE), Err(TraceError::BadOptions(_))));
        assert!(matches!(trace_image(&[RED; 4], 2, 2, &opts(65), &Interrupt::NONE), Err(TraceError::BadOptions(_))));
        let bad_tol = TraceOptions { tolerance: f64::NAN, ..opts(2) };
        assert!(matches!(trace_image(&[RED; 4], 2, 2, &bad_tol, &Interrupt::NONE), Err(TraceError::BadOptions(_))));
        // NaN pixels are transparent (alpha NaN), never a crash.
        assert!(trace_image(&[nan; 9], 3, 3, &opts(2), &Interrupt::NONE).unwrap().is_empty());
        // A one-colour image asks for more colours than it has.
        assert_eq!(trace_image(&[RED; 9], 3, 3, &opts(64), &Interrupt::NONE).unwrap().len(), 1);
    }

    #[test]
    fn a_cancel_stops_the_trace() {
        let px = image(16, 16, |x, _| if x < 8 { RED } else { BLUE });
        let cancel = || true;
        let ctl = Interrupt::cancel_only(&cancel);
        assert_eq!(trace_image(&px, 16, 16, &opts(2), &ctl), Err(TraceError::Cancelled));
    }
}
