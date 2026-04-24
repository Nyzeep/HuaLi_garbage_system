use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoredBBox {
    pub bbox: BBox,
    pub score: f64,
}

impl BBox {
    pub fn width(&self) -> i32 {
        (self.x2 - self.x1).max(0)
    }

    pub fn height(&self) -> i32 {
        (self.y2 - self.y1).max(0)
    }

    pub fn area(&self) -> i32 {
        self.width() * self.height()
    }
}

pub fn iou(a: BBox, b: BBox) -> f64 {
    let inter_x1 = a.x1.max(b.x1);
    let inter_y1 = a.y1.max(b.y1);
    let inter_x2 = a.x2.min(b.x2);
    let inter_y2 = a.y2.min(b.y2);

    let inter_w = (inter_x2 - inter_x1).max(0);
    let inter_h = (inter_y2 - inter_y1).max(0);
    let inter_area = (inter_w * inter_h) as f64;

    let area_a = a.area() as f64;
    let area_b = b.area() as f64;
    let union = area_a + area_b - inter_area;

    if union <= 0.0 {
        0.0
    } else {
        inter_area / union
    }
}

pub fn filter_overlapping_boxes(boxes: Vec<BBox>, threshold: f64) -> Vec<BBox> {
    let scored: Vec<ScoredBBox> = boxes
        .into_iter()
        .map(|bbox| ScoredBBox {
            score: bbox.area() as f64,
            bbox,
        })
        .collect();
    non_max_suppression(scored, threshold)
        .into_iter()
        .map(|item| item.bbox)
        .collect()
}

pub fn non_max_suppression(mut boxes: Vec<ScoredBBox>, threshold: f64) -> Vec<ScoredBBox> {
    boxes.retain(|item| !item.score.is_nan());
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score).reverse());
    let mut kept: Vec<ScoredBBox> = Vec::new();

    'outer: for candidate in boxes {
        for existing in &kept {
            if iou(candidate.bbox, existing.bbox) >= threshold {
                continue 'outer;
            }
        }
        kept.push(candidate);
    }

    kept
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LetterboxTransform {
    pub scale: f64,
    pub pad_w: f64,
    pub pad_h: f64,
    pub original_width: i32,
    pub original_height: i32,
}

pub fn invert_letterbox_bbox(bbox: BBox, transform: LetterboxTransform) -> BBox {
    let x1 = (((bbox.x1 as f64) - transform.pad_w) / transform.scale)
        .clamp(0.0, transform.original_width as f64)
        .round() as i32;
    let y1 = (((bbox.y1 as f64) - transform.pad_h) / transform.scale)
        .clamp(0.0, transform.original_height as f64)
        .round() as i32;
    let x2 = (((bbox.x2 as f64) - transform.pad_w) / transform.scale)
        .clamp(0.0, transform.original_width as f64)
        .round() as i32;
    let y2 = (((bbox.y2 as f64) - transform.pad_h) / transform.scale)
        .clamp(0.0, transform.original_height as f64)
        .round() as i32;
    BBox { x1, y1, x2, y2 }
}

pub fn batch_iou_match(left: Vec<BBox>, right: Vec<BBox>, threshold: f64) -> Vec<(usize, usize, f64)> {
    let mut matches = Vec::new();
    for (left_index, left_bbox) in left.iter().enumerate() {
        let mut best_index: Option<usize> = None;
        let mut best_score = 0.0;
        for (right_index, right_bbox) in right.iter().enumerate() {
            let score = iou(*left_bbox, *right_bbox);
            if score >= threshold && score > best_score {
                best_score = score;
                best_index = Some(right_index);
            }
        }
        if let Some(right_index) = best_index {
            matches.push((left_index, right_index, best_score));
        }
    }
    matches
}

pub fn perceptual_hash(grayscale_pixels: Vec<u8>, width: usize, height: usize) -> u64 {
    if width == 0 || height == 0 || grayscale_pixels.len() != width * height {
        return 0;
    }

    let target_w = 9usize;
    let target_h = 8usize;
    let mut resized = vec![0u8; target_w * target_h];
    for y in 0..target_h {
        for x in 0..target_w {
            let src_x = x * width / target_w;
            let src_y = y * height / target_h;
            resized[y * target_w + x] = grayscale_pixels[src_y * width + src_x];
        }
    }

    let mut hash = 0u64;
    for y in 0..target_h {
        for x in 0..(target_w - 1) {
            let left = resized[y * target_w + x];
            let right = resized[y * target_w + x + 1];
            let bit = if left > right { 1u64 } else { 0u64 };
            let bit_index = y * (target_w - 1) + x;
            hash |= bit << bit_index;
        }
    }
    hash
}

pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackEvent {
    pub class_id: i32,
    pub bbox: BBox,
    pub timestamp_ms: i64,
}

pub fn dedupe_track_events(events: Vec<TrackEvent>, cooldown_ms: i64, iou_threshold: f64) -> Vec<TrackEvent> {
    let mut kept: Vec<TrackEvent> = Vec::new();
    let mut last_seen_by_class: HashMap<i32, usize> = HashMap::new();

    'event_loop: for event in events {
        if let Some(&idx) = last_seen_by_class.get(&event.class_id) {
            let existing = &kept[idx];
            if event.timestamp_ms - existing.timestamp_ms <= cooldown_ms && iou(existing.bbox, event.bbox) >= iou_threshold {
                continue 'event_loop;
            }
        }

        last_seen_by_class.insert(event.class_id, kept.len());
        kept.push(event);
    }

    kept
}

#[cfg(feature = "pyo3")]
mod py_api {
    use super::*;
    use pyo3::prelude::*;
    use pyo3::types::PySequence;

    fn py_to_bbox(obj: &Bound<'_, PyAny>) -> PyResult<BBox> {
        if let Ok(seq) = obj.downcast::<PySequence>() {
            let x1: i32 = seq.get_item(0)?.extract()?;
            let y1: i32 = seq.get_item(1)?.extract()?;
            let x2: i32 = seq.get_item(2)?.extract()?;
            let y2: i32 = seq.get_item(3)?.extract()?;
            return Ok(BBox { x1, y1, x2, y2 });
        }
        let x1: i32 = obj.getattr("x1")?.extract()?;
        let y1: i32 = obj.getattr("y1")?.extract()?;
        let x2: i32 = obj.getattr("x2")?.extract()?;
        let y2: i32 = obj.getattr("y2")?.extract()?;
        Ok(BBox { x1, y1, x2, y2 })
    }

    fn py_to_bbox_list(items: &Bound<'_, PyAny>) -> PyResult<Vec<BBox>> {
        let mut out = Vec::new();
        for item in items.iter()? {
            let value = item?;
            out.push(py_to_bbox(&value)?);
        }
        Ok(out)
    }

    fn py_to_scored_bbox_list(items: &Bound<'_, PyAny>) -> PyResult<Vec<ScoredBBox>> {
        let mut out = Vec::new();
        for item in items.iter()? {
            let value = item?;
            if let Ok(seq) = value.downcast::<PySequence>() {
                let bbox = py_to_bbox(&seq.get_item(0)?)?;
                let score: f64 = seq.get_item(1)?.extract()?;
                out.push(ScoredBBox { bbox, score });
                continue;
            }
            let bbox = py_to_bbox(&value.getattr("bbox")?)?;
            let score: f64 = value.getattr("score")?.extract()?;
            out.push(ScoredBBox { bbox, score });
        }
        Ok(out)
    }

    fn bbox_to_py(py: Python<'_>, bbox: BBox) -> PyResult<Py<PyAny>> {
        Ok((bbox.x1, bbox.y1, bbox.x2, bbox.y2).into_py(py))
    }

    fn event_to_track_event(obj: &Bound<'_, PyAny>) -> PyResult<TrackEvent> {
        if let Ok(seq) = obj.downcast::<PySequence>() {
            let class_id: i32 = seq.get_item(0)?.extract()?;
            let bbox = py_to_bbox(&seq.get_item(1)?)?;
            let timestamp_ms: i64 = seq.get_item(2)?.extract()?;
            return Ok(TrackEvent { class_id, bbox, timestamp_ms });
        }
        let class_id: i32 = obj.getattr("class_id")?.extract()?;
        let timestamp_ms: i64 = obj.getattr("timestamp_ms")?.extract()?;
        let bbox_any = obj.getattr("bbox")?;
        let bbox = py_to_bbox(&bbox_any)?;
        Ok(TrackEvent { class_id, bbox, timestamp_ms })
    }

    fn event_to_py(py: Python<'_>, event: TrackEvent) -> PyResult<Py<PyAny>> {
        Ok((event.class_id, (event.bbox.x1, event.bbox.y1, event.bbox.x2, event.bbox.y2), event.timestamp_ms).into_py(py))
    }

    fn py_to_letterbox_transform(obj: &Bound<'_, PyAny>) -> PyResult<LetterboxTransform> {
        if let Ok(seq) = obj.downcast::<PySequence>() {
            return Ok(LetterboxTransform {
                scale: seq.get_item(0)?.extract()?,
                pad_w: seq.get_item(1)?.extract()?,
                pad_h: seq.get_item(2)?.extract()?,
                original_width: seq.get_item(3)?.extract()?,
                original_height: seq.get_item(4)?.extract()?,
            });
        }
        Ok(LetterboxTransform {
            scale: obj.getattr("scale")?.extract()?,
            pad_w: obj.getattr("pad_w")?.extract()?,
            pad_h: obj.getattr("pad_h")?.extract()?,
            original_width: obj.getattr("original_width")?.extract()?,
            original_height: obj.getattr("original_height")?.extract()?,
        })
    }

    #[pyfunction]
    fn iou_py(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(super::iou(py_to_bbox(a)?, py_to_bbox(b)?))
    }

    #[pyfunction]
    fn invert_letterbox_bbox_py(py: Python<'_>, bbox: &Bound<'_, PyAny>, transform: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let result = super::invert_letterbox_bbox(py_to_bbox(bbox)?, py_to_letterbox_transform(transform)?);
        bbox_to_py(py, result)
    }

    #[pyfunction]
    fn batch_iou_match_py(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Vec<(usize, usize, f64)>> {
        Ok(super::batch_iou_match(py_to_bbox_list(left)?, py_to_bbox_list(right)?, threshold))
    }

    #[pyfunction]
    fn perceptual_hash_py(grayscale_pixels: Vec<u8>, width: usize, height: usize) -> PyResult<u64> {
        Ok(super::perceptual_hash(grayscale_pixels, width, height))
    }

    #[pyfunction]
    fn hamming_distance_py(a: u64, b: u64) -> PyResult<u32> {
        Ok(super::hamming_distance(a, b))
    }

    #[pyfunction]
    fn non_max_suppression_py(py: Python<'_>, boxes: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Vec<Py<PyAny>>> {
        let boxes = py_to_scored_bbox_list(boxes)?;
        Ok(super::non_max_suppression(boxes, threshold)
            .into_iter()
            .map(|item| ((item.bbox.x1, item.bbox.y1, item.bbox.x2, item.bbox.y2), item.score).into_py(py))
            .collect())
    }

    #[pyfunction]
    fn filter_overlapping_boxes_py(py: Python<'_>, boxes: &Bound<'_, PyAny>, threshold: f64) -> PyResult<Vec<Py<PyAny>>> {
        let boxes = py_to_bbox_list(boxes)?;
        Ok(super::filter_overlapping_boxes(boxes, threshold)
            .into_iter()
            .map(|bbox| bbox_to_py(py, bbox))
            .collect::<PyResult<Vec<_>>>()?)
    }

    #[pyfunction]
    fn dedupe_track_events_py(py: Python<'_>, events: &Bound<'_, PyAny>, cooldown_ms: i64, iou_threshold: f64) -> PyResult<Vec<Py<PyAny>>> {
        let mut parsed = Vec::new();
        parsed.reserve(events.len().unwrap_or(0));
        for item in events.iter()? {
            let value = item?;
            parsed.push(event_to_track_event(&value)?);
        }
        Ok(super::dedupe_track_events(parsed, cooldown_ms, iou_threshold)
            .into_iter()
            .map(|event| event_to_py(py, event))
            .collect::<PyResult<Vec<_>>>()?)
    }

    #[pymodule]
    fn huali_garbage_core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(iou_py, m)?)?;
        m.add_function(wrap_pyfunction!(invert_letterbox_bbox_py, m)?)?;
        m.add_function(wrap_pyfunction!(batch_iou_match_py, m)?)?;
        m.add_function(wrap_pyfunction!(perceptual_hash_py, m)?)?;
        m.add_function(wrap_pyfunction!(hamming_distance_py, m)?)?;
        m.add_function(wrap_pyfunction!(non_max_suppression_py, m)?)?;
        m.add_function(wrap_pyfunction!(filter_overlapping_boxes_py, m)?)?;
        m.add_function(wrap_pyfunction!(dedupe_track_events_py, m)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bbox(x1: i32, y1: i32, x2: i32, y2: i32) -> BBox { BBox { x1, y1, x2, y2 } }
    fn ev(class_id: i32, x1: i32, y1: i32, x2: i32, y2: i32, ts: i64) -> TrackEvent {
        TrackEvent { class_id, bbox: bbox(x1, y1, x2, y2), timestamp_ms: ts }
    }

    #[test] fn iou_identical() { assert!((iou(bbox(0,0,100,100), bbox(0,0,100,100)) - 1.0).abs() < 1e-9); }
    #[test] fn iou_no_overlap() { assert_eq!(iou(bbox(0,0,10,10), bbox(20,20,30,30)), 0.0); }
    #[test] fn iou_partial() {
        let v = iou(bbox(0,0,100,100), bbox(50,50,150,150));
        assert!((v - 2500.0/17500.0).abs() < 1e-9);
    }
    #[test] fn iou_zero_area() { assert_eq!(iou(bbox(5,5,5,5), bbox(5,5,5,5)), 0.0); }
    #[test] fn iou_symmetric() {
        let a = bbox(0,0,80,60); let b = bbox(40,30,120,90);
        assert!((iou(a,b) - iou(b,a)).abs() < 1e-12);
    }

    #[test] fn filter_non_overlapping_kept() {
        assert_eq!(filter_overlapping_boxes(vec![bbox(0,0,10,10), bbox(20,20,30,30)], 0.3).len(), 2);
    }
    #[test] fn filter_duplicate_removed() {
        assert_eq!(filter_overlapping_boxes(vec![bbox(0,0,100,100), bbox(0,0,100,100)], 0.5).len(), 1);
    }
    #[test] fn filter_empty() { assert_eq!(filter_overlapping_boxes(vec![], 0.5).len(), 0); }
    #[test] fn nms_ignores_nan_scores() {
        let kept = non_max_suppression(
            vec![
                ScoredBBox { bbox: bbox(0, 0, 10, 10), score: f64::NAN },
                ScoredBBox { bbox: bbox(0, 0, 10, 10), score: 0.9 },
            ],
            0.5,
        );
        assert_eq!(kept.len(), 1);
        assert!(!kept[0].score.is_nan());
    }

    #[test] fn dedupe_single_kept() {
        assert_eq!(dedupe_track_events(vec![ev(0,0,0,100,100,0)], 1000, 0.3).len(), 1);
    }
    #[test] fn dedupe_same_object_within_cooldown() {
        let evs = vec![ev(0,0,0,100,100,0), ev(0,0,0,100,100,500)];
        assert_eq!(dedupe_track_events(evs, 1000, 0.3).len(), 1);
    }
    #[test] fn dedupe_after_cooldown_expires() {
        let evs = vec![ev(0,0,0,100,100,0), ev(0,0,0,100,100,1001)];
        assert_eq!(dedupe_track_events(evs, 1000, 0.3).len(), 2);
    }
    #[test] fn dedupe_different_class_kept() {
        let evs = vec![ev(0,0,0,100,100,0), ev(1,0,0,100,100,0)];
        assert_eq!(dedupe_track_events(evs, 1000, 0.3).len(), 2);
    }
    #[test] fn dedupe_different_location_kept() {
        let evs = vec![ev(0,0,0,100,100,0), ev(0,500,500,600,600,500)];
        assert_eq!(dedupe_track_events(evs, 1000, 0.3).len(), 2);
    }
    #[test] fn dedupe_empty() { assert_eq!(dedupe_track_events(vec![], 1000, 0.3).len(), 0); }
}
