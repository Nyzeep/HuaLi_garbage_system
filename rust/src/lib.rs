use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
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
    let mut kept: Vec<BBox> = Vec::new();

    'outer: for candidate in boxes {
        for existing in &kept {
            if iou(candidate, *existing) >= threshold {
                continue 'outer;
            }
        }
        kept.push(candidate);
    }

    kept
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
            let len = seq.len()?;
            if len != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bbox sequence must contain exactly 4 items, got {}",
                    len
                )));
            }
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

    #[pyfunction]
    fn iou_py(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(super::iou(py_to_bbox(a)?, py_to_bbox(b)?))
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
