use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use huali_garbage_core::{dedupe_track_events, filter_overlapping_boxes, iou, BBox, TrackEvent};
use serde::{Deserialize, Serialize};
use std::{env, net::SocketAddr};

#[derive(Debug, Clone)]
struct AppState {
    service_name: String,
    service_version: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    available: bool,
    healthy: bool,
    error: Option<String>,
    latency_ms: Option<f64>,
    service: String,
    version: String,
}

#[derive(Debug, Deserialize)]
struct IouRequest {
    a: BBox,
    b: BBox,
}

#[derive(Debug, Serialize)]
struct IouResponse {
    value: f64,
}

#[derive(Debug, Deserialize)]
struct FilterBoxesRequest {
    boxes: Vec<BBox>,
    threshold: f64,
}

#[derive(Debug, Serialize)]
struct FilterBoxesResponse {
    boxes: Vec<BBox>,
}

#[derive(Debug, Deserialize)]
struct DedupeEventsRequest {
    events: Vec<TrackEvent>,
    cooldown_ms: i64,
    iou_threshold: f64,
}

#[derive(Debug, Serialize)]
struct DedupeEventsResponse {
    events: Vec<TrackEvent>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    message: String,
}

#[tokio::main]
async fn main() {
    let host = env::var("RUST_SERVICE_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port = env::var("RUST_SERVICE_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(50051);
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .expect("invalid service bind address");

    let state = AppState {
        service_name: "huali-garbage-core".to_string(),
        service_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/iou", post(compute_iou))
        .route("/v1/filter-boxes", post(filter_boxes))
        .route("/v1/dedupe-events", post(dedupe_events))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind rust service port");

    axum::serve(listener, app)
        .await
        .expect("rust service exited unexpectedly");
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        available: true,
        healthy: true,
        error: None,
        latency_ms: None,
        service: state.service_name,
        version: state.service_version,
    })
}

async fn compute_iou(
    Json(payload): Json<IouRequest>,
) -> Result<Json<IouResponse>, (StatusCode, Json<ErrorResponse>)> {
    validate_bbox(payload.a)?;
    validate_bbox(payload.b)?;
    Ok(Json(IouResponse {
        value: iou(payload.a, payload.b),
    }))
}

async fn filter_boxes(
    Json(payload): Json<FilterBoxesRequest>,
) -> Result<Json<FilterBoxesResponse>, (StatusCode, Json<ErrorResponse>)> {
    if !(0.0..=1.0).contains(&payload.threshold) {
        return Err(bad_request("threshold must be between 0 and 1"));
    }
    for bbox in &payload.boxes {
        validate_bbox(*bbox)?;
    }
    Ok(Json(FilterBoxesResponse {
        boxes: filter_overlapping_boxes(payload.boxes, payload.threshold),
    }))
}

async fn dedupe_events(
    Json(payload): Json<DedupeEventsRequest>,
) -> Result<Json<DedupeEventsResponse>, (StatusCode, Json<ErrorResponse>)> {
    if payload.cooldown_ms < 0 {
        return Err(bad_request("cooldown_ms must be >= 0"));
    }
    if !(0.0..=1.0).contains(&payload.iou_threshold) {
        return Err(bad_request("iou_threshold must be between 0 and 1"));
    }
    for event in &payload.events {
        validate_bbox(event.bbox)?;
    }
    Ok(Json(DedupeEventsResponse {
        events: dedupe_track_events(payload.events, payload.cooldown_ms, payload.iou_threshold),
    }))
}

fn validate_bbox(bbox: BBox) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    if bbox.x2 < bbox.x1 || bbox.y2 < bbox.y1 {
        return Err(bad_request("invalid bbox coordinates"));
    }
    Ok(())
}

fn bad_request(message: &str) -> (StatusCode, Json<ErrorResponse>) {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
            message: message.to_string(),
        }),
    )
}
