from __future__ import annotations

"""Benchmark Rust call paths.

This script compares three execution paths for the same geometry/dedupe workload:
1. Pure Python implementation
2. Long-lived Rust HTTP service (current production bridge)
3. PyO3 extension module (optional, if available)

Usage examples:

  # Single-frame micro-benchmark
  python benchmarks/rust_call_path_benchmark.py --mode single --iterations 5000

  # Batch/multi-frame benchmark
  python benchmarks/rust_call_path_benchmark.py --mode batch --batch-size 32 --iterations 2000

  # Save results as JSON
  python benchmarks/rust_call_path_benchmark.py --output benchmark_results.json
"""

import argparse
import importlib
import json
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import requests


BBox = tuple[int, int, int, int]
TrackEvent = tuple[int, BBox, int]


@dataclass
class BenchmarkResult:
    path: str
    mode: str
    iterations: int
    batch_size: int
    avg_ms: float
    p50_ms: float
    p95_ms: float
    throughput_ops_s: float
    notes: str = ""


SAMPLE_BOXES: list[BBox] = [
    (10, 10, 50, 50),
    (12, 12, 48, 48),
    (100, 100, 160, 160),
    (170, 170, 240, 240),
]

SAMPLE_EVENTS: list[TrackEvent] = [
    (3, SAMPLE_BOXES[0], 0),
    (3, SAMPLE_BOXES[1], 500),
    (1, SAMPLE_BOXES[2], 1100),
    (1, SAMPLE_BOXES[3], 1400),
]


def py_iou(a: BBox, b: BBox) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def py_filter_boxes(boxes: list[BBox], threshold: float) -> list[BBox]:
    kept: list[BBox] = []
    for candidate in boxes:
        if any(py_iou(candidate, existing) >= threshold for existing in kept):
            continue
        kept.append(candidate)
    return kept


def py_dedupe_events(events: list[TrackEvent], cooldown_ms: int, iou_threshold: float) -> list[TrackEvent]:
    kept: list[TrackEvent] = []
    for event in events:
        suppressed = False
        for existing in kept:
            if existing[0] != event[0]:
                continue
            if event[2] - existing[2] > cooldown_ms:
                continue
            if py_iou(existing[1], event[1]) >= iou_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(event)
    return kept


def http_get_json(base_url: str, path: str) -> dict:
    response = requests.get(f"{base_url}{path}", timeout=3)
    response.raise_for_status()
    return response.json()


def http_ready(base_url: str) -> tuple[bool, str]:
    try:
        payload = http_get_json(base_url, "/health")
        if payload.get("healthy") is True:
            return True, "health ok"
        return False, f"health endpoint returned unhealthy payload: {payload}"
    except Exception as exc:
        return False, str(exc)


def http_iou(base_url: str, a: BBox, b: BBox) -> float:
    response = requests.post(f"{base_url}/v1/iou", json={"a": bbox_to_obj(a), "b": bbox_to_obj(b)}, timeout=5)
    response.raise_for_status()
    return float(response.json()["value"])


def http_filter_boxes(base_url: str, boxes: list[BBox], threshold: float) -> list[BBox]:
    response = requests.post(
        f"{base_url}/v1/filter-boxes",
        json={"boxes": [bbox_to_obj(b) for b in boxes], "threshold": threshold},
        timeout=5,
    )
    response.raise_for_status()
    return obj_to_boxes(response.json()["boxes"])


def http_dedupe_events(base_url: str, events: list[TrackEvent], cooldown_ms: int, iou_threshold: float) -> list[TrackEvent]:
    response = requests.post(
        f"{base_url}/v1/dedupe-events",
        json={
            "events": [
                {
                    "class_id": int(e[0]),
                    "bbox": bbox_to_obj(e[1]),
                    "timestamp_ms": int(e[2]),
                }
                for e in events
            ],
            "cooldown_ms": cooldown_ms,
            "iou_threshold": iou_threshold,
        },
        timeout=5,
    )
    response.raise_for_status()
    payload = response.json()["events"]
    return [
        (e["class_id"], obj_to_box(e["bbox"]), e["timestamp_ms"])
        for e in payload
    ]


def bbox_to_obj(bbox: BBox) -> dict[str, int]:
    return {"x1": bbox[0], "y1": bbox[1], "x2": bbox[2], "y2": bbox[3]}


def obj_to_box(obj: dict[str, int]) -> BBox:
    return (obj["x1"], obj["y1"], obj["x2"], obj["y2"])


def obj_to_boxes(items: Iterable[dict[str, int]]) -> list[BBox]:
    return [obj_to_box(item) for item in items]


def bench(fn: Callable[[], object], iterations: int) -> list[float]:
    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def summarize(samples: list[float], iterations: int, batch_size: int, path: str, mode: str, notes: str = "") -> BenchmarkResult:
    avg = statistics.mean(samples)
    p50 = statistics.median(samples)
    p95 = statistics.quantiles(samples, n=20)[18] if len(samples) >= 20 else max(samples)
    throughput = (iterations * batch_size) / (sum(samples) / 1000.0) if samples else 0.0
    return BenchmarkResult(
        path=path,
        mode=mode,
        iterations=iterations,
        batch_size=batch_size,
        avg_ms=round(avg, 4),
        p50_ms=round(p50, 4),
        p95_ms=round(p95, 4),
        throughput_ops_s=round(throughput, 2),
        notes=notes,
    )


def maybe_load_pyo3_module() -> object | None:
    candidates = ["huali_garbage_core", "rust_core", "huali_garbage_pyo3"]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def load_module_callables(module: object) -> dict[str, Callable]:
    callables: dict[str, Callable] = {}
    for name in ("iou", "filter_overlapping_boxes", "dedupe_track_events"):
        if hasattr(module, name):
            callables[name] = getattr(module, name)
        else:
            alt_name = f"{name}_py"
            if hasattr(module, alt_name):
                callables[name] = getattr(module, alt_name)
    return callables


def run_single_mode(args: argparse.Namespace) -> list[BenchmarkResult]:
    base_url = args.base_url.rstrip("/")
    results: list[BenchmarkResult] = []

    # Pure Python baseline
    samples = bench(lambda: py_iou(SAMPLE_BOXES[0], SAMPLE_BOXES[1]), args.iterations)
    results.append(summarize(samples, args.iterations, 1, "pure_python", "single", "iou"))

    samples = bench(lambda: py_filter_boxes(SAMPLE_BOXES, 0.5), args.iterations)
    results.append(summarize(samples, args.iterations, 1, "pure_python", "single", "filter_boxes"))

    samples = bench(lambda: py_dedupe_events(SAMPLE_EVENTS, 1000, 0.4), args.iterations)
    results.append(summarize(samples, args.iterations, 1, "pure_python", "single", "dedupe_events"))

    ready, ready_msg = http_ready(base_url)
    if ready:
        samples = bench(lambda: http_iou(base_url, SAMPLE_BOXES[0], SAMPLE_BOXES[1]), args.iterations)
        results.append(summarize(samples, args.iterations, 1, "rust_http", "single", "iou"))

        samples = bench(lambda: http_filter_boxes(base_url, SAMPLE_BOXES, 0.5), args.iterations)
        results.append(summarize(samples, args.iterations, 1, "rust_http", "single", "filter_boxes"))

        samples = bench(lambda: http_dedupe_events(base_url, SAMPLE_EVENTS, 1000, 0.4), args.iterations)
        results.append(summarize(samples, args.iterations, 1, "rust_http", "single", "dedupe_events"))
    else:
        results.append(
            BenchmarkResult(
                path="rust_http",
                mode="single",
                iterations=args.iterations,
                batch_size=1,
                avg_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                throughput_ops_s=0.0,
                notes=f"skipped: {ready_msg}",
            )
        )

    # Optional PyO3 module
    module = maybe_load_pyo3_module()
    if module is None:
        results.append(
            BenchmarkResult(
                path="pyo3",
                mode="single",
                iterations=args.iterations,
                batch_size=1,
                avg_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                throughput_ops_s=0.0,
                notes="not installed",
            )
        )
    else:
        callables = load_module_callables(module)
        if "iou" in callables:
            samples = bench(lambda: callables["iou"](SAMPLE_BOXES[0], SAMPLE_BOXES[1]), args.iterations)
            results.append(summarize(samples, args.iterations, 1, "pyo3", "single", "iou"))
        if "filter_overlapping_boxes" in callables:
            samples = bench(lambda: callables["filter_overlapping_boxes"](SAMPLE_BOXES, 0.5), args.iterations)
            results.append(summarize(samples, args.iterations, 1, "pyo3", "single", "filter_boxes"))
        if "dedupe_track_events" in callables:
            samples = bench(
                lambda: callables["dedupe_track_events"](SAMPLE_EVENTS, 1000, 0.4),
                args.iterations,
            )
            results.append(summarize(samples, args.iterations, 1, "pyo3", "single", "dedupe_events"))

    return results


def run_batch_mode(args: argparse.Namespace) -> list[BenchmarkResult]:
    base_url = args.base_url.rstrip("/")
    batch_boxes = SAMPLE_BOXES * args.batch_size
    batch_events = SAMPLE_EVENTS * args.batch_size
    results: list[BenchmarkResult] = []

    results.append(
        summarize(
            bench(lambda: py_filter_boxes(batch_boxes, 0.5), args.iterations),
            args.iterations,
            args.batch_size,
            "pure_python",
            "batch",
            "filter_boxes",
        )
    )
    results.append(
        summarize(
            bench(lambda: py_dedupe_events(batch_events, 1000, 0.4), args.iterations),
            args.iterations,
            args.batch_size,
            "pure_python",
            "batch",
            "dedupe_events",
        )
    )

    ready, ready_msg = http_ready(base_url)
    if ready:
        results.append(
            summarize(
                bench(lambda: http_filter_boxes(base_url, batch_boxes, 0.5), args.iterations),
                args.iterations,
                args.batch_size,
                "rust_http",
                "batch",
                "filter_boxes",
            )
        )
        results.append(
            summarize(
                bench(lambda: http_dedupe_events(base_url, batch_events, 1000, 0.4), args.iterations),
                args.iterations,
                args.batch_size,
                "rust_http",
                "batch",
                "dedupe_events",
            )
        )
    else:
        results.append(
            BenchmarkResult(
                path="rust_http",
                mode="batch",
                iterations=args.iterations,
                batch_size=args.batch_size,
                avg_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                throughput_ops_s=0.0,
                notes=f"skipped: {ready_msg}",
            )
        )

    module = maybe_load_pyo3_module()
    if module is not None:
        callables = load_module_callables(module)
        if "filter_overlapping_boxes" in callables:
            samples = bench(lambda: callables["filter_overlapping_boxes"](batch_boxes, 0.5), args.iterations)
            results.append(summarize(samples, args.iterations, args.batch_size, "pyo3", "batch", "filter_boxes"))
        if "dedupe_track_events" in callables:
            samples = bench(
                lambda: callables["dedupe_track_events"](batch_events, 1000, 0.4),
                args.iterations,
            )
            results.append(summarize(samples, args.iterations, args.batch_size, "pyo3", "batch", "dedupe_events"))
    else:
        results.append(
            BenchmarkResult(
                path="pyo3",
                mode="batch",
                iterations=args.iterations,
                batch_size=args.batch_size,
                avg_ms=0.0,
                p50_ms=0.0,
                p95_ms=0.0,
                throughput_ops_s=0.0,
                notes="not installed",
            )
        )

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    print("\n=== Rust call path benchmark results ===\n")
    for item in results:
        print(
            f"[{item.mode}] {item.path:10s} | {item.notes:14s} | "
            f"avg={item.avg_ms:.4f} ms | p50={item.p50_ms:.4f} ms | "
            f"p95={item.p95_ms:.4f} ms | throughput={item.throughput_ops_s:.2f} ops/s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Rust call paths")
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--base-url", default="http://127.0.0.1:50051", help="Rust HTTP service base URL")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "single":
        results = run_single_mode(args)
    else:
        results = run_batch_mode(args)

    print_results(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved results to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
