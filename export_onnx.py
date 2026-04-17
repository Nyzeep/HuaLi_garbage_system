from __future__ import annotations

from pathlib import Path


MODEL_PAIRS = [
    ("app/models/garbage_yolov8.pt", "app/models/garbage_yolov8.onnx"),
    ("app/models/fire_yolov8.pt", "app/models/fire_yolov8.onnx"),
    ("app/models/smoke_yolov8.pt", "app/models/smoke_yolov8.onnx"),
]


def export_model(pt_path: Path, onnx_path: Path) -> None:
    from ultralytics import YOLO

    if not pt_path.exists():
        print(f"[skip] missing model: {pt_path}")
        return

    print(f"[export] {pt_path} -> {onnx_path}")
    model = YOLO(str(pt_path))
    exported = model.export(format="onnx", opset=12, simplify=True)
    exported_path = Path(exported)
    if exported_path.resolve() != onnx_path.resolve():
        onnx_path.write_bytes(exported_path.read_bytes())
    print(f"[ok] {onnx_path}")


if __name__ == "__main__":
    for pt_file, onnx_file in MODEL_PAIRS:
        export_model(Path(pt_file), Path(onnx_file))
