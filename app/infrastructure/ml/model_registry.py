from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.ml.backends import InferenceBackend


@dataclass
class ModelDescriptor:
    key: str
    onnx_path: Path
    pt_path: Path
    class_mapping: dict[int, int]


@dataclass
class ModelBundle:
    descriptor: ModelDescriptor
    backend: InferenceBackend | None = None

    @property
    def loaded(self) -> bool:
        return bool(self.backend and self.backend.loaded)


class ModelRegistry:
    def __init__(self) -> None:
        self._bundles: dict[str, ModelBundle] = {}

    def register(self, descriptor: ModelDescriptor, backend: InferenceBackend | None) -> None:
        self._bundles[descriptor.key] = ModelBundle(descriptor=descriptor, backend=backend)

    def get(self, key: str) -> ModelBundle | None:
        return self._bundles.get(key)

    def items(self) -> list[ModelBundle]:
        return list(self._bundles.values())

    def loaded_map(self) -> dict[str, bool]:
        return {key: bundle.loaded for key, bundle in self._bundles.items()}
