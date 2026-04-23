from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.ml.backends import InferenceBackend


@dataclass(frozen=True)
class ModelDescriptor:
    key: str
    onnx_path: Path
    pt_path: Path
    class_mapping: dict[int, int]


@dataclass
class ModelBundle:
    descriptor: ModelDescriptor
    backend: InferenceBackend | None

    @property
    def loaded(self) -> bool:
        return bool(self.backend is not None and self.backend.loaded)


class ModelRegistry:
    def __init__(self) -> None:
        self._bundles: list[ModelBundle] = []
        self._bundle_index_by_key: dict[str, int] = {}

    def register(self, descriptor: ModelDescriptor, backend: InferenceBackend | None) -> None:
        bundle = ModelBundle(descriptor=descriptor, backend=backend)
        index = self._bundle_index_by_key.get(descriptor.key)
        if index is None:
            self._bundle_index_by_key[descriptor.key] = len(self._bundles)
            self._bundles.append(bundle)
            return
        self._bundles[index] = bundle

    def get(self, key: str) -> ModelBundle | None:
        """Return the model bundle for ``key``.

        Args:
            key: Registered model key.

        Returns:
            The matching ``ModelBundle`` when found; otherwise ``None``.

        Raises:
            None.
        """
        index = self._bundle_index_by_key.get(key)
        if index is None:
            return None
        return self._bundles[index]

    def items(self) -> list[ModelBundle]:
        return list(self._bundles)

    def loaded_map(self) -> dict[str, bool]:
        return {bundle.descriptor.key: bundle.loaded for bundle in self._bundles}
