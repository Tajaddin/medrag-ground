"""Pluggable embedders — fastembed when available, hash trigram as zero-dep fallback."""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: Iterable[str]) -> np.ndarray: ...


class HashEmbedder:
    """Character-trigram hash embedder. Deterministic, zero-dep, structural."""

    def __init__(self, dim: int = 384) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        out = []
        for t in texts:
            out.append(self._one(t))
        arr = np.stack(out).astype(np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        s = text.lower()
        for i in range(max(1, len(s) - 2)):
            tri = s[i : i + 3]
            h = int(hashlib.blake2s(tri.encode("utf-8"), digest_size=8).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
        return vec


class FastEmbedEmbedder:
    """fastembed-backed BGE-small (384-dim ONNX, no PyTorch)."""

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "fastembed is not installed. Install with `pip install medrag-ground[fastembed]` "
                "or use HashEmbedder."
            ) from exc
        self.model_name = model_name or os.environ.get("FASTEMBED_MODEL", self.DEFAULT_MODEL)
        self._model = TextEmbedding(model_name=self.model_name)
        probe = next(self._model.embed(["dim probe"]))
        self.dim = len(probe)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        if not texts_list:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.stack(list(self._model.embed(texts_list))).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


def get_embedder(name: str | None = None, *, dim: int = 384) -> Embedder:
    name = (name or os.environ.get("MEDRAG_EMBEDDER") or "").lower().strip()
    if name == "hash":
        return HashEmbedder(dim=dim)
    if name == "fastembed":
        return FastEmbedEmbedder()
    if not name:
        try:
            return FastEmbedEmbedder()
        except RuntimeError:
            return HashEmbedder(dim=dim)
    raise ValueError(f"Unknown embedder: {name!r}")
