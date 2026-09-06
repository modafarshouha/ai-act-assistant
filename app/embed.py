"""Embeddings via fastembed's ONNX runtime."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

from app.config import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_FILE,
    EMBEDDING_SOURCE_REPO,
    QUERY_INSTRUCTION,
    Settings,
    get_settings,
)

_MODELS: dict[int, TextEmbedding] = {}
_registered = False


def register_model(settings: Settings | None = None) -> None:
    """Register the fp32 export of BGE-small.

    fastembed's built-in "BAAI/bge-small-en-v1.5" resolves to an int8 build that
    runs about 8x slower on this CPU; onnxruntime has no VNNI kernel here and
    drops to a reference integer path. fp32 agrees with it to cosine 0.999998,
    so this is free speed. Pooling and normalisation have to be spelled out
    because fastembed can't infer them for a model it doesn't know.
    """
    global _registered
    if _registered:
        return
    settings = settings or get_settings()
    known = {m["model"] for m in TextEmbedding.list_supported_models()}
    if settings.embedding_model not in known:
        TextEmbedding.add_custom_model(
            model=settings.embedding_model,
            pooling=PoolingType.CLS,
            normalization=True,
            sources=ModelSource(hf=EMBEDDING_SOURCE_REPO),
            dim=EMBEDDING_DIM,
            model_file=EMBEDDING_MODEL_FILE,
        )
    _registered = True


def get_model(settings: Settings | None = None, threads: int = 1) -> TextEmbedding:
    settings = settings or get_settings()
    if threads not in _MODELS:
        register_model(settings)
        _MODELS[threads] = TextEmbedding(model_name=settings.embedding_model, threads=threads)
    return _MODELS[threads]


def embed_documents(
    texts: list[str],
    settings: Settings | None = None,
    threads: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed passages for indexing. No prefix; BGE only wants one on queries."""
    settings = settings or get_settings()
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model(settings, threads)

    # Longest first, then put the order back. A batch pads to its longest
    # member, so pairing a 30-token chunk with a 411-token one costs 411 twice.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    collected: list[np.ndarray] = []
    for start in range(0, len(order), 64):
        batch = [texts[i] for i in order[start : start + 64]]
        collected.extend(model.embed(batch, batch_size=len(batch)))
        if on_progress:
            on_progress(len(collected), len(order))

    embedded = np.asarray(collected, dtype=np.float32)
    vectors = np.empty_like(embedded)
    vectors[order] = embedded

    norms = np.linalg.norm(vectors, axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("embeddings are not unit length; the IP index would rank by magnitude")
    return vectors


@lru_cache(maxsize=512)
def _embed_query_cached(text: str) -> np.ndarray:
    settings = get_settings()
    model = get_model(settings)
    vector = next(iter(model.embed([f"{QUERY_INSTRUCTION}{text}"], batch_size=1)))
    return np.asarray(vector, dtype=np.float32)


def embed_query(text: str, settings: Settings | None = None) -> np.ndarray:
    # Copy, or a caller mutating the result corrupts the cache.
    return _embed_query_cached(text).copy()
