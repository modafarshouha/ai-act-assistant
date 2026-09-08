"""Hybrid index: FAISS for dense retrieval, BM25 for sparse, fused by RRF."""

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunker import Chunk
from app.config import EMBEDDING_DIM, RECITAL_WEIGHT, RRF_K, Settings, get_settings

# Keeps "6(2)" and "2016/679" intact. Splitting on every non-alphanumeric
# shreds the identifiers people actually search by.
_TOKEN = re.compile(r"\d+\(\d+[a-z]?\)|[a-z0-9]+(?:/[a-z0-9]+)+|[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class HybridIndex:
    def __init__(self, chunks: list[Chunk], dense, sparse, settings: Settings | None = None):
        self.chunks = chunks
        self.dense = dense
        self.sparse = sparse
        self.settings = settings or get_settings()

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(cls, chunks, vectors, settings=None) -> "HybridIndex":
        settings = settings or get_settings()
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        # Vectors are unit length, so inner product is cosine similarity.
        dense = faiss.IndexFlatIP(EMBEDDING_DIM)
        dense.add(np.asarray(vectors, dtype=np.float32))
        sparse = BM25Okapi([tokenize(chunk.text) for chunk in chunks])
        return cls(chunks, dense, sparse, settings)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.dense, str(directory / "dense.faiss"))
        (directory / "sparse.pkl").write_bytes(pickle.dumps(self.sparse))
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                record = chunk.metadata()
                record["header"] = chunk.header
                record["body"] = chunk.body
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: Path, settings=None) -> "HybridIndex":
        settings = settings or get_settings()
        missing = [
            name
            for name in ("dense.faiss", "sparse.pkl", "chunks.jsonl")
            if not (directory / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"index incomplete, missing {missing}; run scripts/ingest.py")

        chunks = []
        with (directory / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                text = f"{record['header']}\n\n{record['body']}"
                chunks.append(
                    Chunk(
                        chunk_id=record["chunk_id"],
                        source_key=record["source_key"],
                        celex=record["celex"],
                        regulation=record["regulation"],
                        kind=record["kind"],
                        number=record["number"],
                        title=record["title"],
                        paragraphs=tuple(record["paragraphs"]),
                        citation=record["citation"],
                        header=record["header"],
                        body=record["body"],
                        text=text,
                        token_count=record["token_count"],
                        amended=record["amended"],
                        is_current=record["is_current"],
                        content_hash=record["content_hash"],
                    )
                )

        dense = faiss.read_index(str(directory / "dense.faiss"))
        sparse = pickle.loads((directory / "sparse.pkl").read_bytes())
        return cls(chunks, dense, sparse, settings)

    def search_dense(self, vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        scores, ids = self.dense.search(np.asarray([vector], dtype=np.float32), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]

    def search_sparse(self, text: str, k: int) -> list[tuple[int, float]]:
        scores = self.sparse.get_scores(tokenize(text))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def fuse(self, dense, sparse, k: int, include_superseded: bool | None = None) -> list[Hit]:
        """Reciprocal rank fusion, then the currency and recital policies.

        RRF because BM25 scores are unbounded and cosine sits in [-1, 1], so a
        weighted sum needs a weight tuned per corpus. Ranks need no calibration.
        """
        if include_superseded is None:
            include_superseded = not self.settings.prefer_current_law

        scores: dict[int, float] = {}
        dense_rank: dict[int, int] = {}
        sparse_rank: dict[int, int] = {}
        for rank, (row, _) in enumerate(dense, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + rank)
            dense_rank[row] = rank
        for rank, (row, _) in enumerate(sparse, start=1):
            scores[row] = scores.get(row, 0.0) + 1.0 / (RRF_K + rank)
            sparse_rank[row] = rank

        adjusted: dict[int, float] = {}
        for row, score in scores.items():
            chunk = self.chunks[row]
            # Quoting a repealed date as current law is a correctness bug, so
            # drop these outright. Demoting them is not enough.
            if not include_superseded and not chunk.is_current:
                continue
            if chunk.kind == "recital":
                score *= RECITAL_WEIGHT
            adjusted[row] = score

        ordered = sorted(adjusted.items(), key=lambda item: (-item[1], item[0]))
        return [
            Hit(
                chunk=self.chunks[row],
                score=score,
                dense_rank=dense_rank.get(row),
                sparse_rank=sparse_rank.get(row),
            )
            for row, score in ordered[:k]
        ]
