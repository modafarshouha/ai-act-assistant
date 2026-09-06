"""Cross-encoder reranking of the fused candidate pool."""

from dataclasses import dataclass

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.config import RERANK_CANDIDATES, RERANK_MAX_CHARS, RERANK_MODEL, Settings, get_settings
from app.index import Hit

_reranker: TextCrossEncoder | None = None


def get_reranker() -> TextCrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL, threads=4)
    return _reranker


@dataclass(frozen=True)
class Ranked:
    hit: Hit
    rerank_score: float
    fused_rank: int


def scoring_text(hit: Hit, max_chars: int = RERANK_MAX_CHARS) -> str:
    """Truncate the body for scoring, but always keep the header.

    The header holds the citation, and that is most of what tells the model
    whether a passage is even about the right provision. Only the score sees
    the shortened text; synthesis still gets the full chunk.
    """
    header = hit.chunk.header
    room = max_chars - len(header) - 2
    if room <= 0:
        return header[:max_chars]
    return f"{header}\n\n{hit.chunk.body[:room]}"


def rerank(
    query: str,
    hits: list[Hit],
    settings: Settings | None = None,
    top_n: int | None = None,
) -> list[Ranked]:
    settings = settings or get_settings()
    if not hits:
        return []

    candidates = hits[:RERANK_CANDIDATES]
    scores = list(get_reranker().rerank(query, [scoring_text(hit) for hit in candidates]))
    ranked = [
        Ranked(hit=hit, rerank_score=float(score), fused_rank=position)
        for position, (hit, score) in enumerate(zip(candidates, scores, strict=True), start=1)
    ]
    ranked.sort(key=lambda item: (-item.rerank_score, item.fused_rank))
    return ranked[: top_n if top_n is not None else settings.rerank_top_n]
