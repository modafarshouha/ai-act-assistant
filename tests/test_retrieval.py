import pytest

from app.rerank import Ranked
from app.retrieval import Context, build_context, build_retrieval_graph, retrieve


@pytest.fixture(scope="module")
def rag(index, settings):
    return build_retrieval_graph(index, settings)


def test_the_subgraph_runs_every_stage(rag, settings):
    result = retrieve("maximum fine for a prohibited AI practice", rag, settings)
    assert set(result.timings) == {"embed", "search", "rerank", "select"}
    assert result.fused and result.ranked and result.context.selected


def test_a_normal_question_does_not_widen(rag, settings):
    result = retrieve("obligations of deployers of high-risk AI systems", rag, settings)
    assert not result.widened


def test_an_off_topic_question_is_marked_out_of_scope(rag, settings):
    result = retrieve("what is the capital of France", rag, settings)
    assert result.out_of_scope
    assert result.context.selected == []


def test_widening_happens_once_and_then_stops(rag, settings, monkeypatch):
    """Force selection to reject everything and check search reran once.

    This is the loop that hangs if `widened` is not written in the same update
    that triggers the routing decision.
    """
    import app.retrieval as retrieval

    calls = {"n": 0}

    def never_selects(ranked, settings=None):
        calls["n"] += 1
        return Context([], False, len(ranked), 0)

    monkeypatch.setattr(retrieval, "build_context", never_selects)
    result = retrieve("obligations of providers", rag, settings)
    assert result.widened
    assert calls["n"] == 2


def test_out_of_scope_beats_the_relevance_threshold(index, settings):
    """A very low top score means the corpus does not cover the question."""
    hit = type("H", (), {"chunk": index.chunks[0]})()
    ranked = [Ranked(hit=hit, rerank_score=-11.0, fused_rank=1)]
    assert build_context(ranked, settings).out_of_scope


def test_a_weak_but_in_scope_result_is_still_answered(index, settings):
    """In scope, but nothing clears the threshold. Answer from the best hit."""
    hit = type("H", (), {"chunk": index.chunks[0]})()
    ranked = [Ranked(hit=hit, rerank_score=-1.0, fused_rank=1)]
    context = build_context(ranked, settings)
    assert not context.out_of_scope
    assert len(context.selected) == 1


def test_the_token_budget_is_respected(index, settings):
    hits = [type("H", (), {"chunk": chunk})() for chunk in index.chunks[:40]]
    ranked = [Ranked(hit=h, rerank_score=5.0, fused_rank=i) for i, h in enumerate(hits, 1)]
    context = build_context(ranked, settings)
    total = sum(item.hit.chunk.token_count for item in context.selected)
    assert total <= settings.max_context_tokens
