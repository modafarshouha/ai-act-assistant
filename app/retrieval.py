"""RAG subgraph: embed, search, rerank, select, with one widening retry.

Its own graph because the stages carry state the agent wants afterwards (what
was retrieved, what got scored, what was dropped), and because select can send
the search back for a wider second pass. Without that branch it would just be a
function. The agent's retrieve node invokes this and maps the result across.
"""

import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.config import (
    MAX_CHUNKS_PER_PROVISION,
    OUT_OF_SCOPE_SCORE,
    RELEVANCE_THRESHOLD,
    Settings,
    get_settings,
)
from app.embed import embed_query
from app.index import Hit, HybridIndex
from app.rerank import Ranked, rerank


@dataclass(frozen=True)
class Context:
    selected: list[Ranked]
    out_of_scope: bool
    dropped_by_score: int
    dropped_by_budget: int

    @property
    def citations(self) -> list[str]:
        seen: list[str] = []
        for item in self.selected:
            if item.hit.chunk.citation not in seen:
                seen.append(item.hit.chunk.citation)
        return seen

    def as_prompt_block(self) -> str:
        return "\n\n".join(
            f"[{position}] {item.hit.chunk.citation}\n{item.hit.chunk.body}"
            for position, item in enumerate(self.selected, start=1)
        )


def build_context(ranked: list[Ranked], settings: Settings | None = None) -> Context:
    settings = settings or get_settings()
    if not ranked:
        return Context([], True, 0, 0)

    # Off-topic questions score about -11 against every candidate; the weakest
    # in-scope one sits near -3. Wide enough to refuse on, with no LLM call.
    if max(item.rerank_score for item in ranked) < OUT_OF_SCOPE_SCORE:
        return Context([], True, len(ranked), 0)

    selected: list[Ranked] = []
    per_provision: dict[tuple[str, str, str], int] = {}
    dropped_score = 0
    dropped_budget = 0
    tokens = 0

    for item in ranked:
        if item.rerank_score < RELEVANCE_THRESHOLD:
            dropped_score += 1
            continue
        chunk = item.hit.chunk
        key = (chunk.source_key, chunk.kind, chunk.number)
        if per_provision.get(key, 0) >= MAX_CHUNKS_PER_PROVISION:
            continue
        if tokens + chunk.token_count > settings.max_context_tokens:
            dropped_budget += 1
            continue
        selected.append(item)
        per_provision[key] = per_provision.get(key, 0) + 1
        tokens += chunk.token_count

    # In scope, but everything under the threshold. Refusing here would be
    # wrong, so take the best one and let groundedness decide.
    if not selected:
        selected = [ranked[0]]

    return Context(selected, False, dropped_score, dropped_budget)


class RetrievalState(TypedDict, total=False):
    question: str
    include_superseded: bool
    vector: np.ndarray
    fused: list[Hit]
    ranked: list[Ranked]
    context: Context
    widened: bool
    retry: bool
    timings: Annotated[dict[str, float], lambda a, b: {**a, **b}]


def _timed(name: str, started: float) -> dict[str, float]:
    return {name: (time.perf_counter() - started) * 1000}


def build_retrieval_graph(index: HybridIndex, settings: Settings | None = None):
    settings = settings or get_settings()

    def embed(state: RetrievalState) -> dict[str, Any]:
        started = time.perf_counter()
        return {
            "vector": embed_query(state["question"], settings),
            "timings": _timed("embed", started),
        }

    def search(state: RetrievalState) -> dict[str, Any]:
        started = time.perf_counter()
        # Second pass widens the net instead of reshuffling. If the answer was
        # outside the first pool, reordering it won't help.
        spread = 3 if state.get("widened") else 1
        dense = index.search_dense(state["vector"], settings.dense_k * 2 * spread)
        sparse = index.search_sparse(state["question"], settings.sparse_k * 2 * spread)
        fused = index.fuse(
            dense,
            sparse,
            settings.fused_k * spread,
            include_superseded=state.get("include_superseded"),
        )
        return {"fused": fused, "timings": _timed("search", started)}

    def rerank_node(state: RetrievalState) -> dict[str, Any]:
        started = time.perf_counter()
        ranked = rerank(state["question"], state["fused"], settings)
        return {"ranked": ranked, "timings": _timed("rerank", started)}

    def select(state: RetrievalState) -> dict[str, Any]:
        started = time.perf_counter()
        context = build_context(state["ranked"], settings)
        update: dict[str, Any] = {"context": context, "timings": _timed("select", started)}
        if not context.selected and not context.out_of_scope and not state.get("widened"):
            update["widened"] = True
            update["retry"] = True
        else:
            update["retry"] = False
        return update

    def route(state: RetrievalState) -> Literal["search", "__end__"]:
        # widened is set in the same update as retry, and a conditional edge
        # sees the write its own node just made, so this fires at most once.
        return "search" if state.get("retry") else END

    builder = StateGraph(RetrievalState)
    builder.add_node("embed", embed)
    builder.add_node("search", search)
    builder.add_node("rerank", rerank_node)
    builder.add_node("select", select)

    builder.add_edge(START, "embed")
    builder.add_edge("embed", "search")
    builder.add_edge("search", "rerank")
    builder.add_edge("rerank", "select")
    builder.add_conditional_edges("select", route, ["search", END])

    return builder.compile()


@dataclass(frozen=True)
class RetrievalResult:
    context: Context
    ranked: list[Ranked]
    fused: list[Hit]
    timings: dict[str, float]
    widened: bool

    @property
    def out_of_scope(self) -> bool:
        return self.context.out_of_scope

    @property
    def citations(self) -> list[str]:
        return self.context.citations


def retrieve(
    question: str,
    graph,
    settings: Settings | None = None,
    include_superseded: bool | None = None,
) -> RetrievalResult:
    """Run the subgraph for one question.

    Config is passed explicitly. The agent fans out over subtasks in a thread
    pool and ThreadPoolExecutor.submit doesn't carry contextvars, so without
    this the subgraph quietly gets LangGraph's default limit of 10007 and a
    routing bug hangs instead of erroring.
    """
    settings = settings or get_settings()
    final = graph.invoke(
        {
            "question": question,
            "include_superseded": include_superseded,
            "widened": False,
            "timings": {},
        },
        RunnableConfig(recursion_limit=settings.recursion_limit),
    )
    return RetrievalResult(
        context=final["context"],
        ranked=final.get("ranked", []),
        fused=final.get("fused", []),
        timings=final.get("timings", {}),
        widened=bool(final.get("widened")),
    )
