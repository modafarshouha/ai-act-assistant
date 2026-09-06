"""Agent state and the reducers that merge it across parallel branches."""

import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

Scope = Literal["in_scope", "out_of_scope"]
SubtaskKind = Literal["retrieve", "tool", "both"]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class Subtask:
    subtask_id: int
    question: str
    kind: SubtaskKind = "retrieve"
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)

    @property
    def needs_retrieval(self) -> bool:
        return self.kind in ("retrieve", "both")

    @property
    def needs_tool(self) -> bool:
        return self.kind in ("tool", "both") and self.tool_name is not None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    citation: str
    celex: str
    regulation: str
    body: str
    token_count: int
    rerank_score: float
    subtask_id: int
    is_current: bool


@dataclass(frozen=True)
class Gap:
    kind: Literal["unsupported_number", "unresolved_citation", "no_citation"]
    detail: str


@dataclass(frozen=True)
class Verification:
    grounded: bool
    gaps: list[Gap] = field(default_factory=list)


@dataclass(frozen=True)
class Citation:
    marker: int
    citation: str
    chunk_id: str


@dataclass(frozen=True)
class TraceEvent:
    node: str
    detail: str = ""
    duration_ms: float = 0.0


def dedup_chunks(existing: list[RetrievedChunk], incoming: list[RetrievedChunk]):
    """Merge by chunk id, keeping the better score.

    Retrieval fans out over subtasks and runs again on a retry, so the same
    passage arrives more than once. Appending blindly repeats it in the prompt
    and pays for it twice out of the context budget.
    """
    merged: dict[str, RetrievedChunk] = {}
    for chunk in [*existing, *incoming]:
        current = merged.get(chunk.chunk_id)
        if current is None or chunk.rerank_score > current.rerank_score:
            merged[chunk.chunk_id] = chunk
    return sorted(merged.values(), key=lambda chunk: -chunk.rerank_score)


def dedup_tool_results(existing: list[Any], incoming: list[Any]):
    """Merge tool results by tool name and arguments; the newest wins."""
    merged: dict[tuple[str, str], Any] = {}
    for result in [*existing, *incoming]:
        key = (result.tool, json.dumps(result.args, sort_keys=True, default=str))
        merged[key] = result
    return list(merged.values())


def sum_timings(existing: dict[str, float], incoming: dict[str, float]):
    """Add instead of overwriting, so a node that reran shows its total."""
    merged = dict(existing)
    for node, ms in incoming.items():
        merged[node] = merged.get(node, 0.0) + ms
    return merged


class AgentState(TypedDict, total=False):
    # From the caller.
    question: str
    history: list[Message]
    max_retries: int

    # Written by nodes.
    scope: Scope
    out_of_scope_reason: str
    subtasks: list[Subtask]
    contexts: Annotated[list[RetrievedChunk], dedup_chunks]
    tool_results: Annotated[list[Any], dedup_tool_results]
    draft: str
    verification: Verification
    retries: int
    answer: str
    citations: list[Citation]

    trace: Annotated[list[TraceEvent], operator.add]
    timings: Annotated[dict[str, float], sum_timings]
