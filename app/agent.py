"""LangGraph agent. 7 nodes, 3 routers, one retry loop.

    START -> triage -> planner -> retrieve (+ tool_executor) -> synthesize -> verify
                |                                                              |
                +-> finalize <-------------------------------------------------+
                                (ungrounded, retries left -> planner)

Only planner and synthesize call the model. Everything else is plain Python.
Ask a 0.5b model "is EUR 56,000,000 in this context?" and it answers
confidently either way, so I don't ask it.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.config import MAX_SUBTASKS, SUBTASK_WORKERS, Settings, get_settings
from app.index import HybridIndex
from app.llm import LLMError, LLMProvider
from app.prompts import DECOMPOSE_SYSTEM, SYNTHESIS_SYSTEM, decompose_prompt, synthesis_prompt
from app.retrieval import build_retrieval_graph, retrieve
from app.state import (
    AgentState,
    Citation,
    Gap,
    Message,
    RetrievedChunk,
    Subtask,
    TraceEvent,
    Verification,
)
from app.tools import dispatch

# Stems, matched as substrings. "prohibited" does not contain "prohibitions",
# and that alone was refusing questions about "the prohibitions on
# unacceptable-risk AI" as off-topic.
_DOMAIN_TERMS = (
    "ai act", "artificial intelligence act", "gdpr", "data protection", "regulation",
    "article", "annex", "chapter", "recital", "compliance", "fine", "penalt", "prohibit",
    "risk", "deployer", "provider", "conformity", "notified body", "supervisory authority",
    "controller", "processor", "personal data", "biometric", "transparen", "sme", "smc",
    "turnover", "deadline", "obligat", "2024/1689", "2016/679", "2026/1744",
)

_INJECTION = re.compile(
    r"\b(ignore (all |your )?(previous|prior|above)|disregard (the )?(instructions|rules)"
    r"|you are now|system prompt|reveal your)\b",
    re.IGNORECASE,
)

_EU_BODY = r"(?:union institution\w*|eu institution\w*|edps)"
_PROHIBITION = r"(?:prohibit\w*|banned|forbidden|article 5|art\.? 5)"

# First match wins. Specific tiers before the catch-all.
_TIER_PATTERNS = (
    # Art. 100(2) needs an EU body *and* a prohibition, in either order.
    ("aia_eu_body_prohibited",
     re.compile(rf"^(?=.*\b{_EU_BODY}\b)(?=.*\b{_PROHIBITION}\b)", re.I | re.S)),
    ("aia_eu_body_other", re.compile(rf"\b{_EU_BODY}\b", re.I)),
    ("aia_prohibited", re.compile(rf"\b{_PROHIBITION}\b", re.I)),
    ("aia_misinformation",
     re.compile(r"\b(incorrect|misleading|incomplete)\w*\s+information\b", re.I)),
    ("gdpr_upper", re.compile(r"\bgdpr\b.*\b(principle|consent|data subject|transfer)\w*", re.I)),
    ("gdpr_lower", re.compile(r"\bgdpr\b", re.I)),
    ("aia_obligations", re.compile(r"\b(fine\w*|penalt\w*)\b", re.I)),
)

# Plurals spelled out; \bsme\b won't match "SMEs". The entity picks the
# comparator, so a miss here gives a clean-looking answer for the wrong company.
_ENTITY_PATTERNS = (
    ("smc", re.compile(r"\b(smcs?|small mid.?caps?|mid.?caps?)\b", re.I)),
    ("sme", re.compile(r"\b(smes?|small.{0,12}medium|start.?ups?)\b", re.I)),
    ("eu_institution", re.compile(r"\b(union institutions?|eu institutions?|edps)\b", re.I)),
)

_TURNOVER = re.compile(
    r"(?:eur\s*|€\s*)?([\d][\d,. ]{0,30})\s*(bn|billion|m|million|k|thousand)?\b.{0,24}turnover"
    r"|turnover.{0,24}?(?:eur\s*|€\s*)?([\d][\d,. ]{0,30})\s*(bn|billion|m|million|k|thousand)?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"bn": 1e9, "billion": 1e9, "m": 1e6, "million": 1e6, "k": 1e3, "thousand": 1e3}

_ASKS_PENALTY = re.compile(
    r"\b(fine\w*|penalt\w*|how much|maximum\w*|ceiling\w*|cap|caps)\b", re.I
)
# "Did the date for Annex III obligations change?" asks about timing without
# using "when". Over-triggering costs nothing: a miss returns ok=False and the
# answer falls back on the passages.
_ASKS_DEADLINE = re.compile(
    r"\b(when|deadline\w*|date\w*|how long|apply from|applicable from|come into|in force|"
    r"start to apply|starts? applying|days until|chang\w+|defer\w*|postpon\w*|delay\w*)\b",
    re.I,
)

_MARKER = re.compile(r"\[(\d+)\]")
# Grouped amounts ("EUR 35 000 000", "56,000,000") or a plain number. Groups
# after a separator must be exactly 3 digits. Without that it ran past the end
# of a sentence and glued "7 500 000." onto the "4." opening the next paragraph,
# inventing 7500004 out of nothing. Took me a while to spot.
_NUMBER = re.compile(r"\d+(?:[ ,]\d{3})+|\d+(?:\.\d+)?")


def classify_scope(question: str, history: list[Message] | None = None) -> tuple[str, str]:
    """Keyword gate. Loose on purpose.

    Rejects empty input, prompt injection, and questions obviously about
    something else. The real decision is made after retrieval, where the
    cross-encoder separates off-topic queries cleanly. Loose here and strict
    there costs one wasted retrieval on a bad question and avoids refusing a
    good one.
    """
    text = question.lower().strip()
    if not text:
        return "out_of_scope", "No question was asked."
    if _INJECTION.search(text):
        return "out_of_scope", "That asks the assistant to change its instructions."
    if any(term in text for term in _DOMAIN_TERMS):
        return "in_scope", ""
    # A follow-up may carry none of the vocabulary itself ("what about SMEs?").
    if history and any(
        term in " ".join(m.content.lower() for m in history[-4:]) for term in _DOMAIN_TERMS
    ):
        return "in_scope", ""
    return (
        "out_of_scope",
        "This assistant answers questions about the EU AI Act and the GDPR from the "
        "vendored regulation texts. This one falls outside that corpus.",
    )


def parse_turnover(text: str) -> float | None:
    match = _TURNOVER.search(text)
    if not match:
        return None
    digits = match.group(1) or match.group(3)
    if not digits:
        return None
    suffix = (match.group(2) or match.group(4) or "").lower()
    try:
        value = float(digits.replace(",", "").replace(" ", "").rstrip("."))
    except ValueError:
        return None
    return value * _MULTIPLIERS.get(suffix, 1.0)


def classify_subtask(question: str, subtask_id: int) -> Subtask:
    """Pick the tool for a sub-question, if it needs one.

    Regex, not the model. A 0.5b model asked to emit tool JSON gets argument
    names wrong often enough to matter, and the entity argument decides the
    comparator. Would revisit this with something bigger.
    """
    if _ASKS_DEADLINE.search(question) and not _ASKS_PENALTY.search(question):
        return Subtask(subtask_id, question, "both", "lookup_deadline", {"query": question})

    if _ASKS_PENALTY.search(question):
        tier = next((name for name, rx in _TIER_PATTERNS if rx.search(question)), "aia_obligations")
        entity = next((name for name, rx in _ENTITY_PATTERNS if rx.search(question)), "undertaking")
        args: dict[str, Any] = {"tier_id": tier, "entity": entity}
        turnover = parse_turnover(question)
        if turnover is not None:
            args["turnover_eur"] = turnover
        return Subtask(subtask_id, question, "both", "compute_penalty", args)

    return Subtask(subtask_id, question, "retrieve")


def check_groundedness(state: AgentState) -> Verification:
    """Check the draft only claims what the context and tools support.

    Every [n] must resolve to a passage, a non-refusal must cite at least one,
    and every number must appear in the question, a passage or a tool result.
    Numbers are what people act on, so that last check earns its keep.
    """
    draft = state.get("draft", "").strip()
    contexts = state.get("contexts", [])
    if not draft:
        return Verification(False, [Gap("no_citation", "The model produced no answer.")])

    gaps: list[Gap] = []
    markers = sorted({int(m) for m in _MARKER.findall(draft)})
    for marker in markers:
        if not 1 <= marker <= len(contexts):
            gaps.append(Gap("unresolved_citation", f"[{marker}] does not match a passage."))

    refusing = not contexts or "do not contain an answer" in draft
    if not markers and not refusing:
        gaps.append(Gap("no_citation", "The answer cites no passage."))

    tools = [result for result in state.get("tool_results", []) if result.ok]
    supported = set()
    for text in [
        state.get("question", ""),
        *(chunk.body for chunk in contexts),
        *(str(result.value) + result.summary for result in tools),
    ]:
        supported |= _canonical_numbers(text)

    drafted = _canonical_numbers(draft) - {str(m) for m in markers}
    for value in sorted(drafted - supported)[:3]:
        gaps.append(Gap("unsupported_number", f"{value} appears in no passage or tool result."))

    return Verification(not gaps, gaps)


def _canonical_numbers(text: str) -> set[str]:
    found = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").replace(" ", "")
        # Strip a float's trailing ".0" first. Otherwise the tool's own
        # 56000000.0 becomes 560000000 (nine digits) and never matches the
        # 56,000,000 it just produced.
        cleaned = re.sub(r"\.0+$", "", cleaned).replace(".", "")
        stripped = cleaned.lstrip("0") or "0"
        # Single digits are nearly always list markers or article numbers.
        if len(stripped) > 1:
            found.add(stripped)
    return found


def _timed(name: str, started: float, detail: str = "") -> dict[str, Any]:
    ms = (time.perf_counter() - started) * 1000
    return {"timings": {name: ms}, "trace": [TraceEvent(name, detail, round(ms, 1))]}


def build_graph(llm: LLMProvider, index: HybridIndex, settings: Settings | None = None):
    settings = settings or get_settings()
    rag = build_retrieval_graph(index, settings)

    def triage(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        scope, reason = classify_scope(state["question"], state.get("history"))
        return {
            "scope": scope,
            "out_of_scope_reason": reason,
            **_timed("triage", started, scope),
        }

    def planner(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        question = state["question"]
        if state.get("verification") and not state["verification"].grounded:
            gaps = "; ".join(gap.detail for gap in state["verification"].gaps[:2])
            question = f"{question} (focus on finding support for: {gaps})"

        try:
            completion = llm.complete(decompose_prompt(question), system=DECOMPOSE_SYSTEM)
            lines = [line.strip(" -•\t") for line in completion.text.splitlines() if line.strip()]
        except LLMError:
            # Decomposition is optional. The whole question works as one task.
            lines = []

        subtasks = [
            classify_subtask(line, i) for i, line in enumerate(lines[:MAX_SUBTASKS] or [question])
        ]
        detail = f"{len(subtasks)} subtask(s)"
        return {"subtasks": subtasks, **_timed("planner", started, detail)}

    def retrieve_node(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        pending = [task for task in state.get("subtasks", []) if task.needs_retrieval]

        def run(task: Subtask):
            return task, retrieve(task.question, rag, settings)

        if len(pending) > 1:
            with ThreadPoolExecutor(max_workers=SUBTASK_WORKERS) as pool:
                results = list(pool.map(run, pending))
        else:
            results = [run(task) for task in pending]

        chunks: list[RetrievedChunk] = []
        for task, result in results:
            for item in result.context.selected:
                chunk = item.hit.chunk
                chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        citation=chunk.citation,
                        celex=chunk.celex,
                        regulation=chunk.regulation,
                        body=chunk.body,
                        token_count=chunk.token_count,
                        rerank_score=item.rerank_score,
                        subtask_id=task.subtask_id,
                        is_current=chunk.is_current,
                    )
                )

        detail = f"{len(chunks)} passage(s) from {len(pending)} subtask(s)"
        return {"contexts": chunks, **_timed("retrieve", started, detail)}

    def tool_executor(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        results = [
            dispatch(task.tool_name, task.tool_args, settings)
            for task in state.get("subtasks", [])
            if task.needs_tool
        ]
        detail = ", ".join(f"{r.tool}{'' if r.ok else ' (failed)'}" for r in results)
        return {"tool_results": results, **_timed("tool_executor", started, detail)}

    def synthesize(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        contexts = state.get("contexts", [])
        context_block = "\n\n".join(
            f"[{i}] {chunk.citation}\n{chunk.body}" for i, chunk in enumerate(contexts, start=1)
        )
        tool_block = "\n".join(
            f"- {result.summary} ({result.citation})"
            for result in state.get("tool_results", [])
            if result.ok
        )
        try:
            completion = llm.complete(
                synthesis_prompt(state["question"], context_block, tool_block),
                system=SYNTHESIS_SYSTEM,
            )
            draft = completion.text.strip()
        except LLMError as exc:
            draft = ""
            return {"draft": draft, **_timed("synthesize", started, f"failed: {exc}")}
        return {"draft": draft, **_timed("synthesize", started, f"{len(draft)} chars")}

    def verify(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        verification = check_groundedness(state)
        update: dict[str, Any] = {"verification": verification}
        if not verification.grounded:
            update["retries"] = state.get("retries", 0) + 1
        detail = "grounded" if verification.grounded else f"{len(verification.gaps)} gap(s)"
        return {**update, **_timed("verify", started, detail)}

    def finalize(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        if state.get("scope") == "out_of_scope":
            reason = state.get("out_of_scope_reason", "")
            return {"answer": reason, "citations": [], **_timed("finalize", started, "refused")}

        draft = state.get("draft", "")
        contexts = state.get("contexts", [])
        citations = [
            Citation(marker, contexts[marker - 1].citation, contexts[marker - 1].chunk_id)
            for marker in sorted({int(m) for m in _MARKER.findall(draft)})
            if 1 <= marker <= len(contexts)
        ]

        answer = draft or "I could not produce an answer from the retrieved passages."
        verification = state.get("verification")
        if verification and not verification.grounded:
            answer += (
                "\n\nNote: parts of this answer could not be traced to a retrieved "
                "passage. Check it against the cited provisions before relying on it."
            )
        return {
            "answer": answer,
            "citations": citations,
            **_timed("finalize", started, f"{len(citations)} citation(s)"),
        }

    def triage_router(state: AgentState) -> Literal["planner", "finalize"]:
        return "finalize" if state.get("scope") == "out_of_scope" else "planner"

    def route_subtasks(state: AgentState) -> list[str]:
        """Returns a list so both branches can fan out in one superstep.

        Two plain edges out of planner would fire both on every question, and
        synthesize (two incoming edges) would run twice. One conditional edge
        returning a list dispatches only what the plan needs, and they join at
        synthesize once. Easy to get wrong; see test_agent.py.
        """
        subtasks = state.get("subtasks", [])
        branches = []
        if any(task.needs_retrieval for task in subtasks):
            branches.append("retrieve")
        if any(task.needs_tool for task in subtasks):
            branches.append("tool_executor")
        return branches or ["retrieve"]

    def verify_router(state: AgentState) -> Literal["planner", "finalize"]:
        verification = state.get("verification")
        if verification is None or verification.grounded:
            return "finalize"
        # verify already bumped the counter, so this terminates.
        if state.get("retries", 0) > state.get("max_retries", 1):
            return "finalize"
        return "planner"

    builder = StateGraph(AgentState)
    builder.add_node("triage", triage)
    builder.add_node("planner", planner)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("tool_executor", tool_executor)
    builder.add_node("synthesize", synthesize)
    builder.add_node("verify", verify)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", triage_router, ["planner", "finalize"])
    builder.add_conditional_edges("planner", route_subtasks, ["retrieve", "tool_executor"])
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("tool_executor", "synthesize")
    builder.add_edge("synthesize", "verify")
    builder.add_conditional_edges("verify", verify_router, ["planner", "finalize"])
    builder.add_edge("finalize", END)

    return builder.compile()


def initial_state(question: str, history=None, settings: Settings | None = None) -> AgentState:
    settings = settings or get_settings()
    return AgentState(
        question=question,
        history=history or [],
        max_retries=settings.max_retries,
        retries=0,
        contexts=[],
        tool_results=[],
        trace=[],
        timings={},
    )


def run_config(settings: Settings | None = None) -> RunnableConfig:
    """recursion_limit must be top level. Nested under `configurable` it is
    silently ignored and you get LangGraph's default of ~10007, which guards
    nothing."""
    settings = settings or get_settings()
    return RunnableConfig(recursion_limit=settings.recursion_limit)
