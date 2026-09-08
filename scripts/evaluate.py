"""Retrieval and graph metrics against eval/questions.yaml.

    python scripts/evaluate.py
    python scripts/evaluate.py --answers      # needs LLM_PROVIDER=ollama
    python scripts/evaluate.py --json out.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_graph, initial_state, run_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.index import HybridIndex  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.retrieval import build_retrieval_graph, retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class Metric:
    def __init__(self, name: str):
        self.name = name
        self.hits = 0
        self.total = 0
        self.misses: list[str] = []

    def record(self, ok: bool, label: str) -> None:
        self.total += 1
        if ok:
            self.hits += 1
        else:
            self.misses.append(label)

    @property
    def rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {"hits": self.hits, "total": self.total, "rate": round(self.rate, 4)}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def evaluate_retrieval(cases, index, settings) -> dict:
    graph = build_retrieval_graph(index, settings)
    at_1, at_5, at_20 = Metric("recall@1"), Metric("recall@5"), Metric("recall@20")
    # Fused recall@1 is the RRF pool. Reranked recall@1 is what the answer
    # actually gets built from. The gap between them is what the reranker buys.
    reranked_1 = Metric("reranked recall@1")
    reciprocal: list[float] = []
    latencies: list[float] = []
    with_filter = 0
    without_filter = 0

    for case in cases:
        query, relevant = case["query"], case["relevant"]
        started = time.perf_counter()
        result = retrieve(query, graph, settings)
        latencies.append((time.perf_counter() - started) * 1000)

        ranked = [hit.chunk.citation for hit in result.fused]
        rank = next(
            (i + 1 for i, c in enumerate(ranked) if any(c.startswith(p) for p in relevant)), None
        )
        at_1.record(rank == 1, case["id"])
        at_5.record(rank is not None and rank <= 5, case["id"])
        at_20.record(rank is not None, f"{case['id']}: {query[:44]}")
        reciprocal.append(1.0 / rank if rank else 0.0)

        top = result.ranked[0].hit.chunk.citation if result.ranked else ""
        reranked_1.record(any(top.startswith(p) for p in relevant), case["id"])

        with_filter += sum(1 for hit in result.fused if not hit.chunk.is_current)
        # Same query, currency filter off. Counting superseded chunks in the
        # normal pool always gives zero because fuse drops them first, so you
        # need both numbers for the figure to mean anything.
        relaxed = retrieve(query, graph, settings, include_superseded=True)
        without_filter += sum(1 for hit in relaxed.fused if not hit.chunk.is_current)

    return {
        "metrics": [at_1, at_5, at_20, reranked_1],
        "mrr": round(statistics.mean(reciprocal), 4) if reciprocal else 0.0,
        "superseded_with_filter": with_filter,
        "superseded_without_filter": without_filter,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }


def evaluate_functional(cases, graph, settings, score_answers: bool) -> dict:
    scope = Metric("scope classification")
    selection = Metric("tool selection")
    arguments = Metric("tool arguments")
    citations = Metric("citation correctness")
    decomposition = Metric("multi-hop decomposition")
    grounded = Metric("verified grounded")
    answers = Metric("answer contains expected")
    latencies: list[float] = []

    for case in cases:
        expects = case.get("expects", {})
        started = time.perf_counter()
        result = graph.invoke(initial_state(case["question"], None, settings), run_config(settings))
        latencies.append((time.perf_counter() - started) * 1000)

        if "scope" in expects:
            scope.record(result.get("scope") == expects["scope"], case["id"])

        called = [r.tool for r in result.get("tool_results", []) if r.ok]
        if "tool" in expects:
            wanted = expects["tool"]
            selection.record(wanted in called if wanted else not called, case["id"])

        if expects.get("tool_args"):
            actual = next(
                (t.tool_args for t in result.get("subtasks", []) if t.tool_name == expects["tool"]),
                {},
            )
            arguments.record(
                all(actual.get(k) == v for k, v in expects["tool_args"].items()), case["id"]
            )

        if expects.get("citations_any"):
            found = [c.citation for c in result.get("citations", [])]
            citations.record(
                any(c.startswith(p) for c in found for p in expects["citations_any"]), case["id"]
            )

        if expects.get("min_subtasks"):
            decomposition.record(
                len(result.get("subtasks", [])) >= expects["min_subtasks"], case["id"]
            )

        verification = result.get("verification")
        if verification is not None:
            grounded.record(verification.grounded, case["id"])

        if score_answers and (expects.get("contains_any") or expects.get("contains_none")):
            answer = result.get("answer", "")
            ok = all(
                [
                    not expects.get("contains_any")
                    or any(s in answer for s in expects["contains_any"]),
                    not expects.get("contains_none")
                    or all(s not in answer for s in expects["contains_none"]),
                ]
            )
            answers.record(ok, case["id"])

    metrics = [scope, selection, arguments, citations, decomposition, grounded]
    if score_answers:
        metrics.append(answers)
    return {
        "metrics": metrics,
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(percentile(latencies, 95), 1),
    }


def report(title: str, section: dict) -> None:
    print(f"\n{title}")
    for metric in section["metrics"]:
        if metric.total:
            print(f"  {metric.name:<28} {metric.hits}/{metric.total}  {metric.rate:6.1%}")
            for miss in metric.misses[:4]:
                print(f"      miss: {miss}")
    for key, value in section.items():
        if key != "metrics":
            print(f"  {key:<28} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", action="store_true", help="also score answer text")
    parser.add_argument("--json", type=Path, help="write metrics as JSON")
    args = parser.parse_args()

    settings = get_settings()
    if args.answers and settings.llm_provider == "dummy":
        raise SystemExit(
            "answer scoring needs a real provider. The default quotes passages rather "
            "than writing prose, so scoring it would produce a number that looks like a "
            "result and means nothing. Set LLM_PROVIDER=ollama."
        )

    cases = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    index = HybridIndex.load(settings.index_dir, settings)
    graph = build_graph(get_llm(settings), index, settings)

    print(f"provider={settings.llm_provider} chunks={len(index)}")
    retrieval = evaluate_retrieval(cases["retrieval"], index, settings)
    report("retrieval", retrieval)
    functional = evaluate_functional(cases["functional"], graph, settings, args.answers)
    report("graph", functional)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": settings.llm_provider,
            "model": get_llm(settings).model,
            "retrieval": {
                **{m.name: m.as_dict() for m in retrieval["metrics"]},
                **{k: v for k, v in retrieval.items() if k != "metrics"},
            },
            "graph": {
                **{m.name: m.as_dict() for m in functional["metrics"] if m.total},
                **{k: v for k, v in functional.items() if k != "metrics"},
            },
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
