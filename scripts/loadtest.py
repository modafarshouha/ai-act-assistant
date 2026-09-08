"""Latency under load, in process.

    python scripts/loadtest.py                    # 100 requests, serial
    python scripts/loadtest.py -n 100 -c 4        # four at a time
    python scripts/loadtest.py --json out.json    # plus a JSON report

Concurrency here measures contention, not capacity. The embedding and reranking
models get one ONNX thread each, so raising it mostly queues work. Throughput is
printed next to latency to make that obvious: if p95 climbs while throughput
stays flat, you are looking at a queue.
"""

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import cycle, islice
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import build_graph, initial_state, run_config  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.embed import embed_query  # noqa: E402
from app.index import HybridIndex  # noqa: E402
from app.llm import get_llm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Sample:
    total_ms: float
    node_ms: dict[str, float]
    ok: bool


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered) + 0.5) - 1))
    return ordered[index]


def run_one(graph, question: str, settings) -> Sample:
    started = time.perf_counter()
    try:
        result = graph.invoke(initial_state(question, None, settings), run_config(settings))
    except Exception:
        return Sample((time.perf_counter() - started) * 1000, {}, False)
    return Sample(
        (time.perf_counter() - started) * 1000, dict(result.get("timings", {})), True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--requests", type=int, default=100)
    parser.add_argument("-c", "--concurrency", type=int, default=1)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    settings = get_settings()
    cases = yaml.safe_load((ROOT / "eval" / "questions.yaml").read_text(encoding="utf-8"))
    pool = [
        case["question"]
        for case in cases["functional"]
        if case.get("expects", {}).get("scope") != "out_of_scope"
    ]
    questions = list(islice(cycle(pool), args.requests))

    index = HybridIndex.load(settings.index_dir, settings)
    graph = build_graph(get_llm(settings), index, settings)

    # Warm the model and query cache, or sample 1 is an outlier.
    embed_query("warm up")
    run_one(graph, pool[0], settings)

    print(
        f"{args.requests} requests at concurrency {args.concurrency}, "
        f"provider {settings.llm_provider}"
    )
    started = time.perf_counter()
    if args.concurrency == 1:
        samples = [run_one(graph, q, settings) for q in questions]
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            samples = list(executor.map(lambda q: run_one(graph, q, settings), questions))
    wall = time.perf_counter() - started

    ok = [s for s in samples if s.ok]
    latencies = [s.total_ms for s in ok]
    failures = len(samples) - len(ok)

    nodes: dict[str, float] = {}
    for sample in ok:
        for node, ms in sample.node_ms.items():
            nodes[node] = nodes.get(node, 0.0) + ms
    node_mean = {node: round(total / len(ok), 1) for node, total in nodes.items()}
    node_mean = dict(sorted(node_mean.items(), key=lambda item: -item[1]))

    mean_total = statistics.mean(latencies)
    print(f"\nthroughput  {len(ok) / wall:.2f} req/s over {wall:.1f}s, {failures} failures")
    print("latency ms")
    for label, pct in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99)):
        print(f"  {label}  {percentile(latencies, pct):8.1f}")
    print(f"  max   {max(latencies):8.1f}")
    print("\nmean ms per node")
    for node, ms in node_mean.items():
        print(f"  {node:<14} {ms:8.1f}   {ms / mean_total:5.1%}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "provider": settings.llm_provider,
                    "requests": len(samples),
                    "concurrency": args.concurrency,
                    "failures": failures,
                    "throughput_rps": round(len(ok) / wall, 3),
                    "latency_ms": {
                        label: round(percentile(latencies, pct), 1)
                        for label, pct in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))
                    }
                    | {"max": round(max(latencies), 1)},
                    "node_mean_ms": node_mean,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
