# EU AI Act & GDPR compliance assistant

An agentic RAG system that answers compliance questions from the regulation
texts, computes statutory fines deterministically, and shows its working.

## The problem

**Problem statement.** The AI Act carries fines of up to 7% of global turnover,
and its obligations phase in on dates that moved in 2026. Regulation (EU)
2026/1744, the Digital Omnibus, amended the AI Act on 27 July 2026. It pushed
the Annex III high-risk obligations from 2 August 2026 out to 2 December 2027,
and inserted a fine cap for small mid-cap enterprises that had not existed
before. A model trained before those amendments answers with the old dates.

**Users.** Compliance officers, product lawyers, and engineers.

**Agent vs. plain RAG.** A question often contains sub-questions that different
provisions answer: "what is the fine for this, and when does the obligation
start?" Those need splitting and researching separately. Parts of the answer are
also not writing tasks at all. A fine ceiling is arithmetic over a legal
comparator. The workflow therefore routes between retrieval and tools.

### A worked example

```
"What fine can an SMC face for a prohibited practice with EUR 800m turnover?"

  → EUR 56,000,000  (AI Act Art. 99(3))
```

Art. 99(3) caps a fine for a prohibited practice at **EUR 35,000,000 or 7% of
worldwide annual turnover, whichever is higher**. Here 7% of EUR 800m is EUR
56m (56m > the fixed 35m), so EUR 56m is the ceiling.

Smaller companies normally get relief, and the two relief provisions differ:

- Art. 99(6) flips that comparison to *whichever is lower* for an SME (small and
  medium-sized enterprise). On identical facts an SME is capped at EUR 35m.
- Art. 99(6a), inserted by the Digital Omnibus, grants the same relief to an SMC
  (small mid-cap enterprise, a size band above SME), but it names paragraphs 4
  and 5 only. Paragraph 3 is left out on purpose, so an SMC committing a
  prohibited practice gets no relief and stays on higher-of: EUR 56m.

Same facts, EUR 21 million apart, decided by which paragraph numbers one
subsection lists. Art. 99(6a) did not exist when the model was trained. The
system reads it from the amended text and a calculator produces the figure.

**Objective.** Answer a compliance question about the AI Act or the GDPR from
the current text, cite the provision it came from, compute any fine with a
calculator, and show every step. Refuse when the corpus does not cover the
question.

## Quick start

Needs Python 3.12 or newer. No API key and no paid service.

```bash
python -m venv .venv            # python3 on most Linux distributions

# one of these, for your shell:
source .venv/bin/activate       # Linux, WSL, macOS
source .venv/Scripts/activate   # Windows, Git Bash
.venv\Scripts\activate          # Windows, cmd.exe or PowerShell

pip install -r requirements-dev.txt
python scripts/ingest.py        # builds the index, ~9 min, once
streamlit run app/ui.py         # http://localhost:8501
```

The first `ingest.py` run downloads the ~130 MB embedding model; the ~90 MB
reranker is fetched on the first question. After both, nothing touches the
network.

Or skip all of that and use Docker.

## Docker

```bash
docker compose up               # http://localhost:8501
```

The image integrates the models and builds the index during the build, so the
container needs no network at runtime. Verified by running it with
`--network none`. Budget 30-40 minutes for the first build, most of it spent
embedding the corpus.

For real generation instead of the deterministic default, `docker compose
--profile llm up` and set `LLM_PROVIDER=ollama`.

## Architecture

Two graphs. The agent decides *what* to do; the retrieval subgraph does the
RAG.

```
START → triage ─(in scope)─► planner ─┬─► retrieve ──────┐
           │                    ▲     └─► tool_executor ─┴─► synthesize
           │                    │                                  │
           │                    │                                  ▼
           │                    └──(ungrounded, retries left)── verify
           │                                                       │
           │                                    (grounded, or no   │
           │                                     retries left)     │
           │                                                       ▼
           └──────────(out of scope)─────────────────────────► finalize
                                                                   │
                                                                   ▼
                                                                  END
```

Seven nodes, three conditional routers, one bounded retry. What each node does:

- `triage` decides whether the corpus covers the question. Keyword check.
- `planner` splits it into at most four sub-questions. First LLM call.
- `retrieve` runs the RAG subgraph once per sub-question, in parallel.
- `tool_executor` calls the fine calculator or the compliance timeline.
- `synthesize` writes the answer from the passages and the computed values.
  Second LLM call.
- `verify` checks the draft against what was actually retrieved.
- `finalize` resolves the `[n]` markers into citations, or returns a refusal.

The out-of-scope branch is the refusal path, taken for an injection or an
irrelevant topic. If a question passes triage, what constrains it is `verify`,
where every number and every `[n]` marker must resolve to a retrieved passage
or a tool result.

`planner` returns a *list* of branches, so retrieval and tools separate then
rejoin at `synthesize`. `verify` owns the retry counter.

The RAG subgraph is separate and invoked from the `retrieve` node:

```
embed → search → rerank → select
```

**Retrieval.** FAISS dense (`bge-small-en-v1.5`, fp32 ONNX) plus BM25 sparse,
fused by reciprocal rank (RRF), then reranked by a cross-encoder. RRF and not a
weighted score sum, because BM25 is unbounded (a term-frequency score) while
cosine similarity from FAISS sits in [-1, 1], so combining them directly means
tuning a weight per corpus.

**Tools.** Both deterministic, neither retrieval-based. `compute_penalty` reads
`data/penalties.json`, where every tier carries the quote it came from;
`lookup_deadline` reads `data/deadlines.json`, which keeps superseded dates so
"what changed?" is answerable.

**Corpus.** AI Act (consolidated and original), GDPR, Digital Omnibus. 3.3 MB
vendored from the Publications Office, 1,194 chunks, each below the model's
512-token window (i.e., largest chunk is 411 tokens).

## Design decisions

**Two LLM calls per question.** The planner and the synthesiser. Triage, tool
dispatch, fine arithmetic and groundedness are plain Python.

**The model never does arithmetic.** Ceilings come from a calculator reading
`data/penalties.json`.

**Superseded law.** `index.fuse` drops non-current chunks from the candidate
pool. The original AI Act stays indexed, reachable by asking what the text said
before the amendment.

Verification is deterministic. Every `[n]` must resolve to a real passage, a
non-refusal must cite at least one, and every number in the answer must appear
in the question, a passage, or a tool result.

**Model choice: `qwen2.5:0.5b-instruct` via Ollama.** Open weights, Apache-2.0,
CPU-only, no paid API.

The default provider is a deterministic double that quotes retrieved passages.

## Results

22 retrieval queries and 18 functional questions, defined in
[`eval/questions.yaml`](eval/questions.yaml).
Raw output is committed in [`eval/results/`](eval/results).

| Retrieval | | |
|---|---|---|
| recall@1 (fused pool) | right provision ranked first, before reranking | 77.3% |
| recall@1 (after reranking) | right provision ranked first, after the cross-encoder | 86.4% |
| recall@5 / recall@20 | right provision anywhere in the top 5 / top 20 | 100% / 100% |
| MRR | mean of 1/rank of the right provision | 0.871 |
| superseded passages in top-20, filter **off** | repealed law reaching the pool | **181** |
| superseded passages in top-20, filter **on** | the same count, filter on | **0** |

| Agent | | |
|---|---|---|
| scope classification | in-scope or out-of-scope, labelled correctly | 18/18 |
| tool selection | the expected tool ran, or none where none was wanted | 11/11 |
| tool arguments | tier, entity and turnover read off the question | 4/4 |
| citation correctness | a returned citation points at the right provision | 8/8 |
| multi-hop decomposition | a compound question split into parts | 1/1 |
| verified grounded | every number and `[n]` marker resolved to a source | 16/16 |

## Performance

100 queries per run, deterministic provider, measured in process. Raw output in
[`eval/results/perf-c1.json`](eval/results/perf-c1.json) and
[`perf-c4.json`](eval/results/perf-c4.json).

| | concurrency 1 | concurrency 4 |
|---|---|---|
| p50 | 679 ms | 1976 ms |
| p95 | 1380 ms | 2998 ms |
| throughput | 1.31 req/s | 1.96 req/s |
| `retrieve` share of latency | 97.3% | 97.8% |

Bottleneck: the cross-encoder inside `retrieve`. Optimisations: score fewer and
shorter candidates (`RERANK_CANDIDATES`, `RERANK_MAX_CHARS`), and batch at the
reranker so concurrent requests share one forward pass.

## Task requirements

| Requirement | Where |
|---|---|
| LangGraph workflow, at least 5 nodes | `app/agent.py`, `build_graph`, 7 nodes |
| Conditional routing | `triage_router`, `route_subtasks`, `verify_router` |
| Subtask decomposition, independent execution | `planner`, then `ThreadPoolExecutor` in `retrieve_node` |
| State for intermediate results | `app/state.py`, `AgentState` and `dedup_chunks` |
| Two tools, neither retrieval-based | `app/tools.py`, `compute_penalty` and `lookup_deadline` |
| Modular RAG subgraph | `app/retrieval.py`, `build_retrieval_graph` |
| Text-based data source | `data/raw/`, provenance in `data/raw/SOURCES.md` |
| Open-source model, no paid API | `app/llm.py`, `OllamaLLM` with `qwen2.5:0.5b-instruct`; `DummyLLM` by default |
| Streamlit UI showing the agent's steps | `app/ui.py`, the "How this answer was reached" panel |
| Dockerfile | `Dockerfile` |
| docker-compose | `docker-compose.yml` |
| Evaluation set, 10 to 20 questions | `eval/questions.yaml`, 18 functional; results in `eval/results/eval.json` |
| Load test, 50 to 200 queries | `scripts/loadtest.py -n 100`; results in `eval/results/perf-c1.json` and `perf-c4.json` |
| Bottleneck and optimisations | Performance section above |

## Development

```bash
pytest                                                     # 88 tests
ruff check app scripts tests                               # lint
python scripts/evaluate.py --json eval/results/eval.json   # retrieval and graph metrics
python scripts/loadtest.py -n 100 -c 1                     # latency under load
```

## Limitations

- Not legal advice. It reports what the regulations say. Only the Official
  Journal is authentic.
- The corpus has a cut-off: four documents as of September 2026. Nothing outside
  them can be answered, and the system says so instead of guessing.
- Retrieval is imperfect. recall@1 is 86.4%, not 100%. The trace shows every
  passage and its score, so a weak answer is visible instead of hidden.
- Tool selection is keyword-based, so unusual phrasing can miss a tool. A miss
  falls back to retrieval, not to a wrong answer.
- No multi-turn coreference. "What about an SME?" will not resolve against the
  previous question.
- Ollama is wired up but not benchmarked. The latency figures above are for the
  deterministic provider; with a real model, generation dominates.

## Copyright and Legal Notices

Copyright (c) 2026 Modafar Al-Shouha. All rights reserved.

This code was created for evaluation purposes and may not be copied, modified,
distributed, or used commercially without the express written permission
of the author.

Regulation texts © European Union, 1998–2026, reused under Commission Decision
2011/833/EU. Source: [EUR-Lex](https://eur-lex.europa.eu).
See `data/raw/SOURCES.md` for provenance and hashes.

> Only the legislation published in the printed edition of the Official Journal
> of the European Union is deemed authentic.
