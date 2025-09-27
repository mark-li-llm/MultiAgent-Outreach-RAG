# Project Environments and Runners

We run different gates in different Python environments to avoid OpenMP runtime conflicts while keeping FAISS available.

- Main env: `age` (Python 3.13) — all steps except FAISS index build
- FAISS env: `ageFaiss` (Python 3.12) — Gate‑2 (FAISS index + health)

Quick commands:

```
# Gate‑1: Embeddings (age)
conda run -n age python scripts/qa_step01_embeddings.py

# Gate‑2: Index build & integrity (ageFaiss)
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate‑7: Retrieval eval (age)
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

See `docs/envs.md` for environment creation details.

## Gate‑7 Trace and Context

Gate‑7 now emits lightweight run context and per‑query trace to aid debugging without changing gating logic.

- Toggle via env vars (defaults in parentheses):
- `AG7_TRACE=1` — write per‑query JSONL trace (`reports/router/step07_retrieval_trace.jsonl`).
- `AG7_TRACE_TOPK=10` — number of Top‑K items stored per query in the trace.
- `AG7_TRACE_SUCCESSES=1` — include successes as well as failures in the trace.
- `AG7_DEBUG=1` — umbrella switch that turns tracing on by default (this is the default). Set `AG7_DEBUG=0` or `AG7_TRACE=0` to disable.
- The main report `reports/qa/step07_retrieval_eval.json` includes:
  - `run_context`: retrieval mode (offline/online_stub/online_external), router/vector config snapshot, eval settings.
  - `trace_path`: path to the JSONL trace file when tracing is enabled.
  - `router_summary`: counts by backend and reason codes.
