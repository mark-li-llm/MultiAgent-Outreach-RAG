# AGENTS — Working Guidelines for Automation/AI Coders

This file gives agents practical guidance to work safely and productively in this repo. It covers environment setup, where key logic lives, how to run gates, and pitfalls to avoid (e.g., OpenMP conflicts).

## Quick Facts
- Core steps live under `scripts/` and produce machine + human QA reports under `reports/qa/` and `data/final/reports/`.
- MCP stub services (including `kb.search`) listen on localhost ports 7801–7805, configured via `configs/mcp.tools.yaml`.
- Retrieval evaluation (Gate‑7) depends on an embedding space shared by both documents and queries.

## Environments
We use two conda environments to avoid OpenMP runtime conflicts while keeping FAISS available for index builds:

- `age` (Python 3.13): default for most tasks (Gate‑1 embeddings, Gate‑7 eval, routing, stubs).
- `ageFaiss` (Python 3.12): dedicated to Gate‑2 FAISS index build and health checks.

Create from YAMLs (recommended):
```
conda env create -f envs/age.yaml
conda env create -f envs/ageFaiss.yaml
```
See details in `docs/envs.md`.

## Runbook (common commands)
- Gate‑1 — Embeddings (text‑based):
  `conda run -n age python scripts/qa_step01_embeddings.py`
- Gate‑2 — Index build & integrity (FAISS):
  `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
- Gate‑7 — Retrieval evaluation:
  `conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py`

Artifacts land in `reports/qa/` and `data/final/reports/`.

## Retrieval stack (where to look)
- Shared text embedding utility: `scripts/embedding_utils.py`
  - normalize → tokenize (words + bigrams) → signed feature hashing (L2‑normalized)
  - Use `embed_text(text, dim)` for both documents and queries.
- Gate‑1 build (document vectors): `scripts/qa_step01_embeddings.py`
  - Generates `data/vector/embeddings/embeddings.parquet`.
- MCP `kb.search` stub: `scripts/qa_step03_mcp.py`
  - Uses `embed_text` for queries; returns candidates with a lightweight lexical rerank.
- Gate‑7 eval: `scripts/qa_step07_retrieval_eval.py`
  - Computes recall@10, nDCG@5, coverage (optional), freshness, latency. Has offline fallback.

## Router and config
- Router + rerank helpers: `scripts/router_core.py`
- Heuristics: `configs/router.heuristics.yaml`
- MCP endpoints/ports: `configs/mcp.tools.yaml`
- Vector settings: `configs/vector.indexing.yaml` (embedding.model is `hashlex-v1`).

## Quality gates (outputs)
- Gate‑1: `reports/qa/step01_embeddings.{json,md}`
- Gate‑2: `reports/qa/step02_indexes.{json,md}` (+ `data/final/reports/index_health.json`)
- Gate‑7: `reports/qa/step07_retrieval_eval.{json,md}` and failure log `reports/eval/retrieval_failures.jsonl`

## Useful toggles
- Gate‑7:
  - `AG7_IGNORE_COVERAGE=1` — skip coverage gating.
  - `AG7_LATENCY_MULTIPLIER=<float>` — relax latency budgets.
- Gate‑2:
  - Prefer running in `ageFaiss`. Avoid installing pip `faiss-cpu` inside `age`.

## Pitfalls and fixes
- Duplicate OpenMP runtime (libomp) crash during FAISS build:
  - Symptom: `OMP Error #15` / segfault when running Gate‑2 in `age`.
  - Cause: mixing pip `faiss-cpu` (bundles libomp) with conda‑forge OpenBLAS (OpenMP variant) + `llvm-openmp`.
  - Fix: run Gate‑2 in `ageFaiss` (Python 3.12 + conda‑forge FAISS); keep `age` free of pip FAISS.
- Recall==0 in Gate‑7:
  - Ensure both doc and query use the same `embed_text`. Do not revert to ID‑based or random embeddings.
- PDF glyph noise:
  - Some chunks include CID-like tokens. The current embedding’s tokenization and bigrams mitigate but do not fully fix; consider reranker improvements if needed.

## Coding conventions for agents
- Keep changes minimal and focused; prefer adding small, documented switches (env vars) over invasive alterations.
- Update docs when changing run commands, environments, or outputs (see `docs/` and this AGENTS.md).
- Avoid adding network or heavyweight deps; the MCP stub is designed to run locally.
- Do not auto‑install packages in scripts (this can destabilize envs). Prefer documenting env requirements.
- Preserve report paths and schemas used by gates.

## Getting unstuck
- Ports busy (7801): another stub is running — stop it or reuse the external service (`/invoke` health check is built into Gate‑7).
- Env mismatch: recreate from `envs/*.yaml`.
- Where to ask the code: check logs under `logs/` and the machine JSONs under `reports/qa/`.

Happy hacking!
