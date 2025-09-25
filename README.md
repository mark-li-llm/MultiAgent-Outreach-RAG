# Multi‑Agent RAG for Sales/IR/PR Outreach

Automates trusted‑source research and audit‑ready email generation with step‑level traceability, reducing compliance back‑and‑forth from days to hours. All stages emit machine + human‑readable QA reports so you can replay, inspect, and prove what happened at each step.

## 🏗️ System Architecture

```
┌─────────────────┐      ┌──────────────────────────────────────────────────┐
│   User Query    │ ──►  │              LangGraph Orchestration             │
│   (Sales/IR/PR) │      │   Gate-1 → Gate-2 → Gate-3 → Gate-6 → Gate-7    │
└─────────────────┘      │  Collect   Embed   Index    A2A     Eval        │
                         └──────────────┬───────────────────────────────────┘
                                        │
                         ┌──────────────▼───────────────────────────────────┐
                         │           Multi-Agent System (A2A)               │
                         │                                                  │
                         │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
                         │  │Planner  │─►│Retriever│─►│Consolid-│─►│Stylist  │ │
                         │  │Route    │  │Evidence │  │ator     │  │Email    │ │
                         │  │Select   │  │Gather   │  │Rank     │  │Generate │ │
                         │  └─────────┘  └────┬────┘  └─────────┘  └─────────┘ │
                         └───────────────────┼──────────────────────────────────┘
                                            │
                         ┌──────────────────▼──────────────────────────────────┐
                         │                MCP Tools Layer                      │
                         │                                                     │
                         │  ┌─────────────┐              ┌─────────────────┐  │
                         │  │ kb.search   │ ◄──────────► │ Safety Check    │  │
                         │  │ Query       │              │ Compliance      │  │
                         │  │ Interface   │              │ Guard           │  │
                         │  └──────┬──────┘              └─────────────────┘  │
                         └─────────┼─────────────────────────────────────────────┘
                                   │
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
            ▼                      ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     FAISS       │    │    Weaviate     │    │    Pinecone     │
│   Local Vector  │    │  Cloud Vector   │    │ Managed Vector  │
│   <1s latency   │    │ Semantic Search │    │  Scale Ready    │
│   hashlex-v1    │    │   Multi-tenant  │    │   Production    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │   Quality Gates   │
                         │ JSON + Markdown   │
                         │   Audit Trail     │
                         │ reports/qa/*.md   │
                         └───────────────────┘
```

## Core Architecture

- Orchestration (State Machine)
  - 6+ gated stages spanning collection → normalization → embeddings → indexing → retrieval → A2A/compliance → evaluation.
  - Replayable, traceable transitions: each gate writes JSON + Markdown reports under `reports/qa/` with evidence links.
  - Node layout is captured declaratively (see `configs/langgraph.nodes.yaml`) and executed by stage scripts under `scripts/`.

- Multi‑Agent Architecture (A2A)
  - Planner → Retriever → Consolidator → Stylist, with agent‑to‑agent handoffs and guardrails.
  - Planner: routing + policy selection using heuristics in `configs/router.heuristics.yaml` via `scripts/router_core.py`.
  - Retriever: MCP `kb.search` tool (local stub) across FAISS/Weaviate/Pinecone in `scripts/qa_step03_mcp.py`.
  - Consolidator: lightweight lexical rerank + reason capture in `scripts/router_core.py`.
  - Stylist: A2A + compliance checks for email outputs in `scripts/qa_step06_a2a.py` and `scripts/tool_safety_check_server.py`.

- MCP + Multi‑Index Routing
  - Policy‑controlled routing across FAISS, Weaviate, and Pinecone backends with budgeted latencies and fallbacks.
  - Endpoints and timeouts are configured in `configs/mcp.tools.yaml`; router logic in `scripts/router_core.py`.
  - Gate‑3 validates tool health, contracts, and latency budgets; Gate‑7 evaluates retrieval quality end‑to‑end.

- Shared Embeddings (hashlex‑v1)
  - `scripts/embedding_utils.py`: normalize → tokenize (words + bigrams) → signed feature hashing (L2‑normalized).
  - Single `embed_text(text, dim)` used for both documents and queries to preserve a shared space.
  - Vector settings in `configs/vector.indexing.yaml` (`embedding.model: hashlex-v1`).

- Scale & Performance
  - Designed and verified on 100+ documents (≈1.6k chunks). Retrieval evaluation reports recall@10, nDCG@5, coverage, freshness, and latency.
  - Typical local median latency is sub‑second on FAISS.
  - Stages and services are stateless for horizontal scale; indexes can be sharded externally.

## Environments

We use two conda envs to avoid OpenMP conflicts while keeping FAISS available.

- `age` (Python 3.13): default for most tasks (embeddings, routing, MCP stubs, retrieval eval).
- `ageFaiss` (Python 3.12): dedicated to FAISS index build and integrity checks.

Create from YAMLs:

- `conda env create -f envs/age.yaml`
- `conda env create -f envs/ageFaiss.yaml`

Details: `docs/envs.md`.

## Runbook

- Gate‑1 — Embeddings (text vectors)
  - `conda run -n age python scripts/qa_step01_embeddings.py`
  - Outputs: `data/vector/embeddings/embeddings.parquet`, report `reports/qa/step01_embeddings.{json,md}`.

- Gate‑2 — Index build & integrity (FAISS)
  - `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
  - Outputs: `data/final/reports/index_health.json`, report `reports/qa/step02_indexes.{json,md}`.

- Gate‑3 — MCP tool health & contracts
  - `conda run -n age python scripts/qa_step03_mcp.py`
  - Validates `kb.search` (and peers) health, contract adherence, and latency budgets across FAISS/Weaviate/Pinecone.

- Gate‑6 — A2A & Compliance checks (email)
  - `conda run -n age python scripts/qa_step06_a2a.py --session-id <SESSION>`
  - Checks readability, proof‑point references, and compliance (see `scripts/tool_safety_check_server.py`).

- Gate‑7 — Retrieval evaluation
  - `conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py`
  - Computes recall@10, nDCG@5, coverage (optional), freshness, latency; report `reports/qa/step07_retrieval_eval.{json,md}` and failures `reports/eval/retrieval_failures.jsonl`.

Artifacts land in `reports/qa/` and `data/final/reports/`.

## Where to Look

- Shared embeddings: `scripts/embedding_utils.py`
- Router + rerank: `scripts/router_core.py`
- MCP `kb.search` stub + probes: `scripts/qa_step03_mcp.py`, config `configs/mcp.tools.yaml`
- Vector settings: `configs/vector.indexing.yaml`
- Heuristics: `configs/router.heuristics.yaml`
- Retrieval eval: `scripts/qa_step07_retrieval_eval.py`
- A2A + compliance: `scripts/qa_step06_a2a.py`, `scripts/tool_safety_check_server.py`

## Quality Gates & Outputs

- Gate‑1: `reports/qa/step01_embeddings.{json,md}`
- Gate‑2: `reports/qa/step02_indexes.{json,md}` (+ `data/final/reports/index_health.json`)
- Gate‑3: `reports/qa/step03_mcp.{json,md}` (tool health, contracts, latency budgets)
- Gate‑6: `reports/qa/step06_a2a.{json,md}`
- Gate‑7: `reports/qa/step07_retrieval_eval.{json,md}` (+ failures `reports/eval/retrieval_failures.jsonl`)




## Quick Start

1) Create environments: `conda env create -f envs/age.yaml` and `conda env create -f envs/ageFaiss.yaml`.
2) Build embeddings and indexes:
   - `conda run -n age python scripts/qa_step01_embeddings.py`
   - `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
3) Validate MCP tools: `conda run -n age python scripts/qa_step03_mcp.py`
4) Run retrieval eval: `conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py`
5) Inspect results in `reports/qa/`.

## Notes

- This repo ships local MCP stubs so you can run everything offline. Swap stubs for real services by pointing `configs/mcp.tools.yaml` to production endpoints.
- For a Day‑1 milestone summary, see `README_DAY1.md`.
