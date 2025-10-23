# Part 1: System Overview & Architecture

**Research Date**: 2025-10-20 15:18:36 EDT
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

---

## TL;DR

This document describes a **multi-agent RAG system for audit-ready B2B outreach** that transforms unstructured documents into compliance-vetted, persona-specific emails with complete traceability.

**Problem:** Sales/IR/PR teams need personalized outreach emails referencing recent company developments (earnings, product launches, partnerships) with proof of every claim and the ability to recreate outputs for regulatory audits. Traditional LLM generation lacks provenance and cannot prove what sources influenced which claims.

**Solution:** A **13-stage gated pipeline** processes 100+ documents through normalization, chunking, embedding (OpenAI ada-002), and multi-index storage (FAISS/Weaviate/Pinecone). An **8-node LangGraph orchestration** (Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler) executes persona-specific research queries, synthesizes insights with gpt-5-nano, generates email drafts, and validates compliance through agent-to-agent negotiation. Every stage emits dual-format reports (JSON + Markdown) with evidence links for full auditability.

**Requirements:** Two conda environments (`age` for most tasks, `ageFaiss` for FAISS indexing due to OpenMP conflicts), OpenAI API access (ada-002 embeddings + gpt-5-nano LLM), and 9 quality gate passes including ≥80% retrieval recall, 0 critical compliance flags, and ≤160-word emails. Current corpus: 100+ documents → ~1,600 chunks. Performance: sub-second retrieval, 15-30s total runtime (LLM-dominated).

**Quick Start:** (1) Create environments: `conda env create -f envs/{age,ageFaiss}.yaml`, (2) Set `OPENAI_API_KEY` in `.env`, (3) Run critical quality gates: Gate-1 (embeddings), Gate-2 (indexes), Gate-7 (retrieval ≥80% recall), Gate-8 (generation 0 critical flags), (4) Execute graph: `conda run -n age python scripts/run_graph_langgraph.py --company Salesforce --persona vp_customer_experience`, (5) Inspect `outputs/{session-id}/email.json` and `reports/qa/step{07,08}_*.md`.

**Key Differentiator:** Unlike typical RAG systems optimizing for speed or accuracy alone, this architecture prioritizes **traceability and reproducibility**—every insight links to source chunks (chunk_id:line), every routing decision is logged in JSONL traces, every compliance check is documented with evidence paths, and every pipeline stage is replayable from intermediate checkpoints. Designed for regulated industries where "how did we get this result?" matters as much as the result itself.

---

## 1. Overview

### Executive Summary

This system implements a **multi-agent RAG (Retrieval-Augmented Generation) pipeline** for Sales/IR/PR outreach that automates trusted-source research and generates audit-ready, compliance-vetted emails with complete step-level traceability. The architecture consists of a 13-stage gated data pipeline feeding into an 8-node LangGraph orchestration that processes company research requests through specialized agents (Planner, Retriever, Consolidator, Stylist, A2A Compliance) to produce persona-specific outreach emails backed by verifiable evidence chains.

The system prioritizes **traceability and reproducibility** over raw performance. Every stage—from document collection through normalization, chunking, embedding, indexing, retrieval, synthesis, and generation—emits dual-format reports (JSON for machines, Markdown for humans) with evidence links, timestamps, and quality metrics. This design enables compliance teams to reconstruct exactly what happened at each step, replay pipelines from intermediate checkpoints, and prove data provenance for regulatory audits.

Built on **Python 3.13** with **LangGraph** state machine orchestration, **OpenAI ada-002** embeddings (cached via SHA-256 keys), and a **multi-index routing system** (FAISS for local speed, Weaviate for semantic dev docs, Pinecone for press/financial content), the pipeline achieves **sub-second median retrieval latency** on a 100+ document corpus (~1.6k chunks) while maintaining strict quality gates: ≥80% recall@10, ≥60% nDCG@5, ≥98% publish date coverage, zero critical compliance flags, and ≤160-word emails with ≤10.0 Flesch-Kincaid grade.

### Problem Statement

The system solves the problem of **audit-ready B2B outreach at scale** where:
- Sales/IR/PR teams need **personalized emails** referencing recent company developments (earnings, product launches, partnerships)
- Compliance requires **proof of claims** (no hallucinations, no uncited assertions)
- Regulatory frameworks demand **reproducibility** (ability to recreate outputs from historical state)
- Traditional LLM generation lacks **provenance** (cannot trace which source documents influenced which claims)

The solution transforms unstructured documents (SEC filings, press releases, product docs, help articles, Wikipedia) into a **queryable knowledge base** with vector search, then generates emails where every insight is linked back to specific chunks (chunk_id:line) in source documents, with compliance negotiation logs showing how critical violations were detected and remediated.

### Key Capabilities

**Data Pipeline** (13 gates across scripts/):
1. **Collection** — Fetch from 7 source types (SEC, product, dev_docs, help_docs, newsroom, investor_news, wikipedia)
2. **Normalization** — Clean HTML with BeautifulSoup using rule-based selectors (configs/normalization.rules.yaml)
3. **Metadata** — Extract publish dates, titles, URLs, doctypes with 98%+ coverage
4. **Chunking** — Semantic splitting (800 target tokens, 120 overlap, cl100k_base tokenizer)
5. **Deduplication** — Hash-based dedup to remove exact chunk duplicates
6. **Embedding** (Gate-1) — OpenAI text-embedding-ada-002 (1536-dim) with SHA-256 caching
7. **Indexing** (Gate-2) — Build FAISS HNSW + Weaviate + Pinecone manifests with 98%+ upsert
8. **MCP Tools** (Gate-3) — Validate 5 local stub services (kb.search, web.fetch, link.resolve, crm.lookup, safety.check)
9. **Routing** (Gate-4) — Keyword-based backend selection (FAISS/Weaviate/Pinecone) with fallback
10. **Graph** (Gate-5) — End-to-end workflow validation (8 nodes in correct order, 5 insights, ≥4 domains)
11. **A2A** (Gate-6) — Agent-to-agent compliance negotiation (≤2 rounds, 0 critical flags)
12. **Retrieval Eval** (Gate-7) — recall@10 ≥80%, nDCG@5 ≥60%, latency ≤budgets
13. **Generation Eval** (Gate-8) — 10-run validation (100% structural pass, 0 critical flags, ≥9/10 readability)

**Agent Orchestration** (8 LangGraph nodes):
- **Intake** — Validate company/persona inputs
- **Planner** — Generate 5 persona-specific queries from eval seed
- **Retriever** — Execute vector search with router-based backend selection
- **Synthesizer** — Convert chunks to candidate insight objects
- **Consolidator** — LLM-enhance 5 insights with gpt-5-nano (persona_relevance, metric_impact, action_suggestion)
- **Stylist** — Generate email draft with gpt-5-nano (100-140 words, contextual opening, proof points)
- **A2A** — Compliance check with safety.check service (max 2 revision rounds)
- **Assembler** — Attach proof points and finalize output

**Multi-Index Routing**:
- **Keyword rules** — Press/financial → Pinecone, dev docs → Weaviate, general → FAISS
- **Persona bias** — CIO prefers dev_docs, VP Sales Ops prefers product/help, VP CX prefers press/financial
- **Weighted scoring** — Similarity 0.5 + Recency 0.3 + Diversity 0.2
- **Diversity enforcement** — Max 2 results per domain in top-10, merge from alternates if <3 unique domains
- **Fallback order** — [faiss, weaviate, pinecone] for empty results

**Quality Assurance**:
- **Dual-format reports** — JSON (machine-readable) + Markdown (human-readable) for every gate
- **Evidence linking** — Every check includes file path to supporting data (reports/eval/, reports/router/)
- **Status colors** — GREEN (all pass), AMBER (warnings only), RED (failures)
- **Trace logs** — JSONL format for per-query routing decisions, retrieval failures, MCP probes

### Quick Stats

**Codebase Size**:
- **63 Python scripts** (41 in scripts/, 22 in subdirectories)
  - 10 Quality gates (qa_step00-08 + debug)
  - 7 Data fetchers (fetch_*.py)
  - 2 Manual ingesters (ingest_*.py)
  - 5 Core utilities (embedding_utils, router_core, langgraph_nodes, langgraph_state, common)
  - 6 Data processors (normalize, extract_metadata, chunk, dedupe, parse_sec, link_health)
  - 10 Verifiers (qa_verify_*.py)
  - 3 Graph executors (run_graph, run_graph_langgraph, visualize_graph)
  - 20+ Helper scripts (build_*, debug_*, fix_*, test_*, tool_servers)

**Configuration**:
- **10 config files** (9 YAML + 1 JSON in configs/)

**Data Pipeline**:
- **13 gated stages** (Gates 0-8 implemented)
- **7 source types** (SEC, product, dev_docs, help_docs, newsroom, investor_news, wikipedia)
- **100+ documents** → **~1,600 chunks** (expansion ratio ~16×)
- **98%+ publish date coverage** (temporal metadata)
- **≥3 unique domains** in retrieval results (diversity enforcement)

**Agent System**:
- **8 LangGraph nodes** (Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler)
- **2 LLM calls** (Consolidator + Stylist using gpt-5-nano at temp 0.3)
- **5 insight cards** per session (≥4 unique domains, ≥2 within 12 months)
- **≤2 A2A revision rounds** (compliance negotiation with critical flag detection)

**Performance**:
- **Sub-second FAISS latency** (median p50 ~50-100ms local)
- **Weaviate**: ~40-80ms simulated (manifest-based stub)
- **Pinecone**: ~80-160ms simulated (manifest-based stub)
- **Total runtime**: ~15-30s for end-to-end graph execution (dominated by LLM calls)

**Quality Metrics**:
- **Recall@10**: ≥80% (proportion of ground truth in top 10)
- **nDCG@5**: ≥60% (ranking quality via discounted cumulative gain)
- **Email length**: ≤160 words (hard limit)
- **Readability**: ≤10.0 Flesch-Kincaid grade (college level)
- **Structural pass**: 100% (all runs have valid schema)
- **Critical flags**: 0 (no compliance violations)
- **Persona keywords**: ≥2.0 avg hits per email

**Environments**:
- **2 conda environments** (age for most tasks, ageFaiss for FAISS-only to avoid OpenMP conflicts)
- **Python 3.13** (primary), **Python 3.12** (FAISS env)

**Documentation**:
- **22+ Markdown files** in docs/ (architecture, commands, configuration, troubleshooting, evaluation, envs)
- **3 root docs** (README.md, CLAUDE.md, AGENTS.md)
- **70+ log files** (execution traces in logs/)

---

## 2. Architecture & Design

### System Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          13-STAGE GATED PIPELINE                             │
│                                                                              │
│  Gate-0    Gate-1     Gate-2      Gate-3      Gate-4      Gate-5            │
│ Baseline  Embeddings  Indexing  MCP Tools    Router      Graph              │
│  (100+    (OpenAI     (FAISS     (5 local   (Heuristic  (8 nodes            │
│   docs)   ada-002)    HNSW)      stubs)     routing)    seq check)          │
│                                                                              │
│  └─────────┬────────────┬──────────┬───────────┬──────────┬─────────┘       │
│            │            │          │           │          │                  │
│            ▼            ▼          ▼           ▼          ▼                  │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                   MULTI-INDEX STORAGE                           │        │
│  │  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐         │        │
│  │  │   FAISS     │   │  Weaviate   │   │  Pinecone    │         │        │
│  │  │  (local)    │   │  (cloud)    │   │  (managed)   │         │        │
│  │  │  HNSW/L2    │   │  semantic   │   │  production  │         │        │
│  │  │  sub-1s     │   │  dev docs   │   │  scale-ready │         │        │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬───────┘         │        │
│  │         │                 │                  │                  │        │
│  │         └─────────────────┴──────────────────┘                  │        │
│  │                           │                                      │        │
│  │                           ▼                                      │        │
│  │                  ┌─────────────────┐                            │        │
│  │                  │  Router Core    │                            │        │
│  │                  │  (Heuristics)   │                            │        │
│  │                  └────────┬────────┘                            │        │
│  └───────────────────────────┼─────────────────────────────────────┘        │
│                              │                                               │
└──────────────────────────────┼───────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                       LANGGRAPH ORCHESTRATION                                │
│                                                                              │
│    ┌─────────┐                                                              │
│    │ Intake  │  Validate company/persona                                    │
│    └────┬────┘                                                              │
│         │                                                                    │
│         ▼                                                                    │
│    ┌─────────┐                                                              │
│    │ Planner │  Generate 5 persona-specific queries                         │
│    └────┬────┘                                                              │
│         │                                                                    │
│         ▼                                                                    │
│    ┌──────────┐                                                             │
│    │Retriever │  Execute vector search via MCP kb.search                    │
│    └────┬─────┘  (5 queries × top 10 = ~50 chunks)                         │
│         │                                                                    │
│         ▼                                                                    │
│    ┌────────────┐                                                           │
│    │Synthesizer │  Convert chunks → candidate insight objects               │
│    └─────┬──────┘                                                           │
│          │                                                                   │
│          ▼                                                                   │
│    ┌──────────────┐                                                         │
│    │Consolidator  │  LLM-enhance 5 insights (gpt-5-nano)                    │
│    └──────┬───────┘  + persona_relevance + metric_impact                    │
│           │                                                                  │
│           ▼                                                                  │
│    ┌──────────┐                                                             │
│    │ Stylist  │  Generate email draft (gpt-5-nano)                          │
│    └────┬─────┘  100-140 words, proof points                               │
│         │                                                                    │
│         ▼                                                                    │
│    ┌─────────┐                                                              │
│    │   A2A   │  Compliance check (safety.check MCP)                         │
│    └────┬────┘  Max 2 rounds, critical flag detection                       │
│         │                                                                    │
│         ├───────────┐ (critical flags + rounds<2)                           │
│         │           └─────► Stylist (revision)                              │
│         │                                                                    │
│         ▼                                                                    │
│    ┌───────────┐                                                            │
│    │ Assembler │  Attach proof points, finalize                             │
│    └─────┬─────┘                                                            │
│          │                                                                   │
│          ▼                                                                   │
│    ┌──────────────────────────────────────────────────────┐                │
│    │ Outputs: email.json, insights.json, timing.json,    │                │
│    │          compliance_report.json, router_trace.jsonl │                │
│    └──────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        QUALITY EVALUATION GATES                              │
│                                                                              │
│  Gate-6: A2A ≤2 rounds, 0 critical, ≤160 words, ≤10.0 grade                │
│  Gate-7: Recall@10 ≥80%, nDCG@5 ≥60%, coverage ≥3.0, freshness ≤540d       │
│  Gate-8: 10 runs → 100% structural, 0 critical, ≥9/10 length/read, ≥2.0 kw │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ reports/qa/step{NN}_{name}.{json,md}                               │   │
│  │ reports/eval/{retrieval_failures,generation_metrics,compliance}.json│   │
│  │ reports/router/step{NN}_trace.jsonl                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Major Subsystems

#### 1. Data Pipeline (scripts/)

**Purpose**: Transform raw documents into queryable vector representations with quality gates

**Stages**:
1. **Collection** (fetch_*.py, ingest_*.py) — Gather documents from 7 source types
2. **Normalization** (normalize_html.py) — Clean HTML using BeautifulSoup with rule-based selectors
3. **Metadata** (extract_metadata.py) — Extract publish_date, title, url, doctype (98%+ coverage)
4. **Chunking** (chunk_documents.py) — Semantic splitting (800 tokens, 120 overlap, cl100k_base)
5. **Deduplication** (dedupe_chunks.py) — Hash-based exact match removal
6. **Embedding** (qa_step01_embeddings.py) — OpenAI ada-002 with SHA-256 caching
7. **Indexing** (qa_step02_indexes.py) — Build FAISS HNSW (M=32, efConstruction=40, efSearch=16)
8. **MCP Tools** (qa_step03_mcp.py) — Validate 5 local stub services

**Key files**:
- `scripts/embedding_utils.py:86-211` — Unified embedding with caching and retry
- `scripts/common.py:1-400` — Shared utilities (ensure_dir, now_iso, rate limiter, fetch_with_retries)
- `configs/normalization.rules.yaml` — HTML cleaning selectors
- `configs/chunking.config.json` — Token targets and overlap
- `configs/vector.indexing.yaml` — FAISS params and embedding model

**Outputs**:
- `data/vector/embeddings/embeddings.parquet` — 1,600+ rows × (chunk_id, doc_id, vector[1536])
- `data/vector/faiss/index.faiss` — HNSW index (~2MB for 1.6k vectors)
- `data/vector/faiss/idmap.parquet` — Internal ID → chunk_id mapping

#### 2. Multi-Index Routing (scripts/router_core.py)

**Purpose**: Dynamically select backend (FAISS/Weaviate/Pinecone) per query

**Decision tree** (router_core.py:decide_backend):
1. **Keyword rules** (first match wins):
   - "earnings", "revenue", "financial", "sec filing" → Pinecone
   - "press release", "announcement", "newsroom" → Pinecone
   - "developer", "api", "sdk", "code", "integration" → Weaviate
   - "definition", "what is", "how to" → FAISS
2. **Persona bias** (optional):
   - CIO → prefer Weaviate (technical docs)
   - VP Sales Ops → prefer FAISS/Weaviate (product/help)
   - VP CX → prefer Pinecone (press/financial)
3. **Weighted scoring** (if no rule matches):
   - Similarity: 0.5, Recency: 0.3, Diversity: 0.2
4. **Fallback** (if empty results):
   - Try alternates in order: [faiss, weaviate, pinecone]

**Reranking** (router_core.py:rerank):
- **Diversity enforcement**: Max 2 results per domain in top-10
- **Recency boost**: Newer docs get higher scores
- **Deduplication**: Remove duplicate chunk_ids

**Configuration**:
- `configs/router.heuristics.yaml` — Keyword rules, persona bias, weights
- `configs/mcp.tools.yaml` — Backend endpoints and timeouts

**Trace logging**:
- `reports/router/step{NN}_trace.jsonl` — Per-query decisions with backend + reasons

#### 3. MCP (Model Context Protocol) Tools (scripts/qa_step03_mcp.py)

**Purpose**: Standardized interface for tool invocation across agents

**5 local stub services** (ports 7801-7805):
1. **kb.search** (7801) — Vector search with lexical rerank
2. **web.fetch** (7802) — HTTP fetching (stubbed, returns ok)
3. **link.resolve** (7803) — URL resolution (stubbed)
4. **crm.lookup** (7804) — CRM data lookup (stubbed)
5. **safety.check** (7805) — Compliance validation (tool_safety_check_server.py)

**Stub architecture** (qa_step03_mcp.py:40-205):
- Async aiohttp servers on localhost
- `/healthz` GET endpoint for health checks
- `/invoke` POST endpoint for tool calls
- Simulated backend latencies (FAISS 5-10ms, Weaviate 40-80ms, Pinecone 80-160ms)

**kb.search handler** (qa_step03_mcp.py:82-156):
1. Validate request (method="search", query + backend + top_k)
2. Embed query via embed_text()
3. L2 vector search (numpy)
4. Lexical rerank (70% vector + 30% token overlap)
5. Return top_k with score + snippet

**Configuration**:
- `configs/mcp.tools.yaml` — Service endpoints, timeouts, fallback behavior

**Gate-3 validation** (qa_step03_mcp.py:223-378):
- G3-01: All 5 services respond to /healthz (==5)
- G3-02: Invalid requests rejected (contract conformance ==1.0)
- G3-03: Latency budgets (p50/p95 ≤ budget × multiplier)
- G3-04: Timeout rate ==0.0 (stability)

#### 4. LangGraph Orchestration (scripts/run_graph_langgraph.py, langgraph_nodes.py, langgraph_state.py)

**Purpose**: Type-safe agent workflow with conditional routing

**StateGraph** (langgraph_state.py:7-42):
```python
class AgentState(TypedDict):
    # Input fields (replaced)
    company: str
    persona: str
    session_id: str
    timestamp: str

    # Planning (replaced)
    queries: List[str]  # 5 queries
    persona_keywords: List[str]

    # Retrieval (accumulated with Annotated[..., add])
    retrieved_chunks: Annotated[List[Dict], add]
    retrieval_logs: Annotated[List[Dict], add]
    route_decisions: Annotated[List[Dict], add]

    # Synthesis (replaced)
    insight_candidates: List[Dict]
    insight_cards: List[Dict]  # 5 final cards

    # Generation (replaced)
    email_draft: Dict

    # Compliance (accumulated + replaced)
    compliance_flags: Annotated[List[str], add]
    a2a_rounds: int

    # Observability (replaced)
    metrics: Dict
    errors: Annotated[List[str], add]
```

**8 nodes** (langgraph_nodes.py):
1. **intake_node** (166-171) — Validates company/persona exist
2. **planner_node** (174-211) — Loads eval seed, generates 5 queries, loads persona keywords
3. **retriever_node** (214-244) — Calls decide_backend(), executes kb.search, reranks
4. **synthesizer_node** (247-277) — Converts chunks to candidates with metadata
5. **consolidator_node** (280-447) — Domain diversity selection (≥4 domains), LLM enhancement (gpt-5-nano), 3-attempt retry on ID hallucination
6. **stylist_node** (450-468) — Generates email via gpt-5-nano (100-140 words, natural opening, proof points)
7. **a2a_node** (471-567) — Calls safety.check, applies critical fixes (missing blocks, prohibited phrases, uncited claims), increments a2a_rounds
8. **assembler_node** (570-583) — Adds safety defaults (unsubscribe, company_info), attaches proof_points array

**Conditional routing** (run_graph_langgraph.py:25-33):
```python
def should_revise_email(state: AgentState) -> str:
    critical_flags = [f for f in state["compliance_flags"] if f.startswith("CRITICAL:")]
    rounds = state.get("a2a_rounds", 0)
    if critical_flags and rounds < 2:
        return "revise"  # → Stylist (regenerate email)
    return "assemble"  # → Assembler (finalize)
```

**Execution flow** (run_graph_langgraph.py:73-205):
1. Build graph: `workflow = build_graph()` → `app = workflow.compile()`
2. Initialize state with company/persona/session_id
3. Execute: `result = await app.ainvoke(initial_state)`
4. Post-process: Word count enforcement (160 max), readability check (≤10.0 grade)
5. Write 5 artifacts: insights.json, email.json, timing.json, compliance_report.json, router_trace.jsonl
6. State snapshot: state/session-{session_id}.json

**Outputs** (outputs/{session_id}/):
- `insights.json` — 5 LLM-enhanced insight cards
- `email.json` — Final email (subject, body, unsubscribe_block, company_info_block, proof_points[])
- `timing.json` — Total runtime milliseconds
- `compliance_report.json` — A2A rounds + flags (critical, warning)
- `router_trace.jsonl` — 5 routing decisions (query → backend → reasons)

#### 5. Quality Evaluation Gates (scripts/qa_step*.py)

**Purpose**: Validate pipeline quality with automated thresholds

**Gate hierarchy**:
- **Gates 0-5**: Data preparation (baseline, embeddings, indexes, MCP, router, graph)
- **Gates 6-8**: End-to-end evaluation (A2A compliance, retrieval quality, generation quality)

**Common patterns** (all gates):
- **Check structure**: `{"id": "GN-NN", "metric": str, "actual": value, "threshold": str, "status": "PASS"|"WARN"|"FAIL", "evidence": path}`
- **Status logic**: GREEN (all pass), AMBER (1 warning), RED (failures)
- **Dual reports**: reports/qa/step{NN}_{name}.{json,md}

**Gate-7 metrics** (qa_step07_retrieval_eval.py:637-642):
- **G7-01**: Recall@10 ≥80% (hits in top 10 / total queries)
- **G7-02**: nDCG@5 ≥60% (ranking quality via DCG)
- **G7-03**: Coverage ≥3.0 unique domains per query
- **G7-04**: Freshness ≤540 days average doc age
- **G7-05**: Latency p50/p95 ≤ budget × multiplier

**Gate-8 metrics** (qa_step08_generation_eval.py:403-434):
- **G8-01**: Structural pass rate ==100% (insights==5, sources≥4, recent≥2, schema ok, proof points resolve)
- **G8-02**: Critical flags total ==0 (across 10 runs)
- **G8-03**: Length/readability pass ≥9/10 runs (words≤160, grade≤10.0)
- **G8-04**: Persona keyword hits avg ≥2.0 (keywords from eval.prompts.yaml)

**Environment variables**:
- `AG1_AUTO_CONFIRM=1` — Skip embedding cost confirmation
- `AG7_IGNORE_COVERAGE=1` — Skip coverage gating
- `AG7_LATENCY_MULTIPLIER=3.0` — Relax latency budgets
- `AG7_TRACE=1` — Enable trace logging (default ON)

### Component Relationships

**Data flow**:
```
Raw Docs → Normalize → Metadata → Chunk → Dedupe → Embed → Index
    ↓
FAISS/Weaviate/Pinecone ← Router ← Queries (Planner)
    ↓
Retrieved Chunks → Synthesizer → Consolidator (LLM) → Insights
    ↓
Stylist (LLM) → Email Draft → A2A (Safety) → Assembler → Final Email
```

**Dependency graph**:
- **Embedding generation** (Gate-1) depends on chunked documents
- **Indexing** (Gate-2) depends on embeddings
- **MCP tools** (Gate-3) depend on embeddings for kb.search stub
- **Retrieval eval** (Gate-7) depends on embeddings, indexes, MCP tools, eval seed
- **Generation eval** (Gate-8) depends on all upstream gates + graph execution

**Configuration cascade**:
- `vector.indexing.yaml` → embedding model, FAISS params
- `router.heuristics.yaml` → routing logic (used by Retriever, qa_step04, qa_step07)
- `mcp.tools.yaml` → service endpoints (used by all MCP clients)
- `eval.prompts.yaml` → persona keywords, evaluation criteria
- `compliance.template.yaml` → A2A validation rules

### Data Flow (End-to-End)

**Input**: `--company Salesforce --persona vp_customer_experience --session-id test-run`

**Pipeline execution**:
1. **Intake** validates company="Salesforce", persona="vp_customer_experience" exist
2. **Planner** loads `data/interim/eval/salesforce_eval_seed.jsonl`, filters persona="vp_customer_experience", extracts 5 queries:
   - "Agentforce product announcement"
   - "latest earnings results"
   - "remaining performance obligation definition"
   - "customer experience AI"
   - "Data Cloud recent updates"
3. **Retriever** for each query:
   - Calls `decide_backend(query, persona, None)`:
     - "Agentforce" → matches "product" → Pinecone
     - "earnings" → matches "financial" → Pinecone
     - "rpo definition" → matches "definition" → FAISS
     - "customer experience AI" → no rule → weighted score → FAISS
     - "Data Cloud updates" → matches "product" → Pinecone
   - Executes `kb.search(backend, query, top_k=10)` via MCP
   - Reranks with diversity (max 2 per domain)
   - Accumulates ~50 total chunks (5 queries × 10 results)
4. **Synthesizer** converts chunks to candidates:
   - Deduplicates by chunk_id
   - Enriches with doc metadata (title, url, publish_date, source_domain)
   - Truncates snippets to 320 chars
   - Returns ~40 unique candidates
5. **Consolidator** selects 5 diverse insights:
   - Phase 1: Domain diversity (≥4 unique domains, synthesizes from docmeta if needed)
   - Phase 2: LLM enhancement (gpt-5-nano):
     ```
     Input: 5 base cards (id, title, summary)
     Output: 5 enhanced cards with:
       - persona_relevance: {why_it_matters, relevance_score, keywords_hit}
       - metric_impact: {metric, direction, magnitude}
       - action_suggestion: "Next step..."
     ```
   - Retry up to 3× on ID hallucination (if LLM invents unknown ID)
6. **Stylist** generates email:
   - Input: company, persona, persona_keywords, 5 insight_cards
   - LLM call (gpt-5-nano, temp 0.3):
     ```
     System: "You are writing B2B email. 100-140 words. Natural opening (no 'Dear [Title]').
              Reference recent developments. 1-3 bullets paraphrasing insights. Subject ≤12 words."
     User: "Company: Salesforce, Persona: VP Customer Experience, Keywords: [NPS, CSAT, omnichannel, ...]
            Insights: [5 cards with persona_relevance + metric_impact + action_suggestion]"
     ```
   - Output: `{"subject": str, "body": str, "unsubscribe_block": str, "company_info_block": str}`
7. **A2A** compliance check:
   - Calls MCP safety.check with email body + fields + insights
   - Returns (critical_flags, warning_flags):
     - Example: ["CRITICAL:UNCITED_CLAIM", "WARN:EXCESS_LENGTH"]
   - Applies critical fixes:
     - UNCITED_CLAIM → Appends "See: [insight_cards[0].title]"
     - MISSING_UNSUBSCRIBE → Adds default block
     - PROHIBITED_PHRASE → String replacements (guaranteed→designed to)
   - Increments a2a_rounds
   - Conditional routing:
     - If critical_flags AND rounds < 2 → return "revise" → back to Stylist
     - Else → return "assemble" → to Assembler
8. **Assembler** finalizes:
   - Adds safety defaults (unsubscribe, company_info) via setdefault()
   - Attaches proof_points: [{"id": card["id"], "title": card["title"]} for card in insight_cards]
   - Returns final email_draft

**Output**:
- `outputs/test-run/email.json`:
  ```json
  {
    "subject": "Salesforce's Agentforce redefines customer engagement",
    "body": "Recent Agentforce launch signals a major shift...\n\n• ...",
    "unsubscribe_block": "...",
    "company_info_block": "...",
    "proof_points": [{"id": "card-001", "title": "Agentforce Product Launch"}]
  }
  ```
- `outputs/test-run/insights.json`: 5 cards with persona_relevance + metric_impact
- `outputs/test-run/compliance_report.json`: `{"rounds": 1, "flags": {"critical": [], "warning": []}}`
- `state/session-test-run.json`: Full state snapshot with all intermediate artifacts

---

## 3. File Inventory

### Directory Structure

```
agent-weaviate/                           # Repository root
├── configs/                              # Configuration files (10 files)
│   ├── agents.schema.yaml                # Agent role definitions
│   ├── chunking.config.json              # Document chunking parameters
│   ├── compliance.template.yaml          # Compliance checking rules
│   ├── eval.prompts.yaml                 # Evaluation prompts and personas
│   ├── langgraph.nodes.yaml              # LangGraph topology
│   ├── mcp.tools.yaml                    # MCP service endpoints
│   ├── metadata.dictionary.yaml          # Metadata extraction rules
│   ├── normalization.rules.yaml          # HTML/text cleaning rules
│   ├── router.heuristics.yaml            # Multi-index query routing
│   └── vector.indexing.yaml              # Embedding model and FAISS params
│
├── scripts/                              # Processing and QA scripts (63 Python files)
│   ├── qa_step00_baseline.py             # Gate-0: Baseline checks
│   ├── qa_step01_embeddings.py           # Gate-1: Generate embeddings
│   ├── qa_step02_indexes.py              # Gate-2: Build FAISS index
│   ├── qa_step03_mcp.py                  # Gate-3: Validate MCP tools
│   ├── qa_step04_router.py               # Gate-4: Router verification
│   ├── qa_step05_graph.py                # Gate-5: Graph execution
│   ├── qa_step06_a2a.py                  # Gate-6: Agent-to-agent validation
│   ├── qa_step07_retrieval_eval.py       # Gate-7: Retrieval evaluation
│   ├── qa_step08_generation_eval.py      # Gate-8: Generation evaluation
│   ├── qa_step08_debug.py                # Gate-8: Debug tool
│   ├── fetch_*.py                        # Data collection (7 scripts)
│   ├── ingest_*.py                       # Manual ingestion (2 scripts)
│   ├── embedding_utils.py                # OpenAI ada-002 with caching
│   ├── router_core.py                    # Query routing logic
│   ├── langgraph_nodes.py                # 8 node implementations
│   ├── langgraph_state.py                # AgentState TypedDict
│   ├── run_graph.py                      # Original graph implementation
│   ├── run_graph_langgraph.py            # LangGraph implementation (recommended)
│   ├── visualize_graph.py                # Graph visualization
│   ├── common.py                         # Common utilities
│   ├── tool_safety_check_server.py       # MCP safety check server
│   └── [39 other processing/verification scripts]
│
├── data/                                 # Data artifacts (organized by stage)
│   ├── raw/                              # Original documents (7 source types)
│   ├── interim/                          # Processing artifacts
│   │   ├── normalized/                   # Normalized documents
│   │   ├── chunks/                       # Chunked documents (*.chunks.jsonl)
│   │   ├── dedup/                        # Deduplicated chunks
│   │   └── eval/                         # Evaluation datasets
│   ├── vector/                           # Vector embeddings and indexes
│   │   ├── embeddings/                   # embeddings.parquet (1536-dim vectors)
│   │   ├── faiss/                        # index.faiss, idmap.parquet
│   │   ├── weaviate/                     # index_manifest.json
│   │   └── pinecone/                     # index_manifest.json
│   ├── cache/                            # Caching layer
│   │   └── embeddings/                   # SHA-256 keyed embedding cache
│   ├── final/                            # Production-ready artifacts
│   │   ├── reports/                      # index_health.json
│   │   ├── inventory/                    # salesforce_inventory.csv
│   │   └── dictionaries/                 # Metadata dictionaries
│   └── backup/                           # Historical backups
│
├── reports/                              # Quality assurance reports (52 files)
│   ├── qa/                               # Gate reports (40 files: JSON + MD)
│   │   ├── step00_baseline.{json,md}     # Gate-0 baseline report
│   │   ├── step01_embeddings.{json,md}   # Gate-1 embedding quality
│   │   ├── step02_indexes.{json,md}      # Gate-2 index integrity
│   │   ├── step03_mcp.{json,md}          # Gate-3 MCP health
│   │   ├── step04_router.{json,md}       # Gate-4 routing quality
│   │   ├── step05_graph.{json,md}        # Gate-5 graph execution
│   │   ├── step06_a2a.{json,md}          # Gate-6 compliance
│   │   ├── step07_retrieval_eval.{json,md}# Gate-7 retrieval metrics
│   │   └── step08_generation_eval.{json,md}# Gate-8 generation metrics
│   ├── eval/                             # Evaluation metrics (6 files)
│   │   ├── retrieval_failures.jsonl      # Failed retrieval traces
│   │   ├── generation_metrics.json       # Per-run structural data
│   │   └── compliance_metrics.json       # Per-run compliance data
│   └── router/                           # Router decision logs (2 files)
│       ├── step04_router_trace.jsonl     # Gate-4 routing decisions
│       └── step07_retrieval_trace.jsonl  # Gate-7 routing decisions
│
├── outputs/                              # Generated outputs (6 sessions × 5 files)
│   ├── official-cio/                     # Official CIO session
│   │   ├── email.json                    # Final email
│   │   ├── insights.json                 # 5 insight cards
│   │   ├── timing.json                   # Execution timing
│   │   ├── compliance_report.json        # A2A results
│   │   └── router_trace.jsonl            # Routing decisions
│   └── [5 more sessions: vp-cx, vp-sales-ops, test-runs]
│
├── state/                                # Persistent state (6 files)
│   ├── session-official-cio.json         # Complete state snapshot
│   └── [5 more session snapshots]
│
├── logs/                                 # Execution logs (70+ files)
│   ├── fetch/                            # Data fetching logs (51 files)
│   ├── chunk/                            # Chunking logs (12 files)
│   ├── dedupe/                           # Deduplication logs (4 files)
│   ├── metadata/                         # Metadata extraction (5 files)
│   ├── langgraph/                        # LLM retry events (1 file)
│   └── mcp/                              # MCP probe logs (1 file)
│
├── docs/                                 # Documentation (22 Markdown files)
│   ├── README.md                         # Documentation index
│   ├── architecture.md                   # Detailed system architecture
│   ├── commands.md                       # Complete command reference
│   ├── configuration.md                  # Config file deep dive
│   ├── evaluation.md                     # Quality gates and metrics
│   ├── envs.md                           # Environment setup
│   ├── troubleshooting.md                # Debug playbook
│   ├── langgraph-edge-cases.md           # LangGraph edge cases
│   ├── archive/                          # Historical features/fixes/milestones
│   └── langgraph/                        # LangGraph-specific docs
│
├── envs/                                 # Conda environment definitions (2 files)
│   ├── age.yaml                          # Primary environment (Python 3.13)
│   └── ageFaiss.yaml                     # FAISS environment (Python 3.12)
│
├── icl/                                  # In-context learning templates (2 files)
│   ├── persona/                          # Persona definitions
│   │   └── vp_customer_experience.yaml   # VP CX persona
│   └── templates/                        # Prompt templates
│       └── email.yaml                    # Email generation template
│
├── roadmap/                              # Roadmap and issues (3 files)
│   ├── 0.md                              # Roadmap version 0
│   ├── 1.md                              # Roadmap version 1
│   └── issue/                            # Issue tracking
│       └── issue001.md                   # Part 1 research spec
│
├── README.md                             # Main project documentation
├── CLAUDE.md                             # Claude Code guidance
├── AGENTS.md                             # Agent/automation guidelines
└── README_DAY1.md                        # Day-1 milestone documentation
```

### File Counts by Category

| Category | Count | Location |
|----------|-------|----------|
| **Python Scripts** | 63 | scripts/ |
| - Quality Gates | 10 | scripts/qa_step*.py |
| - Data Fetchers | 7 | scripts/fetch_*.py |
| - Core Utilities | 5 | scripts/{embedding_utils,router_core,langgraph_*,common}.py |
| - Graph Executors | 3 | scripts/run_graph*.py, visualize_graph.py |
| - Processors | 6 | scripts/{normalize,extract_metadata,chunk,dedupe,parse_sec,link_health}.py |
| - Verifiers | 10 | scripts/qa_verify_*.py |
| - Manual Ingesters | 2 | scripts/ingest_*.py |
| - Other Helpers | 20 | scripts/{build_*,debug_*,fix_*,test_*,tool_*}.py |
| **Configuration Files** | 10 | configs/ (9 YAML + 1 JSON) |
| **Documentation** | 22 | docs/ (Markdown) |
| **QA Reports** | 40 | reports/qa/ (JSON + MD pairs) |
| **Eval Metrics** | 6 | reports/eval/ |
| **Router Traces** | 2 | reports/router/ |
| **Session Outputs** | 30 | outputs/ (6 sessions × 5 files) |
| **State Snapshots** | 6 | state/ |
| **Execution Logs** | 70+ | logs/ |
| **Conda Environments** | 2 | envs/ |
| **ICL Templates** | 2 | icl/ |
| **Root Documentation** | 4 | README*.md, CLAUDE.md, AGENTS.md |

**Total files**: 300+

### Key Files and Their Roles

**Critical entry points**:
- `scripts/run_graph_langgraph.py:1-223` — LangGraph implementation (recommended)
- `scripts/run_graph.py:1-819` — Original implementation (for comparison)
- `scripts/qa_step00_baseline.py` through `scripts/qa_step08_generation_eval.py` — 9 quality gates

**Core utilities**:
- `scripts/embedding_utils.py:86-211` — OpenAI ada-002 embedding with SHA-256 caching, retry logic (tenacity), batch processing
- `scripts/router_core.py:72-160` — Routing decision tree (decide_backend), reranking with diversity (rerank)
- `scripts/langgraph_nodes.py:1-583` — 8 agent node implementations
- `scripts/langgraph_state.py:7-42` — AgentState TypedDict schema
- `scripts/common.py:1-400` — Shared utilities (ensure_dir, now_iso, RateLimiter, fetch_with_retries, build_doc_id)

**Data artifacts**:
- `data/vector/embeddings/embeddings.parquet` — 1,600+ rows × (chunk_id, doc_id, seq_no, token_count, l2_norm, vector[1536])
- `data/vector/faiss/index.faiss` — HNSW index (~2MB for 1.6k 1536-dim vectors)
- `data/vector/faiss/idmap.parquet` — Internal FAISS ID → chunk_id mapping
- `data/interim/eval/salesforce_eval_seed.jsonl` — 40+ query-answer pairs for evaluation
- `data/final/inventory/salesforce_inventory.csv` — Document inventory with metadata

**Configuration**:
- `configs/vector.indexing.yaml` — Embedding model (openai-ada-002, dim 1536), FAISS params (M=32, efConstruction=40)
- `configs/router.heuristics.yaml` — Keyword rules, persona bias, weights, fallback order
- `configs/mcp.tools.yaml` — 5 tool endpoints (kb.search:7801, web.fetch:7802, link.resolve:7803, crm.lookup:7804, safety.check:7805)
- `configs/eval.prompts.yaml` — Persona keywords (vp_customer_experience: ["NPS", "CSAT", "omnichannel", ...])
- `configs/compliance.template.yaml` — Critical rules (missing blocks, prohibited phrases, uncited claims)

**Reports**:
- `reports/qa/step07_retrieval_eval.{json,md}` — Recall@10, nDCG@5, coverage, freshness, latency metrics
- `reports/qa/step08_generation_eval.{json,md}` — Structural pass rate, critical flags, length/readability, persona keywords
- `reports/eval/retrieval_failures.jsonl` — Per-query miss diagnostics (expected chunk/doc, retrieved top-10, classification)

---

## 4. Core Components Deep Dive

*Deferred to later roadmap parts (Part 2: Data Pipeline, Part 3: Vector Search & Routing, Part 4: Agent Orchestration, Part 5: LLM Integration, Part 6: Compliance & Safety).*

For this overview, see **Section 2: Architecture & Design** for high-level component descriptions.

---

## 5. Configuration & Settings

### Configuration Files Overview

All configuration files live in `configs/` directory (10 files total: 9 YAML + 1 JSON).

| File | Purpose | Key Settings | Used By |
|------|---------|--------------|---------|
| `vector.indexing.yaml` | Embedding model and index parameters | `embedding.model: openai-ada-002`<br>`embedding.dim: 1536`<br>`faiss.type: HNSW`<br>`faiss.M: 32`<br>`faiss.efConstruction: 40` | qa_step01, qa_step02, qa_step07, embedding_utils, run_graph |
| `router.heuristics.yaml` | Query routing rules | `weights: {similarity: 0.6, recency: 0.3, diversity: 0.1}`<br>`persona_bias: {cio: weaviate, ...}`<br>`rules: [{keywords: [...], backend: pinecone}]`<br>`fallback_order: [faiss, weaviate, pinecone]` | router_core, qa_step04, qa_step07 |
| `mcp.tools.yaml` | MCP service endpoints | `kb.search: {host: 127.0.0.1, port: 7801, timeout_ms: 2000}`<br>`[4 more tools: web.fetch, link.resolve, crm.lookup, safety.check]` | qa_step03, qa_step04, qa_step07, qa_step08, router_core, run_graph, langgraph_nodes |
| `langgraph.nodes.yaml` | Agent graph topology | `nodes: [Intake, Planner, Retriever, Synthesizer, Consolidator, Stylist, A2A, Assembler]`<br>`timeouts_ms: {Consolidator: 30000, Stylist: 30000, ...}` | run_graph, qa_step08_debug |
| `eval.prompts.yaml` | Evaluation prompts and personas | `personas.vp_customer_experience: [NPS, CSAT, omnichannel, ...]`<br>`personas.cio: [ROI, API, integration, ...]` | build_eval_generation_prompts, qa_step08, langgraph_nodes, run_graph |
| `compliance.template.yaml` | Compliance checking rules | `critical_rules: [MISSING_UNSUBSCRIBE, MISSING_COMPANY_INFO, PROHIBITED_PHRASE, UNCITED_CLAIM]`<br>`prohibited_phrases: {guaranteed: "designed to", ...}` | tool_safety_check_server, qa_step08, langgraph_nodes, run_graph |
| `normalization.rules.yaml` | HTML/text cleaning | `remove_selectors: [nav, footer, script, style, ...]`<br>`preserve_selectors: [article, main, ...]`<br>`newline_blocks: [p, div, li, ...]` | normalize_html, qa_verify_normalization |
| `chunking.config.json` | Document chunking parameters | `{"tokenizer": "cl100k_base", "target_tokens": 800, "overlap_tokens": 120, "short_doc_threshold_tokens": 350}` | chunk_documents, qa_verify_chunking |
| `metadata.dictionary.yaml` | Metadata extraction fields | Field definitions for extraction | (Documented, may be used by future scripts) |
| `agents.schema.yaml` | Agent role definitions | Agent roles, message schema, system limits | (Documented, may be used for schema validation) |

### Environment Variables

**Gate-1 (Embeddings)**:
- `AG1_AUTO_CONFIRM=1` — Skip cost confirmation prompt
- `OPENAI_API_KEY` — OpenAI API key (required)

**Gate-2 (Indexing)**:
- `AG2_DISABLE_FAISS=1` — Skip FAISS import (writes disabled manifest)

**Gate-7 (Retrieval)**:
- `AG7_IGNORE_COVERAGE=1` — Skip G7-03 coverage check
- `AG7_LATENCY_MULTIPLIER=<float>` — Relax latency budgets (e.g., `3.0` = 3×)
- `AG7_DEBUG=1` — Enable debug mode (default ON)
- `AG7_TRACE=1` — Enable trace logging (default ON if debug)
- `AG7_TRACE_TOPK=10` — Trace retrieval depth
- `AG7_TRACE_SUCCESSES=1` — Include successful queries in trace
- `AG7_ANALYZE_TOPK=10` — Evaluation retrieval depth
- `AG7_TOPK_SLICES="1,3,5,10"` — Multi-k recall curves
- `AG7_NEAR_SEQ_TOL=1` — Near-miss seq_no tolerance

**Global**:
- `AR_GLOBAL_RPS` — Rate limit for fetch operations (default 6.0 req/s)

*Detailed configuration documentation available in `docs/configuration.md`.*

---

## 6. Data Structures & Schemas

### High-Level Data Types

**Document Metadata** (data/interim/normalized/*.json):
```python
{
    "doc_id": "crm::dev_docs::2024-01-15::agent-api-get-started::a1b2c3d4",
    "source_domain": "developer.salesforce.com",
    "source_bucket": "dev_docs",
    "doctype": "dev_docs",
    "requested_url": "https://developer.salesforce.com/...",
    "final_url": "https://developer.salesforce.com/...",
    "redirect_chain": ["..."],
    "http_status": 200,
    "content_type": "text/html",
    "content_length": 45678,
    "fetched_at": "2024-01-15T10:30:00Z",
    "sha256_raw": "a1b2c3d4...",
    "visible_title": "Get Started with Agent API",
    "visible_date": "2024-01-15",
    "publish_date": "2024-01-15",
    "headline": "Get Started with Agent API"
}
```

**Chunk JSONL** (data/interim/chunks/*.chunks.jsonl):
```python
{
    "chunk_id": "crm::dev_docs::2024-01-15::agent-api-get-started::a1b2c3d4::0000",
    "doc_id": "crm::dev_docs::2024-01-15::agent-api-get-started::a1b2c3d4",
    "seq_no": 0,
    "text": "Agent API allows you to build conversational AI...",
    "token_count": 128,
    "start_char": 0
}
```

**Embedding Parquet** (data/vector/embeddings/embeddings.parquet):
```python
pa.table({
    "chunk_id": pa.string(),
    "doc_id": pa.string(),
    "seq_no": pa.int32(),
    "token_count": pa.int32(),
    "l2_norm": pa.float32(),
    "vector": pa.list_(pa.float32())  # 1536 dimensions
})
```

**AgentState** (scripts/langgraph_state.py:7-42):
```python
class AgentState(TypedDict):
    # Input
    company: str
    persona: str
    session_id: str
    timestamp: str

    # Planning
    queries: List[str]
    persona_keywords: List[str]

    # Retrieval (accumulated)
    retrieved_chunks: Annotated[List[Dict], add]
    retrieval_logs: Annotated[List[Dict], add]
    route_decisions: Annotated[List[Dict], add]

    # Synthesis
    insight_candidates: List[Dict]
    insight_cards: List[Dict]

    # Generation
    email_draft: Dict

    # Compliance
    compliance_flags: Annotated[List[str], add]
    a2a_rounds: int

    # Observability
    metrics: Dict
    errors: Annotated[List[str], add]
```

**Insight Card** (outputs/{session_id}/insights.json):
```python
{
    "id": "card-001",
    "title": "Agentforce Product Launch",
    "summary": "Salesforce announced Agentforce...",
    "url": "https://www.salesforce.com/news/...",
    "published_date": "2024-09-12",
    "doc_id": "crm::newsroom::2024-09-12::agentforce-launch::b3c4d5e6",
    "source_domain": "www.salesforce.com",
    "evidence_snippet": "...",
    "confidence": 0.9,
    "persona_relevance": {
        "why_it_matters": "VP CX cares about AI-driven customer engagement",
        "relevance_score": 5,
        "keywords_hit": ["AI", "customer experience", "automation"]
    },
    "metric_impact": {
        "metric": "CSAT",
        "direction": "increase",
        "magnitude": "significant"
    },
    "action_suggestion": "Explore Agentforce pilot for support automation"
}
```

**Email JSON** (outputs/{session_id}/email.json):
```python
{
    "subject": "Salesforce's Agentforce redefines customer engagement",
    "body": "Recent Agentforce launch signals...\n\n• ...\n\n• ...\n\n• ...",
    "unsubscribe_block": "You can unsubscribe at any time...",
    "company_info_block": "Sent by ACME AI, 123 Market St...",
    "proof_points": [
        {"id": "card-001", "title": "Agentforce Product Launch"},
        {"id": "card-002", "title": "Q3 2024 Earnings Results"}
    ]
}
```

**Quality Check** (reports/qa/step{NN}_{name}.json):
```python
{
    "id": "G7-01",
    "metric": "recall@10",
    "actual": 0.85,
    "threshold": ">=0.80",
    "status": "PASS",
    "evidence": "reports/eval/retrieval_failures.jsonl"
}
```

*Detailed schema documentation available in `docs/architecture.md` and individual script docstrings.*

---

## 7. External Dependencies

### Technology Stack

**Core Runtime**:
- **Python 3.13** (primary environment: `age`)
- **Python 3.12** (FAISS-only environment: `ageFaiss`)
- **Conda** package manager (environment isolation)

**Web Framework**:
- **aiohttp 3.x** — Async HTTP client/server
- **asyncio** — Async/await event loop

**LLM & Embeddings**:
- **OpenAI API** — text-embedding-ada-002 (1536-dim embeddings)
- **OpenAI API** — gpt-5-nano (Consolidator + Stylist LLM calls)
- **LangChain** — ChatOpenAI client, ChatPromptTemplate
- **tenacity** — Retry logic with exponential backoff

**Vector Search**:
- **FAISS** (conda-forge, CPU-only) — HNSW index build (Python 3.12 env only)
- **numpy** — Vector operations (L2 distance, array manipulation)
- **PyArrow** — Parquet serialization for embeddings

**State Machine**:
- **LangGraph** — StateGraph orchestration with conditional routing
- **operator.add** — Annotated field accumulation

**Data Processing**:
- **BeautifulSoup4** — HTML parsing and normalization
- **pandas** — CSV/DataFrame operations
- **PyYAML** — YAML configuration loading

**Utilities**:
- **tiktoken** — OpenAI tokenizer (cl100k_base for chunking)
- **hashlib** — SHA-256 caching keys, document ID hashing
- **argparse** — CLI argument parsing
- **logging** — Structured logging with timestamps
- **uuid** — Session ID generation

**Testing/Evaluation**:
- **json** — Report serialization
- **jsonlines** — JSONL trace logs
- **pathlib** — Path manipulation
- **subprocess** — Graph execution in Gate-5/Gate-8

### External Services

**Required**:
- **OpenAI API** (api.openai.com)
  - **text-embedding-ada-002** for document/query embeddings
  - **gpt-5-nano** (configured model name) for LLM calls in Consolidator/Stylist
  - Requires `OPENAI_API_KEY` in `.env` file
  - Cost estimation in Gate-1 before embedding generation
  - SHA-256 caching reduces API calls (typical 80%+ cache hit rate after initial run)

**Optional** (manifest-based stubs for development):
- **Weaviate** — Cloud vector database for dev docs (simulated via manifest)
- **Pinecone** — Managed vector database for press/financial (simulated via manifest)

**Local MCP Stubs** (ports 7801-7805):
- **kb.search** (7801) — Vector search with lexical rerank (fully functional local stub)
- **web.fetch** (7802) — HTTP fetching (stubbed, returns ok)
- **link.resolve** (7803) — URL resolution (stubbed)
- **crm.lookup** (7804) — CRM data lookup (stubbed)
- **safety.check** (7805) — Compliance validation (fully functional local stub)

### Environment Management

**Two-environment architecture** (critical for OpenMP conflict avoidance):

**age** (Python 3.13) — Primary environment:
```yaml
dependencies:
  - python=3.13
  - numpy
  - pandas
  - pyyaml
  - aiohttp
  - beautifulsoup4
  - tiktoken
  - tenacity
  - pyarrow
  - pip:
    - openai
    - langgraph
    - langchain-openai
```
**Used for**: Gates 0-1, 3-8, routing, MCP stubs, graph execution

**ageFaiss** (Python 3.12) — FAISS-only environment:
```yaml
dependencies:
  - python=3.12
  - numpy
  - pandas
  - pyyaml
  - pyarrow
  - faiss-cpu  # conda-forge (NOT pip)
```
**Used for**: Gate-2 (FAISS index builds) ONLY

**Critical constraint**: NEVER install pip `faiss-cpu` in `age` environment (causes OMP Error #15 due to duplicate libomp). Always use conda-forge FAISS in dedicated `ageFaiss` environment.

*Detailed dependency lists in `docs/envs.md` and environment YAMLs in `envs/`.*

---

## 8. Execution & Usage

### Main Entry Points

**LangGraph Implementation** (recommended):
```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Original Implementation** (for comparison):
```bash
conda run -n age python scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

Both produce identical outputs in `outputs/<session-id>/`.

**Arguments**:
- `--company` — Target company name (default: "Salesforce")
- `--persona` — Recipient persona (default: "vp_customer_experience")
  - Valid: `vp_customer_experience`, `cio`, `vp_sales_ops`
- `--session-id` — Unique session identifier (default: auto-generated 12-char hex)

**Outputs**:
- `outputs/{session_id}/insights.json` — 5 LLM-enhanced insight cards
- `outputs/{session_id}/email.json` — Final email with proof points
- `outputs/{session_id}/timing.json` — Total runtime milliseconds
- `outputs/{session_id}/compliance_report.json` — A2A rounds + flags
- `outputs/{session_id}/router_trace.jsonl` — 5 routing decisions
- `state/session-{session_id}.json` — Complete state snapshot

### Quick Start Guide

**1. Create environments**:
```bash
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
```

**2. Set up OpenAI API key**:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

**3. Run quality gates** (sequential execution):
```bash
# Gate-1: Generate embeddings
conda run -n age python scripts/qa_step01_embeddings.py

# Gate-2: Build FAISS index (USE ageFaiss environment!)
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate-3: Validate MCP tools
conda run -n age python scripts/qa_step03_mcp.py

# Gate-7: Retrieval evaluation (with relaxed budgets)
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py

# Gate-8: Generation evaluation (10 runs, ~5 minutes)
conda run -n age python scripts/qa_step08_generation_eval.py
```

**4. Execute graph workflow**:
```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id test-session
```

**5. Inspect results**:
```bash
# Retrieval quality
cat reports/qa/step07_retrieval_eval.md

# Generation quality
cat reports/qa/step08_generation_eval.md

# Generated email
cat outputs/test-session/email.json | jq .

# Compliance results
cat outputs/test-session/compliance_report.json | jq .
```

### Common Workflows

**Data collection** (run once or when source data changes):
```bash
# Fetch SEC filings
conda run -n age python scripts/fetch_sec_filings.py --limit 10

# Fetch developer docs
conda run -n age python scripts/fetch_dev_docs.py --limit 20

# Fetch product pages
conda run -n age python scripts/fetch_product_docs.py --limit 15
```

**Data processing** (run after collection):
```bash
# Normalize HTML
conda run -n age python scripts/normalize_html.py

# Extract metadata
conda run -n age python scripts/extract_metadata.py

# Chunk documents
conda run -n age python scripts/chunk_documents.py

# Deduplicate chunks
conda run -n age python scripts/dedupe_chunks.py
```

**Verification** (optional quality checks):
```bash
# Verify normalization quality
conda run -n age python scripts/qa_verify_normalization.py

# Verify chunking quality
conda run -n age python scripts/qa_verify_chunking.py

# Verify metadata coverage
conda run -n age python scripts/qa_verify_metadata.py
```

**Graph visualization**:
```bash
conda run -n age python scripts/visualize_graph.py
# Generates: reports/graphs/agent_workflow.{mmd,png}
```

**Debug tools**:
```bash
# Debug Gate-8 failures
conda run -n age python scripts/qa_step08_debug.py \
  --company Salesforce \
  --persona vp_customer_experience

# Debug SEC retrieval issues
conda run -n age python scripts/debug_sec_retrieval.py
```

### Typical Development Cycle

1. **Modify source code** (e.g., update LangGraph nodes, routing heuristics, prompts)
2. **Run affected gates** to validate changes:
   - If changed embedding logic → Gate-1
   - If changed index build → Gate-2
   - If changed routing → Gate-4, Gate-7
   - If changed LLM prompts → Gate-8
3. **Run full graph** with test inputs to verify end-to-end
4. **Inspect outputs** in `outputs/{session_id}/` and `reports/qa/`
5. **Check traces** in `reports/eval/` and `reports/router/` for debugging

### Performance Expectations

**Quality gates** (age environment):
- Gate-0: <1s (baseline checks)
- Gate-1: ~2-5 minutes (1,600 chunks, with 80%+ cache hit rate)
- Gate-3: ~5-10s (MCP health checks)
- Gate-4: ~10-20s (40+ queries, routing + retrieval)
- Gate-5: ~20-30s (single graph run)
- Gate-6: <1s (single session compliance check)
- Gate-7: ~30-60s (40+ queries, full retrieval eval)
- Gate-8: ~5-10 minutes (10 graph runs with LLM calls)

**FAISS index build** (ageFaiss environment):
- Gate-2: ~5-10s (1,600 vectors, HNSW build)

**End-to-end graph execution**:
- Original: ~15-25s (dominated by 2 LLM calls at temp 0.3)
- LangGraph: ~15-30s (similar performance, +5s overhead for state management)

---

## 9. Code Patterns & Conventions

### File Naming Conventions

**Quality gate scripts**:
- Pattern: `qa_step{NN}_{descriptor}.py` (zero-padded gate number)
- Examples: `qa_step00_baseline.py`, `qa_step01_embeddings.py`, `qa_step07_retrieval_eval.py`
- All emit dual-format reports: `reports/qa/step{NN}_{name}.{json,md}`

**Data collection scripts**:
- Pattern: `fetch_{source_type}.py`
- Examples: `fetch_dev_docs.py`, `fetch_sec_filings.py`, `fetch_newsroom_rss.py`
- Async operations using aiohttp
- Rate limiting via `RateLimiter` from common.py

**Manual ingestion scripts**:
- Pattern: `ingest_manual_{descriptor}.py`
- Examples: `ingest_manual_html.py`, `ingest_manual_ir_html.py`
- Process browser-saved HTML files

**Tool server scripts**:
- Pattern: `tool_{service_name}_server.py`
- Example: `tool_safety_check_server.py` (port 7805)
- aiohttp.web framework with `/healthz` and `/invoke` endpoints

### Script Organization Patterns

**Async data collection** (all fetch_*.py):
```python
async def main_async(args):
    logger, log_path = build_logger()
    ensure_dir("data/raw/{bucket}")
    limiter = RateLimiter()
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_and_save(session, limiter, url, ...) for url in urls]
        results = await asyncio.gather(*tasks)

    logger.info(f"Fetched {len(results)} pages. Logs: {log_path}")

def parse_args():
    p = argparse.ArgumentParser(description="...")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=4)
    return p.parse_args()

def main():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
```

**Quality gate structure** (all qa_step*.py):
```python
def main():
    ensure_dir(VEC_DIR)
    # ... load inputs ...

    # Build checks list
    checks: List[Dict[str, Any]] = []
    checks.append({
        "id": "G{N}-{NN}",
        "metric": "metric_name",
        "actual": actual_value,
        "threshold": ">=0.80",
        "status": "PASS" if condition else "FAIL",
        "evidence": "path/to/evidence"
    })

    # Status logic
    if all(c["status"] == "PASS" for c in checks):
        status = "GREEN"
    elif <amber_condition>:
        status = "AMBER"
    else:
        status = "RED"

    # Write dual-format reports
    ensure_dir("reports/qa")
    with open("reports/qa/step{NN}_{name}.json", "w") as f:
        json.dump({"step": "...", "status": status, "checks": checks, ...}, f, indent=2)

    with open("reports/qa/step{NN}_{name}.md", "w") as f:
        f.write(f"# STEP {N} — {Title} — {status}\n\nChecks:\n...")

if __name__ == "__main__":
    main()
```

### Embedding Consistency

**Unified function** (embedding_utils.py:86-133):
```python
def embed_text(text: str, dim: int) -> List[float]:
    """OpenAI ada-002 with SHA-256 caching."""
    if dim != 1536:
        raise ValueError(f"ada-002 requires dim=1536, got {dim}")

    if not text.strip():
        return [0.001] * 1536

    cached = _load_from_cache(text)
    if cached:
        return cached

    embedding = _call_openai_api(text)  # retry with tenacity
    _save_to_cache(text, embedding)
    return embedding
```

**Caching pattern** (embedding_utils.py:34-68):
```python
CACHE_DIR = Path("data/cache/embeddings")

def _get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def _load_from_cache(text: str) -> Optional[List[float]]:
    cache_path = CACHE_DIR / f"{_get_cache_key(text)}.json"
    if cache_path.exists():
        data = json.load(open(cache_path))
        if data.get("text_hash") == hashlib.md5(text.encode()).hexdigest():
            return data["embedding"]
    return None

def _save_to_cache(text: str, embedding: List[float]):
    cache_path = CACHE_DIR / f"{_get_cache_key(text)}.json"
    json.dump({"text_hash": hashlib.md5(text.encode()).hexdigest(), "embedding": embedding}, open(cache_path, "w"))
```

**Retry logic** (embedding_utils.py:70-84):
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import APIError, APIConnectionError, RateLimitError

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError))
)
def _call_openai_api(text: str) -> List[float]:
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding
```

### Configuration Loading

**YAML with defaults** (router_core.py:20-42):
```python
def load_router_config(path: str = "configs/router.heuristics.yaml") -> Dict:
    if not os.path.exists(path):
        return {
            "weights": {"similarity": 0.6, "recency": 0.3, "diversity": 0.1},
            "persona_bias": {},
            "rules": [],
            "fallback_order": ["faiss", "weaviate", "pinecone"]
        }
    return yaml.safe_load(open(path)) or {}
```

**Environment variable overrides** (qa_step07_retrieval_eval.py:260-279):
```python
ignore_coverage = os.getenv("AG7_IGNORE_COVERAGE", "0") == "1"
latency_multiplier = float(os.getenv("AG7_LATENCY_MULTIPLIER", "1.0"))
topk_eval = int(os.getenv("AG7_ANALYZE_TOPK", "10"))
debug_flag = os.getenv("AG7_DEBUG", "1") == "1"
```

### Traceability

**Document ID construction** (common.py:322-326):
```python
def build_doc_id(doctype: str, date_str: Optional[str], slug_base: str, url_for_hash: str) -> str:
    date_part = date_str or "unknown"
    slug = slugify(slug_base or "document")
    tail = sha1_8(strip_tracking_params(url_for_hash))
    return f"crm::{doctype}::{date_part}::{slug}::{tail}"
```

**Evidence links in checks** (all qa_step*.py):
```python
checks.append({
    "id": "G7-01",
    "metric": "recall@10",
    "actual": 0.85,
    "threshold": ">=0.80",
    "status": "PASS",
    "evidence": "reports/eval/retrieval_failures.jsonl"  # Links to supporting data
})
```

**Chunk provenance** (langgraph_nodes.py:270):
```python
card = {
    "id": f"card-{i:03d}",
    "doc_id": chunk.get("doc_id"),
    "chunk_id": chunk.get("chunk_id"),
    "url": doc_meta.get("url"),
    "published_date": doc_meta.get("publish_date"),
    "source_domain": doc_meta.get("source_domain"),
    "evidence_snippet": chunk.get("snippet")[:320]
}
```

**State snapshots** (run_graph_langgraph.py:201-202):
```python
with open(f"state/session-{session_id}.json", "w") as f:
    json.dump(dict(result), f, ensure_ascii=False, indent=2)
```

### Common Utility Functions

**Directory creation**:
```python
from common import ensure_dir
ensure_dir("data/vector/embeddings")  # Creates if not exists
```

**Timestamp generation**:
```python
from common import now_iso
timestamp = now_iso()  # Returns "2024-01-15T10:30:00Z"
```

**URL normalization**:
```python
from common import strip_tracking_params
clean_url = strip_tracking_params(url)  # Removes utm_*, gclid, fbclid, etc.
```

**Rate limiting**:
```python
from common import RateLimiter
limiter = RateLimiter(rps=6.0)
await limiter.wait()  # Enforces 6 req/s
```

**Retry-enabled fetch**:
```python
from common import fetch_with_retries
result = await fetch_with_retries(session, limiter, url, timeout_s=30)
# Returns: FetchResult(status, final_url, redirect_chain, content_type, body, latency_ms, err)
```

---

## 10. Testing & Verification

### Quality Gate System Overview

The system implements **9 quality gates** (Gates 0-8) across the data pipeline and agent execution, each validating specific artifacts with automated pass/fail thresholds.

**Gate Categories**:
- **Gates 0-5**: Data preparation (baseline, embeddings, indexes, MCP, router, graph)
- **Gates 6-8**: End-to-end evaluation (A2A compliance, retrieval quality, generation quality)

### Gate Summary Table

| Gate | Script | Purpose | Key Thresholds | Environment |
|------|--------|---------|----------------|-------------|
| **0** | qa_step00_baseline.py | Corpus baseline | docs≥80, publish_date≥98%, eval≥40, chunks≥docs, domains≥3 | age |
| **1** | qa_step01_embeddings.py | Embedding quality | rows==chunks, zero==0, nan==0, outliers≤0.5% | age |
| **2** | qa_step02_indexes.py | Index integrity | upsert≥98%, metadata≤2%, roundtrip≤0.001, sanity≥3 | **ageFaiss** |
| **3** | qa_step03_mcp.py | MCP tool health | health==5, contracts==1.0, timeout==0, latency≤budget | age |
| **4** | qa_step04_router.py | Routing quality | coverage≥10%, empty≤2%, retry≥95%, age≤p50, diversity≥2.4 | age |
| **5** | qa_step05_graph.py | Graph execution | nodes==8, runtime≤30s, insights==5, sources≥4, recent≥2 | age |
| **6** | qa_step06_a2a.py | A2A compliance | rounds≤2, critical==0, words≤160, grade≤10 | age |
| **7** | qa_step07_retrieval_eval.py | Retrieval metrics | recall≥80%, ndcg≥60%, coverage≥3.0, age≤540d, latency≤budget | age |
| **8** | qa_step08_generation_eval.py | Generation metrics | structural==100%, critical==0, length≥9/10, keywords≥2.0 | age |

### Status Color System

All gates use a **three-tier status**:
- **GREEN**: All checks pass → Proceed to next stage
- **AMBER**: 1 non-critical check fails within tolerance → Proceed with caution
- **RED**: Critical check fails or multiple failures → Fix and rerun

### Dual-Format Reports

Every gate emits two report formats:
- **JSON** (`reports/qa/step{NN}_{name}.json`) — Machine-readable, full state
- **Markdown** (`reports/qa/step{NN}_{name}.md`) — Human-readable summary

### Critical Gates

**Gate-1** (Embeddings):
- Validates OpenAI ada-002 embeddings with SHA-256 caching
- Checks: row count, dimension conformance, zero vectors, NaN vectors, norm outliers
- Cost estimation before API calls (skippable with `AG1_AUTO_CONFIRM=1`)
- **Must pass** before indexing

**Gate-2** (Indexing):
- Builds FAISS HNSW index with roundtrip validation
- Checks: upsert rates (98%+), metadata completeness, roundtrip error (≤0.001), sanity search (≥3 results)
- **CRITICAL**: Must run in `ageFaiss` environment (OpenMP conflict)
- **Must pass** before retrieval

**Gate-7** (Retrieval):
- End-to-end retrieval evaluation on 40+ query-answer pairs
- Checks: recall@10 (≥80%), nDCG@5 (≥60%), coverage (≥3.0 domains), freshness (≤540d avg age), latency (≤budget)
- Failure trace: `reports/eval/retrieval_failures.jsonl` with diagnostics
- Environment variables: `AG7_IGNORE_COVERAGE=1`, `AG7_LATENCY_MULTIPLIER=3.0`
- **Must pass** before generation evaluation

**Gate-8** (Generation):
- 10-run validation across ≥3 personas
- Checks: structural pass rate (==100%), critical flags (==0), length/readability (≥9/10), persona keywords (≥2.0 avg)
- Execution time: ~5-10 minutes (10 graph runs with LLM calls)
- **Final validation** before deployment

### Trace Logging

**Retrieval traces** (`reports/router/step07_retrieval_trace.jsonl`):
```json
{"timestamp": "2024-01-15T10:30:00Z", "query_text": "Agentforce product", "decision_backend": "pinecone", "reason_codes": ["keyword_match:product"]}
```

**Failure traces** (`reports/eval/retrieval_failures.jsonl`):
```json
{"query": "latest earnings", "expected_chunk": "crm::sec::2024-01-15::q3-earnings::a1b2::0000", "expected_doc": "crm::sec::2024-01-15::q3-earnings::a1b2", "retrieved_top10": ["crm::newsroom::...", ...], "classification": "chunk_miss_doc_hit_near", "nearest_same_doc": {"chunk_id": "...", "seq_no": 1, "delta": 1}}
```

**MCP probes** (`logs/mcp/step03_probes.jsonl`):
```json
{"timestamp": "2024-01-15T10:30:00Z", "tool": "kb.search", "method": "search", "latency_ms": 45, "status": "ok"}
```

### Verification Scripts

**Optional quality checks** (qa_verify_*.py):
- `qa_verify_collection.py` — Validates raw document counts
- `qa_verify_normalization.py` — Checks HTML cleaning quality
- `qa_verify_metadata.py` — Validates metadata coverage
- `qa_verify_chunking.py` — Checks chunk token counts
- `qa_verify_dedupe.py` — Validates deduplication
- `qa_verify_link_health.py` — Checks URL accessibility
- `qa_verify_eval_seed.py` — Validates evaluation dataset
- `qa_verify_day1_signoff.py` — Day-1 milestone checks

### Debug Tools

**Gate-8 debug** (qa_step08_debug.py):
```bash
conda run -n age python scripts/qa_step08_debug.py \
  --company Salesforce \
  --persona vp_customer_experience
```
Single-run execution with detailed output for debugging generation failures.

**SEC retrieval debug** (debug_sec_retrieval.py):
```bash
conda run -n age python scripts/debug_sec_retrieval.py
```
Diagnoses SEC filing retrieval issues (XBRL noise, chunk quality).

*Detailed gate documentation in `docs/evaluation.md`.*

---

## 11. Known Issues & Limitations

### System-Level Constraints

**Corpus Scale**:
- **Current**: Designed and verified on 100+ documents (~1.6k chunks)
- **Scaling**: FAISS HNSW scales to millions, but LLM costs and latency increase linearly with queries
- **Mitigation**: Shard indexes externally for horizontal scaling

**Embedding Dependency**:
- **OpenAI API required**: Gate-1 cannot run without `OPENAI_API_KEY`
- **Cost**: ~$0.0001 per 1k tokens (cached 80%+ after initial run)
- **Mitigation**: SHA-256 caching minimizes API calls

**Environment Complexity**:
- **Two conda environments required**: `age` (primary) + `ageFaiss` (FAISS-only)
- **OpenMP conflict**: Mixing pip faiss-cpu with conda OpenBLAS causes OMP Error #15
- **Mitigation**: Strict environment discipline documented in CLAUDE.md

**LLM Model Configuration**:
- **gpt-5-nano**: OpenAI multimodal model (2025) used in Consolidator/Stylist nodes (run_graph.py:176, langgraph_nodes.py:384, 452)
- **Temperature**: 0.3 for consistent, focused outputs
- **Use cases**: LLM-enhanced insights (Consolidator), email draft generation (Stylist)
- **Fallback**: None configured (model availability assumed)

**Weaviate/Pinecone Stubs**:
- **Simulated backends**: Weaviate and Pinecone use manifest-based stubs (no real cloud connection)
- **Latency simulation**: 40-80ms (Weaviate), 80-160ms (Pinecone) via asyncio.sleep
- **Mitigation**: Update `configs/mcp.tools.yaml` to point to production services

### Known Technical Issues

**PDF Glyph Noise** (docs/troubleshooting.md):
- **Symptom**: SEC filings contain CID-like tokens (e.g., "(cid:123)")
- **Cause**: PyMuPDF/pdfplumber glyph extraction issues
- **Impact**: Reduced retrieval recall on PDF-heavy queries
- **Mitigation**: Current tokenizer (cl100k_base) with bigrams mitigates but doesn't fully fix

**Recall=0 in Gate-7** (docs/troubleshooting.md):
- **Symptom**: Retrieval evaluation shows 0% recall despite indexed documents
- **Cause**: Document/query embeddings use different functions (e.g., random vectors vs. embed_text)
- **Solution**: Ensure both documents and queries use `embed_text()` from embedding_utils.py

**MCP Port Conflicts** (docs/troubleshooting.md):
- **Symptom**: Gate-3 fails with "Address already in use"
- **Cause**: Another stub server is running on ports 7801-7805
- **Solution**: Kill conflicting processes or change ports in `configs/mcp.tools.yaml`

**Graph Runtime Variance**:
- **Symptom**: Gate-5 runtime check fails (>30s)
- **Cause**: OpenAI API latency spikes (LLM calls in Consolidator/Stylist)
- **Impact**: Amber/Red status on slow API responses
- **Mitigation**: Relax timeout in Gate-5 or retry on transient failures

### Design Limitations

**Single-turn Generation**:
- **Current**: Email generation is one-shot (no iterative refinement beyond 2 A2A rounds)
- **Limitation**: Cannot handle complex multi-turn dialogues or user feedback
- **Roadmap**: Future multi-turn orchestration in Part 6

**Persona Coverage**:
- **Current**: 3 personas (vp_customer_experience, cio, vp_sales_ops)
- **Limitation**: Hardcoded in `configs/eval.prompts.yaml`
- **Roadmap**: Dynamic persona loading in future versions

**Language Support**:
- **Current**: English-only (OpenAI ada-002 tokenizer, prompts)
- **Limitation**: No i18n for non-English content
- **Roadmap**: Not planned

**Compliance Rules**:
- **Current**: Hardcoded in `configs/compliance.template.yaml`
- **Limitation**: Static rule set, no dynamic policy updates
- **Roadmap**: External compliance service integration in Part 6

### Performance Bottlenecks

**LLM Calls**:
- **Consolidator**: ~5-10s per call (gpt-5-nano at temp 0.3)
- **Stylist**: ~5-10s per call
- **Total**: ~15-25s per graph run (dominated by LLM latency)
- **Mitigation**: Batch graph runs for evaluation, cache LLM responses (future)

**Embedding Generation**:
- **Gate-1**: ~2-5 minutes for 1,600 chunks (with 80%+ cache hit)
- **Cold start**: ~10-15 minutes without cache
- **Mitigation**: SHA-256 caching, batch API calls (100 texts per request)

**FAISS Index Build**:
- **Gate-2**: ~5-10s for 1,600 vectors (HNSW M=32, efConstruction=40)
- **Scaling**: O(n log n) → ~1 minute for 100k vectors
- **Mitigation**: Use GPU FAISS for larger corpora (not currently supported)

*Detailed troubleshooting in `docs/troubleshooting.md`.*

---

## 12. References

### Internal Documentation

**Core Documentation**:
- **README.md** — Main project documentation (architecture overview, quick start)
- **CLAUDE.md** — Claude Code guidance (critical reference for agents)
- **AGENTS.md** — Agent/automation guidelines (runbook for autonomous execution)
- **README_DAY1.md** — Day-1 milestone documentation (initial verification)

**Detailed Architecture**:
- **docs/architecture.md** — Complete system design (pipeline stages, LangGraph orchestration, embedding system, multi-index routing, MCP tools)
- **docs/commands.md** — Command reference for all 41 scripts (quality gates, data collection, processing, verification)
- **docs/configuration.md** — Config file deep dive (all 10 config files with examples and tuning guidelines)
- **docs/troubleshooting.md** — Debug playbook (OpenMP conflicts, recall=0, API errors, port conflicts, debugging tools)
- **docs/evaluation.md** — Quality gates and metrics (Gate-7 retrieval, Gate-8 generation, thresholds, status colors)
- **docs/envs.md** — Environment setup (package details, conflict resolution, two-environment architecture)

**Roadmap**:
- **roadmap/0.md** — Initial roadmap (high-level milestones)
- **roadmap/1.md** — Updated roadmap (current scope)
- **roadmap/issue/issue001.md** — Part 1 research spec (this document's requirements)
- **roadmap/part1-overview/README.md** — This document

### External Documentation

**LangGraph**:
- https://langchain-ai.github.io/langgraph/ — Official LangGraph docs
- https://langchain-ai.github.io/langgraph/concepts/low_level/ — StateGraph API reference

**OpenAI**:
- https://platform.openai.com/docs/guides/embeddings — text-embedding-ada-002 guide
- https://platform.openai.com/docs/api-reference/embeddings — Embeddings API reference
- https://platform.openai.com/docs/models — Model documentation (check gpt-5-nano availability)

**FAISS**:
- https://github.com/facebookresearch/faiss — Official FAISS repo
- https://github.com/facebookresearch/faiss/wiki — FAISS wiki (HNSW parameters, index types)

**Vector Databases**:
- https://weaviate.io/developers/weaviate — Weaviate documentation
- https://docs.pinecone.io/ — Pinecone documentation

**Python Libraries**:
- https://docs.aiohttp.org/ — aiohttp async HTTP client/server
- https://www.crummy.com/software/BeautifulSoup/bs4/doc/ — BeautifulSoup4 HTML parsing
- https://arrow.apache.org/docs/python/parquet.html — PyArrow Parquet serialization

### Related Roadmap Parts

**Upcoming Parts** (planned):
- **Part 2**: Data Pipeline Deep Dive (collection, normalization, chunking, deduplication)
- **Part 3**: Vector Search & Routing (embedding generation, FAISS/Weaviate/Pinecone, routing heuristics)
- **Part 4**: Agent Orchestration (LangGraph nodes, state management, conditional routing)
- **Part 5**: LLM Integration (Consolidator, Stylist, prompt engineering, retry logic)
- **Part 6**: Compliance & Safety (A2A negotiation, safety.check service, compliance rules)
- **Part 7**: Quality Gates & Evaluation (all 9 gates in detail, metrics, thresholds)
- **Part 8**: Configuration & Tuning (all config files, environment variables, performance tuning)

---

## Appendix: File References

### Key File Locations (with line numbers)

**Entry Points**:
- `scripts/run_graph_langgraph.py:1-223` — LangGraph implementation main
- `scripts/run_graph.py:1-819` — Original implementation
- `scripts/qa_step00_baseline.py:1-229` — Gate-0
- `scripts/qa_step07_retrieval_eval.py:1-862` — Gate-7
- `scripts/qa_step08_generation_eval.py:1-562` — Gate-8

**Core Utilities**:
- `scripts/embedding_utils.py:86-211` — embed_text(), embed_batch() with caching
- `scripts/router_core.py:72-160` — decide_backend(), rerank()
- `scripts/langgraph_nodes.py:166-583` — 8 node implementations
- `scripts/langgraph_state.py:7-42` — AgentState TypedDict
- `scripts/common.py:23-400` — ensure_dir(), now_iso(), RateLimiter, fetch_with_retries()

**Configuration**:
- `configs/vector.indexing.yaml:1-30` — Embedding model, FAISS params
- `configs/router.heuristics.yaml:1-40` — Routing rules, persona bias
- `configs/mcp.tools.yaml:1-50` — MCP service endpoints
- `configs/eval.prompts.yaml:1-100` — Persona keywords
- `configs/compliance.template.yaml:1-80` — Compliance rules

**Data Artifacts**:
- `data/vector/embeddings/embeddings.parquet` — 1,600+ embedding rows
- `data/vector/faiss/index.faiss` — HNSW index
- `data/interim/eval/salesforce_eval_seed.jsonl` — 40+ evaluation queries

**Reports**:
- `reports/qa/step07_retrieval_eval.{json,md}` — Retrieval metrics
- `reports/qa/step08_generation_eval.{json,md}` — Generation metrics
- `reports/eval/retrieval_failures.jsonl` — Failed query diagnostics

---

**End of Part 1: System Overview & Architecture**

*For detailed component documentation, see Parts 2-8 of the roadmap.*
