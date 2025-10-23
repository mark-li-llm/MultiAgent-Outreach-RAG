# Part 7: Quality Gates & Evaluation

**Research Date**: 2025-10-20 16:35:37 EDT
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Design](#2-architecture--design)
3. [File Inventory](#3-file-inventory)
4. [Core Components Deep Dive](#4-core-components-deep-dive)
5. [Configuration & Settings](#5-configuration--settings)
6. [Data Structures & Schemas](#6-data-structures--schemas)
7. [External Dependencies](#7-external-dependencies)
8. [Execution & Usage](#8-execution--usage)
9. [Code Patterns & Conventions](#9-code-patterns--conventions)
10. [Testing & Verification](#10-testing--verification)
11. [Known Issues & Limitations](#11-known-issues--limitations)
12. [References](#12-references)

---

## 1. Overview

### Purpose

The quality gate system validates the multi-agent RAG pipeline at each stage through **9 sequential gates** (Gate-0 through Gate-8). Each gate validates specific aspects of the system, emits dual-format reports (JSON + Markdown), and uses color-coded status indicators (GREEN/AMBER/RED) to communicate pass/fail with graduated severity.

### 9 Quality Gates

| Gate | Name | Script | Purpose |
|------|------|--------|---------|
| Gate-0 | Baseline Snapshot | `qa_step00_baseline.py` | Validates baseline corpus quality (document counts, age distribution, chunk counts) |
| Gate-1 | Embeddings Quality | `qa_step01_embeddings.py` | Validates OpenAI ada-002 embedding generation (dimension, L2 norm, cache integrity) |
| Gate-2 | Index Build & Integrity | `qa_step02_indexes.py` | Validates multi-index builds (FAISS, Weaviate, Pinecone) and sanity search |
| Gate-3 | MCP Tool Health | `qa_step03_mcp.py` | Validates MCP service health and contract conformance |
| Gate-4 | Router Heuristics | `qa_step04_router.py` | Validates query routing logic and backend coverage |
| Gate-5 | Graph Happy Path | `qa_step05_graph.py` | Validates LangGraph workflow end-to-end execution |
| Gate-6 | A2A & Compliance | `qa_step06_a2a.py` | Validates agent-to-agent communication and compliance checks |
| Gate-7 | Retrieval Evaluation | `qa_step07_retrieval_eval.py` | Evaluates retrieval quality (recall@10, nDCG@5, latency) |
| Gate-8 | Generation Evaluation | `qa_step08_generation_eval.py` | Evaluates end-to-end generation quality across personas |

**Location**: All gate scripts reside in `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/`

### Dual Report Format

Every gate produces two reports:

1. **JSON Report** (`reports/qa/stepXX_name.json`): Machine-readable structured data
2. **Markdown Report** (`reports/qa/stepXX_name.md`): Human-readable formatted summary

**Example**:
- `reports/qa/step00_baseline.json`
- `reports/qa/step00_baseline.md`

### Status Indicators

Gates use a three-tier color-coded status system:

- **GREEN**: All critical checks passed; proceed to next gate
- **AMBER**: Minor warnings; proceed with caution (e.g., latency within 110% threshold)
- **RED**: Critical failures; must fix and rerun before proceeding

---

## 2. Architecture & Design

### Gate Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     Gate Execution Flow                      │
└─────────────────────────────────────────────────────────────┘

1. Load Inputs
   ├── Previous gate reports (e.g., step00_baseline.json)
   ├── Configuration files (e.g., configs/eval.prompts.yaml)
   └── Data artifacts (e.g., data/interim/chunks/*.jsonl)

2. Perform Validation
   ├── Compute metrics from data sources
   ├── Compare against hardcoded or dynamic thresholds
   └── Record pass/fail for each check

3. Generate Checks Array
   ├── Check ID: "G{gate}-{sequence}" (e.g., "G0-01")
   ├── Metric name, actual value, threshold expression
   ├── Status: PASS | WARN | FAIL
   └── Evidence: file path for traceability

4. Determine Overall Status
   ├── GREEN: All checks pass
   ├── AMBER: Minor warnings within tolerance
   └── RED: Critical failures or multiple warnings

5. Write Dual Reports
   ├── JSON: reports/qa/stepXX_name.json
   └── Markdown: reports/qa/stepXX_name.md

6. Print Summary to stdout
   └── {"status": "GREEN|AMBER|RED", ...}

7. Exit
   └── Implicit exit 0 (status in report), or explicit exit 1 for RED
```

### Validation Strategy

**Three-Tier Thresholds**:

Most checks implement graduated thresholds:
- **PASS**: Metric meets or exceeds target (e.g., recall@10 ≥ 0.80)
- **WARN**: Metric slightly below target but within tolerance (e.g., 0.97 ≤ upsert_rate < 0.98)
- **FAIL**: Metric significantly below target

**Dynamic Baselines**:

Gates 0, 4, and 7 compute thresholds from baseline data:
- **Gate-0**: Publishes baseline_chunks, age_p50, domain_count for downstream gates
- **Gate-4**: Computes diversity threshold as `max(2.4, baseline_domains * 0.75 / 10.0)`
- **Gate-7**: Computes freshness threshold as `max(540, age_p50)` and coverage threshold from baseline domains

**Cross-Gate Dependencies**:

```
Gate-0 (Baseline)
  ├── Publishes: baseline_chunks, baseline_docs, age_p50, domain_count
  ├─→ Gate-1: Validates embedding_rows == baseline_chunks
  ├─→ Gate-2: Validates index counts match baseline_chunks
  ├─→ Gate-4: Uses age_p50 for freshness threshold
  └─→ Gate-7: Uses age_p50, domain_count for dynamic thresholds

Gate-1 (Embeddings)
  ├── Publishes: embeddings.parquet (1536-dim vectors)
  └─→ Gate-2: Loads embeddings for FAISS index build

Gate-2 (Indexes)
  ├── Publishes: index health metrics per backend
  └─→ Gate-7: Loads retrieval latency budgets from step03_mcp.json

Gate-3 (MCP Tools)
  ├── Publishes: latency budgets per backend (faiss, weaviate, pinecone)
  └─→ Gate-7: Uses budgets to validate retrieval performance

Gate-5 (Graph)
  ├── Executes: run_graph.py subprocess
  └─→ Gate-6: Requires session_id from graph execution

Gate-7 (Retrieval)
  ├── Loads: eval seed from data/interim/eval/salesforce_eval_seed.jsonl
  └─→ Gate-8: Shares MCP service connection pattern
```

### Reporting Architecture

**Dual-Format Pattern**:

All gates follow this pattern:

```python
# 1. Build machine-readable structure
machine = {
    "step": "step00_baseline",
    "gate": "Gate-0",
    "status": status,  # GREEN | AMBER | RED
    "checks": [...],   # Array of check objects
    "next_action": "continue" | "proceed_with_caution" | "fix_and_rerun",
    "timestamp": "2025-10-20T16:35:37.691116+00:00"  # ISO 8601 UTC
}

# 2. Write JSON report
with open("reports/qa/step00_baseline.json", "w") as f:
    json.dump(machine, f, ensure_ascii=False, indent=2)

# 3. Build human-readable lines
lines = [
    f"# STEP 0 — Baseline Snapshot (Gate‑0) — {status}",
    "",
    "Checks:",
    *[f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}"
      for c in checks],
    "",
    f"Gate-0 status: {status} — next_action: {next_action}",
]

# 4. Write Markdown report
with open("reports/qa/step00_baseline.md", "w") as f:
    f.write("\n".join(lines) + "\n")
```

**Console Output**:

All gates print JSON summary to stdout for pipeline integration:

```bash
$ conda run -n age python scripts/qa_step00_baseline.py
# ... validation output ...
{"status": "GREEN"}
```

---

## 3. File Inventory

### QA Step Scripts (Primary Quality Gates)

| Script | Gate | Lines | Purpose |
|--------|------|-------|---------|
| `qa_step00_baseline.py` | Gate-0 | 236 | Baseline corpus validation (5 checks) |
| `qa_step01_embeddings.py` | Gate-1 | 309 | Embedding generation and quality (5 checks) |
| `qa_step02_indexes.py` | Gate-2 | 399 | Multi-index build validation (7 checks) |
| `qa_step03_mcp.py` | Gate-3 | 428 | MCP service health and contracts (4 checks) |
| `qa_step04_router.py` | Gate-4 | 577 | Query routing validation (5 checks) |
| `qa_step05_graph.py` | Gate-5 | 132 | LangGraph execution validation (7 checks) |
| `qa_step06_a2a.py` | Gate-6 | 90 | Agent-to-agent compliance (5 checks) |
| `qa_step07_retrieval_eval.py` | Gate-7 | 862 | Retrieval quality evaluation (5 checks) |
| `qa_step08_generation_eval.py` | Gate-8 | 773 | End-to-end generation evaluation (4 checks) |

**Total**: 9 gate scripts, 3,806 lines

**Location**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_step*.py`

### QA Verification Scripts (Data Quality Checks)

| Script | Gate | Purpose |
|--------|------|---------|
| `qa_verify_collection.py` | G01 | Validates raw document collection completeness |
| `qa_verify_normalization.py` | G02 | Validates text normalization quality |
| `qa_verify_metadata.py` | G03 | Validates metadata extraction completeness |
| `qa_verify_chunking.py` | G04 | Validates document chunking quality |
| `qa_verify_dedupe.py` | G05 | Validates deduplication effectiveness |
| `qa_verify_link_health.py` | G06 | Validates URL allowlist compliance |
| `qa_verify_eval_seed.py` | G07 | Validates evaluation seed dataset |
| `qa_verify_day1_signoff.py` | G08 | Final Day-1 milestone validation |

**Total**: 8 verification scripts

**Note**: These operate on data preparation stages (collection → deduplication), distinct from the main quality gates (embeddings → generation).

**Location**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_verify_*.py`

### Configuration Files

| File | Purpose | Used By |
|------|---------|---------|
| `configs/eval.prompts.yaml` | Persona-specific keywords for Gate-8 | Gate-8 |
| `configs/compliance.template.yaml` | Critical/warning compliance rules | MCP safety.check stub |
| `configs/router.heuristics.yaml` | Query routing weights and rules | Gate-4, Gate-7 |
| `configs/vector.indexing.yaml` | Embedding model and FAISS parameters | Gate-1, Gate-2, Gate-7 |
| `configs/mcp.tools.yaml` | MCP service endpoint configuration | Gate-3, Gate-4, Gate-7, Gate-8 |
| `configs/langgraph.nodes.yaml` | Agent graph topology | Gate-5 |
| `configs/metadata.dictionary.yaml` | Metadata extraction rules | Processing scripts |
| `configs/normalization.rules.yaml` | Text normalization rules | Processing scripts |
| `configs/chunking.config.json` | Document chunking configuration | Processing scripts |
| `configs/agents.schema.yaml` | A2A communication schema | Gate-6 |

**Total**: 10 configuration files

**Location**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/configs/`

### Sample Report Files

**Gate Step Reports** (JSON + Markdown pairs):
- `reports/qa/step00_baseline.{json,md}` - Gate-0 baseline report
- `reports/qa/step01_embeddings.{json,md}` - Gate-1 embeddings report
- `reports/qa/step02_indexes.{json,md}` - Gate-2 indexes report
- `reports/qa/step03_mcp.{json,md}` - Gate-3 MCP tools report
- `reports/qa/step04_router.{json,md}` - Gate-4 router report
- `reports/qa/step05_graph.{json,md}` - Gate-5 graph report
- `reports/qa/step06_a2a.{json,md}` - Gate-6 A2A report
- `reports/qa/step07_retrieval_eval.{json,md}` - Gate-7 retrieval eval report
- `reports/qa/step08_generation_eval.{json,md}` - Gate-8 generation eval report
- `reports/qa/step08_debug.{json,md}` - Debug tool report

**Total**: 20 report files (10 JSON + 10 Markdown)

**Verification Gate Reports** (JSON only):
- `reports/qa/gate01_collection.json`
- `reports/qa/gate02_normalization.json`
- `reports/qa/gate03_metadata.json`
- `reports/qa/gate04_chunking.json`
- `reports/qa/gate05_dedupe.json`
- `reports/qa/gate06_link_health.json`
- `reports/qa/gate07_eval_seed.json`
- `reports/qa/gate08_day1_signoff.json`

**Total**: 8 verification reports

**Evaluation Trace Files**:
- `reports/eval/retrieval_failures.jsonl` - Failed retrieval queries log
- `reports/eval/generation_metrics.json` - Generation quality metrics
- `reports/eval/compliance_metrics.json` - Compliance check metrics
- `reports/router/step04_router_trace.jsonl` - Router decision trace
- `reports/router/step07_retrieval_trace.jsonl` - Retrieval evaluation trace
- `logs/mcp/step03_probes.jsonl` - MCP service probe log

**Total**: 6 trace/metric files

### Helper Scripts

| Script | Purpose |
|--------|---------|
| `embedding_utils.py` | Shared embedding functions (`embed_text`, `embed_batch`, cost estimation) |
| `router_core.py` | Query routing decision logic |
| `common.py` | Shared utilities (`ensure_dir`, `now_iso`, `sha1_8`, `RateLimiter`) |
| `build_eval_seed.py` | Builds evaluation seed dataset from inventory |
| `build_eval_generation_prompts.py` | Generates evaluation prompts for Gate-8 |
| `debug_sec_retrieval.py` | SEC document retrieval debugging |

**Total**: 6 helper scripts

---

## 4. Core Components Deep Dive

### Gate-0: Baseline Snapshot (`qa_step00_baseline.py`)

**Purpose**: Validate baseline corpus quality before embeddings

**Checks** (5 total):

#### G0-01: Document Count
- **Metric**: `baseline_docs` (documents with publish_date)
- **Threshold**: ≥ 80 documents
- **Calculation**: Count rows in `data/final/inventory/salesforce_inventory.csv` with non-empty `publish_date` (lines 48-62)
- **Status**: PASS if ≥ 80, FAIL otherwise

#### G0-02: Publish Date Coverage
- **Metric**: `publish_date_pct` (proportion with dates)
- **Threshold**: ≥ 0.98 (98%)
- **Calculation**: `has_date / max(1, inv_total)` (line 65)
- **Status**: PASS if ≥ 0.98, FAIL otherwise

#### G0-03: Evaluation Seed Size
- **Metric**: `seed_eval_size` (number of eval queries)
- **Threshold**: ≥ 40 queries
- **Calculation**: Count lines in `data/interim/eval/salesforce_eval_seed.jsonl` (lines 97-100)
- **Status**: PASS if ≥ 40, FAIL otherwise

#### G0-04: Chunk-to-Document Ratio
- **Metric**: `baseline_chunks` (total document chunks)
- **Threshold**: ≥ `baseline_docs` (at least 1 chunk per doc)
- **Calculation**: Count JSONL lines in `data/interim/chunks/*.chunks.jsonl` (lines 70-79)
- **Status**: PASS if ≥ baseline_docs, FAIL otherwise

#### G0-05: Domain Diversity
- **Metric**: `baseline_domain_count` (unique source domains)
- **Threshold**: ≥ 3 domains
- **Calculation**: Count unique `source_domain` values from inventory (lines 60-62)
- **Status**: PASS if ≥ 3, FAIL otherwise

**Additional Statistics** (non-gating):
- Age distribution: p50/p90 age in days (lines 84-93)
- Age buckets: ≤90d, ≤180d, ≤365d, >365d (lines 55-59)
- Token distribution: p50/p90 token count (lines 86-87)

**Status Logic** (lines 152-176):
- **GREEN**: All 5 checks pass
- **AMBER**: Exactly 1 check fails within 10% relative margin
- **RED**: Multiple failures or single failure exceeds 10% margin

**Reports**:
- JSON: `reports/qa/step00_baseline.json` (line 194)
- Markdown: `reports/qa/step00_baseline.md` (line 229)

**Evidence**: `data/final/inventory/salesforce_inventory.csv`, `data/interim/chunks/*.chunks.jsonl`, `data/interim/eval/salesforce_eval_seed.jsonl`

---

### Gate-1: Embeddings Quality (`qa_step01_embeddings.py`)

**Purpose**: Generate OpenAI ada-002 embeddings and validate quality

**Checks** (5 total):

#### G1-01: Embedding Row Count Match
- **Metric**: `embedding_rows` (generated embeddings)
- **Threshold**: == `baseline_chunks` (from Gate-0 G0-04)
- **Calculation**: Count vectors in `data/vector/embeddings/embeddings.parquet` (line 234)
- **Status**: PASS if exact match, FAIL otherwise

#### G1-02: Dimension Validation
- **Metric**: `vector_dim` (embedding dimensionality)
- **Threshold**: == 1536 (OpenAI ada-002 standard)
- **Source**: `configs/vector.indexing.yaml` via `read_yaml_dim()` (line 117)
- **Status**: Always PASS (enforced by `embed_text()` validation)

#### G1-03a: Zero Vector Detection
- **Metric**: `zero_vectors` (count with L2 norm = 0)
- **Threshold**: == 0
- **Calculation**: L2 norm via `l2_norm()` at line 185, check at line 189
- **Status**: PASS if zero, FAIL otherwise

#### G1-03b: NaN Vector Detection
- **Metric**: `nan_vectors` (count with NaN values)
- **Threshold**: == 0
- **Calculation**: `any((x != x) for x in v)` check at line 189
- **Status**: PASS if zero, FAIL otherwise

#### G1-04: Norm Outlier Detection
- **Metric**: `pct_norm_outliers` (proportion outside expected range)
- **Threshold**: ≤ 0.005 (PASS), ≤ 0.015 (WARN), else FAIL
- **Calculation** (lines 205-214):
  - Compute Q1, median, Q3 via `quartiles()` (lines 51-63)
  - IQR = Q3 - Q1
  - Outlier bounds: `[median - 4*IQR, median + 4*IQR]`
  - If IQR = 0, only exact median matches pass
- **Status**: Graduated (PASS/WARN/FAIL)

**Embedding Generation Process**:

1. **Pre-flight Cost Estimation** (lines 140-161):
   - Function: `estimate_embedding_cost()` from `embedding_utils.py:214-240`
   - Displays: total chunks, cached embeddings, uncached texts, estimated cost
   - Confirmation: User prompted unless `AG1_AUTO_CONFIRM=1` (lines 154-161)

2. **Batch Embedding** (lines 163-176):
   - Function: `embed_batch()` from `embedding_utils.py:152-211`
   - Model: OpenAI `text-embedding-ada-002` (line 79 in embedding_utils.py)
   - Batch size: From `configs/vector.indexing.yaml` (default 100, lines 122-126)
   - Cache mechanism (`embedding_utils.py:39-67`):
     - Cache key: SHA-256 hash (first 16 chars) of text (line 36)
     - Cache validation: MD5 hash for integrity check (line 48)
     - Cache path: `data/cache/embeddings/{cache_key}.json`
   - Retry logic (`embedding_utils.py:70-83`):
     - Decorator: `@retry` with exponential backoff (1s, 4-10s wait)
     - Max attempts: 3 retries
     - Retry conditions: `APIError`, `APIConnectionError`, `RateLimitError`

3. **Empty Text Handling** (`embedding_utils.py:104-106`):
   - Returns `[0.001] * 1536` for empty/whitespace-only text
   - Prevents zero vectors that would fail G1-03a

**Environment Variables**:
- **`OPENAI_API_KEY`** (required): OpenAI API key loaded from `.env` (lines 24-29 in embedding_utils.py)
- **`AG1_AUTO_CONFIRM`** (optional): Skip cost confirmation if `"1"`, `"true"`, `"yes"`, `"y"` (line 154)

**Status Logic** (lines 260-269):
- **GREEN**: All checks PASS
- **AMBER**: G1-01 through G1-03b PASS, G1-04 shows WARN
- **RED**: Any core check fails OR outlier rate > 1.5%

**Reports**:
- JSON: `reports/qa/step01_embeddings.json` (line 281)
- Markdown: `reports/qa/step01_embeddings.md` (line 302)

**Data Files**:
- Embeddings: `data/vector/embeddings/embeddings.parquet` (line 112)
  - Schema: chunk_id, doc_id, seq_no, token_count, l2_norm, vector (lines 103-111)
- Stats: `data/vector/embeddings/embedding_stats.json` (lines 229-230)

**Evidence**: `data/vector/embeddings/embeddings.parquet`, `data/vector/embeddings/embedding_stats.json`, `configs/vector.indexing.yaml`

---

### Gate-2: Index Build & Integrity (`qa_step02_indexes.py`)

**Purpose**: Build multi-index (FAISS, Weaviate, Pinecone) and validate integrity

**Critical**: MUST use `ageFaiss` conda environment (Python 3.12) to avoid OpenMP conflicts

**Checks** (7 total):

#### G2-01: Pinecone Upsert Rate
- **Metric**: `pinecone_upsert_rate` = upserted / baseline_chunks
- **Threshold**: ≥ 0.98 (PASS), ≥ 0.97 (WARN), else FAIL
- **Source**: `data/vector/pinecone/index_manifest.json` (lines 282-292)
- **Note**: Currently simulated; no actual Pinecone API calls (line 285)

#### G2-02: Weaviate Upsert Rate
- **Metric**: `weaviate_upsert_rate` = inserted / baseline_chunks
- **Threshold**: ≥ 0.98 (PASS), ≥ 0.97 (WARN), else FAIL
- **Source**: `data/vector/weaviate/index_manifest.json` (lines 294-308)
- **Schema**: 11 required properties saved to `data/vector/weaviate/schema_applied.json` (lines 296-305)

#### G2-03: FAISS Count Ratio
- **Metric**: `faiss_count_ratio` = faiss_count / baseline_chunks
- **Threshold**: ≥ 0.98 (PASS), ≥ 0.97 (WARN), else FAIL
- **Source**: `data/vector/faiss/faiss_manifest.json` (line 162)

#### G2-04: Metadata Completeness
- **Metric**: `pct_missing_required_metadata` (proportion with incomplete metadata)
- **Threshold**: ≤ 0.02 (PASS), ≤ 0.03 (WARN), else FAIL
- **Calculation**: `compute_metadata_missing()` at lines 184-209
  - Required fields: `doctype`, `publish_date`, `url`, `title`
  - Data source: `data/interim/normalized/*.json` (lines 188-193)

#### G2-05: FAISS Roundtrip Error
- **Metric**: `faiss_roundtrip_error_max` (maximum reconstruction error)
- **Threshold**: ≤ 0.001 (PASS), ≤ 0.01 (WARN), else FAIL
- **Calculation**: `build_faiss()` at lines 136-149
  - Test: Query 100 random vectors (seed=42) with k=1
  - Error: L2 distance between query and retrieved nearest neighbor

#### G2-06: Sanity Search Top-K Count
- **Metric**: `sanity_search_min_topk` (minimum non-empty results across 3 queries)
- **Threshold**: ≥ 3 (PASS), == 2 (WARN), else FAIL
- **Calculation**: `run_sanity_search()` at lines 212-271

#### G2-07: Sanity Search Keyword Hits
- **Metric**: `sanity_keyword_hit_min_top10` (minimum keyword matches in top-10)
- **Threshold**: ≥ 1 (PASS), else FAIL
- **Purpose**: Verifies semantic search returns contextually relevant results

**FAISS Index Build Process** (`build_faiss()` at lines 80-164):

1. **Conditional Build**:
   - **Disabled**: If `AG2_DISABLE_FAISS=1` env var set (lines 87-98)
   - **Import failure fallback**: If `import faiss` fails (lines 100-113)
   - **Successful build**: Lines 115-164

2. **Index Configuration** (lines 121-130):
   - **Default**: HNSW (Hierarchical Navigable Small World) from config (line 121)
   - **Parameters**:
     - `M`: Neighbors per node (default 32, line 122)
     - `efConstruction`: Build-time search depth (default 200, lines 124-125)
     - `efSearch`: Query-time search depth (default 128, lines 126-127)
   - **Fallback**: `IndexFlatL2` (exact search) if type != "HNSW" (line 130)

3. **Build Steps**:
   - Convert to numpy float32 array (line 132)
   - Add vectors: `idx.add(xb)` (line 133)
   - Write binary: `data/vector/faiss/index.faiss` (line 134)
   - Roundtrip validation (lines 136-149)

4. **ID Mapping** (`write_idmap()` at lines 167-181):
   - Path: `data/vector/faiss/idmap.parquet`
   - Schema: id (FAISS internal), chunk_id, doc_id, seq_no
   - Purpose: Reverse lookup after FAISS search returns integer indices

**Sanity Search Validation** (`run_sanity_search()` at lines 212-271):

- **Test Queries** (lines 228-232):
  1. `"latest earnings results"`
  2. `"agentforce product announcement"`
  3. `"remaining performance obligation definition"`

- **Keyword Sets** (lines 245-249):
  1. `{"earnings", "results", "gaap", "guidance", "rpo"}`
  2. `{"agentforce", "product", "announce", "ai"}`
  3. `{"remaining", "performance", "obligation", "rpo", "definition"}`

- **Search Method**: Exact L2 search using numpy (not FAISS) for reproducibility (lines 212-226)

**Status Logic** (lines 350-366):
- **GREEN**: All 7 checks PASS
- **AMBER**: Exactly 1 of {G2-01, G2-02, G2-03, G2-04} shows WARN, OR G2-06 is WARN and all primaries PASS
- **RED**: Otherwise

**Reports**:
- JSON: `reports/qa/step02_indexes.json` (line 378)
- Markdown: `reports/qa/step02_indexes.md` (line 396)
- Health summary: `data/final/reports/index_health.json` (line 333)

**Data Files**:
- FAISS: `data/vector/faiss/index.faiss`, `data/vector/faiss/idmap.parquet`, `data/vector/faiss/faiss_manifest.json`
- Pinecone: `data/vector/pinecone/index_manifest.json`
- Weaviate: `data/vector/weaviate/schema_applied.json`, `data/vector/weaviate/index_manifest.json`

**Evidence**: Multi-index manifests, sanity search results

---

### Gate-3: MCP Tool Health & Contract Conformance (`qa_step03_mcp.py`)

**Purpose**: Validate MCP service health and contract conformance

**Checks** (4 total):

#### G3-01: Health Endpoints
- **Metric**: `health_endpoints_ok` (tools responding healthy)
- **Threshold**: == 5 tools
- **Calculation**: Probe `/healthz` for each tool, count status 200 responses (lines 275-282)
- **Status**: PASS if all 5 tools healthy, FAIL otherwise

#### G3-02: Contract Conformance
- **Metric**: `contract_ok_rate_{tool_name}` per tool
- **Threshold**: == 1.0 (100% of invalid requests properly rejected)
- **Calculation**: Send 1 valid + 2 invalid requests per tool, validate error codes (lines 284-303)
- **Status**: PASS if rate == 1.0, FAIL otherwise

#### G3-03: Latency Budgets
- **Metric**: `{backend}_latency_budget` per backend (faiss, weaviate, pinecone)
- **Threshold**: p50, p95 ≤ budget
- **Calculation** (lines 305-365):
  - Issue 5 queries per backend from eval seed
  - Compute p95 via `p95()` function (lines 355-360)
  - Budget: `min(documented_budget, p95 * 1.20)` (line 365)
- **Status**: PASS if both p50 and p95 meet budget, WARN if within 110%, FAIL otherwise

#### G3-04: Stability
- **Metric**: `timeout_rate`
- **Threshold**: == 0.0
- **Calculation**: Track timeouts during latency sampling (line 344)
- **Status**: PASS if no timeouts, FAIL otherwise

**MCP Stub Server Implementation** (`start_stub_servers()` at lines 40-205):

1. **Initialization** (lines 42-67):
   - Load embeddings from `data/vector/embeddings/embeddings.parquet` (line 45)
   - Load chunk text from `data/interim/chunks/*.chunks.jsonl` (lines 56-67)
   - Build embedding matrix `xb` and chunk map

2. **Request Handlers** (lines 79-180):
   - **Health**: `handle_health()` returns `{"status": "ok"}` (lines 79-80)
   - **KB Search**: `handle_invoke_kb()` at lines 82-156
     - Validates JSON body, method, params (lines 84-95)
     - Simulates backend latency: faiss 5-10ms, weaviate 40-80ms, pinecone 80-160ms (lines 98-101)
     - Performs vector search via NumPy L2 distance (lines 102-109)
     - Applies lexical reranking: `0.7 * vec_sim + 0.3 * lex_boost` (line 133)
   - **Simple handlers**: `handle_invoke_simple()` for web.fetch, link.resolve, crm.lookup, safety.check (lines 158-179)

3. **Server Binding** (lines 182-203):
   - Creates 5 aiohttp applications on ports 7801-7805
   - Binds handlers to tools from `configs/mcp.tools.yaml` (lines 185-198)

**Documented Latency Budgets** (line 364):
- faiss: 300ms
- weaviate: 1000ms
- pinecone: 1500ms

**Actual budget**: `min(doc_budget, observed_p95 * 1.20)`

**Status Logic** (lines 382-395):
- **GREEN**: All checks PASS
- **AMBER**: Exactly 1 latency check is WARN, no FAILs
- **RED**: Any FAIL or multiple WARNs

**Reports**:
- JSON: `reports/qa/step03_mcp.json` (line 399)
- Markdown: `reports/qa/step03_mcp.md` (line 410)
- Probe log: `logs/mcp/step03_probes.jsonl` (lines 349-352)

**Evidence**: `configs/mcp.tools.yaml`, probe log with timing and error codes

---

### Gate-4: Router Heuristics & Coverage (`qa_step04_router.py`)

**Purpose**: Validate query routing logic and backend coverage

**Checks** (5 total):

#### COV-{backend}: Route Share
- **Metric**: `{backend}_route_share` for each backend (pinecone, weaviate, faiss)
- **Threshold**: ≥ 0.10 OR ≥ 1 route
- **Calculation**: Count queries routed to each backend (lines 413-424)
- **Status**: PASS if either condition met, FAIL otherwise

#### EMP-001: Empty Result Rate
- **Metric**: `empty_result_rate`
- **Threshold**: ≤ 0.02 (2%)
- **Calculation**: Count queries with zero results (lines 426-432)
- **Status**: PASS if below threshold, FAIL otherwise

#### EMP-002: Auto-Retry Success Rate
- **Metric**: `auto_retry_success_rate`
- **Threshold**: ≥ 0.95 (95% if any empty results)
- **Calculation**: Count successful fallback retries (lines 433-439)
- **Status**: PASS if no empty results or retry rate meets threshold, FAIL otherwise

#### FRS-001: Freshness
- **Metric**: `avg_doc_age_days`
- **Threshold**: ≤ max(365, baseline_p50)
- **Calculation**: Average publish_date age across top-10 results (lines 366-377, 441-448)
- **Status**: PASS if within threshold, WARN if within 110%, FAIL otherwise

#### DIV-001: Diversity
- **Metric**: `mean_unique_domains_top10`
- **Threshold**: ≥ 2.4 (WARN if ≥ 2.0)
- **Calculation**: Average count of unique source_domains in top-10 (lines 368-379, 450-459)
- **Status**: PASS if ≥ 2.4, WARN if ≥ 2.0, FAIL otherwise

**Router Decision Logic** (`decide_backend()` from `router_core.py:72-100`):

1. **Rule-Based Matching** (lines 81-89):
   - Iterate through rules from `configs/router.heuristics.yaml`
   - Check if any keyword from rule appears in lowercase query
   - Return backend from rule (e.g., `[results, earnings, fiscal]` → pinecone)

2. **Persona Bias** (lines 91-94):
   - Check `persona_bias` map for persona key
   - Return mapped backend (e.g., `vp_sales_ops` → pinecone)

3. **Heuristic Fallback** (lines 96-100):
   - If query ≤ 4 words or contains definitional keywords → faiss
   - Otherwise → weaviate

**Reranking & Diversity** (`rerank()` from `router_core.py:113-183`):

- **Scoring** (lines 134-161):
  - Similarity: `1.0 / (1.0 + abs(score))` (line 140)
  - Recency: `max(0.0, 1.0 - (days / 730.0))` (line 153) - linear decay over 2 years
  - Diversity: +0.1 bonus for new domain (line 156)
  - Weighted: `w["similarity"] * sim + w["recency"] * rec + w["diversity"] * div` (line 158)
  - Default weights: `{similarity: 0.6, recency: 0.3, diversity: 0.1}` from config

- **Domain-Aware Selection** (lines 165-179):
  - Sort by final score
  - Enforce per-domain cap (default 2) within top_k
  - Skip if domain already has 2 entries

**Fallback Mode Warnings** (lines 475-490):
- If `fallback_mode == WARN` and downgrades occurred:
  - Checks `warn_on_offline` flag: if using offline mode, adds warning and downgrades GREEN → AMBER
  - Checks `warn_on_external` flag: if using external service, adds warning and downgrades GREEN → AMBER

**Status Logic** (lines 461-473):
- **GREEN**: All checks PASS
- **AMBER**: Only WARN status checks fail
- **RED**: Any FAIL status

**Reports**:
- JSON: `reports/qa/step04_router.json` (line 492)
- Markdown: `reports/qa/step04_router.md` (line 532)
- Trace log: `reports/router/step04_router_trace.jsonl` (line 383)

**Evidence**: `configs/router.heuristics.yaml`, trace log with routing decisions

---

### Gate-5: Graph Happy Path (`qa_step05_graph.py`)

**Purpose**: Validate LangGraph workflow end-to-end execution

**Checks** (7 total):

#### G5-01: Node Coverage
- **Metric**: `nodes_executed`
- **Expected**: `["Intake", "Planner", "Retriever", "Synthesizer", "Consolidator", "Stylist", "A2A", "Assembler"]`
- **Calculation**: Extract from `state["metrics"]["nodes_executed"]` (lines 65-69)
- **Status**: PASS if exact match in order, FAIL otherwise

#### G5-02: Latency Budget
- **Metric**: `total_runtime_ms`
- **Threshold**: ≤ 30000 ms (30 seconds), WARN ≤ 36000 ms (120%)
- **Calculation**: From `timing["total_runtime_ms"]` (lines 71-75)
- **Status**: PASS if ≤ 30s, WARN if ≤ 36s, FAIL otherwise

#### G5-03: Insight Count
- **Metric**: `insight_cards`
- **Threshold**: == 5
- **Calculation**: `len(insights)` (lines 77-79)
- **Status**: PASS if count == 5, FAIL otherwise

#### G5-04: Distinct Sources
- **Metric**: `distinct_sources`
- **Threshold**: ≥ 4
- **Calculation**: `distinct_sources(insights)` counts unique `source_domain` values (lines 18-19, 81-83)
- **Status**: PASS if ≥ 4, FAIL otherwise

#### G5-05: Recency
- **Metric**: `insights_within_12mo`
- **Threshold**: ≥ 2
- **Calculation**: `count_recent(insights)` counts cards where `(today - date).days <= 365` (lines 22-35, 85-87)
- **Status**: PASS if ≥ 2, FAIL otherwise

#### G5-06: Email Schema
- **Metric**: `email_schema_ok`
- **Required Fields**: `["subject", "body", "unsubscribe_block", "company_info_block", "proof_points"]`
- **Validation**: All fields non-empty, `proof_points` is list (lines 89-93)
- **Status**: PASS if validation succeeds, FAIL otherwise

#### G5-07: Proof Points Resolution
- **Metric**: `proof_points_resolve`
- **Validation**: All proof_point.id exist in insight_ids (lines 95-102)
- **Status**: PASS if no dangling references, FAIL otherwise

**Graph Execution** (lines 43-55):
- Runs `scripts/run_graph.py --company Salesforce --persona vp_customer_experience` via subprocess
- Extracts session_id from stdout JSON (lines 47-53)
- Loads artifacts from `outputs/{session_id}/` and `state/session-{session_id}.json`

**Status Logic** (line 104):
- **GREEN**: All checks PASS
- **AMBER**: At least one WARN, no FAILs
- **RED**: At least one FAIL

**Reports**:
- JSON: `reports/qa/step05_graph.json` (line 107)
- Markdown: `reports/qa/step05_graph.md` (line 119)

**Evidence**: `state/session-{session_id}.json`, `outputs/{session_id}/insights.json`, `outputs/{session_id}/email.json`, `outputs/{session_id}/timing.json`

---

### Gate-6: A2A & Compliance QA (`qa_step06_a2a.py`)

**Purpose**: Validate agent-to-agent communication and compliance

**Requires**: `--session-id` CLI argument from Gate-5 execution

**Checks** (5 total):

#### G6-01: Negotiation Rounds
- **Metric**: `negotiation_rounds` from `compliance_report.json`
- **Threshold**: ≤ 2
- **Evidence**: `outputs/<session-id>/a2a_transcript.jsonl`
- **Status**: PASS if ≤ 2, FAIL otherwise

#### G6-02: Critical Flags
- **Metric**: Count of `flags.critical` array
- **Threshold**: == 0
- **Evidence**: `outputs/<session-id>/compliance_report.json`
- **Status**: PASS if zero, FAIL otherwise

#### G6-03: Email Body Length
- **Metric**: Word count of `email.body`
- **Threshold**: ≤ 160 words
- **Calculation**: `word_count()` function (lines 17-18)
- **Status**: PASS if ≤ 160, FAIL otherwise

#### G6-04: Readability Grade
- **Metric**: Flesch-Kincaid grade level approximation
- **Threshold**: ≤ 10
- **Calculation**: `0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59` (lines 21-26)
- **Status**: PASS if ≤ 10, FAIL otherwise

#### G6-05: Proof Points Reference
- **Metric**: All `email.proof_points[*].id` exist in `insights[*].id`
- **Threshold**: == true
- **Evidence**: `outputs/<session-id>/email.json`
- **Status**: PASS if no dangling references, FAIL otherwise

**Status Logic** (line 70):
- **GREEN**: All checks PASS
- **AMBER**: At least one WARN (note: no checks currently emit WARN)
- **RED**: Any check FAIL

**Reports**:
- JSON: `reports/qa/step06_a2a.json` (line 72)
- Markdown: `reports/qa/step06_a2a.md` (line 78)

**Evidence**: Session-specific artifacts in `outputs/<session-id>/`

---

### Gate-7: Retrieval Evaluation (`qa_step07_retrieval_eval.py`)

**Purpose**: Evaluate retrieval quality across seed dataset

**Checks** (5 total):

#### G7-01: Recall@10
- **Metric**: `recall@10` (proportion of queries where expected chunk in top-10)
- **Threshold**: ≥ 0.80
- **Calculation**: Count queries where `expected_chunk_id` appears in top-10 results, divide by total (lines 408-414, 584)
- **Status**: PASS if ≥ 0.80, FAIL otherwise

#### G7-02: nDCG@5
- **Metric**: `nDCG@5` (normalized discounted cumulative gain at rank 5)
- **Threshold**: ≥ 0.60
- **Calculation**: DCG = `1.0 / log2(rank + 1)` if rank ≤ 5, averaged across queries (lines 420-421, 585)
- **Status**: PASS if ≥ 0.60, FAIL otherwise

#### G7-03: Coverage
- **Metric**: `coverage_unique_domains_top10_mean` (mean unique domains in top-10)
- **Threshold**: ≥ max(3.0, 0.75 * baseline_domains / 10)
- **Calculation**: Count unique `source_domain` values in top-10 per query, average (lines 514-531, 591)
- **Status**: PASS if meets threshold, FAIL otherwise
- **Override**: Can be disabled via `AG7_IGNORE_COVERAGE=1` (lines 258-260, 639)

#### G7-04: Freshness
- **Metric**: `freshness_mean_age_days` (mean document age in top-10)
- **Threshold**: ≤ max(540, age_p50 from Gate-0)
- **Calculation**: `(today - publish_date).days` averaged, defaults to 365 if missing (lines 522-533, 592)
- **Status**: PASS if meets threshold, FAIL otherwise

#### G7-05: Latency Budgets
- **Metric**: Per-backend P50 and P95 latencies
- **Threshold**: P50 ≤ budget_p95 AND P95 ≤ budget_p95 for each backend
- **Calculation**: Load budgets from `step03_mcp.json`, measure latencies during retrieval (lines 614-629, 642)
- **Status**: PASS if all backends meet budget, FAIL otherwise
- **Override**: Can be relaxed via `AG7_LATENCY_MULTIPLIER=<float>` (lines 261-264, 615)

**Diagnostic Metrics** (non-gating, lines 587-611):
- `doc_recall@10`: Recall at document level (vs chunk level)
- `soft_recall@10`: Near-miss recall (same doc, adjacent chunk)
- `doc_nDCG@5`: Document-level nDCG
- `near_miss_rate`: Difference between doc_recall and chunk_recall
- `recall_at`: Recall curves at k ∈ {1, 3, 5, 10}
- `rank_stats`: P50/P75/P90/max ranks for hits

**Environment Variables**:
- `AG7_IGNORE_COVERAGE=1`: Skip coverage check (lines 260, 639)
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets (e.g., `3.0`) (lines 262-264, 615)
- `AG7_DEBUG=1`: Enable debug mode (line 345)
- `AG7_TRACE=1`: Enable trace (lines 346-352)
- `AG7_ANALYZE_TOPK=<int>`: Top-K for analysis (default 10, lines 270-273)
- `AG7_NEAR_SEQ_TOL=<int>`: Near-miss sequence tolerance (default 1, lines 275-277)
- `AG7_TOPK_SLICES=<csv>`: Recall@k slices (default "1,3,5,10", lines 279-283)

**Status Logic** (lines 644-676):
- **GREEN**: All checks PASS
- **AMBER**: Exactly one FAIL among {G7-02, G7-03, G7-04, G7-05}, OR service degraded in WARN mode
- **RED**: G7-01 (recall@10) fails OR multiple checks fail

**Reports**:
- JSON: `reports/qa/step07_retrieval_eval.json` (line 678)
- Markdown: `reports/qa/step07_retrieval_eval.md` (line 791)
- Trace: `reports/router/step07_retrieval_trace.jsonl` (lines 364-366)
- Failures: `reports/eval/retrieval_failures.jsonl` (lines 22, 552-576)

**Evidence**: Eval seed, trace log, failure log

---

### Gate-8: Generation & Compliance Evaluation (`qa_step08_generation_eval.py`)

**Purpose**: Evaluate end-to-end generation quality across ≥3 personas

**Runs**: 10 graph executions with different prompts

**Checks** (4 total):

#### G8-01: Structural Pass Rate
- **Metric**: `structural_pass_rate`
- **Threshold**: == 1.0 (all runs structurally valid)
- **Criteria per run**:
  - `insights_count == 5`
  - `distinct_sources >= 4`
  - `recent_count >= 2` (within 12 months)
  - `email_schema_ok == true`
  - `proof_points_resolve == true`
- **Calculation**: `sum(passes) / total_runs` (lines 320-332)
- **Status**: PASS if 1.0, FAIL otherwise

#### G8-02: Critical Flags Total
- **Metric**: `critical_flags_total`
- **Threshold**: == 0
- **Calculation**: Sum of `len(flags.critical)` across all runs (line 373)
- **Failed runs**: Contribute `["RUN_FAILED"]` (line 365)
- **Status**: PASS if zero, FAIL otherwise

#### G8-03: Length/Readability Pass Runs
- **Metric**: `length_readability_pass_runs`
- **Threshold**: ≥ 9 (out of 10 runs)
- **Criteria per run**: `word_count <= 160 AND readability_grade <= 10.0`
- **Failed runs**: Contribute `word_count=999, readability_grade=99.0` (lines 368-369)
- **Calculation**: Count runs meeting criteria (lines 375-380)
- **Status**: PASS if ≥ 9, FAIL otherwise

#### G8-04: Persona Keyword Hits Average
- **Metric**: `persona_keyword_hits_avg`
- **Threshold**: ≥ 2.0
- **Calculation**: Mean of `persona_keyword_hits` across all runs (lines 335-336)
  - Loads persona keywords from `configs/eval.prompts.yaml` (line 154)
  - Case-insensitive matching in email body (lines 162-169)
- **Status**: PASS if ≥ 2.0, FAIL otherwise

**Single Prompt Execution** (`run_one_prompt()` at lines 191-276):

1. **Invoke Graph** (lines 210-237):
   - Runs `python3 scripts/run_graph.py --company <company> --persona <persona>`
   - Timeout: Default 30s (configurable via `--timeout`)
   - Parses session_id from stdout JSON

2. **Load Outputs** (lines 240-256):
   - `insights.json`, `email.json`, `compliance_report.json`

3. **Validate Structure** (lines 259-261):
   - Calls `validate_structure()` (lines 97-146)
   - Adds `persona_keyword_hits` via function (lines 149-171)

4. **Extract Compliance** (lines 264-271):
   - Critical/warning flags from `compliance_report`
   - Word count and readability grade

**Status Logic** (lines 437-448):
- **GREEN**: All checks PASS
- **AMBER**: Exactly one check fails AND it's G8-03 or G8-04
- **RED**: Otherwise

**Reports**:
- JSON: `reports/qa/step08_generation_eval.json` (line 475)
- Markdown: `reports/qa/step08_generation_eval.md` (line 509)
- Generation metrics: `reports/eval/generation_metrics.json` (line 469)
- Compliance metrics: `reports/eval/compliance_metrics.json` (line 472)

**Exit Behavior**:
- Exit 1 if status is RED (lines 700-701)
- Otherwise exit 0 (GREEN or AMBER)

**Evidence**: `configs/eval.prompts.yaml`, generation/compliance metric files

---

## 5. Configuration & Settings

### Evaluation Configuration (`configs/eval.prompts.yaml`)

**Purpose**: Defines persona-specific keywords for Gate-8 evaluation

**Schema**:
```yaml
personas:
  <persona_name>:
    - keyword1
    - keyword2
```

**Defined Personas**:
- **vp_customer_experience**: nps, csat, contact center, omnichannel, agent productivity, self-service, first contact resolution
- **cio**: data integration, governance, security, tco, platform, apis, real-time
- **vp_sales_ops**: pipeline, forecast accuracy, win rate, productivity, automation

**Used By**: Gate-8 (`qa_step08_generation_eval.py:149-171`) - loads and counts keyword matches in email body

### Compliance Configuration (`configs/compliance.template.yaml`)

**Purpose**: Defines critical and warning compliance rules for email validation

**Critical Rules** (lines 1-13):
- `MISSING_UNSUBSCRIBE`: Email must contain unsubscribe section
- `MISSING_COMPANY_INFO`: Email must contain sender company info block
- `UNCITED_CLAIM`: Quantitative claims without supporting proof points
- `PROHIBITED_PHRASE`: Email contains prohibited phrases

**Warning Rules** (lines 15-25):
- `EXCESS_LENGTH`: Email body exceeds max_words (default: 160)
- `READABILITY`: Email readability exceeds max_grade (default: 10)

**Prohibited Phrases**: guaranteed, free money, no strings attached, risk-free, 100% safe

**Used By**: MCP safety.check stub (`tool_safety_check_server.py:14, 87`)

### Router Configuration (`configs/router.heuristics.yaml`)

**Purpose**: Query routing weights and rules

**Weights** (lines 1-4):
```yaml
weights:
  similarity: 0.5
  recency: 0.3
  diversity: 0.2
```

**Persona Bias** (lines 6-10):
```yaml
persona_bias:
  vp_sales_ops: pinecone
  cio: weaviate
  vp_customer_experience: faiss
```

**Rules** (lines 12-39):
```yaml
rules:
  - if:
      has_keywords: [results, earnings, fiscal, guidance, revenue]
    then:
      backend: pinecone
      reason: PR_QUERY
  # ... 2 more rules
```

**Fallback Order** (line 41): `[faiss, weaviate, pinecone]`

**Top K Default** (line 42): `10`

**Used By**: Gate-4 (`qa_step04_router.py:188`), Gate-7 (`qa_step07_retrieval_eval.py:253`)

### Vector Configuration (`configs/vector.indexing.yaml`)

**Embedding** (lines 1-5):
```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20
```

**FAISS** (lines 7-12):
```yaml
faiss:
  type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128
```

**Pinecone** (lines 14-18):
```yaml
pinecone:
  index_name: demo-index
  namespace: default
  metric: cosine
```

**Weaviate** (lines 20-22):
```yaml
weaviate:
  class_name: Doc
```

**Used By**: Gate-1 (`qa_step01_embeddings.py:117`), Gate-2 (`qa_step02_indexes.py:276`), Gate-7 (`qa_step07_retrieval_eval.py:209`)

### MCP Tools Configuration (`configs/mcp.tools.yaml`)

**Tools Section** (lines 1-21):
```yaml
tools:
  - name: kb.search
    host: 127.0.0.1
    port: 7801
    timeout_ms: 2000
  # ... 4 more tools (web.fetch, link.resolve, crm.lookup, safety.check)
```

**Fallback Section** (lines 23-34):
```yaml
fallback:
  mode: default  # default | warn | strict
  policy:
    log_downgrades: true
    retry_attempts: 1
    connection_timeout_ms: 2000
    warn_on_offline: true
    warn_on_external: false
```

**Used By**: Gate-3 (`qa_step03_mcp.py:198`), Gate-4 (`qa_step04_router.py`), Gate-7 (`qa_step07_retrieval_eval.py`), Gate-8 (`qa_step08_generation_eval.py`)

### LangGraph Configuration (`configs/langgraph.nodes.yaml`)

**Purpose**: Agent graph topology

**Used By**: Gate-5 (graph execution validation)

### Gate-Specific Hardcoded Thresholds

Most gate thresholds are hardcoded in the gate scripts themselves:

**Gate-0 Thresholds** (`qa_step00_baseline.py`):
- baseline_docs ≥ 80 (line 109)
- publish_date_pct ≥ 0.98 (line 118)
- seed_eval_size ≥ 40 (line 127)
- baseline_chunks ≥ baseline_docs (line 136)
- baseline_domain_count ≥ 3 (line 147)

**Gate-1 Thresholds** (`qa_step01_embeddings.py`):
- embedding_rows == baseline_chunks (line 236)
- vector_dim == 1536 (line 242)
- zero_vectors == 0 (line 247)
- nan_vectors == 0 (line 251)
- pct_norm_outliers ≤ 0.005 (WARN ≤ 0.015) (line 255)

**Gate-2 Thresholds** (`qa_step02_indexes.py`):
- pinecone_upsert_rate ≥ 0.98 (WARN ≥ 0.97) (line 341)
- weaviate_upsert_rate ≥ 0.98 (WARN ≥ 0.97) (line 342)
- faiss_count_ratio ≥ 0.98 (WARN ≥ 0.97) (line 343)
- pct_missing_required_metadata ≤ 0.02 (WARN ≤ 0.03) (line 344)
- faiss_roundtrip_error_max ≤ 0.001 (WARN ≤ 0.01) (line 345)
- sanity_search_min_topk ≥ 3 (WARN == 2) (line 346)
- sanity_keyword_hit_min_top10 ≥ 1 (line 347)

**Gate-3 Thresholds** (`qa_step03_mcp.py`):
- health_endpoints_ok == 5 (line 369)
- contract_ok_rate == 1.0 (line 372)
- latency_budget: p50, p95 ≤ budget (line 378)
- timeout_rate == 0.0 (line 379+)

**Gate-4 Thresholds** (`qa_step04_router.py`):
- route_share ≥ 0.10 OR ≥ 1 route (lines 416-424)
- empty_result_rate ≤ 0.02 (line 429)
- auto_retry_success_rate ≥ 0.95 (line 437)
- avg_doc_age_days ≤ max(365, age_p50) (line 447)
- mean_unique_domains_top10 ≥ 2.4 (WARN ≥ 2.0) (lines 456-458)

**Gate-5 Thresholds** (`qa_step05_graph.py`):
- nodes_executed == all 8 in order (line 66)
- total_runtime_ms ≤ 30000 (WARN ≤ 36000) (line 73)
- insight_cards == 5 (line 78)
- distinct_sources ≥ 4 (line 82)
- insights_within_12mo ≥ 2 (line 86)
- email_schema_ok == true (line 92)
- proof_points_resolve == true (line 102)

**Gate-6 Thresholds** (`qa_step06_a2a.py`):
- negotiation_rounds ≤ 2 (line 45)
- critical_flags == 0 (line 49)
- email_body_length ≤ 160 (line 54)
- readability_grade ≤ 10 (line 59)
- proof_points_resolve == true (line 68)

**Gate-7 Thresholds** (`qa_step07_retrieval_eval.py`):
- recall@10 ≥ 0.80 (line 637)
- nDCG@5 ≥ 0.60 (line 638)
- coverage ≥ max(3.0, 0.75 * baseline_domains / 10) (line 640)
- freshness ≤ max(540, age_p50) (line 641)
- latency: p50, p95 ≤ budget per backend (line 642)

**Gate-8 Thresholds** (`qa_step08_generation_eval.py`):
- structural_pass_rate == 1.0 (line 407)
- critical_flags_total == 0 (line 414)
- length_readability_pass_runs ≥ 9 (line 423)
- persona_keyword_hits_avg ≥ 2.0 (line 431)

---

## 6. Data Structures & Schemas

### Gate Report Structure (JSON)

**Standard Envelope** (all gates):

```json
{
  "step": "step00_baseline",
  "gate": "Gate-0",
  "status": "GREEN|AMBER|RED",
  "checks": [
    {
      "id": "G0-01",
      "metric": "baseline_docs",
      "actual": 97,
      "threshold": ">=80",
      "status": "PASS|WARN|FAIL",
      "evidence": "data/final/inventory/salesforce_inventory.csv"
    }
  ],
  "next_action": "continue|proceed_with_caution|fix_and_rerun|stop",
  "timestamp": "2025-10-20T16:35:37.691116+00:00"
}
```

**Extended Fields** (gate-specific):

- **Gate-0**: Includes `baseline` object with domains, age buckets, token stats
- **Gate-7**: Includes `service_mode`, `fallback_mode`, `summary`, `run_context`, `trace_path`, `router_summary`
- **Gate-8**: Includes `summary` with aggregated run metrics

**File Locations**:
- JSON: `reports/qa/stepXX_name.json`
- Markdown: `reports/qa/stepXX_name.md`

**Source**: All gate scripts write dual reports using same pattern

### Check Object Schema

**Standard Fields**:

```json
{
  "id": "G0-01",          // Gate number + sequence
  "metric": "baseline_docs",
  "actual": 97,
  "threshold": ">=80",
  "status": "PASS|WARN|FAIL",
  "evidence": "path/to/file"
}
```

**Status Values**:
- `PASS`: Check met threshold
- `WARN`: Check slightly below threshold but within tolerance
- `FAIL`: Check significantly below threshold

**ID Format**: `G{gate}-{sequence}` (e.g., `G0-01`, `G1-03a`, `G7-05`)

### Evaluation Metrics Structure

**Retrieval Metrics** (Gate-7):

```json
{
  "recall@10": 0.7174,
  "nDCG@5": 0.3591,
  "coverage_unique_domains_top10_mean": 2.239,
  "freshness_mean_age_days": 337.54,
  "latency": [
    {"backend": "faiss", "p50": 12.3, "p95": 18.7, "budget_p95": 300.0},
    {"backend": "weaviate", "p50": 56.2, "p95": 89.1, "budget_p95": 1000.0},
    {"backend": "pinecone", "p50": 123.4, "p95": 187.6, "budget_p95": 1500.0}
  ],
  "queries": 46,
  "recall_at": {"@1": 0.1739, "@3": 0.413, "@5": 0.5435, "@10": 0.7174},
  "doc_recall@10": 0.8478,
  "soft_recall@10": 0.0652,
  "doc_nDCG@5": 0.6922,
  "near_miss_rate": 0.1304,
  "rank_stats": {
    "chunk": {"count": 33, "p50": 3, "p75": 5, "p90": 7, "max": 10},
    "doc": {"count": 39, "p50": 1, "p75": 2, "p90": 4, "max": 7}
  },
  "by_doctype": {...},
  "by_backend": {...}
}
```

**Generation Metrics** (Gate-8):

```json
{
  "runs": [
    {
      "eval_id": "...",
      "persona": "vp_customer_experience",
      "session_id": "...",
      "out_dir": "outputs/...",
      "structural": {
        "insights_count": 5,
        "distinct_sources": 4,
        "recent_count": 2,
        "email_schema_ok": true,
        "proof_points_resolve": true,
        "persona_keyword_hits": 0
      },
      "perf_ms": 12345,
      "error": null
    }
  ],
  "aggregates": {
    "structural_pass_rate": 1.0,
    "persona_keyword_hits_avg": 0.0
  }
}
```

**Compliance Metrics** (Gate-8):

```json
{
  "runs": [
    {
      "eval_id": "...",
      "flags": {
        "critical": [],
        "warning": []
      },
      "word_count": 145,
      "readability_grade": 8.2
    }
  ],
  "aggregates": {
    "critical_flags_total": 0,
    "length_readability_pass_runs": 10
  }
}
```

### Embedding Data Schema

**Parquet File** (`data/vector/embeddings/embeddings.parquet`):

```python
schema = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("doc_id", pa.string()),
    pa.field("seq_no", pa.int32()),
    pa.field("token_count", pa.int32()),
    pa.field("l2_norm", pa.float32()),
    pa.field("vector", pa.list_(pa.float32()))  # Length 1536
])
```

**Source**: Gate-1 (`qa_step01_embeddings.py:103-111`)

### FAISS Manifest Schema

**File** (`data/vector/faiss/faiss_manifest.json`):

```json
{
  "index_type": "HNSW",
  "metric": "L2",
  "dim": 1536,
  "count": 536,
  "roundtrip_error_max": 0.0,
  "paths": {
    "index": "data/vector/faiss/index.faiss",
    "idmap": "data/vector/faiss/idmap.parquet"
  },
  "params": {
    "M": 32,
    "efConstruction": 200,
    "efSearch": 128
  }
}
```

**Source**: Gate-2 (`qa_step02_indexes.py:151-162`)

### Trace File Schema

**Retrieval Trace** (`reports/router/step07_retrieval_trace.jsonl`):

```json
{
  "eval_id": "...",
  "persona": "vp_customer_experience",
  "query_text": "latest earnings results",
  "router_backend": "pinecone",
  "reason_codes": ["PR_QUERY"],
  "retrieval_mode": "internal_stub",
  "latency_ms": 56.2,
  "topk": [
    {"chunk_id": "...", "score": 0.87, "text": "..."},
    // ... up to AG7_TRACE_TOPK entries
  ],
  "hit": {
    "chunk_rank": 3,
    "doc_rank": 1,
    "near_hit": false
  }
}
```

**Router Trace** (`reports/router/step04_router_trace.jsonl`):

```json
{
  "timestamp": "2025-10-20T16:35:37Z",
  "query_text": "agentforce product announcement",
  "persona": "cio",
  "decision_backend": "weaviate",
  "fallback_used": false,
  "reason_codes": ["FILTER_MATCH"],
  "latency_ms": 67.3,
  "top_k": 10,
  "n_unique_domains": 3,
  "avg_doc_age_days": 245.6,
  "empty_result": false
}
```

**Source**: Gate-4 (`qa_step04_router.py:383-395`), Gate-7 (`qa_step07_retrieval_eval.py:481-512`)

---

## 7. External Dependencies

### Python Libraries

**Core Dependencies** (all gates):
- `json`: JSON parsing and writing
- `os`: File system operations
- `sys`: System operations (exit codes, stdin/stdout)
- `argparse`: Command-line argument parsing
- `datetime`, `timezone`: Timestamp generation
- `typing`: Type hints

**Async Operations** (Gates 3, 4, 7, 8):
- `asyncio`: Async/await for network I/O
- `aiohttp`: HTTP client/server for MCP stubs

**Data Processing**:
- `numpy`: Vector operations, L2 distance, percentile calculation
- `pyarrow`: Parquet file I/O for embeddings
- `pandas`: (optional) CSV reading

**YAML Configuration**:
- `pyyaml`: YAML config file parsing (optional, graceful fallback)

**OpenAI API** (Gate-1):
- `openai`: OpenAI client for ada-002 embeddings
- `dotenv`: Load `.env` file for API key

**FAISS** (Gate-2):
- `faiss-cpu`: Vector index builds (conda install, NOT pip)
- **Critical**: Must use conda-installed FAISS in `ageFaiss` environment to avoid OpenMP conflicts

**Retry Logic** (Gate-1):
- `tenacity`: Exponential backoff retry decorator

**Graph Execution** (Gates 5, 6, 8):
- `subprocess`: Run `scripts/run_graph.py` as subprocess

### External Tools

**Conda Environments**:
- `age` (Python 3.13): Primary environment for Gates 0, 1, 3-8
- `ageFaiss` (Python 3.12): FAISS-only environment for Gate-2

**Git**:
- `git status`, `git log`: Check repository state
- `git branch`: Determine current branch

**Bash Scripts**:
- `hack/spec_metadata.sh`: Generate metadata for research documents

### OpenAI API

**Model**: `text-embedding-ada-002`
- **Dimension**: 1536
- **Cost**: $0.0001 per 1K tokens
- **Rate Limits**: Handled by `@retry` decorator with exponential backoff

**Usage**: Gate-1 (`qa_step01_embeddings.py:163-176`)

**Environment Variable**: `OPENAI_API_KEY` (loaded from `.env` file)

### MCP Services

**Internal Stubs** (development):
- **kb.search**: Port 7801
- **web.fetch**: Port 7802
- **link.resolve**: Port 7803
- **crm.lookup**: Port 7804
- **safety.check**: Port 7805

**Configuration**: `configs/mcp.tools.yaml`

**Used By**: Gates 3, 4, 7, 8

### No External Metrics Libraries

All metrics are computed inline:
- **Recall@10**: Simple count of hits in top-10
- **nDCG@5**: DCG calculation via `1.0 / log2(rank + 1)`
- **Jaccard similarity**: Set intersection / union
- **L2 norm**: `sqrt(sum(x^2))`
- **Percentiles**: Sorted array indexing

**Reason**: Minimal dependencies, lightweight implementation

---

## 8. Execution & Usage

### Running All Gates Sequentially

**Full quality gate run**:

```bash
# Set up environments first (one-time)
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml

# Set up API key (one-time)
echo "OPENAI_API_KEY=your-api-key-here" > .env

# Run all gates in sequence
conda run -n age python scripts/qa_step00_baseline.py
conda run -n age python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
conda run -n age python scripts/qa_step03_mcp.py
conda run -n age python scripts/qa_step04_router.py
conda run -n age python scripts/qa_step05_graph.py
conda run -n age python scripts/qa_step06_a2a.py --session-id <from-gate-5>
conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py
conda run -n age python scripts/qa_step08_generation_eval.py
```

**Source**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/commands.md:526-536`

### Running Individual Gates

**Gate-0: Baseline**:
```bash
conda run -n age python scripts/qa_step00_baseline.py
```

**Gate-1: Embeddings** (with auto-confirm):
```bash
conda run -n age AG1_AUTO_CONFIRM=1 python scripts/qa_step01_embeddings.py
```

**Gate-2: Indexes** (CRITICAL: use ageFaiss):
```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Gate-3: MCP Tools**:
```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**Gate-4: Router**:
```bash
conda run -n age python scripts/qa_step04_router.py
```

**Gate-5: Graph**:
```bash
conda run -n age python scripts/qa_step05_graph.py
```

**Gate-6: A2A** (requires session-id):
```bash
conda run -n age python scripts/qa_step06_a2a.py --session-id <session-id>
```

**Gate-7: Retrieval** (with relaxed budgets):
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Gate-8: Generation** (with custom prompts):
```bash
conda run -n age python scripts/qa_step08_generation_eval.py --prompts data/interim/eval/custom_prompts.jsonl --timeout 60
```

### Environment Variables Reference

#### Gate-1 (Embeddings)
- **`AG1_AUTO_CONFIRM=1`**: Skip cost confirmation prompt

#### Gate-2 (Indexes)
- **`AG2_DISABLE_FAISS=1`**: Skip FAISS index build (fallback mode)

#### Gate-7 (Retrieval Evaluation)
- **`AG7_IGNORE_COVERAGE=1`**: Skip coverage gating (relaxed mode)
- **`AG7_LATENCY_MULTIPLIER=<float>`**: Relax latency budgets (e.g., `3.0` for 3x budgets)
- **`AG7_DEBUG=1`**: Enable debug mode (default)
- **`AG7_TRACE=1`**: Enable trace logging (default follows debug)
- **`AG7_ANALYZE_TOPK=<int>`**: Top-K for analysis (default `10`)
- **`AG7_NEAR_SEQ_TOL=<int>`**: Near-miss sequence tolerance (default `1`)
- **`AG7_TOPK_SLICES=<csv>`**: Recall@k slices (default `"1,3,5,10"`)
- **`AG7_TRACE_TOPK=<int>`**: Top-K for trace (default `10`)
- **`AG7_TRACE_SUCCESSES=<0|1>`**: Trace successful queries (default follows debug)

#### Gate-8 (Generation Evaluation)
- **`--prompts <path>`**: Override prompts file (default: `data/interim/eval/generation_prompts.jsonl`)
- **`--timeout <seconds>`**: Timeout per run (default: 30s)
- **`--self-test`**: Run internal self-tests and exit

#### Global
- **`OPENAI_API_KEY`**: OpenAI API key (required for Gate-1, loaded from `.env`)

### Conda Environment Discipline

**CRITICAL**: Use correct environment for each gate

**`age` environment** (Python 3.13):
- Gates: 0, 1, 3, 4, 5, 6, 7, 8
- Reason: Primary development environment with all dependencies

**`ageFaiss` environment** (Python 3.12):
- Gates: 2 (ONLY)
- Reason: Conda-installed FAISS to avoid OpenMP conflicts
- **DO NOT** install pip `faiss-cpu` in `age` environment

**Error if wrong env**: `OMP Error #15` or segmentation fault

**Source**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/CLAUDE.md:22-29`

### Checking Gate Status

**Console Output** (all gates):
```bash
$ conda run -n age python scripts/qa_step00_baseline.py
# ... validation output ...
{"status": "GREEN"}
```

**JSON Report**:
```bash
$ cat reports/qa/step00_baseline.json | jq .status
"GREEN"
```

**Markdown Report** (human-readable):
```bash
$ cat reports/qa/step00_baseline.md
# STEP 0 — Baseline Snapshot (Gate‑0) — GREEN
...
```

### Pipeline Integration

**Scripted Execution**:
```bash
#!/bin/bash
set -e  # Exit on any error

# Run gates sequentially
conda run -n age python scripts/qa_step00_baseline.py
STATUS_0=$(cat reports/qa/step00_baseline.json | jq -r .status)
if [ "$STATUS_0" != "GREEN" ]; then
  echo "Gate-0 failed: $STATUS_0"
  exit 1
fi

conda run -n age python scripts/qa_step01_embeddings.py
STATUS_1=$(cat reports/qa/step01_embeddings.json | jq -r .status)
if [ "$STATUS_1" != "GREEN" ] && [ "$STATUS_1" != "AMBER" ]; then
  echo "Gate-1 failed: $STATUS_1"
  exit 1
fi

# ... continue for all gates
```

---

## 9. Code Patterns & Conventions

### Dual-Format Reporting Pattern

**Used By**: All gates

**Implementation**:
```python
# 1. Build machine-readable structure
machine = {
    "step": "step00_baseline",
    "gate": "Gate-0",
    "status": status,
    "checks": checks,
    "next_action": next_action,
    "timestamp": now_iso(),
}

# 2. Write JSON report
ensure_dir("reports/qa")
with open("reports/qa/step00_baseline.json", "w", encoding="utf-8") as f:
    json.dump(machine, f, ensure_ascii=False, indent=2)

# 3. Build Markdown lines
lines = [
    f"# STEP 0 — Baseline Snapshot (Gate‑0) — {status}",
    "",
    "Checks:",
    *[f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}"
      for c in checks],
    "",
    f"Gate-0 status: {status} — next_action: {next_action}",
]

# 4. Write Markdown report
with open("reports/qa/step00_baseline.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
```

**Key Aspects**:
- JSON for automation and pipeline integration
- Markdown for human review and debugging
- Both created in same execution
- Use `ensure_dir()` before writing

**Source**: Gates 0-8 (`qa_step*.py` scripts)

### Consistent Exit Codes

**Pattern**: Implicit exit 0, status communicated via reports

**Most Gates** (0, 1, 2, 3, 4, 5, 6, 7):
```python
def main():
    # ... validation logic ...
    # Write reports
    print(json.dumps({"status": status}, indent=2))
    # Implicit: exit 0

if __name__ == "__main__":
    main()
```

**Gate-8 Exception** (explicit exit 1 on RED):
```python
if status == "RED":
    sys.exit(1)
else:
    sys.exit(0)
```

**Reason**: Most gates allow RED status to be recorded without failing CI/CD pipeline. Gate-8 fails explicitly because it's the final gate.

**Source**: All gate scripts

### Cross-Gate Loading Pattern

**Pattern**: Load previous gate results via JSON reports

**Example** (Gate-1 loading Gate-0):
```python
def load_baseline_chunks() -> int:
    """Load baseline_chunks from Gate-0 report."""
    try:
        j = json.load(open("reports/qa/step00_baseline.json", "r", encoding="utf-8"))
        for c in j.get("checks", []):
            if c.get("id") == "G0-04":
                return int(c.get("actual") or 0)
    except Exception:
        pass
    # Fallback: count directly from chunks
    cnt = 0
    for path in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                cnt += 1
    return cnt
```

**Key Aspects**:
- Try to load from previous gate's JSON report
- Extract specific check results by check ID
- Always provide fallback by computing directly from source data
- Use exception handling to gracefully handle missing files

**Source**: Gate-1 (`qa_step01_embeddings.py:66-80`), Gate-2 (`qa_step02_indexes.py:46-60`)

### Check ID Naming Convention

**Pattern**: `G{gate}-{sequence}`

**Examples**:
- `G0-01`, `G0-02`, ... `G0-05` (Gate-0, 5 checks)
- `G1-01`, `G1-02`, `G1-03a`, `G1-03b`, `G1-04` (Gate-1, 5 checks with sub-checks)
- `G7-01`, `G7-02`, ... `G7-05` (Gate-7, 5 checks)

**Source**: All gate scripts

### Status Determination Pattern

**Pattern**: GREEN/AMBER/RED based on check results

**Simple All-or-Nothing** (Gates 5, 6):
```python
status = "GREEN" if all(c["status"] == "PASS" for c in checks) else \
         ("AMBER" if any(c["status"] == "WARN" for c in checks) and
                     not any(c["status"] == "FAIL" for c in checks)
          else "RED")
```

**Complex Tiered Logic** (Gate-1):
```python
passes = {c["id"]: c for c in checks}
if all(c["status"] == "PASS" for c in checks):
    status = "GREEN"
    next_action = "continue"
elif all(passes[k]["status"] == "PASS" for k in ("G1-01", "G1-02", "G1-03a", "G1-03b")) \
     and passes["G1-04"]["status"] == "WARN":
    status = "AMBER"
    next_action = "proceed_with_caution"
else:
    status = "RED"
    next_action = "fix_and_rerun"
```

**Source**: All gate scripts

### Shared Utilities Pattern

**Common Module** (`scripts/common.py`):
```python
from common import ensure_dir, now_iso, sha1_8, RateLimiter

# Create directory if needed
ensure_dir("reports/qa")

# Get ISO timestamp
timestamp = now_iso()  # Returns datetime.now(timezone.utc).isoformat()

# 8-char SHA1 hash
hash_val = sha1_8(text)

# Async rate limiting
async with RateLimiter(max_per_second=10):
    # rate-limited code
    pass
```

**Embedding Utilities** (`scripts/embedding_utils.py`):
```python
from embedding_utils import embed_text, embed_batch, estimate_embedding_cost

# Single text embedding
vector = embed_text(text, dim=1536)

# Batch embedding with caching
vectors = embed_batch(texts, dim=1536, batch_size=100)

# Cost estimation
cost_info = estimate_embedding_cost(num_texts, avg_text_length)
```

**Key Aspects**:
- **CRITICAL**: Both documents and queries MUST use `embed_text()` for consistency
- Built-in caching to minimize API costs
- Supports cost estimation before execution

**Source**: `scripts/common.py:23-28`, `scripts/embedding_utils.py`

### YAML Configuration Loading Pattern

**Pattern**: Load YAML with graceful fallback

```python
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
```

**Key Aspects**:
- Try to import `yaml` at runtime (optional dependency)
- Return empty dict on any error (missing file, parse error, import error)
- Allows gates to run with default values even if config missing

**Source**: Gate-1 (`qa_step01_embeddings.py:31-38`), Gate-7 (`qa_step07_retrieval_eval.py:31-37`)

### JSONL Processing Pattern

**Pattern**: Load JSONL with skip-malformed

```python
def load_seed(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue  # Skip malformed lines
    return items
```

**Key Aspects**:
- Read JSONL format (one JSON object per line)
- Skip malformed lines with try/except
- Return list of dictionaries

**Source**: Gate-7 (`qa_step07_retrieval_eval.py:51-59`), Gate-8 (`qa_step08_generation_eval.py:53-62`)

### Path Constants at Module Level

**Pattern**: Define all paths as module-level constants

```python
# All paths defined at top of file
INV_PATH = "data/final/inventory/salesforce_inventory.csv"
CHUNK_GLOB = "data/interim/chunks/*.chunks.jsonl"
EVAL_PATH = "data/interim/eval/salesforce_eval_seed.jsonl"
OUT_JSON = os.path.join("reports", "qa", "step00_baseline.json")
OUT_MD = os.path.join("reports", "qa", "step00_baseline.md")
```

**Key Aspects**:
- Use `os.path.join()` for cross-platform compatibility
- Constants in UPPER_CASE
- Globs use string literals (not os.path.join for patterns)

**Source**: All gate scripts

### Import Organization Pattern

**Pattern**: Shebang, standard lib, third-party, local

```python
#!/usr/bin/env python3
# Standard library imports (alphabetical)
import argparse
import asyncio
import glob
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Third-party imports
import aiohttp
import numpy as np

# Local imports
from common import ensure_dir, now_iso
from embedding_utils import embed_text
```

**Key Aspects**:
- Shebang first: `#!/usr/bin/env python3`
- Standard library grouped
- Third-party libraries grouped
- Local modules last
- Type hints from `typing` module

**Source**: All gate scripts

---

## 10. Testing & Verification

### How Gates Are Tested

**Self-Testing**:
- Gate-8 includes `--self-test` mode to validate internal functions
- Tests `persona_keyword_hits()`, `validate_structure()`, `word_count()`, `readability_grade()`
- Exits 0 if all tests pass

**Example**:
```bash
conda run -n age python scripts/qa_step08_generation_eval.py --self-test
```

**Source**: Gate-8 (`qa_step08_generation_eval.py:729-770`)

**Integration Testing**:
- Gates tested by running full pipeline on sample data
- Each gate validates outputs from previous gates
- Cross-gate dependencies ensure consistency

**Validation Test Cases**:
- Gate-7 uses evaluation seed dataset (`data/interim/eval/salesforce_eval_seed.jsonl`)
- Gate-8 runs 10 end-to-end executions across personas
- Gate-2 includes sanity search with 3 predefined queries

### Verification Scripts (qa_verify_*)

**Purpose**: Validate data preparation stages (collection → deduplication)

**8 Verification Gates** (G01-G08):
1. **G01 (Collection)**: Validates raw document collection completeness
2. **G02 (Normalization)**: Validates text normalization quality
3. **G03 (Metadata)**: Validates metadata extraction completeness
4. **G04 (Chunking)**: Validates document chunking quality
5. **G05 (Deduplication)**: Validates deduplication effectiveness
6. **G06 (Link Health)**: Validates URL allowlist compliance
7. **G07 (Eval Seed)**: Validates evaluation seed dataset
8. **G08 (Day-1 Signoff)**: Final Day-1 milestone validation

**Relationship to Main Gates**:
- Verification gates (G01-G08) validate data preparation
- Main gates (Gate-0 to Gate-8) validate vector pipeline
- No direct overlap; complementary validation tracks

**Reports**: `reports/qa/gate0X_*.json` and `reports/qa/human_readable/gate0X_*.md`

**Source**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_verify_*.py`

### Trace Logging for Debugging

**Gate-3 Probe Log** (`logs/mcp/step03_probes.jsonl`):
- Records all MCP service probes with timing and error codes
- One JSONL record per probe

**Gate-4 Router Trace** (`reports/router/step04_router_trace.jsonl`):
- Records routing decisions per query
- Fields: backend, reason_codes, fallback_used, latency_ms, top_k, diversity, freshness

**Gate-7 Retrieval Trace** (`reports/router/step07_retrieval_trace.jsonl`):
- Records retrieval results per query
- Fields: backend, latency_ms, topk results, hit status (chunk_rank, doc_rank, near_hit)
- Controlled by `AG7_TRACE=1` env var

**Gate-7 Failure Log** (`reports/eval/retrieval_failures.jsonl`):
- Records all retrieval misses with diagnostic info
- Classification: `chunk_miss_doc_miss`, `chunk_miss_doc_hit_near`, `chunk_miss_doc_hit_far`
- Includes nearest same-doc chunk diagnostics

**Source**: Gates 3, 4, 7

### Sanity Checks

**Gate-2 Sanity Search**:
- 3 predefined queries with keyword sets
- Validates semantic search returns contextually relevant results
- Checks minimum top-k count and keyword hit count

**Gate-3 Contract Validation**:
- Sends 1 valid + 2 invalid requests per MCP tool
- Validates error codes match expectations (`InvalidParams`, `BackendUnavailable`, `InvalidMethod`)
- Ensures 100% contract conformance rate

**Gate-5 Graph Execution**:
- Runs full LangGraph workflow end-to-end
- Validates all 8 nodes executed in order
- Checks output structure (insights, email, compliance)

**Source**: Gates 2, 3, 5

---

## 11. Known Issues & Limitations

### OpenMP Conflicts (Gate-2)

**Issue**: `OMP Error #15` or segfault during FAISS operations

**Cause**: Mixing pip `faiss-cpu` with conda OpenMP runtime in `age` environment (Python 3.13)

**Solution**:
- Always use `ageFaiss` environment (Python 3.12) for Gate-2
- NEVER install pip `faiss-cpu` in `age` environment
- If already installed: `conda env remove -n age && conda env create -f envs/age.yaml`

**Source**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/CLAUDE.md:22-29`, `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/troubleshooting.md`

### Embedding Consistency (Gates 1, 2, 7)

**Issue**: Retrieval recall = 0% despite having indexed documents

**Cause**: Documents and queries use different embedding functions or random vectors

**Solution**:
- Both documents AND queries MUST use `embed_text()` from `scripts/embedding_utils.py`
- Never use different embedding functions or random vectors
- Dimension must be 1536 for both

**Source**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/CLAUDE.md:30-36`, `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/troubleshooting.md`

### OpenAI API Key (Gate-1)

**Issue**: Embedding generation fails with authentication error

**Cause**: Missing or invalid `OPENAI_API_KEY` in `.env` file

**Solution**:
- Create `.env` file: `echo "OPENAI_API_KEY=your-key" > .env`
- Verify key works before running Gate-1
- Check API quotas and rate limits

**Source**: `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/CLAUDE.md:37-43`

### Gate-7 Relaxed Budgets

**Issue**: Gate-7 fails due to strict latency or coverage thresholds

**Cause**: Development environment or limited baseline data

**Solution**:
- Use `AG7_IGNORE_COVERAGE=1` to skip coverage gating
- Use `AG7_LATENCY_MULTIPLIER=3.0` to relax latency budgets 3x
- Adjust thresholds based on actual performance characteristics

**Example**:
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Source**: Gate-7 documentation, env var handling at lines 258-264

### Gate-2 Disabled FAISS Mode

**Issue**: Cannot build FAISS index due to import failure or environment conflicts

**Cause**: FAISS not installed or incompatible version

**Solution**:
- Set `AG2_DISABLE_FAISS=1` to skip FAISS build
- System will write manifest with `"disabled": true`
- Index build continues with Weaviate and Pinecone only

**Example**:
```bash
conda run -n ageFaiss AG2_DISABLE_FAISS=1 python scripts/qa_step02_indexes.py
```

**Source**: Gate-2 (`qa_step02_indexes.py:87-98`)

### Simulated Backends (Gate-2)

**Limitation**: Pinecone and Weaviate index builds are currently simulated

**Behavior**:
- Manifest files written with simulated counts
- No actual API calls to Pinecone or Weaviate cloud services
- Sanity search uses local numpy-based exact search, not cloud indexes

**Future**: Replace with actual cloud API integration

**Source**: Gate-2 (`qa_step02_indexes.py:285`, lines 282-308)

### Session-Specific Gates (Gate-6)

**Limitation**: Gate-6 requires `--session-id` from Gate-5 execution

**Reason**: Gate-6 validates outputs from specific graph session

**Workaround**: Always run Gate-5 before Gate-6, capture session-id from stdout

**Example**:
```bash
# Run Gate-5, capture session-id
OUTPUT=$(conda run -n age python scripts/qa_step05_graph.py)
SESSION_ID=$(echo "$OUTPUT" | jq -r .session_id)

# Run Gate-6 with session-id
conda run -n age python scripts/qa_step06_a2a.py --session-id "$SESSION_ID"
```

**Source**: Gate-6 (`qa_step06_a2a.py:31`)

### Async Gate Execution Order

**Limitation**: Gates 3, 4, 7, 8 must run with MCP services available

**Reason**: These gates connect to MCP service stubs

**Dependency**:
- Gate-3 starts internal stub servers
- Gates 4, 7, 8 can use internal stubs, external service, or offline mode
- Fallback mode controls degradation behavior (default, warn, strict)

**Solution**: Run Gate-3 first to validate MCP health, then run dependent gates

**Source**: Gates 3, 4, 7, 8 (MCP connection logic)

---

## 12. References

### Related Documentation Parts

**Part 2: Pipeline Stages**
- Documents the 13 pipeline stages being validated by gates
- Gate-0 validates stage 0 (baseline snapshot)
- Verification gates (qa_verify_*) validate stages 1-5 (collection → deduplication)
- Main gates validate stages 6-13 (embeddings → generation evaluation)

**Part 3: Vector Pipeline**
- Documents Gate-1 (embeddings) and Gate-2 (indexing) in detail
- Covers OpenAI ada-002 embedding model
- Describes FAISS HNSW index parameters
- Explains multi-index architecture (FAISS, Weaviate, Pinecone)

**Part 5: MCP Tools**
- Documents Gate-3 (MCP service validation)
- Describes MCP stub server implementation
- Covers kb.search, web.fetch, link.resolve, crm.lookup, safety.check tools
- Explains contract conformance validation

**Part 6: Agent Graph**
- Documents Gate-5 (graph execution) and Gate-6 (A2A compliance)
- Describes LangGraph workflow (8 nodes)
- Covers agent-to-agent communication schema
- Explains compliance rule validation

### Related Files

**Environment Setup**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/envs/age.yaml` - Primary environment (Python 3.13)
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/envs/ageFaiss.yaml` - FAISS environment (Python 3.12)

**Processing Scripts**:
- `scripts/normalize_html.py` - Text normalization (validated by verification gate G02)
- `scripts/extract_metadata.py` - Metadata extraction (validated by G03)
- `scripts/chunk_documents.py` - Document chunking (validated by G04)
- `scripts/dedupe_chunks.py` - Deduplication (validated by G05)
- `scripts/run_graph.py` - LangGraph execution (validated by Gate-5)

**Configuration Documentation**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/configuration.md` - All config files detailed
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/commands.md` - Complete command reference

**Troubleshooting**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/troubleshooting.md` - Debug playbook

**Architecture**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/architecture.md` - System design overview

**Project Guidelines**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/CLAUDE.md` - Project conventions and critical gotchas
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/AGENTS.md` - Automation-friendly runbook

### Sample Reports

**Successful Gate Run** (`reports/qa/step00_baseline.md`):
```markdown
# STEP 0 — Baseline Snapshot (Gate‑0) — GREEN

Inputs:
- Inventory: data/final/inventory/salesforce_inventory.csv
- Chunks: data/interim/chunks/*.chunks.jsonl
- Eval seed: data/interim/eval/salesforce_eval_seed.jsonl

Counts:
- baseline_docs: 97
- publish_date_pct: 1.0
- baseline_chunks: 536
- seed_eval_size: 46
- baseline_domain_count: 7

Checks:
- G0-01: baseline_docs = 97 (threshold >=80) -> PASS
- G0-02: publish_date_pct = 1.0 (threshold >=0.98) -> PASS
- G0-03: seed_eval_size = 46 (threshold >=40) -> PASS
- G0-04: baseline_chunks = 536 (threshold >=baseline_docs (97)) -> PASS
- G0-05: baseline_domain_count = 7 (threshold >=3) -> PASS

Gate-0 status: GREEN — next_action: continue
Timestamp: 2025-10-10T16:26:44.691116+00:00
```

**Failed Gate Run** (`reports/qa/step07_retrieval_eval.md`):
```markdown
# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

**Service Mode**: internal_stub (fallback mode: default)

**Checks:**
- G7-01: recall@10 = 0.7174 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.3591 (threshold >=0.60) -> FAIL
- G7-04: freshness_mean_age_days = 337.54 (threshold <=540) -> PASS
- G7-05: latency_budgets = {...} (threshold p50,p95 <= budget_p95 per backend) -> PASS

Diagnostics (not gating):
- recall@k: {'@1': 0.1739, '@3': 0.413, '@5': 0.5435, '@10': 0.7174}
- doc_recall@10: 0.8478
- soft_recall@10: 0.0652

Gate-7 status: RED — next_action: fix_and_rerun
```

---

## Appendix: Quick Reference

### Gate Summary Table

| Gate | Script | Checks | Key Metrics | Status Criteria |
|------|--------|--------|-------------|-----------------|
| Gate-0 | `qa_step00_baseline.py` | 5 | docs, date coverage, chunks, eval seed, domains | GREEN: all pass, AMBER: 1 fail within 10% margin, RED: otherwise |
| Gate-1 | `qa_step01_embeddings.py` | 5 | embedding rows, dimension, zero/nan vectors, norm outliers | GREEN: all pass, AMBER: only G1-04 WARN, RED: otherwise |
| Gate-2 | `qa_step02_indexes.py` | 7 | upsert rates, FAISS count, metadata, roundtrip error, sanity search | GREEN: all pass, AMBER: 1 primary WARN or G2-06 WARN, RED: otherwise |
| Gate-3 | `qa_step03_mcp.py` | 4 | health endpoints, contracts, latency budgets, timeout rate | GREEN: all pass, AMBER: 1 latency WARN, RED: otherwise |
| Gate-4 | `qa_step04_router.py` | 5 | route shares, empty rate, retry rate, freshness, diversity | GREEN: all pass, AMBER: only WARN, RED: any FAIL |
| Gate-5 | `qa_step05_graph.py` | 7 | node coverage, latency, insight count, sources, recency, schema, resolution | GREEN: all pass, AMBER: any WARN, RED: any FAIL |
| Gate-6 | `qa_step06_a2a.py` | 5 | negotiation rounds, critical flags, length, readability, resolution | GREEN: all pass, AMBER: any WARN, RED: any FAIL |
| Gate-7 | `qa_step07_retrieval_eval.py` | 5 | recall@10, nDCG@5, coverage, freshness, latency | GREEN: all pass, AMBER: 1 non-critical FAIL, RED: recall fails or multiple |
| Gate-8 | `qa_step08_generation_eval.py` | 4 | structural pass rate, critical flags, length/readability, keywords | GREEN: all pass, AMBER: only G8-03 or G8-04 fail, RED: otherwise |

### Command Quick Reference

```bash
# Environment setup (one-time)
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
echo "OPENAI_API_KEY=your-key" > .env

# Run all gates
conda run -n age python scripts/qa_step00_baseline.py
conda run -n age AG1_AUTO_CONFIRM=1 python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
conda run -n age python scripts/qa_step03_mcp.py
conda run -n age python scripts/qa_step04_router.py
conda run -n age python scripts/qa_step05_graph.py
conda run -n age python scripts/qa_step06_a2a.py --session-id <from-gate-5>
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
conda run -n age python scripts/qa_step08_generation_eval.py

# Check gate status
cat reports/qa/step00_baseline.json | jq .status
cat reports/qa/step00_baseline.md
```

### File Locations Quick Reference

```
scripts/
  ├── qa_step00_baseline.py            # Gate-0
  ├── qa_step01_embeddings.py          # Gate-1
  ├── qa_step02_indexes.py             # Gate-2
  ├── qa_step03_mcp.py                 # Gate-3
  ├── qa_step04_router.py              # Gate-4
  ├── qa_step05_graph.py               # Gate-5
  ├── qa_step06_a2a.py                 # Gate-6
  ├── qa_step07_retrieval_eval.py      # Gate-7
  ├── qa_step08_generation_eval.py     # Gate-8
  ├── qa_verify_*.py                   # 8 verification scripts
  ├── embedding_utils.py               # Shared embedding functions
  ├── router_core.py                   # Query routing logic
  └── common.py                        # Shared utilities

configs/
  ├── eval.prompts.yaml                # Persona keywords (Gate-8)
  ├── compliance.template.yaml         # Compliance rules (MCP safety)
  ├── router.heuristics.yaml           # Routing rules (Gate-4, Gate-7)
  ├── vector.indexing.yaml             # Embedding/FAISS params (Gate-1, Gate-2)
  ├── mcp.tools.yaml                   # MCP endpoints (Gate-3, Gate-4, Gate-7, Gate-8)
  └── langgraph.nodes.yaml             # Graph topology (Gate-5)

reports/qa/
  ├── step00_baseline.{json,md}        # Gate-0 reports
  ├── step01_embeddings.{json,md}      # Gate-1 reports
  ├── step02_indexes.{json,md}         # Gate-2 reports
  ├── step03_mcp.{json,md}             # Gate-3 reports
  ├── step04_router.{json,md}          # Gate-4 reports
  ├── step05_graph.{json,md}           # Gate-5 reports
  ├── step06_a2a.{json,md}             # Gate-6 reports
  ├── step07_retrieval_eval.{json,md}  # Gate-7 reports
  ├── step08_generation_eval.{json,md} # Gate-8 reports
  └── gate01_*.json                    # Verification reports

reports/eval/
  ├── retrieval_failures.jsonl         # Gate-7 failure log
  ├── generation_metrics.json          # Gate-8 generation metrics
  └── compliance_metrics.json          # Gate-8 compliance metrics

reports/router/
  ├── step04_router_trace.jsonl        # Gate-4 trace
  └── step07_retrieval_trace.jsonl     # Gate-7 trace

data/vector/
  ├── embeddings/
  │   ├── embeddings.parquet           # Gate-1 output
  │   └── embedding_stats.json         # Gate-1 stats
  ├── faiss/
  │   ├── index.faiss                  # Gate-2 FAISS index
  │   ├── idmap.parquet                # Gate-2 ID mapping
  │   └── faiss_manifest.json          # Gate-2 manifest
  ├── weaviate/
  │   ├── schema_applied.json          # Gate-2 Weaviate schema
  │   └── index_manifest.json          # Gate-2 Weaviate manifest
  └── pinecone/
      └── index_manifest.json          # Gate-2 Pinecone manifest
```

---

**End of Document**

Total Lines: ~1,950

**Research Date**: 2025-10-20 16:35:37 EDT
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate
