---
date: 2025-10-09T16:35:36-04:00
researcher: liyunxiao
git_commit: f734a2dac18482528a23595ee35033fbf7bc2a37
branch: agent-weaviate
repository: ag3
topic: "Gate-7 Retrieval Quality Improvement Analysis for Issue003"
tags: [research, retrieval, gate-7, recall, ndcg, weaviate, pinecone, faiss, sec-filings, backend-simulation]
status: complete
last_updated: 2025-10-09
last_updated_by: liyunxiao
last_updated_note: "Major correction: All backends use identical code (simulated stubs), performance differences due to query difficulty distribution only"
---

# Research: Gate-7 Retrieval Quality Improvement Analysis (Issue003)

**Date**: 2025-10-09T16:35:36-04:00
**Researcher**: liyunxiao
**Git Commit**: f734a2dac18482528a23595ee35033fbf7bc2a37
**Branch**: agent-weaviate
**Repository**: ag3

## Research Question

How can we improve recall@10 (currently 71.74%, target ≥80%) and nDCG@5 (currently 0.3591, target ≥0.60) in the Gate-7 retrieval evaluation?

## Executive Summary

The Gate-7 retrieval evaluation fails both thresholds with **recall@10 of 71.74%** and **nDCG@5 of 35.91%**. After comprehensive analysis of the retrieval pipeline, evaluation methodology, and backend performance, I identified **six root causes** with varying impact levels.

**CRITICAL FINDING**: All three backends (FAISS, Weaviate, Pinecone) use **identical retrieval code** (`qa_step03_mcp.py:96-156`) - they are simulated stubs sharing the same L2 distance search implementation. The only difference is simulated latency (FAISS: 5-10ms, Weaviate: 40-80ms, Pinecone: 80-160ms). **Performance differences are 100% due to query difficulty distribution**, not backend implementation.

The highest-impact issue is **uneven query difficulty distribution**: Pinecone receives the hardest queries (SEC filing table data) via the `PR_QUERY` routing rule, resulting in 50% recall and 5% nDCG@5. FAISS receives the easiest queries (definitional), achieving 80% recall and 51% nDCG. The second major issue is **missing lexical reranking** due to an undefined `tokenize()` function, which causes all backends to lose their 30% lexical boost.

**Quick Wins (Highest Impact)**:
1. Implement missing `tokenize()` function (expect +3-5% overall metrics, affects all backends equally)
2. Add SEC filing document type filters (expect +10-15% SEC-specific recall)
3. Rebalance query routing to avoid sending hardest queries to one backend (expect +5-8% overall metrics)

## Detailed Findings

### Current System State

**Overall Performance** (from `reports/qa/step07_retrieval_eval.json`):
- **recall@10**: 71.74% (threshold: ≥80%) - **FAIL** ❌
- **nDCG@5**: 35.91% (threshold: ≥60%) - **FAIL** ❌
- **doc_recall@10**: 84.78% - Right documents, wrong chunks (13% gap)
- **near_miss_rate**: 13.04% - Adjacent chunks retrieved instead of exact targets

**Per-Backend Quality**:

| Backend | Queries | Chunk Recall@10 | Doc Recall@10 | nDCG@5 | Performance |
|---------|---------|-----------------|---------------|---------|-------------|
| **FAISS** | 10 (21.7%) | 80% | 100% | 0.5131 | **Best** ✅ |
| **Weaviate** | 26 (56.5%) | 76.92% | 76.92% | 0.4188 | Acceptable ⚠️ |
| **Pinecone** | 10 (21.7%) | 50% | 90% | **0.05** | **Catastrophic** ❌ |

**By Document Type**:

| Doctype | Queries | Chunk Recall | Doc Recall | Notes |
|---------|---------|--------------|------------|-------|
| **Press** | 26 | 81% | 92% | Natural language, ideal for ada-002 ✅ |
| **Product** | 6 | 100% | 100% | Marketing prose, perfect match ✅ |
| **10-Q** | 6 | **33%** | 50% | Table data, poor embeddings ❌ |
| **10-K** | 3 | **0%** | 33% | Worst performer ❌ |
| **8-K** | 1 | 100% | 100% | Narrative-driven ✅ |

---

### Root Cause 1: Uneven Query Difficulty Distribution Across Backends

**Impact**: 🔴 **VERY HIGH** - Hardest 21.7% of queries routed to single backend

**CRITICAL INSIGHT**: All backends use **identical code** (`qa_step03_mcp.py:96-156`). The performance disparity is NOT due to backend implementation differences, but due to **which queries each backend receives**.

**Evidence**: `scripts/qa_step03_mcp.py:96-107`
```python
if backend not in ("faiss", "weaviate", "pinecone"):
    return web.json_response({"error": ...}, status=503)

# ONLY difference: simulated latency
delay_ms = {"faiss": (5, 10), "weaviate": (40, 80), "pinecone": (80, 160)}[backend]
await asyncio.sleep(d)

# ALL backends use SAME search logic:
xb = state["xb"]  # Same embedding matrix
qv = state["embed_query"](q)  # Same query embedding
dists = ((xb - qv)**2).sum(axis=1)  # Same L2 distance calculation
```

**Query Difficulty Analysis** (`reports/qa/step07_retrieval_eval.json:182-219`):

| Backend | Queries | Query Types | Chunk Recall | nDCG@5 | Difficulty |
|---------|---------|-------------|--------------|---------|------------|
| **FAISS** | 10 | Definitional ("what is...", "overview") | 80% | 0.5131 | **Easy** ✅ |
| **Weaviate** | 26 | Mixed (long queries, no rule match) | 76.92% | 0.4188 | **Medium** ⚠️ |
| **Pinecone** | 10 | **SEC filing table queries** (financial data) | 50% | 0.05 | **Hard** ❌ |

**Why Pinecone's Queries Are Hardest**:
1. **Routed via PR_QUERY rule** (`configs/router.heuristics.yaml:21-25`)
   - Keywords: `[results, earnings, fiscal, guidance, 10-k, 10-q]` → Pinecone
   - These queries expect SEC filing **table data** (worst embedding quality)

2. **SEC filing challenges**:
   - Expected: 10-Q chunk with table cells ("Europe 2,337  Asia Pacific 1,023  $9,829")
   - Retrieved: Press release with natural language ("First quarter revenue of $9.8 billion")
   - Root issue: Tables embed poorly with OpenAI ada-002 (trained on natural language)

3. **Temporal homogeneity**:
   - "Q1 FY26 results" vs "Q2 FY25 results" vs "Q4 FY24 results" have similar embeddings
   - OpenAI ada-002 doesn't distinguish quarterly/fiscal period nuances well

4. **Cross-doctype confusion**:
   - Query targets 10-Q filing, system returns semantically similar press releases
   - No document type filtering to prefer authoritative sources

**Key Takeaway**: The issue is NOT "Pinecone implementation is bad" - it's "Pinecone was assigned the hardest queries in the evaluation set". Any backend receiving these queries would show poor performance.

**Location**: `scripts/router_core.py:84`, `configs/router.heuristics.yaml:21-25`, `scripts/qa_step03_mcp.py:96-156`

---

### Root Cause 2: Missing Lexical Reranking (tokenize() Undefined)

**Impact**: 🔴 **HIGH** - Affects all backends, loses 30% scoring component

**Evidence**: `scripts/qa_step03_mcp.py:122`
```python
from embedding_utils import tokenize as _tok  # ImportError: tokenize doesn't exist!
```

**How Reranking Should Work** (lines 133-134):
```python
final_score = 0.7 * vector_similarity + 0.3 * lexical_boost
```

**What Actually Happens** (fallback path, lines 146-155):
- Falls back to **pure vector similarity** (100% semantic, 0% lexical)
- Keyword-heavy queries (e.g., "senior notes maturity dates") lose precision
- Financial jargon queries suffer most

**Expected Impact of Fix**:
- Financial/technical queries: +5-10% recall
- Overall nDCG@5: +3-5 points (especially for Pinecone queries)

**Location**: `scripts/qa_step03_mcp.py:122`, `scripts/embedding_utils.py` (function missing)

---

### Root Cause 3: Routing Rule Doesn't Distinguish Query Difficulty

**Impact**: 🟡 **MEDIUM-HIGH** - 10 hardest queries concentrated on one "backend"

**Important Note**: Since all backends use identical code, this isn't about routing to the "wrong backend" - it's about **uneven difficulty distribution** making performance metrics hard to interpret.

**Evidence**: `reports/qa/step07_retrieval_eval.json:269-284`
- Router reason codes: `PR_QUERY` (10 queries) → Pinecone
- These 10 queries have 50% recall, 0.05 nDCG (hardest in eval set)
- FAISS's 10 queries have 80% recall, 0.5131 nDCG (easiest in eval set)

**Current Routing Logic** (`configs/router.heuristics.yaml:21-25`):
```yaml
- if:
    has_keywords: [results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k]
  then:
    backend: pinecone
    reason: PR_QUERY
```

**Problem**: This single rule captures two very different query types:
1. **Press release queries** ("Q1 results announcement") - Medium difficulty
   - Natural language content, good embeddings
   - Expected answer: Press release chunks

2. **SEC filing queries** ("10-Q revenue tables") - **Very high difficulty**
   - Table-formatted data, poor embeddings
   - Expected answer: SEC filing chunks with tabular data
   - Cross-doctype confusion (system returns press releases instead)

**Why This Matters**:
- Concentrating hard queries on one backend makes overall metrics look worse
- Makes it impossible to diagnose whether poor performance is due to backend or query difficulty
- Since backends are identical, this is purely a **reporting/analysis issue**, not a functional issue

**Improvement Opportunity**:
- Split rule to distinguish SEC filing queries from press release queries
- Add document type filtering (see Root Cause 4)
- Expected impact: +10-15% overall recall, +8-12 points overall nDCG@5 (by solving the underlying SEC table data problem, not by changing backends)

**Location**: `scripts/router_core.py:80-89`, `configs/router.heuristics.yaml:21-25`

---

### Root Cause 4: SEC Filing Table Data Semantic Mismatch

**Impact**: 🟡 **MEDIUM** - Affects 19.5% of queries (SEC filings)

**Evidence**: `reports/eval/retrieval_failures.jsonl` (first 6 failures all target SEC filings)

**Example Failure**:
- **Query**: "What was Salesforce's total revenue for Q1 FY26?"
- **Expected**: `10-Q::2025-04-30::fy26-q1-form-10-q::chunk0015` (financial tables)
- **Retrieved (Top 3)**:
  1. Press release chunk: "First quarter revenue of $9.8 billion" (score: 0.829)
  2. Press release chunk: "Q1 FY26 results exceed guidance" (score: 0.827)
  3. Press release chunk: "Salesforce reports Q1 earnings" (score: 0.824)
- **Expected chunk score**: 0.797 (rank 193, not in top-10)

**Why Tables Embed Poorly**:

**10-Q Chunk Content** (chunk0015):
```
crm-20250430

Europe          2,337    2,145
Asia Pacific    1,023      926
              $ 9,829  $ 9,133

Revenues by geography are determined based on...
```
- Title boost: "crm-20250430" (filing ID, no semantic value)
- Body: Table cells without sentence structure
- Missing query terms: No "Q1 FY26", "first quarter", "revenue" adjacent to numbers

**Press Release Content** (chunk0000):
```
Salesforce Reports Record First Quarter Fiscal 2026 Results

First quarter revenue of $9.8 billion, up 8% year-over-year...
```
- Title boost: "Salesforce Reports Record First Quarter Fiscal 2026 Results" (all query keywords)
- Body: Natural language narrative
- Perfect query match: "First Quarter Fiscal 2026", "revenue", "$9.8 billion"

**OpenAI ada-002 Limitation**: Trained on natural language, not optimized for tables

**Location**: `data/interim/chunks/*.chunks.jsonl`, `scripts/chunk_documents.py:130` (title boost)

---

### Root Cause 5: Weaviate as Default Catch-All

**Impact**: 🟢 **LOW-MEDIUM** - Handles 56.5% of queries with acceptable performance

**Evidence**: `reports/qa/step07_retrieval_eval.json:195-205`
- Weaviate: 26 queries, 76.92% recall, 0.4188 nDCG
- Reason codes: `DEFAULT_WEAVIATE` (23), `FILTER_MATCH` (2), `PERSONA_BIAS` (1)

**Current Fallback Logic** (`scripts/router_core.py:96-100`):
```python
if len(ql.split()) <= 4 or any(kw in ql for kw in ["what is", "define", "definition", "overview"]):
    return "faiss", reasons + ["DEFAULT_SHORT_FAISS"]
else:
    return "weaviate", reasons + ["DEFAULT_WEAVIATE"]
```

**Characteristics**:
- Weaviate gets all "long, non-definitional" queries
- Performance is acceptable but not optimal
- **Critical limitation**: Weaviate is **simulated only** - uses same L2 search as other backends

**Improvement Opportunity**:
- Deploy actual Weaviate cluster with hybrid search (BM25 + vector)
- Add metadata filtering for document type disambiguation
- Expected impact: +3-5% Weaviate-specific recall

**Location**: `scripts/router_core.py:99`, `scripts/qa_step03_mcp.py:96-97` (simulation)

---

### Root Cause 6: Chunk Boundary Misalignment

**Impact**: 🟢 **LOW-MEDIUM** - 13% doc-found-but-chunk-missed gap

**Evidence**: `reports/qa/step07_retrieval_eval.json:96-99`
- Doc recall@10: 84.78% (39/46 queries found correct document)
- Chunk recall@10: 71.74% (33/46 queries found correct chunk)
- Gap: 13.04% (6 queries)

**Near-Miss Analysis** (`reports/qa/step07_retrieval_eval.json:100-115`):
- Near-miss rate: 13.04% (6 queries retrieve adjacent chunk, delta_seq ≤ 1)
- Examples: Expected `chunk0020`, got `chunk0021` (delta_seq=1)

**Why Chunk Boundaries Don't Align**:
1. **Fixed token limits** (~800 tokens/chunk) don't respect semantic coherence
2. **SEC Item segmentation** breaks at regulatory boundaries (Item 1, 1A, 7, 7A, 8), not information units
3. **Multi-topic chunks**: Chunk 15 contains revenue tables + contract balances + unearned revenue (dilutes relevance)

**Ground Truth Re-Labeling History** (from `data/interim/eval/salesforce_eval_seed.jsonl`):
- `10q_q1_revenue`: chunk0004 → chunk0008 → chunk0015 (shifted 11 chunks after re-chunking)
- Suggests chunking boundaries are unstable across iterations

**Location**: `scripts/chunk_documents.py:170-186`, `configs/chunking.config.json:3-4`

---

## Code References

### Retrieval Pipeline
- **Evaluation script**: `scripts/qa_step07_retrieval_eval.py:116-862` (`main_async()`)
- **MCP kb.search**: `scripts/qa_step03_mcp.py:82-156` (`handle_invoke_kb()`)
- **Router logic**: `scripts/router_core.py:72-100` (`decide_backend()`)
- **Embedding generation**: `scripts/embedding_utils.py:86-133` (`embed_text()`)

### Configuration
- **Router heuristics**: `configs/router.heuristics.yaml:1-43`
- **Vector indexing**: `configs/vector.indexing.yaml:1-22`
- **Chunking settings**: `configs/chunking.config.json:1-6`
- **MCP tools**: `configs/mcp.tools.yaml` (port 7801-7805)

### Data Artifacts
- **Eval seed**: `data/interim/eval/salesforce_eval_seed.jsonl` (46 queries, human-labeled)
- **Embeddings**: `data/vector/embeddings/embeddings.parquet` (1536-dim ada-002)
- **FAISS index**: `data/vector/faiss/index.faiss` (HNSW, M=32, efSearch=128)
- **Failure log**: `reports/eval/retrieval_failures.jsonl` (13 failures)

### Reports
- **Gate-7 JSON**: `reports/qa/step07_retrieval_eval.json:1-287`
- **Gate-7 Markdown**: `reports/qa/step07_retrieval_eval.md:1-39`
- **Trace log**: `reports/router/step07_retrieval_trace.jsonl`

---

## Architecture Documentation

### Retrieval Data Flow

1. **Query arrives** → `qa_step07_retrieval_eval.py:375` calls `decide_backend(q, persona, None)`
2. **Router processes rules** → `router_core.py:80-100`:
   - Check keyword rules sequentially (first match wins)
   - Fallback to persona bias
   - Fallback to heuristic (short=FAISS, long=Weaviate)
3. **Backend selected** → Reason codes recorded (e.g., `PR_QUERY`, `DEFAULT_WEAVIATE`)
4. **Query embedded** → `embed_text()` → OpenAI ada-002 API → 1536-dim vector (cached)
5. **MCP kb.search invoked** → HTTP POST to `localhost:7801/invoke`
6. **Stub simulates backend**:
   - Load embeddings from Parquet
   - Compute L2 distances: `((xb - qv)**2).sum(axis=1)`
   - Retrieve top 100 candidates
   - **Lexical rerank** (BROKEN: `tokenize()` missing) → Falls back to pure vector similarity
   - Return top-k results
7. **Metrics computed** → `qa_step07_retrieval_eval.py:406-576`:
   - Find rank of expected chunk (0 if not found)
   - Compute recall@k for k ∈ {1, 3, 5, 10}
   - Compute nDCG@5: `sum(1.0 / log2(rank+1))` for ranks ≤ 5
   - Detect near-misses (delta_seq ≤ 1)
8. **Failures logged** → `reports/eval/retrieval_failures.jsonl`

### Embedding Consistency

**Critical: All paths use the same `embed_text()` function**:
- **Documents** (Gate-1): `qa_step01_embeddings.py:173` → `embed_batch()` → `embed_text()`
- **Query (online)**: `qa_step03_mcp.py:71` → `embed_text()`
- **Query (offline)**: `qa_step07_retrieval_eval.py:235` → `embed_text()`

**No preprocessing differences** - identical OpenAI ada-002 API calls

---

## Improvement Opportunities (Prioritized by Impact)

### 🔴 Priority 1: Fix Missing Lexical Reranking (High Impact, Low Effort)

**Problem**: `tokenize()` function doesn't exist, causing 30% lexical boost to be lost

**Solution**:
1. Add `tokenize()` function to `scripts/embedding_utils.py`:
   ```python
   def tokenize(text: str) -> List[str]:
       """Extract word tokens for lexical matching."""
       import re
       return [w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 2]
   ```

2. Verify import in `qa_step03_mcp.py:122` succeeds

**Expected Impact**:
- Financial/technical queries: +5-10% recall
- Overall nDCG@5: +3-5 points
- Pinecone nDCG@5: +0.10-0.15 (from 0.05 to 0.15-0.20)

**Files to Modify**:
- `scripts/embedding_utils.py` (add function)
- Test: `conda run -n age python scripts/qa_step03_mcp.py` (verify import)

---

### 🔴 Priority 2: Split Routing Rule + Add Document Type Filtering (High Impact, Medium Effort)

**Problem**: PR_QUERY rule lumps SEC filing queries (very hard) with press release queries (medium), concentrating hardest queries on one backend. Since all backends use identical code, the backend name doesn't matter - but we need **document type filtering** to improve SEC query recall.

**Solution** (Combined with Priority 3):
1. **Split routing rule** in `configs/router.heuristics.yaml:21-25`:
   ```yaml
   # OLD (lumps all financial queries together):
   - if:
       has_keywords: [results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k]
     then:
       backend: pinecone
       reason: PR_QUERY

   # NEW (separate SEC filings from press releases):
   # SEC filings - add doctype filter
   - if:
       has_keywords: [10-k, 10-q, 8-k, form 10, sec filing, filing]
     then:
       backend: faiss
       reason: SEC_FILING
       filters:
         doctype: [10-K, 10-Q, 8-K]

   # Press releases - add doctype filter
   - if:
       has_keywords: [results, earnings, fiscal, guidance, announce, quarter, reports]
     then:
       backend: weaviate
       reason: PRESS_RELEASE
       filters:
         doctype: [press]
   ```

2. **Implement doctype filtering** in `scripts/qa_step03_mcp.py:handle_invoke_kb()`:
   ```python
   # After line 119, before lexical boost:
   doctype_filter = params.get("filters", {}).get("doctype", [])
   if doctype_filter:
       # Filter candidates by doctype
       res = [r for r in res if state["rows"][...].get("doctype") in doctype_filter]
   ```

3. **Update router to pass filters** in `scripts/router_core.py:decide_backend()`:
   ```python
   # Return filters from routing rules
   return backend, reasons, rule.get("then", {}).get("filters", {})
   ```

**Expected Impact**:
- **SEC filing recall**: 0-33% → **50-60%** (+20-30 points for SEC queries)
- **Overall recall@10**: 71.74% → **~78-82%** (+6-10 points)
- **Overall nDCG@5**: 0.3591 → **~0.50-0.55** (+14-19 points)

**Key Insight**: The impact comes from **document type filtering** (preventing press releases from dominating SEC filing queries), NOT from changing which backend processes the query (since all backends use identical code).

**Files to Modify**:
- `configs/router.heuristics.yaml:21-25` (split rules + add filters)
- `scripts/qa_step03_mcp.py:82-156` (implement filtering)
- `scripts/router_core.py:72-100` (pass filters from routing rules)

---

### 🟡 Priority 3: Enhance SEC Chunk Title Boost (Medium Impact, Low Effort)

**Problem**: SEC chunks boosted with "crm-20250430" instead of descriptive names

**Solution**:
1. Extract descriptive title from SEC filing metadata:
   ```python
   # In chunk_documents.py:130
   if doctype in ("10-k", "10-q", "8-k"):
       fiscal_year = d.get("fiscal_year") or "unknown"
       period = d.get("period_end_date") or "unknown"
       title = f"Salesforce {doctype.upper()} Filing for {fiscal_year} (period ending {period})"
   ```

2. Update normalization to extract fiscal metadata from SEC forms

**Expected Impact**:
- SEC embedding quality: +5-8% similarity to queries mentioning "Salesforce", "fiscal year"
- SEC chunk recall: +3-5 points

**Files to Modify**:
- `scripts/chunk_documents.py:130` (title generation)
- `scripts/normalize_html.py` (extract fiscal metadata)

---

### 🟢 Priority 5: Tune Reranking Weights (Low Impact, Low Effort)

**Problem**: Current weights may not prioritize semantic similarity enough

**Current Weights** (`configs/router.heuristics.yaml:1-4`):
```yaml
weights:
  similarity: 0.5
  recency: 0.3
  diversity: 0.2
```

**Proposed Adjustment**:
```yaml
weights:
  similarity: 0.7  # Increase from 0.5 (emphasize relevance)
  recency: 0.2     # Decrease from 0.3
  diversity: 0.1   # Decrease from 0.2
```

**Rationale**: Low nDCG suggests ranking quality is poor. Prioritizing similarity may help surface correct chunks within correct documents.

**Expected Impact**:
- nDCG@5: +1-2 points (better ranking of relevant chunks)
- Near-miss rate: May decrease slightly (better chunk-level precision)

**Files to Modify**:
- `configs/router.heuristics.yaml:1-4`

**Note**: Reranking is currently NOT applied in Gate-7 evaluation (`qa_step07_retrieval_eval.py:382` calls MCP directly without rerank step). To activate:
1. Add reranking call: `rerank(results, query, weights)` after MCP search
2. Expected additional impact: +2-3 nDCG points

---

### 🟢 Priority 6: Expand FAISS Usage (Low Impact, Low Effort)

**Problem**: FAISS handles only 10 queries despite being the best backend

**Solution**:
1. Broaden DEFINITION rule in `configs/router.heuristics.yaml:35-39`:
   ```yaml
   # OLD:
   - if:
       has_keywords: [definition, what is, overview]
     then:
       backend: faiss
       reason: DEFINITION

   # NEW (add more keywords):
   - if:
       has_keywords: [definition, what is, overview, explain, describe, capabilities, features, how does, what are]
     then:
       backend: faiss
       reason: DEFINITION
   ```

2. Add structured data rule:
   ```yaml
   - if:
       has_keywords: [table, schedule, list, breakdown, comparison, vs, versus]
     then:
       backend: faiss
       reason: STRUCTURED_DATA
   ```

**Expected Impact**:
- Shift 5-8 queries from Weaviate (76.92% recall) to FAISS (80% recall)
- Overall recall@10: +0.5-1 point
- Overall nDCG@5: +1-2 points

**Files to Modify**:
- `configs/router.heuristics.yaml:35-39`

---

## Historical Context (from thoughts/)

### Related Issues
- **Issue001**: Migration from hashlex-v1 (768-dim) to OpenAI ada-002 (1536-dim) - **COMPLETED**
- **Issue002**: SEC filing XBRL metadata pollution - **FIXED** in commit f734a2d
- **Issue003**: Current issue - Gate-7 retrieval quality failures
- **Issue004**: LangGraph integration planning
- **Issue005**: End-to-end flow documentation

### Related Research
- `thoughts/shared/research/2025-10-07-issue002-sec-filing-retrieval-pipeline.md` - Comprehensive SEC pipeline documentation
- `thoughts/shared/research/2025-10-07-issue002-xbrl-selector-fix.md` - XBRL CSS selector fix validation
- `thoughts/shared/research/2025-10-06-issue001-embedding-model-architecture.md` - Hashlex-v1 architecture analysis

### Related Plans
- `thoughts/shared/plans/issue001-OpenAI Ada-002 Migration Plan v2 (Unified).md` - Embedding migration plan (completed)
- `thoughts/shared/plans/issue002-2025-10-07-fix-xbrl-metadata-pollution.md` - XBRL cleanup plan (completed)

---

## Next Steps (Recommended Sequence)

1. **Immediate (Today - 1-2 hours)**:
   - Implement Priority 1: Add `tokenize()` function (30 min)
   - Test lexical reranking working
   - Re-run Gate-7 evaluation
   - **Expected result**: recall@10 ≈ 74-76%, nDCG@5 ≈ 0.40-0.45

2. **Short-term (This Week - 3-4 hours)**:
   - Implement Priority 2: Split routing rules + document type filtering
   - Implement Priority 3: SEC chunk title boost
   - Re-run Gate-7 evaluation
   - **Expected result**: recall@10 ≈ 78-82%, nDCG@5 ≈ 0.50-0.60 (passing thresholds!)

3. **Medium-term (Next Sprint)**:
   - Implement Priority 4: Tune reranking weights + activate reranking in Gate-7
   - Implement Priority 5: Expand FAISS usage (or re-label as "expand definitional routing")
   - **Deploy actual Weaviate cluster** (replace simulation) - This is the big one
   - **Expected result**: recall@10 ≈ 83-85%, nDCG@5 ≈ 0.62-0.68 (comfortably passing)

4. **Long-term (Future Iterations)**:
   - Investigate table-to-text preprocessing for SEC filings (e.g., markdown table formatting)
   - Explore fine-tuning ada-002 on financial domain or switching to a financial-specific embedding model
   - Add temporal disambiguation (FY26 Q1 vs FY25 Q1) via metadata filters
   - Implement chunk boundary optimization based on semantic coherence
   - Consider replacing simulated backends with actual FAISS/Weaviate/Pinecone deployments

**Note on Backend Deployment**: Since all backends currently use identical code, deploying actual FAISS/Weaviate/Pinecone instances would only provide value if they offer features beyond basic L2 distance search (e.g., Weaviate's hybrid BM25+vector search, Pinecone's metadata filtering, FAISS's GPU acceleration).

---

## Summary

The Gate-7 retrieval system is failing primarily due to **uneven query difficulty distribution** (hardest queries concentrated on one backend) and **missing lexical reranking** (affects all backends equally).

**CRITICAL CORRECTION**: All three backends (FAISS, Weaviate, Pinecone) use **identical retrieval code** - they are simulated stubs with the same L2 distance search implementation. Performance differences are 100% due to which queries each backend receives, not implementation quality. "Pinecone's poor performance" is actually "SEC filing table queries have poor embeddings."

**Root Cause Breakdown**:
1. **Missing `tokenize()` function** - Affects all backends, loses 30% lexical boost
2. **SEC filing table data** - Poor embedding quality (OpenAI ada-002 not optimized for tables)
3. **No document type filtering** - SEC queries return press releases instead of filings
4. **Routing concentrates hard queries** - Makes metrics look worse than system actually performs

**Quick Wins**:
1. Implement `tokenize()` function (+3-5% all queries)
2. Add document type filtering (+15-25% SEC queries)
3. Split routing rules to separate SEC from press release queries (better diagnostics)

**Confidence Level**: High - All root causes validated with specific code locations, failure examples, and quantitative evidence. Backend uniformity confirmed by direct code inspection.

**Risk Assessment**: Low - Proposed fixes are configuration changes or small code additions, no architectural refactoring required.
