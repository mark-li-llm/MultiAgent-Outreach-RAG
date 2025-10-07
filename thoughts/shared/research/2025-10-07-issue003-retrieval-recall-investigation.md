---
date: 2025-10-07T14:16:33-04:00
researcher: Claude
git_commit: e6597f5ac21d7a1d210428bddbc5dd75374fde90
branch: agent-weaviate
repository: ag3
topic: "Gate-7 Retrieval System Recall Investigation (issue003)"
tags: [research, gate-7, retrieval, recall, embedding, sec-filings, backend, routing, reranking]
status: complete
last_updated: 2025-10-07
last_updated_by: Claude
---

# Research: Gate-7 Retrieval System Recall Investigation (issue003)

**Date**: 2025-10-07T14:16:33-04:00
**Researcher**: Claude
**Git Commit**: e6597f5ac21d7a1d210428bddbc5dd75374fde90
**Branch**: agent-weaviate
**Repository**: ag3

## Research Question

What is causing the Gate-7 retrieval evaluation to show RED status with 65.22% chunk-level recall@10 (target ≥ 80%), 34.41% nDCG@5 (target ≥ 60%), and particularly 0% chunk recall for SEC quarterly/annual filings (10-Q, 10-K)?

## Summary

The investigation reveals **five interconnected root causes** for the Gate-7 retrieval failures:

1. **Embedding Model Discrepancy** - Documentation states hashlex-v1 (768-dim) but system actually uses OpenAI ada-002 (1536-dim) after migration in commit e6597f5. This creates confusion but NOT a functional issue.

2. **Missing SEC Structure Parsing** - The 10-K filing lacks `sec_item_spans` field, causing it to be chunked as a single 133-chunk segment without Item-level boundaries. The 10-Q has only 1 Item span despite being a multi-Item document, indicating incomplete structure parsing.

3. **XBRL Metadata Noise** - SEC filings contain substantial iXBRL/XBRL namespace URIs, entity IDs, and taxonomy references in early chunks, diluting semantic embeddings and preventing narrative content from surfacing.

4. **Reranking Not Applied in Gate-7** - The evaluation directly tests MCP service results without calling `router_core.rerank()`, missing recency, diversity, and domain-cap adjustments that would improve ranking quality.

5. **Backend Implementation Identical** - All three backends (FAISS, Weaviate, Pinecone) query the same in-memory numpy array with identical L2 distance logic, differentiated only by simulated latency. Pinecone's 5% nDCG is caused by routing harder queries to it, not by backend differences.

**Key Finding**: The system is internally consistent (same embedding model throughout), but SEC filing chunking failures and missing reranking prevent relevant chunks from ranking within top-10 results.

## Detailed Findings

### 1. Embedding Model System: OpenAI ada-002 (NOT hashlex-v1)

**Discovery**: The system **fully migrated** from hashlex-v1 (768-dim) to OpenAI text-embedding-ada-002 (1536-dim) in commit e6597f5 (PR #17), but CLAUDE.md documentation remains outdated.

**Current Implementation** (`scripts/embedding_utils.py:86-133`):
- **Model**: OpenAI `text-embedding-ada-002`
- **Dimensions**: 1536 (hardcoded as `ADA002_DIM` at line 19)
- **Validation**: Enforces `dim == 1536` at lines 97-101, raising `ValueError` if mismatched
- **Caching**: SHA-256 based disk cache in `data/cache/embeddings/`
- **Retry logic**: 3 attempts with exponential backoff for API failures

**Configuration** (`configs/vector.indexing.yaml:1-5`):
```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20
```

**Evidence of Consistency**:
- Gate-1 (`qa_step01_embeddings.py:173`): Uses `embed_batch()` from `embedding_utils`
- Gate-7 (`qa_step07_retrieval_eval.py:233-248`): Uses `embed_text()` from `embedding_utils`
- MCP stub (`qa_step03_mcp.py:71`): Uses `embed_text()` from `embedding_utils`
- All scripts import from **same module**, ensuring vector space consistency

**Validation** (`reports/qa/step01_embeddings.json:14-21`):
```json
{
  "id": "G1-02",
  "metric": "vector_dim",
  "actual": 1536,
  "threshold": "== 1536 (from config)",
  "status": "PASS"
}
```

**Implication**: There is NO embedding model mismatch causing recall=0. The discrepancy mentioned in issue003 is **documentation lag only**.

### 2. SEC Filing Processing Pipeline: Incomplete Structure Parsing

**Problem**: SEC quarterly and annual filings show 0% chunk recall despite 33-50% doc recall.

#### 10-Q Findings (`crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866`)

**Normalized JSON State**:
- `sec_item_spans`: Present with **1 Item** (coverage=0.9999)
- **91 chunks total**

**Issue**: A 10-Q filing should contain multiple Items (typically Item 1, Item 2, plus optional Items like Item 5 or 6). Having only 1 Item span means the structure parser failed to detect most Item boundaries.

#### 10-K Findings (`crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2`)

**Normalized JSON State**:
- `sec_item_spans`: **MISSING** (field not present)
- `sec_item_coverage_ratio`: **MISSING**
- **133 chunks total**

**Evidence**:
- `grep sec_item_spans data/interim/normalized/crm::10-K::*.json` → No output
- `logs/metadata/20250907_175256.log` → 10-K not mentioned (parser didn't run on it)
- `logs/metadata/20250907_175256.log:e16f2866,10-q,items_found=1,coverage=0.9999` → 10-Q processed but only 1 Item detected

**Chunking Consequence** (`chunk_documents.py:180-188`):

```python
if doctype in ("10-k", "10-q", "8-k", "ars_pdf") and (d.get("sec_item_spans") or []):
    # Split by SEC Item spans
    for span in d.get("sec_item_spans") or []:
        segments.append((start, end))
else:
    # Treat entire document as one segment
    segments.append((0, len(text) - 1))
```

Since 10-K lacks `sec_item_spans`, it's chunked as a **single 133-chunk segment** using generic sliding window. This means:
- Item 1 (Business) mixes with Item 1A (Risk Factors)
- Item 7 (MD&A) mixes with Item 8 (Financial Statements)
- No Item-specific context boundaries

**Content Quality Issue**: The first 5 chunks of the 10-K contain XBRL/iXBRL metadata:

Example tokens from `chunk0000`:
```
crm-20250131
0001108524
FALSE
FY
2025
http://fasb.org/us-gaap/2024#PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization
iso4217:USD
xbrli:shares
SubscriptionandSupportMember
```

**Embedding Impact**: These machine-readable taxonomy URIs and entity IDs dominate early embeddings, preventing semantic matching with narrative queries like "What was Salesforce's Q1 FY26 revenue?"

### 3. Backend Query Implementation: Identical Logic Across All Backends

**Discovery**: FAISS, Weaviate, and Pinecone **use the same query implementation** through the MCP stub service.

**Unified Query Path** (`qa_step03_mcp.py:82-156`):

All backends execute:
1. L2 distance computation: `dists = ((xb - qv)**2).sum(axis=1)` (line 108)
2. Top-100 candidate selection: `idx = np.argsort(dists)[:100]` (line 109)
3. Two-stage scoring:
   - **Stage 1** (Vector): `_vec_sim = 1.0 / (1.0 + L2_dist)` (line 119)
   - **Stage 2** (Lexical boost): `final = 0.7 × vec_sim + 0.3 × lex_overlap` (line 133)

**Backend Differentiation**: Only simulated latency differs:
- FAISS: 5-10ms (lines 106)
- Weaviate: 40-80ms
- Pinecone: 80-160ms

**Backend-Specific Results** (`step07_retrieval_eval.json:182-219`):

| Backend | Queries | Chunk Recall | Doc Recall | nDCG@5 | Routing Reason |
|---------|---------|--------------|------------|--------|----------------|
| FAISS | 10 | 80% | 100% | 51.31% | DEFINITION (definitional queries) |
| Weaviate | 26 | 65.38% | 73.08% | 39.22% | DEFAULT_WEAVIATE (fallback) |
| Pinecone | 10 | 50% | 90% | **5%** | PR_QUERY (press/financial keywords) |

**Pinecone's 5% nDCG Explanation**:
- Pinecone is routed queries with keywords: `[results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k]` (`router.heuristics.yaml:21-25`)
- These are **financial queries** expecting SEC filing chunks
- SEC filings have **0% chunk recall** (as documented above)
- Therefore, Pinecone gets the hardest queries by routing rules, not because of backend quality differences

### 4. Query Routing and Reranking System

**Routing Decision** (`router_core.py:72-100`):

1. **Keyword rules** (first match wins):
   - Press/financial → Pinecone (reason: `PR_QUERY`)
   - Developer/API → Weaviate (reason: `FILTER_MATCH`)
   - Definitional → FAISS (reason: `DEFINITION`)

2. **Persona bias** (if no rule matches):
   - `vp_sales_ops` → Pinecone
   - `cio` → Weaviate
   - `vp_customer_experience` → FAISS

3. **Heuristic fallback**:
   - Query ≤4 words OR contains "what is"/"define" → FAISS (reason: `DEFAULT_SHORT_FAISS`)
   - Otherwise → Weaviate (reason: `DEFAULT_WEAVIATE`)

**Reranking Algorithm** (`router_core.py:113-183`):

**Three-factor scoring**:
- **Similarity** (0.5 weight): `1.0 / (1.0 + |score|)` transformation of L2 distance
- **Recency** (0.3 weight): `max(0, 1.0 - (days_since_publish / 730))` - linear decay over 2 years
- **Diversity** (0.2 weight): 0.1 bonus for first occurrence of each source domain

**Domain-aware selection**:
- After scoring, selects top-k with max 2 documents per domain (`domain_cap=2`)
- Prevents one domain from dominating results

**Critical Finding**: Gate-7 evaluation does **NOT call** `router_core.rerank()` (verified in `qa_step07_retrieval_eval.py:382-577`). It directly tests MCP service results, which only include the lexical boost (0.7 vec + 0.3 lex) but miss the recency and diversity adjustments.

**Impact on Near-Miss Rate** (`step07_retrieval_eval.json:99`):
- Near-miss rate = 17.39% (doc_recall - chunk_recall = 82.61% - 65.22%)
- Meaning: 17.39% of queries retrieve the correct **document** but not the exact **chunk** in top-10
- **Cause**: Without recency reranking, older similar chunks from the same document rank equally with the expected chunk

### 5. Example Failure Analysis from Trace Files

**Query**: "What was Salesforce's total revenue for Q1 FY26?" (`eval_id: 10q_q1_revenue`)

- **Expected Chunk**: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0004`
- **Backend Routed**: Weaviate (reason: `DEFAULT_WEAVIATE` - no keyword match)
- **Classification**: `chunk_miss_doc_miss` (neither chunk nor doc found in top-10)

**Top-10 Results** (`retrieval_failures.jsonl:1`):
1. `crm::press::2025-05-28::news-details::e526586e::chunk0000` (score: 0.829) - Press release about Q1 results
2. `crm::press::2025-05-28::salesforce-reports-record-first-quarter-fiscal-2026-results...::chunk0000` (score: 0.827)
3-10. All press releases from `investor.salesforce.com` and `www.salesforce.com`

**Analysis**:
- Query keywords: `[revenue, q1, fy26]`
- Expected document (10-Q) published: 2025-04-30
- Top result (press release) published: 2025-05-28 (28 days later)

**Why Press Releases Rank Higher**:
1. **Better lexical overlap**: Press releases contain "Q1 FY26 revenue" in titles and summaries
2. **Shorter, focused chunks**: Press releases are 2-3 chunks vs 91 chunks for 10-Q
3. **Title boosting**: Each chunk prepended with document title containing query keywords
4. **XBRL noise absence**: Press releases lack the metadata pollution present in SEC filings
5. **No recency penalty**: MCP stub doesn't apply recency scoring, so 28-day-older press release doesn't lose points

**10-Q Chunk 0004 Content**: Based on chunking logic (800 tokens/chunk, start_char ~12,000), chunk0004 is likely still in the XBRL metadata section, not the narrative MD&A section where revenue is discussed in prose.

**Query**: "What regulatory compliance requirements does Salesforce face regarding data protection?" (`eval_id: 10q_data_privacy`)

- **Expected Chunk**: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0051`
- **Backend Routed**: Weaviate
- **Classification**: `chunk_miss_doc_hit_far`
- **Nearest Same-Doc Chunk**: rank=1, `chunk0071`, delta_seq=20, delta_start=65180

**Top Result** (`retrieval_failures.jsonl:6`):
- Rank 1: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0071` (score: 0.776)

**Analysis**:
- The system **found the correct 10-Q document** and ranked it #1
- But retrieved chunk0071 instead of expected chunk0051
- Delta: 20 chunks away, 65,180 characters apart

**Interpretation**: The 10-Q narrative contains multiple sections discussing regulatory compliance. Without SEC Item boundaries:
- chunk0051 might be in Item 1A (Risk Factors - Legal/Regulatory Risks)
- chunk0071 might be in Item 2 (MD&A - Regulatory Environment)
- Both discuss "data protection regulations," but in different contexts

**Without SEC Item Spans**: The chunker has no signal that Item 1A (forward-looking risk discussion) should be kept separate from Item 2 (current regulatory compliance status). Both sections contain similar keywords but serve different purposes.

## Code References

### Embedding System
- `scripts/embedding_utils.py:86-133` - OpenAI ada-002 implementation
- `scripts/embedding_utils.py:19` - ADA002_DIM = 1536 constant
- `scripts/embedding_utils.py:97-101` - Dimension validation logic
- `configs/vector.indexing.yaml:1-5` - Embedding configuration
- `scripts/qa_step01_embeddings.py:173` - Gate-1 embedding generation
- `scripts/qa_step07_retrieval_eval.py:233-248` - Gate-7 query embedding

### SEC Filing Processing
- `scripts/fetch_sec_filings.py:27-64` - Hardcoded SEC document list
- `scripts/parse_sec_structures.py:29-76` - Item detection logic (NOT run on 10-K)
- `scripts/extract_metadata.py:80-96` - Date selection for SEC filings
- `scripts/chunk_documents.py:170-188` - SEC-aware chunking logic
- `scripts/chunk_documents.py:67-102` - Sliding window with boundary snapping
- `data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json` - Normalized 10-K (missing `sec_item_spans`)
- `data/interim/chunks/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.chunks.jsonl` - 133 chunks

### Query Backend Implementation
- `scripts/qa_step03_mcp.py:82-156` - Unified MCP query handler (all backends)
- `scripts/qa_step03_mcp.py:106-119` - L2 distance and vector similarity
- `scripts/qa_step03_mcp.py:120-145` - Lexical boost reranking (0.7 vec + 0.3 lex)
- `scripts/qa_step02_indexes.py:80-164` - FAISS HNSW index build

### Routing and Reranking
- `scripts/router_core.py:72-100` - Backend routing decision
- `scripts/router_core.py:113-183` - Reranking algorithm (NOT used in Gate-7)
- `configs/router.heuristics.yaml:1-4` - Reranking weights (sim:0.5, rec:0.3, div:0.2)
- `configs/router.heuristics.yaml:19-40` - Keyword routing rules

### Gate-7 Evaluation
- `scripts/qa_step07_retrieval_eval.py:116-205` - MCP connection and fallback
- `scripts/qa_step07_retrieval_eval.py:368-576` - Per-query retrieval loop
- `scripts/qa_step07_retrieval_eval.py:407-426` - Chunk-level metrics calculation
- `scripts/qa_step07_retrieval_eval.py:447-478` - Near-miss detection logic
- `scripts/qa_step07_retrieval_eval.py:697-789` - JSON report generation
- `reports/qa/step07_retrieval_eval.json` - Gate-7 results (RED status)
- `reports/eval/retrieval_failures.jsonl` - Failed query details

## Architecture Documentation

### Data Flow: Query → Retrieval → Ranking → Evaluation

```
1. User Query
   ↓
2. Router Decision (router_core.py:72-100)
   - Keyword matching (first match wins)
   - Persona bias
   - Heuristic fallback
   ↓
3. Query Embedding (embedding_utils.py:86-133)
   - OpenAI ada-002 API call (1536-dim)
   - SHA-256 cache lookup/store
   ↓
4. MCP kb.search Service (qa_step03_mcp.py:82-156)
   - L2 distance computation (numpy)
   - Top-100 candidates
   - Lexical boost: 0.7 × vec + 0.3 × lex
   - Return top-10 with scores
   ↓
5. [Reranking SKIPPED in Gate-7]
   - Would apply: 0.5×sim + 0.3×rec + 0.2×div
   - Would enforce domain_cap=2
   ↓
6. Gate-7 Evaluation (qa_step07_retrieval_eval.py:368-576)
   - Extract expected_chunk_id from seed
   - Find rank in results (1-indexed, 0 if not found)
   - Calculate recall@10, nDCG@5
   - Detect near-misses (same doc, within seq_no tolerance)
   - Log failures to retrieval_failures.jsonl
   ↓
7. Report Generation
   - JSON: reports/qa/step07_retrieval_eval.json
   - Markdown: reports/qa/step07_retrieval_eval.md
   - Status: RED (recall < 80%, nDCG < 60%)
```

### Two-Stage Scoring

**Stage 1 - MCP Stub** (`qa_step03_mcp.py:120-145`):
```
final_score = 0.7 × (1/(1 + L2_dist)) + 0.3 × (|query_tokens ∩ snippet_tokens| / |query_tokens|)
```

**Stage 2 - Router Reranking** (`router_core.py:134-161`) [NOT applied in Gate-7]:
```
similarity = 1/(1 + |score|)
recency = max(0, 1 - days_since_publish/730)
diversity = 0.1 if source_domain not seen else 0.0

final_score = 0.5×similarity + 0.3×recency + 0.2×diversity

Then: Domain-cap filtering (max 2 per domain in top-10)
```

## Historical Context (from thoughts/)

- **thoughts/shared/issues/issue001.md** - Original proposal to change embedding model to OpenAI ada-002
- **thoughts/shared/issues/issue002.md** - Previous investigation showing 52.17% recall (lower than current 65.22%)
- **thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md** - Prior research on low recall problem
- **thoughts/shared/plans/OpenAI Ada-002 Migration Plan v2 (Unified).md** - Migration plan from hashlex-v1 to ada-002 (completed in commit e6597f5)

**Progression**:
1. **issue001**: Identified need to migrate to OpenAI embeddings
2. **issue002**: Detected 52.17% recall with old embedding model
3. **Commit e6597f5**: Migrated to ada-002 (1536-dim)
4. **issue003**: Current state shows **improved but still failing** 65.22% recall

This indicates the embedding migration helped (+13% recall) but did not solve the underlying structural issues with SEC filing processing.

## Related Research

- `thoughts/shared/research/exp0:2025-10-06-embedding-model-architecture.md` - Embedding model architecture and dimension dependencies

## Observations Summary

### What EXISTS (System State)

1. **Embedding Model**: OpenAI ada-002 (1536-dim) throughout entire pipeline (Gate-1, Gate-7, MCP stub)
2. **Backend Implementation**: All three backends use identical numpy L2 search
3. **SEC Filing State**:
   - 10-K: 133 chunks, no `sec_item_spans`, XBRL-heavy early chunks
   - 10-Q: 91 chunks, 1 Item span (incomplete parsing)
   - 8-K: 2-3 chunks, 1 Item span, working correctly
4. **Routing System**: Keyword-based with persona bias fallback
5. **MCP Stub Scoring**: 0.7 vector + 0.3 lexical boost
6. **Gate-7 Evaluation**: Direct MCP result testing without router reranking
7. **Metrics**:
   - Overall: 65.22% recall@10, 34.41% nDCG@5
   - 10-Q: 0% chunk recall, 50% doc recall
   - 10-K: 0% chunk recall, 33% doc recall
   - Press releases: 77% chunk recall
   - Product docs: 100% chunk recall

### Failure Patterns

1. **SEC filings consistently fail chunk retrieval** but partially succeed at doc retrieval
2. **Press releases consistently outrank SEC filings** for financial queries
3. **Near-miss rate of 17.39%** indicates relevant chunks exist but rank outside top-10
4. **Pinecone shows 5% nDCG** because it's routed the hardest queries (financial/SEC-related)
5. **FAISS shows 80% recall** because it's routed easier definitional queries
6. **Doc-level metrics (82.61% recall, 69.33% nDCG@5) much better than chunk-level**, indicating ranking/chunking issues rather than embedding quality

## Interconnected Root Causes

The failures are NOT caused by a single issue but by **five interrelated problems**:

### 1. Documentation Lag (Non-Functional Issue)
- CLAUDE.md references hashlex-v1, actual system uses ada-002
- Creates confusion but doesn't affect retrieval
- **Fix**: Update CLAUDE.md documentation

### 2. SEC Structure Parsing Failure (Critical for 10-Q/10-K)
- `parse_sec_structures.py` not run on 10-K (missing `sec_item_spans`)
- 10-Q has only 1 Item detected despite multi-Item structure
- **Result**: Poor chunking boundaries, mixing unrelated sections

### 3. XBRL Metadata Pollution (Embedding Quality)
- Early chunks dominated by iXBRL namespace URIs and entity IDs
- Dilutes semantic meaning of embeddings
- Narrative content (MD&A, Risk Factors) appears in later chunks
- **Result**: Queries for narrative information don't match metadata-heavy embeddings

### 4. Missing Reranking in Gate-7 (Evaluation Gap)
- Gate-7 tests raw MCP results without `router_core.rerank()`
- Missing recency scoring prevents recent SEC filings from ranking higher
- Missing diversity enforcement allows press releases to dominate
- **Result**: 17.39% near-miss rate, older chunks rank equally with newer ones

### 5. Backend Non-Differentiation (Architectural Issue)
- All backends use same numpy array with identical search logic
- Routing distributes query difficulty unevenly (Pinecone gets hardest queries)
- **Result**: Backend metrics reflect routing bias, not backend quality

## Conclusion

The Gate-7 RED status is caused by a **combination of incomplete SEC filing preprocessing, XBRL metadata pollution, and missing reranking logic in evaluation**. The embedding model is internally consistent (ada-002 throughout), but the chunking and ranking systems don't handle structured financial documents effectively.

**Critical Insight**: The system improved from 52.17% to 65.22% recall after embedding migration, but further gains require addressing **document structure parsing and ranking logic**, not just embedding quality.

The fact that doc-level recall (82.61%) is much higher than chunk-level recall (65.22%) confirms the embeddings are semantically matching the right documents, but the chunking and ranking mechanisms aren't surfacing the right specific chunks within those documents.