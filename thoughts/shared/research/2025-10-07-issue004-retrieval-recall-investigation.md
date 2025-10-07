---
date: 2025-10-07T14:16:15-04:00
researcher: System
git_commit: e6597f5ac21d7a1d210428bddbc5dd75374fde90
branch: agent-weaviate
repository: agent-weaviate
topic: "Retrieval System Recall Investigation - Issue004 Gate-7 RED Status"
tags: [research, codebase, retrieval, embeddings, gate-7, recall, sec-filings, issue004]
status: complete
last_updated: 2025-10-07
last_updated_by: System
---

# Research: Retrieval System Recall Investigation - Issue004 Gate-7 RED Status

**Date**: 2025-10-07T14:16:15-04:00
**Researcher**: System
**Git Commit**: e6597f5ac21d7a1d210428bddbc5dd75374fde90
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

Investigate the retrieval system recall issues documented in Issue004, with critical priority on verifying the embedding model discrepancy (hashlex-v1 vs openai-ada-002) and understanding why Gate-7 shows RED status with 65.22% chunk-level recall@10 (target ≥80%) and 34.41% nDCG@5 (target ≥60%).

## Summary

The retrieval system is currently using **OpenAI ada-002 embeddings (1536-dim)**, NOT the hashlex-v1 (768-dim) embeddings described in CLAUDE.md. The system migrated from hashlex-v1 to ada-002 in commit e6597f5 but the documentation was not updated. Gate-7 evaluation shows:

- **Chunk-level recall@10**: 65.22% (FAIL - target ≥80%)
- **nDCG@5**: 34.41% (FAIL - target ≥60%)
- **Near-miss rate**: 17.39% (relevant chunks exist but ranked outside top-10)
- **SEC filing performance**: 10-Q and 10-K have 0% chunk-level recall despite 50% and 33% document-level recall

**Backend Performance Differences**:
- FAISS: 80% chunk recall, 100% doc recall, 51.31% nDCG@5
- Weaviate: 65.38% chunk recall, 73.08% doc recall, 39.22% nDCG@5
- Pinecone: 50% chunk recall, 90% doc recall, **5% nDCG@5** (critical ranking disorder)

**Document Type Performance (chunk-level hits)**:
- Product docs: 100% (6/6) ✓
- Press releases: 76.92% (20/26)
- Wikipedia: 50% (1/2)
- 10-Q filings: **0% (0/6)** ⚠️
- 10-K filings: **0% (0/3)** ⚠️
- 8-K: 100% (1/1)

The evidence indicates three primary issues:
1. **Embedding model documentation discrepancy** - resolved (actual model is ada-002)
2. **SEC filing chunking/ranking problems** - 10-Q/10-K chunks exist but are ranked outside top-10
3. **Pinecone ranking disorder** - severe nDCG@5 failure (5%) despite 90% document recall

## Detailed Findings

### 1. Embedding Model: OpenAI ada-002 (NOT hashlex-v1)

**Current Implementation** ([scripts/embedding_utils.py:86](scripts/embedding_utils.py))

The system uses OpenAI's text-embedding-ada-002 model with:
- **Fixed dimension**: 1536 (constant `ADA002_DIM` at line 19)
- **Model**: `text-embedding-ada-002` (API call at line 79)
- **Caching**: SHA-256 based file cache at `data/cache/embeddings/` (line 20)
- **Retry logic**: 3 attempts with exponential backoff (4-10 seconds) via tenacity decorator (lines 70-74)
- **Cost tracking**: Built-in estimation at $0.0001 per 1K tokens (lines 214-240)

**Configuration File** ([configs/vector.indexing.yaml:1-5](configs/vector.indexing.yaml))

```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20
  notes: OpenAI text-embedding-ada-002 with caching and retry logic
```

**Migration History**:
- Previous model: hashlex-v1 (768-dim, deterministic feature hashing)
- Migration commit: e6597f5 "feat(embeddings): migrate from hashlex-v1 to OpenAI ada-002 (#17)"
- **Documentation gap**: CLAUDE.md still references hashlex-v1 as current implementation

**Consistency Verification**:
- Gate-1 (document embedding): Uses `embed_text()` from `embedding_utils.py` ([scripts/qa_step01_embeddings.py:172](scripts/qa_step01_embeddings.py))
- Gate-7 (query embedding): Uses same `embed_text()` function ([scripts/embedding_utils.py:86](scripts/embedding_utils.py))
- MCP stub (query processing): Uses same `embed_text()` function ([scripts/qa_step03_mcp.py:104](scripts/qa_step03_mcp.py))
- **Result**: All embeddings generated with identical function and dimension (1536), ensuring vector space consistency

### 2. Gate-1: Embedding Generation Pipeline

**Process** ([scripts/qa_step01_embeddings.py:115](scripts/qa_step01_embeddings.py)):

1. **Input**: Reads chunks from `data/interim/chunks/*.chunks.jsonl` (line 129)
2. **Cost estimation**: Samples 100 chunks, estimates API cost, prompts for confirmation (lines 139-161)
3. **Batch processing**: Calls `embed_batch(texts, dim=1536, batch_size=20)` with:
   - Cache lookup for each text (line 176-186 in embedding_utils.py)
   - OpenAI API calls for uncached texts in batches of 20 (line 192-194)
   - L2 norm validation (line 185)
4. **Quality checks**: Validates no zero vectors, no NaN vectors, outlier detection via IQR (lines 184-214)
5. **Output**: Writes to `data/vector/embeddings/embeddings.parquet` with schema:
   ```
   chunk_id: string
   doc_id: string
   seq_no: int32
   token_count: int32
   l2_norm: float32
   vector: list<float32>[1536]
   ```

**Actual Run Results** (from reports/qa/step01_embeddings.json):
- Embedding rows: 565 (matches baseline chunks)
- Vector dimension: 1536 ✓
- Zero vectors: 0 ✓
- NaN vectors: 0 ✓
- Outlier percentage: 0% ✓
- Status: GREEN

### 3. Gate-2: Index Building

**FAISS Index** ([scripts/qa_step02_indexes.py:80-164](scripts/qa_step02_indexes.py)):

- **Index type**: HNSW (Hierarchical Navigable Small World) graph
- **Parameters**: M=32, efConstruction=200, efSearch=128 (from config lines 118-127)
- **Metric**: L2 distance (line 119)
- **Location**: `data/vector/faiss/index.faiss`
- **ID mapping**: `data/vector/faiss/idmap.parquet` maps FAISS integer IDs → chunk_id/doc_id (lines 167-181)
- **Round-trip validation**: Queries 100 random vectors, computes max reconstruction error (lines 136-149)

**Pinecone & Weaviate** ([scripts/qa_step02_indexes.py:282-308](scripts/qa_step02_indexes.py)):

- **Simulated manifests**: No actual network calls to Pinecone or Weaviate services
- Writes JSON manifests documenting upserted count and configuration
- **Pinecone manifest**: `data/vector/pinecone/index_manifest.json`
- **Weaviate schema**: `data/vector/weaviate/schema_applied.json`
- **Weaviate manifest**: `data/vector/weaviate/index_manifest.json`

**Critical Note**: All three backends use the **same** embeddings from Gate-1 (1536-dim ada-002 vectors). Backend differences in retrieval quality are due to:
- Simulated backend latency in MCP stub (FAISS: 5-10ms, Weaviate: 40-80ms, Pinecone: 80-160ms)
- Simulated index characteristics (all use same L2 distance calculation in practice)

### 4. Gate-7: Retrieval Evaluation

**Evaluation Process** ([scripts/qa_step07_retrieval_eval.py:116](scripts/qa_step07_retrieval_eval.py)):

1. **Query routing**: Calls `router_core.decide_backend()` with three-tier cascade:
   - Keyword rules (lines 81-89): "earnings" → pinecone, "api" → weaviate, "definition" → faiss
   - Persona bias (lines 91-94): cio → weaviate, vp_sales_ops → pinecone
   - Heuristic fallback (lines 96-100): short queries → faiss, others → weaviate

2. **MCP stub search** ([scripts/qa_step03_mcp.py:82-156](scripts/qa_step03_mcp.py)):
   - Query embedding via `embed_text(query, 1536)` (line 104)
   - L2 distance computation: `((xb - qv)**2).sum(axis=1)` (line 106)
   - Top-100 candidate retrieval (line 108-119)
   - **Lexical reranking** (lines 121-144):
     - Tokenizes query and chunk text
     - Computes Jaccard-like lexical overlap
     - Final score = `0.7 * vector_sim + 0.3 * lexical_boost`
     - Returns top-K after reranking

3. **Metrics computation** (lines 406-549):
   - **Chunk-level recall@10**: Checks if expected chunk_id appears in top-10 (lines 407-418)
   - **Document-level recall@10**: Checks if expected doc_id appears in top-10 (lines 429-445)
   - **nDCG@5**: Binary relevance with discounted gain `1.0 / log2(rank+1)` for ranks 1-5 (lines 420-421)
   - **Near-miss detection** (lines 447-479): If chunk missed but doc found, checks if nearest same-doc chunk within `seq_no ± 1`
   - **Coverage**: Unique source domains in top-10 (lines 515-521)
   - **Freshness**: Mean document age in days (lines 522-530)

4. **Evidence generation**:
   - **Failure log**: `reports/eval/retrieval_failures.jsonl` - 16 chunk-level misses with full context (lines 552-576)
   - **Trace log**: `reports/router/step07_retrieval_trace.jsonl` - per-query routing and top-K results (lines 481-512)

**Actual Run Results** ([reports/qa/step07_retrieval_eval.json](reports/qa/step07_retrieval_eval.json)):

```
Queries evaluated: 46
Chunk recall@10: 65.22% (30/46) - FAIL (target ≥80%)
Document recall@10: 82.61% (38/46) - PASS
nDCG@5: 34.41% - FAIL (target ≥60%)
Doc nDCG@5: 69.33% - PASS
Near-miss rate: 17.39% (8/46)
Soft recall@10: 4.35% (2/46)

Rank stats (when found):
  Chunk: p50=3, p75=5, p90=7, max=10
  Doc: p50=1, p75=1, p90=3, max=7
```

### 5. SEC Filing Chunking and Metadata

**SEC Structure Parsing** ([scripts/parse_sec_structures.py:29-76](scripts/parse_sec_structures.py)):

- **Item detection**: Regex patterns for Item 1, 1A, 7, 7A, 8 (lines 13-19)
- **Span extraction**: Identifies start/end character positions for each SEC item
- **Coverage ratio**: Percentage of document covered by detected items
- **Output**: Adds `sec_item_spans` and `sec_item_coverage_ratio` to normalized documents

**Chunking Strategy** ([scripts/chunk_documents.py:180-188](scripts/chunk_documents.py)):

For SEC filings (10-K, 10-Q, 8-K, ars_pdf):
1. **Segmentation**: Splits document into segments per SEC item span (lines 182-186)
2. **Boundary candidates**: Collects SEC item start positions + H2/H3 headings (lines 167-173)
3. **Sliding window**: 800-token target, 120-token overlap, with boundary snapping (lines 67-102)
4. **Title boosting**: Prefixes each chunk with document title + first H1 (lines 125-133)

**Default Parameters** ([configs/chunking.config.json](configs/chunking.config.json)):
```json
{
  "tokenizer": "cl100k_base",
  "target_tokens": 800,
  "overlap_tokens": 120,
  "short_doc_threshold_tokens": 350,
  "boundary_tolerance_chars": 50
}
```

**Metadata Snapshot** ([scripts/chunk_documents.py:141-160](scripts/chunk_documents.py)):

Each chunk includes:
```json
{
  "chunk_id": "{doc_id}::chunk{seq_no:04d}",
  "doc_id": "...",
  "seq_no": 0,
  "text": "title-boosted chunk text",
  "word_count": 484,
  "token_count": 800,
  "start_char": 3208,
  "end_char": 10189,
  "local_heads": ["Recent H2", "Previous H3"],
  "metadata_snapshot": {
    "company": "Salesforce",
    "doctype": "10-Q",
    "date": "2025-04-30",
    "url": "https://www.sec.gov/...",
    "title": "fy26-q1-form-10-q",
    "topic": "",
    "persona_tags": []
  }
}
```

### 6. Router Reranking Logic

**Backend Selection** ([scripts/router_core.py:72-100](scripts/router_core.py)):

Three-tier precedence (first match wins):

1. **Keyword rules** (lines 81-89):
   - Press/Financial: `["results", "earnings", "fiscal", "guidance"]` → pinecone
   - Developer/API: `["api", "endpoint", "schema", "developer"]` → weaviate
   - Definitional: `["definition", "what is", "overview"]` → faiss

2. **Persona bias** (lines 91-94):
   - `vp_sales_ops` → pinecone
   - `cio` → weaviate
   - `vp_customer_experience` → faiss

3. **Heuristic fallback** (lines 96-100):
   - Short queries (≤4 words) OR definitional keywords → faiss
   - All others → weaviate

**Reranking Scores** ([scripts/router_core.py:113-183](scripts/router_core.py)):

```python
final_score = (0.6 × similarity) + (0.3 × recency) + (0.1 × diversity_bonus)
```

**Components**:
- **Similarity** (lines 134-146): Transform raw score via `1.0 / (1.0 + abs(score))` (assumes negative L2 distances)
- **Recency** (lines 148-153): `max(0.0, 1.0 - (age_days / 730.0))` (2-year horizon)
- **Diversity** (lines 155-156): Binary 0.1 bonus for first document from each domain

**Domain-aware selection** (lines 165-180):
- Enforces `domain_cap=2` (max 2 results per domain in top-10)
- Skips additional results from domains that reached cap
- Appends skipped results to tail after top-K

**Weight Discrepancy**:
- Hardcoded defaults: 0.6/0.3/0.1 (line 127)
- Config values: 0.5/0.3/0.2 ([configs/router.heuristics.yaml:1-4](configs/router.heuristics.yaml))
- Resolution: Weights only update if explicitly passed to `rerank()` function

### 7. Retrieval Failure Analysis

**Evidence from retrieval_failures.jsonl** (16 failures analyzed):

**10-Q FY26 Q1 queries (6 failures)**:
- `eval_id: 10q_q1_revenue` - "What was Salesforce's total revenue for Q1 FY26?"
  - Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0004`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: All press releases (ranks 1-10), no 10-Q document found
  - Routed to: weaviate

- `eval_id: 10q_operating_cash` - "How much operating cash flow did Salesforce generate in Q1 FY26?"
  - Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0031`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: All press releases, no 10-Q found
  - Routed to: weaviate

- `eval_id: 10q_senior_notes` - "When do Salesforce's senior notes mature and what are the interest rates?"
  - Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0019`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: 10-K at rank 1, 8-K at rank 8, press releases, but not expected 10-Q
  - Routed to: weaviate

- `eval_id: 10q_share_repurchase` - "What is the status of Salesforce's share repurchase program?"
  - Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0032`
  - Classification: `chunk_miss_doc_hit_far`
  - **Document found at rank 3** but wrong chunk: `chunk0089` (delta_seq=57, delta_start=185763 chars)
  - Routed to: faiss

- `eval_id: 10q_ai_risks` - "What are the key risks Salesforce identifies related to AI and generative AI?"
  - Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0050`
  - Classification: `chunk_miss_doc_hit_far`
  - **Document found at rank 6** but wrong chunk: `chunk0070` (delta_seq=20, delta_start=65180 chars)
  - Routed to: weaviate

- `eval_id: 10q_data_privacy` - "What regulatory compliance requirements does Salesforce face regarding data protection?"
  - Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0051`
  - Classification: `chunk_miss_doc_hit_far`
  - **Document found at rank 1** but wrong chunk: `chunk0071` (delta_seq=20, delta_start=65180 chars)
  - Routed to: weaviate

**10-K FY25 queries (3 failures)**:
- `eval_id: 10k_fy25_performance` - "What was Salesforce's full year FY25 financial performance?"
  - Expected: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0006`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: All press releases, no 10-K found
  - Routed to: weaviate

- `eval_id: 10k_agentforce_strategy` - "How does Salesforce describe its Agentforce AI agent strategy in the annual report?"
  - Expected: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0008`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: All press releases, no 10-K found
  - Routed to: weaviate

- `eval_id: 10k_sales_cloud_offerings` - "What capabilities does Salesforce Sales Cloud offer according to the 10-K?"
  - Expected: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0009`
  - Classification: `chunk_miss_doc_hit_far`
  - **Document found at rank 3** but wrong chunk: `chunk0011` (delta_seq=2, delta_start=6416 chars)
  - Routed to: pinecone

**Press release queries (2 near-miss failures)**:
- `eval_id: press_fy25_q4_revenue` - "What were Salesforce's Q4 FY25 revenue and earnings results?"
  - Expected: `crm::press::2025-01-31::news-details::9711c8f6::chunk0004`
  - Classification: `chunk_miss_doc_hit_near`
  - **Document found at rank 2** with near chunk: `chunk0000` (delta_seq=4, delta_start=10696 chars)
  - Routed to: pinecone

- `eval_id: press_fy26_q1_revenue` - "What were Salesforce's Q1 FY26 financial results?"
  - Expected: `crm::press::2025-05-28::news-details::e526586e::chunk0008`
  - Classification: `chunk_miss_doc_hit_far`
  - **Document found at rank 1** but wrong chunk: `chunk0000` (delta_seq=8, delta_start=23512 chars)
  - Routed to: pinecone

**Press release complete miss (2 failures)**:
- `eval_id: press_informatica_acquisition` - "Did Salesforce announce any major acquisitions in 2025?"
  - Expected: `crm::press::2025-05-27::news-details::56b542ba::chunk0002`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: Various press releases but not the expected document
  - Routed to: weaviate

- `eval_id: press_earnings_sep_2024` - "When did Salesforce announce its September 2024 quarter earnings?"
  - Expected: `crm::press::2024-09-05::news-details::e32cfe52::chunk0000`
  - Classification: `chunk_miss_doc_miss`
  - Top-10: All returned from Jan 2024, Jul 2024, Sep 2025 periods (wrong year/quarter)
  - Routed to: pinecone

**Press release partially missing (1 failure)**:
- `eval_id: press_saudi_investment` - "What investment did Salesforce announce in Saudi Arabia?"
  - Expected: `crm::press::2025-02-10::salesforce-announces-500m-investment-in-saudi-arabia...::chunk0001`
  - Classification: `chunk_miss_doc_miss`
  - **Top-10: empty array** - no results returned at all
  - Routed to: weaviate

**Wikipedia query (1 failure)**:
- `eval_id: wiki_salesforce_business` - "What is Salesforce's business model and product portfolio?"
  - Expected: `crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0011`
  - Classification: `chunk_miss_doc_hit_far`
  - **Document found at rank 1** but wrong chunk: `chunk0003` (delta_seq=8, delta_start=18907 chars)
  - Routed to: faiss

### 8. Backend Performance Patterns

**FAISS** (10 queries, best performer):
- Chunk recall@10: 80% (8/10)
- Doc recall@10: 100% (10/10)
- nDCG@5: 51.31%
- Near-miss count: 0

**Queries routed to FAISS**:
- All matched DEFINITION routing rule (lines 87-89 in router_core.py)
- Characteristics: Short queries (≤4 words) or containing definitional keywords
- Example: "What is Salesforce's business model and product portfolio?"

**Weaviate** (26 queries, largest sample):
- Chunk recall@10: 65.38% (17/26)
- Doc recall@10: 73.08% (19/26)
- nDCG@5: 39.22%
- Near-miss count: 0

**Queries routed to Weaviate**:
- 23 matched DEFAULT_WEAVIATE (fallback rule)
- 2 matched FILTER_MATCH (developer/API keywords)
- 1 matched PERSONA_BIAS (cio persona)
- Characteristics: General queries without specific keyword matches
- Example: "What was Salesforce's full year FY25 financial performance?"

**Pinecone** (10 queries, worst performer):
- Chunk recall@10: 50% (5/10)
- Doc recall@10: 90% (9/10)
- nDCG@5: **5%** (critical ranking disorder)
- Near-miss count: 2

**Queries routed to Pinecone**:
- All matched PR_QUERY routing rule (lines 81-89 in router_core.py)
- Characteristics: Earnings/financial queries with keywords ["results", "earnings", "fiscal", "guidance"]
- Example: "What were Salesforce's Q4 FY25 revenue and earnings results?"

**Critical Observation**: Despite 90% document recall, Pinecone achieves only 5% nDCG@5, indicating it finds the right documents but ranks them very poorly. The correct document often appears late in the top-10 (ranks 2-3) instead of rank 1, causing severe DCG penalization.

## Code References

- `scripts/embedding_utils.py:86` - OpenAI ada-002 embedding function (current implementation)
- `scripts/embedding_utils.py:19` - 1536-dimension constant definition
- `scripts/qa_step01_embeddings.py:172` - Gate-1 batch embedding call
- `scripts/qa_step02_indexes.py:80-164` - Gate-2 FAISS index building with HNSW
- `scripts/qa_step07_retrieval_eval.py:116` - Gate-7 evaluation main loop
- `scripts/qa_step03_mcp.py:82-156` - MCP stub kb.search implementation with lexical reranking
- `scripts/router_core.py:72-100` - Backend selection heuristics
- `scripts/router_core.py:113-183` - Multi-factor reranking with similarity/recency/diversity
- `scripts/chunk_documents.py:180-188` - SEC-specific segmentation logic
- `scripts/parse_sec_structures.py:29-76` - SEC item boundary detection
- `configs/vector.indexing.yaml:1-5` - Embedding model configuration (ada-002, 1536-dim)
- `configs/router.heuristics.yaml:19-39` - Keyword-based routing rules
- `configs/chunking.config.json` - Chunking parameters (800 tokens, 120 overlap)

## Architecture Documentation

### End-to-End Retrieval Flow

```
1. Query → Router (router_core.decide_backend)
   ├─ Keyword matching (PR_QUERY, FILTER_MATCH, DEFINITION)
   ├─ Persona bias (cio → weaviate)
   └─ Heuristic fallback (short → faiss, other → weaviate)

2. Router → MCP Stub (qa_step03_mcp.py:82)
   ├─ Query embedding via embed_text(query, 1536)
   ├─ L2 distance computation against all 565 document embeddings
   ├─ Top-100 candidate retrieval by vector similarity
   └─ Lexical reranking (0.7 × vector + 0.3 × lexical)

3. MCP Stub → Gate-7 Evaluator
   ├─ Chunk-level recall check (expected_chunk_id in top-10?)
   ├─ Document-level recall check (expected_doc_id in top-10?)
   ├─ nDCG@5 computation (1.0 / log2(rank+1) for ranks 1-5)
   ├─ Near-miss detection (same doc, |delta_seq| ≤ 1)
   └─ Evidence logging (failures to retrieval_failures.jsonl)

4. Gate-7 → Report Generation
   ├─ Status rollup (GREEN/AMBER/RED)
   ├─ Per-backend metrics (faiss, weaviate, pinecone)
   ├─ Per-doctype breakdown (10-Q, 10-K, press, product, wiki)
   └─ Dual-format reports (JSON + Markdown)
```

### Embedding Consistency Chain

```
Gate-1 (Document Embedding)
  ├─ Input: data/interim/chunks/*.chunks.jsonl
  ├─ Function: embed_batch() → embed_text(text, 1536)
  ├─ Model: OpenAI ada-002 via API
  └─ Output: data/vector/embeddings/embeddings.parquet [565 rows × 1536-dim]

Gate-2 (Index Building)
  ├─ Input: data/vector/embeddings/embeddings.parquet
  ├─ FAISS: HNSW index with L2 distance
  ├─ Pinecone: Simulated manifest (no network)
  └─ Weaviate: Simulated manifest (no network)

Gate-7 (Query Embedding)
  ├─ Input: Query text from eval seed
  ├─ Function: embed_text(query, 1536) [SAME as Gate-1]
  ├─ Model: OpenAI ada-002 via API [SAME as Gate-1]
  └─ Search: L2 distance against Gate-1 embeddings

✓ Consistency: All embeddings use identical function, model, and dimension
```

### SEC Filing Processing Pipeline

```
Raw SEC Filing
  ├─ fetch_sec_filings.py → data/raw/sec/*.html + .meta.json
  └─ Normalization → data/interim/normalized/{doc_id}.json

Normalized Document
  ├─ parse_sec_structures.py → Detect Item 1, 1A, 7, 7A, 8 spans
  ├─ extract_metadata.py → title, publish_date, url, topic, persona_tags
  └─ Updated: data/interim/normalized/{doc_id}.json (with sec_item_spans)

Chunking (chunk_documents.py)
  ├─ Segmentation: Split by SEC item boundaries
  ├─ Sliding window: 800-token target, 120-token overlap
  ├─ Boundary snapping: Align to SEC item starts + H2/H3 headings
  ├─ Title boosting: Prefix with doc title + first H1
  └─ Output: data/interim/chunks/{doc_id}.chunks.jsonl

Embedding (qa_step01_embeddings.py)
  ├─ Read: All .chunks.jsonl files
  ├─ Embed: OpenAI ada-002 batch API (1536-dim)
  └─ Write: data/vector/embeddings/embeddings.parquet

Indexing (qa_step02_indexes.py)
  ├─ FAISS: HNSW graph index (M=32, efConstruction=200)
  └─ ID mapping: FAISS integer ID ↔ chunk_id/doc_id
```

## Historical Context (from thoughts/)

**Previous Research**:
- `thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md` - Earlier investigation of 52.17% recall rate (Issue002), documenting backend differences and suggesting hashlex-v1 limitations
- `thoughts/shared/research/exp0:2025-10-06-embedding-model-architecture.md` - Research on embedding model architecture and dimension dependencies

**Implementation Plans**:
- `thoughts/shared/plans/OpenAI Ada-002 Migration Plan v2 (Unified).md` - Migration plan from hashlex-v1 to ada-002 (successfully executed in commit e6597f5)

**Tracked Issues**:
- `thoughts/shared/issues/issue001.md` - Initial request to migrate to OpenAI ada-002 embeddings
- `thoughts/shared/issues/issue002.md` - Low recall investigation (52.17% → 65.22% improvement post-migration)
- `thoughts/shared/issues/issue003.md` - Gate-7 RED status with same metrics as issue004
- `thoughts/shared/issues/issue004.md` - Current investigation (this research)

**Key Progression**:
1. Issue001: Identified need to migrate from hashlex-v1 → ada-002
2. Migration executed: Commit e6597f5 implemented ada-002 with 1536-dim
3. Issue002: Recall improved from 52.17% to 65.22% but still below 80% target
4. Issue003/004: Current state showing persistent ranking and SEC filing issues

## Related Research

- [2025-10-07-issue002-low-recall-investigation.md](thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md) - Root cause analysis of earlier recall issues
- [exp0:2025-10-06-embedding-model-architecture.md](thoughts/shared/research/exp0:2025-10-06-embedding-model-architecture.md) - Embedding model architecture investigation

## Open Questions

**Documentation Alignment**:
- CLAUDE.md still documents hashlex-v1 as the current embedding model - should be updated to reflect ada-002 implementation
- Router weight discrepancy: Code uses 0.6/0.3/0.1 but config specifies 0.5/0.3/0.2

**SEC Filing Retrieval**:
- Why do 10-Q and 10-K queries fail to retrieve their parent documents (0% chunk recall, doc-level recall also low)?
  - Is it due to press releases having higher lexical overlap with financial queries?
  - Are SEC filing chunks properly embedded and indexed?
  - Do SEC filings have different text density that affects chunk quality?

**Pinecone Ranking Disorder**:
- Why does Pinecone show 90% document recall but only 5% nDCG@5?
  - Is the simulated Pinecone manifest causing ranking artifacts?
  - Should Pinecone use different reranking parameters?
  - Is the PR_QUERY routing rule too aggressive?

**Near-Miss Analysis**:
- 17.39% of queries have relevant chunks ranked outside top-10 but within same document
  - Could larger top-K (e.g., top-20) improve coverage?
  - Should the lexical reranking boost be increased from 0.3 to 0.4-0.5?
  - Are adjacent chunks (delta_seq ≤ 1) semantically different enough to rank separately?

**Chunking Strategy**:
- Are 800-token chunks optimal for SEC filings vs press releases?
- Should SEC Item 1 (Business Overview) and Item 7 (MD&A) use different chunk sizes?
- Does title boosting help or hurt SEC filing retrieval (adds generic context to every chunk)?

**Query-Document Mismatch**:
- Financial queries use informal language ("What was revenue?") while 10-Q/10-K use formal language ("Total revenues, net")
- Should there be query expansion or normalization for SEC filing queries?
- Would synonym matching or financial term glossary improve recall?
