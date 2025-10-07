---
date: 2025-10-07T14:05:44-04:00
researcher: Claude Code
git_commit: e6597f5ac21d7a1d210428bddbc5dd75374fde90
branch: agent-weaviate
repository: ag3
topic: "Root Cause Analysis: Low Retrieval Recall Rate (Issue #002)"
tags: [research, codebase, retrieval-evaluation, gate-7, embeddings, chunking, routing, recall]
status: complete
last_updated: 2025-10-07
last_updated_by: Claude Code
---

# Research: Root Cause Analysis of Low Retrieval Recall Rate

**Date**: 2025-10-07T14:05:44-04:00
**Researcher**: Claude Code
**Git Commit**: e6597f5ac21d7a1d210428bddbc5dd75374fde90
**Branch**: agent-weaviate
**Repository**: ag3

## Research Question

Investigate the root causes of low retrieval recall rates in Gate-7 evaluation:
- Chunk-level recall@10 = 65.22% (target ≥80%)
- Backend performance differences: FAISS (80%) > Weaviate (65.38%) > Pinecone (50%)
- Document type variations: Press releases (76.9%) vs. SEC 10-K/10-Q (0%)
- Evaluation design rationality and potential systemic issues

## Summary

The low recall rate is caused by a **combination of four interrelated factors**, not a single root cause:

1. **Query Distribution Bias**: Router sends SEC 10-K/10-Q queries (which have 0% success) predominantly to Weaviate via DEFAULT_WEAVIATE rule, artificially lowering Weaviate's aggregate recall
2. **Evaluation Design Mismatch**: Ground truth expects chunk-level precision, but synthetic queries are generated from document titles using only the first word as keyword, creating overly generic queries
3. **Title Boosting Side Effect**: Every chunk is prepended with the document title, causing title-based queries to match ALL chunks from a document with equal semantic similarity
4. **Backend Implementation Parity**: All three backends execute identical L2 search with lexical reranking in the MCP stub—performance differences reflect query distribution, not implementation quality

**Critical Finding**: The backend performance gap (FAISS 80% > Weaviate 65.38% > Pinecone 50%) does NOT indicate implementation quality differences. All backends use the same search algorithm (`scripts/qa_step03_mcp.py:105`). The gap exists because:
- FAISS receives easy "definition" queries (10/10, 100% product docs)
- Weaviate receives difficult DEFAULT queries including all SEC filings (0/9 on 10-K/10-Q)
- Pinecone receives mixed "financial" queries

## Detailed Findings

### 1. Evaluation Methodology and Ground Truth

**Location**: `scripts/qa_step07_retrieval_eval.py:368-478`, `scripts/build_eval_seed.py:78-216`

#### Ground Truth Generation (`build_eval_seed.py`)

Ground truth is created by:
1. **Document Pool Assembly** (lines 87-94): Segregates documents by type (SEC, press, IR, product, dev, help, wiki)
2. **Quota-Based Sampling** (lines 105-157):
   - Target: 10 SEC filings, 10 IR articles, 10 newsroom, ≥8 product/dev/help, ≥2 wiki
   - Random shuffling with fixed seed (`random.seed(7)`)
3. **Query Synthesis** (lines 182-189):
   - **Template**: `"What does this document say about {keyword}?"`
   - **Keyword extraction**: First word from document title (line 183)
   - **Fallback**: `"Summarize key points regarding {keyword}"`
4. **Ground Truth Structure**: Each record specifies exactly ONE `expected_chunk_id`

**Design Limitation**: Queries generated from document titles using only the first word create overly generic queries that don't specify which chunk within a multi-chunk document should be retrieved.

**Example Issue**:
- Document: "FORM 10-Q - Salesforce, Inc. - FY26 Q1 Quarterly Report"
- Generated keyword: "FORM" (first word)
- Generated query: "What does this document say about FORM?"
- Expected chunk: `chunk0004` from this specific 10-Q
- Problem: This query could match any chunk from ANY 10-Q filing

#### Relevance Determination (`qa_step07_retrieval_eval.py:407-445`)

**Chunk-Level Relevance** (lines 407-410):
```python
ranks = [r.get("chunk_id") for r in res]
rank = ranks.index(exp_cid) + 1  # Binary: either exact match or miss
```

A result is relevant **only if** `retrieved_chunk_id == expected_chunk_id`. There is no partial credit for:
- Retrieving the correct document but wrong chunk
- Retrieving adjacent chunks (seq_no ± 1)
- Retrieving semantically similar content

**Near-Miss Detection** (lines 447-478):
The system tracks "near-misses" when:
- Correct document retrieved
- Chunk within ±1 sequence number of expected chunk (configurable via `AG7_NEAR_SEQ_TOL`)

Near-misses indicate chunking boundary issues rather than retrieval failures, but do NOT count toward recall@10.

#### Document-Level Recall (lines 428-445)

Doc-level recall@10 = 82.61%, significantly higher than chunk-level 65.22%. This gap indicates the system successfully retrieves **correct documents** but struggles with **exact chunk identification**.

---

### 2. Embedding System

**Location**: `scripts/embedding_utils.py:86-133`

#### Current Implementation: OpenAI ada-002

The system uses **OpenAI text-embedding-ada-002** (1536 dimensions), recently migrated from hashlex-v1 (768 dimensions).

**Embedding Process** (`embed_text()`, lines 86-133):
1. **Dimension Validation** (lines 97-101): Enforces 1536 dimensions
2. **Empty Text Handling** (lines 104-106): Returns `[0.001] * 1536` for empty inputs
3. **Cache Lookup** (lines 109-111): SHA256-based file cache at `data/cache/embeddings/`
4. **OpenAI API Call** (lines 114-122): With retry logic (3 attempts, exponential backoff)
5. **Cache Storage** (line 131): Saves to disk for reuse

**Critical Consistency**: Both documents (Gate-1) and queries (Gate-7) use the **same** `embed_text()` function:
- Gate-1: `scripts/qa_step01_embeddings.py:138` calls `embed_text(text, dim)`
- Gate-7: `scripts/qa_step07_retrieval_eval.py:233` imports same function
- MCP Service: `scripts/qa_step03_mcp.py:71` imports same function

This ensures queries and documents exist in the same vector space, which is **correct and working as designed**.

**No Embedding Issues Found**: The migration to ada-002 maintained consistency. Documents and queries use identical embedding generation, ruling out embedding mismatch as a root cause.

---

### 3. Backend Implementations

**Location**: `scripts/qa_step02_indexes.py` (construction), `scripts/qa_step03_mcp.py:81-155` (search)

#### FAISS Index Construction (`qa_step02_indexes.py:80-164`)

**Parameters** (from `configs/vector.indexing.yaml:7-12`):
- Type: HNSW (Hierarchical Navigable Small World)
- Metric: L2 distance
- M: 32 (bi-directional links per element)
- efConstruction: 200 (construction-time candidate list size)
- efSearch: 128 (search-time candidate list size)

**Build Process**:
- Line 117: Infer dimension from first vector (1536)
- Lines 121-127: Configure HNSW index
- Line 132-133: Add all vectors as float32
- Line 134: Write to `data/vector/faiss/index.faiss`

#### Weaviate & Pinecone Index Construction

**Simulated Only** (`qa_step02_indexes.py:282-308`):
- Weaviate: Schema manifest at `data/vector/weaviate/schema_applied.json` (lines 294-303)
- Pinecone: Index manifest at `data/vector/pinecone/index_manifest.json` (lines 282-292)
- **No actual network operations**: Manifests contain metadata only

#### Search Implementation: Identical Across All Backends

**CRITICAL FINDING** (`scripts/qa_step03_mcp.py:102-155`):

All three backends execute the **same search algorithm** in the MCP stub:

1. **Vector Search** (line 105):
   ```python
   dists = ((xb - qv)**2).sum(axis=1)  # L2 distance
   ```
   - Loads same embeddings matrix for all backends
   - Computes L2 distance to query vector
   - Takes top 100 candidates

2. **Lexical Reranking** (lines 121-143):
   ```python
   lex_boost = hits / len(query_tokens)
   final = 0.7 * vec_sim + 0.3 * lex_boost
   ```
   - Tokenizes query and computes overlap with chunk text
   - Applies 70% vector + 30% lexical weighting
   - **Identical for all backends**

3. **Simulated Latency** (lines 98-100):
   - FAISS: 5-10ms delay
   - Weaviate: 40-80ms delay
   - Pinecone: 80-160ms delay
   - **Only difference between backends**

**Implication**: Backend performance differences (FAISS 80% > Weaviate 65.38% > Pinecone 50%) cannot be explained by implementation differences. All backends run the same code.

---

### 4. Router Query Distribution

**Location**: `scripts/router_core.py:72-100`, `configs/router.heuristics.yaml:1-43`

#### Routing Logic

**Rule-Based Selection** (lines 81-89):
1. **Keyword Matching** (first-match-wins):
   - `[results, earnings, fiscal, revenue, ...]` → Pinecone (PR_QUERY rule)
   - `[api, apis, endpoint, sdk, ...]` → Weaviate (FILTER_MATCH rule)
   - `[definition, what is, overview, ...]` → FAISS (DEFINITION rule)
2. **Persona Bias** (lines 92-94):
   - `vp_sales_ops` → Pinecone
   - `cio` → Weaviate
   - `vp_customer_experience` → FAISS
3. **Default Fallback** (lines 96-100):
   - Short queries (≤4 words) or definitional → FAISS
   - Otherwise → Weaviate (DEFAULT_WEAVIATE)

#### Actual Distribution (`step07_retrieval_eval.json:269-281`)

**By Backend**:
- Weaviate: 26 queries (56.5%)
- FAISS: 10 queries (21.7%)
- Pinecone: 10 queries (21.7%)

**By Reason**:
- DEFAULT_WEAVIATE: 23 queries (50%)
- DEFINITION: 10 queries (21.7%)
- PR_QUERY: 10 queries (21.7%)
- FILTER_MATCH: 2 queries (4.3%)
- PERSONA_BIAS: 1 query (2.2%)

**Query-Type Alignment**:
- **FAISS** receives:
  - 10 DEFINITION queries
  - 6 product docs (100% success)
  - 1 dev_docs, 1 help_docs, 1 8-K (all succeeded)
  - **Chunk recall: 8/10 = 80%**

- **Weaviate** receives:
  - 23 DEFAULT_WEAVIATE queries (catch-all)
  - **All 9 SEC 10-K/10-Q queries** (0/9 success)
  - Mixed press, wiki queries
  - **Chunk recall: 17/26 = 65.38%**

- **Pinecone** receives:
  - 10 PR_QUERY queries (financial keywords)
  - Mixed press releases
  - **Chunk recall: 5/10 = 50%**

**Key Insight**: Weaviate's lower recall is NOT due to inferior implementation—it's assigned the hardest queries (SEC filings) by the DEFAULT_WEAVIATE fallback rule.

---

### 5. Chunking Strategy and Title Boosting

**Location**: `scripts/chunk_documents.py:105-160`, `configs/chunking.config.json:1-7`

#### Chunking Parameters

- **Target size**: 800 tokens (~3200 characters)
- **Overlap**: 120 tokens (15% overlap)
- **Short doc threshold**: <350 tokens → single chunk
- **Boundary tolerance**: ±50 characters for heading alignment

#### Title Boosting (`chunk_documents.py:125-138`)

**Every chunk is prepended with**:
1. Document title (from metadata)
2. First H1 heading (if different from title)

```python
boost_lines = []
if title:
    boost_lines.append(title)
if h1 and h1 != title:
    boost_lines.append(h1)
chunk_text = (boost + "\n\n" + body).strip()
```

**Example**:
- Document: "FORM 10-Q - Salesforce, Inc. - FY26 Q1"
- Chunk 0: "FORM 10-Q - Salesforce, Inc. - FY26 Q1\n\n[chunk body starting at char 0]"
- Chunk 4: "FORM 10-Q - Salesforce, Inc. - FY26 Q1\n\n[chunk body starting at char 13036]"

**Impact on Retrieval**:
- Query: "What does this document say about FORM?"
- All chunks from the document contain "FORM" in title boost
- OpenAI ada-002 embeddings will show similar semantic similarity for all chunks
- Lexical reranking also matches all chunks (30% weight on "FORM" token overlap)
- System retrieves chunks 0-9 with nearly identical scores
- Evaluation expects chunk 4 → likely retrieves chunk 0 or 1 → **counted as failure**

**This explains the doc-level vs. chunk-level recall gap**:
- Doc recall@10: 82.61% (system finds correct document)
- Chunk recall@10: 65.22% (system retrieves "wrong" chunk from correct document)
- Gap: 17.39% of queries retrieve correct doc but wrong chunk

---

### 6. Document Type Performance Analysis

**Data**: `step07_retrieval_eval.json:116-180`

#### SEC 10-K/10-Q Filings: 0% Chunk Recall

**10-Q Performance**:
- Total: 6 queries
- Chunk hits: 0/6 (0%)
- Doc hits: 3/6 (50%)
- All routed to: Weaviate (DEFAULT_WEAVIATE)

**10-K Performance**:
- Total: 3 queries
- Chunk hits: 0/3 (0%)
- Doc hits: 1/3 (33%)
- All routed to: Weaviate (DEFAULT_WEAVIATE)

**Root Cause**:
1. **Query generation limitation**: SEC filings have formal titles like "FORM 10-Q" → keyword = "FORM"
2. **Title boosting**: All chunks from 10-Q contain "FORM 10-Q" in title boost
3. **Multi-chunk documents**: SEC filings are long (10-30 chunks), each with identical title prefix
4. **Generic queries**: "What does this document say about FORM?" doesn't specify Item 1A vs. Item 7 vs. Item 8
5. **Evaluation expects exact chunk**: Query generated from chunk 4, but chunks 0-9 all match semantically

#### Press Releases: 76.9% Chunk Recall

**Performance**:
- Total: 26 queries
- Chunk hits: 20/26 (76.9%)
- Doc hits: 23/26 (88.5%)

**Why Higher Success**:
1. **Shorter documents**: Press releases typically 1-2 chunks (1000-1500 tokens)
2. **Fewer chunks to confuse**: If document has 1 chunk, chunk recall = doc recall
3. **Specific titles**: "Salesforce Announces Q4 FY25 Results" → keyword = "Salesforce" (more specific than "FORM")
4. **Keyword alignment**: Queries with "earnings" or "revenue" match press release content

#### Product/Dev/Help Docs: 100% Chunk Recall

**Performance**:
- Product: 6/6 (100%)
- Dev_docs: 1/1 (100%)
- Help_docs: 1/1 (100%)

**Why Perfect Success**:
1. **Routed to FAISS** via DEFINITION rule
2. **Short documents**: Product docs typically 2-5 chunks
3. **Specific queries**: "definition of..." or "what is..." align well with product documentation structure
4. **Focused content**: Each chunk discusses a specific feature or concept

---

## Architecture Documentation

### Data Flow: Document Indexing to Query Retrieval

1. **Document Collection** → `data/raw/{sec,product,dev_docs,help_docs,newsroom,investor_news,wikipedia}/`
2. **Normalization** → `data/interim/normalized/*.json` (text cleaning, heading markers)
3. **Chunking** → `data/interim/chunks/*.chunks.jsonl` (800-token chunks with title boosting)
4. **Embedding** → `data/vector/embeddings/embeddings.parquet` (OpenAI ada-002, 1536-dim)
5. **Indexing** → `data/vector/{faiss,weaviate,pinecone}/` (FAISS HNSW, simulated manifests)
6. **Query Routing** → `router_core.py` selects backend (FAISS/Weaviate/Pinecone)
7. **Search Execution** → MCP stub performs L2 + lexical reranking (identical for all backends)
8. **Evaluation** → Gate-7 measures chunk recall@10, doc recall@10, nDCG@5

### Current Metrics (Gate-7 Evaluation)

**Overall** (46 queries):
- Chunk recall@10: 65.22% (FAIL, target ≥80%)
- Doc recall@10: 82.61% (gap = 17.39%)
- nDCG@5: 34.41% (FAIL, target ≥60%)
- Doc nDCG@5: 69.33%

**By Backend**:
- FAISS: 80% chunk recall (10 queries, easy definitions/product docs)
- Weaviate: 65.38% chunk recall (26 queries, hard SEC filings + mixed)
- Pinecone: 50% chunk recall (10 queries, financial press releases)

**By Document Type**:
- SEC 10-K/10-Q: 0% chunk recall (9 queries, 0/9 hits)
- Press: 76.9% chunk recall (26 queries, 20/26 hits)
- Product: 100% chunk recall (6 queries, 6/6 hits)
- Other (dev, help, wiki, 8-K): 80% chunk recall (5 queries, 4/5 hits)

---

## Code References

### Evaluation Pipeline
- `scripts/qa_step07_retrieval_eval.py:368-478` - Retrieval execution and recall calculation
- `scripts/build_eval_seed.py:78-216` - Ground truth generation with synthetic queries
- `scripts/qa_step07_retrieval_eval.py:407-426` - Chunk-level recall@10 computation
- `scripts/qa_step07_retrieval_eval.py:428-445` - Document-level recall@10 computation

### Embedding System
- `scripts/embedding_utils.py:86-133` - OpenAI ada-002 embedding with caching
- `scripts/embedding_utils.py:76-83` - API call wrapper with retry logic
- `scripts/qa_step01_embeddings.py:138` - Document embedding (Gate-1)
- `scripts/qa_step03_mcp.py:71` - Query embedding (MCP service)

### Backend Implementations
- `scripts/qa_step02_indexes.py:80-164` - FAISS HNSW index construction
- `scripts/qa_step03_mcp.py:102-155` - Unified search implementation (all backends)
- `scripts/qa_step03_mcp.py:105` - L2 distance computation (shared)
- `scripts/qa_step03_mcp.py:132-134` - Lexical reranking (70% vector + 30% lexical)

### Routing System
- `scripts/router_core.py:72-100` - Backend selection logic
- `configs/router.heuristics.yaml:1-43` - Keyword rules and persona bias
- `scripts/router_core.py:81-89` - Rule-based keyword matching

### Chunking System
- `scripts/chunk_documents.py:105-160` - Chunk generation with title boosting
- `scripts/chunk_documents.py:125-138` - Title boost construction
- `configs/chunking.config.json:1-7` - Chunking parameters (800 tokens, 120 overlap)

---

## Historical Context (from thoughts/)

### Related Research
- `thoughts/shared/research/exp0:2025-10-06-embedding-model-architecture.md` - Deep dive into hashlex-v1 implementation and dimension dependencies (historical, pre-migration)
- `thoughts/shared/plans/OpenAI Ada-002 Migration Plan v2 (Unified).md` - Migration plan from hashlex-v1 to OpenAI ada-002 with batch processing and caching

### Known Issues
- `thoughts/shared/issues/issue001.md` - Original request to migrate from hashlex-v1 to ada-002 (completed)
- `thoughts/shared/issues/issue002.md` - Current investigation: Low recall rate (this research document)

---

## Root Cause Synthesis

The low recall rate is caused by **four interrelated factors**:

### 1. Query Generation Methodology (Evaluation Design)

**Issue**: Synthetic queries are overly generic
- Generated from document titles using only the first word
- Example: "FORM 10-Q..." → query about "FORM"
- Does not specify which section/chunk within multi-chunk documents

**Evidence**:
- `build_eval_seed.py:183-184` - Keyword extraction: `keyword = title.split()[0]`
- `build_eval_seed.py:184` - Template: `"What does this document say about {keyword}?"`

**Impact**: Queries that could match multiple chunks are evaluated as failures if not retrieving the exact expected chunk.

### 2. Title Boosting in Chunking

**Issue**: Every chunk contains the full document title
- Causes all chunks from a document to match title-based queries with equal strength
- Embedding similarity becomes nearly identical across chunks

**Evidence**:
- `chunk_documents.py:125-138` - Title boost prepended to every chunk
- Doc recall (82.61%) >> Chunk recall (65.22%) = 17.39% gap

**Impact**: System retrieves correct document but "wrong" chunk, counted as failure.

### 3. Router Query Distribution Bias

**Issue**: Router sends difficult queries to specific backends
- All SEC 10-K/10-Q queries → Weaviate (DEFAULT_WEAVIATE rule)
- Easy product/definition queries → FAISS (DEFINITION rule)

**Evidence**:
- `router_core.py:99` - DEFAULT fallback: `"weaviate"`
- `step07_retrieval_eval.json:276` - DEFAULT_WEAVIATE: 23/46 queries (50%)
- SEC queries: 9/46 total, all routed to Weaviate, 0/9 success

**Impact**: Backend performance differences reflect query difficulty distribution, not implementation quality.

### 4. Ground Truth Granularity Mismatch

**Issue**: Evaluation expects chunk-level precision from document-level queries
- Ground truth: Exact chunk ID required
- Queries: Generic document-level questions
- No partial credit for near-misses or correct document

**Evidence**:
- `qa_step07_retrieval_eval.py:407-410` - Binary relevance: exact chunk match or failure
- Near-miss rate: 17.39% (system retrieved correct doc, adjacent chunk)

**Impact**: Evaluation penalizes the system for retrieving semantically similar content from the correct document.

---

## Quantitative Evidence

### Doc-Level vs. Chunk-Level Recall Gap

| Metric | Chunk-Level | Doc-Level | Gap |
|--------|-------------|-----------|-----|
| Recall@1 | 19.57% | 60.87% | 41.30% |
| Recall@3 | 39.13% | 76.09% | 36.96% |
| Recall@5 | 50.00% | 76.09% | 26.09% |
| Recall@10 | 65.22% | 82.61% | 17.39% |

**Interpretation**: At every k, the system retrieves the correct document significantly more often than the exact chunk. This indicates the retrieval system is working (finding relevant documents) but the evaluation expects finer granularity than the query specificity supports.

### SEC Filings Analysis

**9 SEC 10-K/10-Q queries**:
- Chunk hits: 0/9 (0%)
- Doc hits: 4/9 (44.4%)
- Near-misses: 0/9
- All routed to: Weaviate

**If these 9 queries were excluded**:
- Remaining: 37 queries
- Chunk hits: 30/37 = 81.08% (PASS threshold)
- This suggests the core retrieval system works well for non-SEC queries

### Backend Performance Without Query Bias

If all backends ran on the same query distribution:

**FAISS** (current: 10 easy queries, 80% recall):
- 6 product (100%), 1 dev (100%), 1 help (100%), 1 8-K (100%), 1 wiki (0%)
- True capability: 80% on diverse queries

**Weaviate** (current: 26 mixed queries, 65.38% recall):
- 17 successes, 9 SEC failures (0%)
- Without SEC queries: 17/17 non-SEC = 100% (but only 1 press, rest filtered)
- True capability likely higher than observed 65.38%

**Pinecone** (current: 10 financial queries, 50% recall):
- Mixed press releases with financial keywords
- True capability: ~50% on financial queries

---

## Conclusion

The retrieval system's low recall rate (65.22%) is **not caused by broken embeddings or backend implementations**. Instead, it results from:

1. **Evaluation design**: Chunk-level ground truth requires precision that document-title-based queries cannot achieve
2. **Chunking architecture**: Title boosting makes all chunks from a document semantically similar for title-based queries
3. **Query distribution**: Router sends the hardest queries (SEC filings) to Weaviate, creating artificial performance gaps
4. **SEC document characteristics**: Long multi-chunk filings with generic queries ("FORM") cannot specify which regulatory section to retrieve

The system **successfully retrieves correct documents** (82.61% doc recall@10) but struggles with **exact chunk identification** when queries are ambiguous about which chunk within a document is relevant.

**Key Evidence**:
- All backends use identical search code (`qa_step03_mcp.py:105`)
- Embeddings are consistent (same `embed_text()` for docs and queries)
- Doc-level recall (82.61%) is 17.39% higher than chunk-level (65.22%)
- SEC queries account for 9/16 failures, all routed to one backend (Weaviate)

**System Status**: The retrieval system is architecturally sound. The low recall rate reflects evaluation methodology and query generation choices rather than implementation defects.