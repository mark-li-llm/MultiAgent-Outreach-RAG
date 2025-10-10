---
date: 2025-10-07T14:31:48-04:00
researcher: Claude
git_commit: e6597f5ac21d7a1d210428bddbc5dd75374fde90
branch: agent-weaviate
repository: agent-weaviate
topic: "SEC Filing Retrieval Pipeline: End-to-End Documentation (Issue005)"
tags: [research, codebase, sec-filings, retrieval-pipeline, gate-7, embeddings, chunking, issue005]
status: complete
last_updated: 2025-10-07
last_updated_by: Claude
---

# Research: SEC Filing Retrieval Pipeline - End-to-End Documentation

**Date**: 2025-10-07T14:31:48-04:00
**Researcher**: Claude
**Git Commit**: e6597f5ac21d7a1d210428bddbc5dd75374fde90
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

**Issue005**: SEC 10-K/10-Q filings achieve 0% chunk-level recall (0/9 successful queries) in Gate-7 evaluation, while maintaining 44.4% document-level recall (4/9).

**Research Goal**: Document the complete retrieval pipeline for SEC filings from collection through evaluation, explaining how documents flow through each stage and where chunk-level vs document-level recall diverge.

## Summary

The SEC filing retrieval pipeline processes 10-K, 10-Q, and 8-K filings through 13 stages from collection to evaluation. The system uses **OpenAI ada-002 embeddings** (1536 dimensions) - not the hashlex-v1 model documented in CLAUDE.md. SEC filings receive special handling at three key points:

1. **Structure Parsing** (`parse_sec_structures.py`): Identifies Item boundaries (Item 1, 1A, 7, 7A, 8) using regex patterns
2. **SEC-Aware Chunking** (`chunk_documents.py:170-188`): Segments documents by Item spans before applying sliding-window chunking
3. **Router Keyword Rules** (`router.heuristics.yaml:22-25`): Routes SEC-related queries to Pinecone backend

The 0% chunk-level recall with 44.4% document-level recall indicates that **the router finds the correct SEC documents but retrieves chunks that don't contain the expected answers**. Gate-7 evaluation (`qa_step07_retrieval_eval.py`) distinguishes these cases as "chunk_miss_doc_hit" failures.

## Detailed Findings

### 1. SEC Filing Collection Pipeline

**Entry Point**: `scripts/fetch_sec_filings.py:27-64`

**Hardcoded Filing Inventory**:
- FY25 10-K: `sec.gov/Archives/edgar/.../crm-20250131.htm`
- FY26 Q1 10-Q: `sec.gov/Archives/edgar/.../crm-20250430.htm`
- 8-K filings: Quarterly results and proxy meetings
- Annual report PDF: FY25 annual report

**Document ID Format**: `crm::<doctype>::<date>::<slug>::<hash8>`
- Example: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2`

**Output Artifacts** (at `data/raw/sec/`):
- `<doc_id>.raw.html` - Full HTTP response body
- `<doc_id>.meta.json` - Fetch metadata (URL, date, SHA256, content-type)

**Deduplication**: Checks for existing `.meta.json` with `status=200` before downloading (`fetch_sec_filings.py:146-162`)

**Key Pattern**: No dynamic SEC API integration - filings are manually curated in `SEC_ITEMS` list

---

### 2. HTML Normalization

**Entry Point**: `scripts/normalize_html.py:194-327`

**Process**:
1. Raw HTML parsed with BeautifulSoup (`normalize_html.py:84-85`)
2. Removes unwanted elements per `configs/normalization.rules.yaml:1-11` (nav, footer, scripts, styles)
3. Converts headings to markers: `<h1>` → `"H1: {text}\n"` (lines 116-119)
4. Collapses whitespace: horizontal `[ \t]+` → `" "`, vertical `\n{3,}` → `"\n\n"` (lines 130-132)
5. Language detection bypassed for `sec.gov` domain (whitelisted as English at lines 249-258)
6. Word count computed via `\b\w+\b` regex, token count via cl100k_base encoding (lines 287-292)

**Critical Limitation**: No table structure preservation - financial tables become linearized text without column alignment or cell boundaries.

**Output Schema** (at `data/interim/normalized/<doc_id>.json`):
```json
{
  "doc_id": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2",
  "doctype": "10-K",
  "text": "H1: crm-20250131\n\ncrm-20250131\n0001108524\nFALSE...",
  "word_count": 68423,
  "token_count": 106812,
  "publish_date": "2025-03-05",
  "source_domain": "sec.gov",
  "url": "https://www.sec.gov/..."
}
```

**No SEC-specific rules**: Uses same normalization as other document types (no special handling for XBRL tags or financial statements)

---

### 3. SEC Item Structure Parsing

**Entry Point**: `scripts/parse_sec_structures.py:29-76`

**Item Detection Patterns** (lines 13-19):
- `Item 1` (Business): `r"(^|\b)item\s*[\xa0\s]*1(?![0-9])\.?\s*(.*)$"`
- `Item 1A` (Risk Factors): `r"(^|\b)item\s*[\xa0\s]*1\s*a\.?\s*(.*)$"`
- `Item 7` (MD&A): `r"(^|\b)item\s*[\xa0\s]*7(?![0-9])\.?\s*(.*)$"`
- `Item 7A` (Market Risk): `r"(^|\b)item\s*[\xa0\s]*7\s*a\.?\s*(.*)$"`
- `Item 8` (Financial Statements): `r"(^|\b)item\s*[\xa0\s]*8(?![0-9])\.?\s*(.*)$"`

**Span Extraction** (lines 57-72):
- Computes character offsets (`start_char`, `end_char`) for each Item section
- Stores as `sec_item_spans` array in normalized JSON
- Calculates `sec_item_coverage_ratio` (proportion of document covered by Items)

**Example Span**:
```json
{
  "label": "Item 1A",
  "title": "Risk Factors",
  "start_char": 12450,
  "end_char": 98372
}
```

**Integration**: Updates normalized JSON files in-place (lines 103-111)

---

### 4. Document Chunking

**Entry Point**: `scripts/chunk_documents.py:105-228`

**Configuration** (`configs/chunking.config.json`):
- `target_tokens`: 800
- `overlap_tokens`: 120
- `boundary_tolerance_chars`: 50
- `short_doc_threshold_tokens`: 350

**SEC-Specific Segmentation** (lines 170-188):
```python
if doctype in ("10-k", "10-q", "8-k", "ars_pdf") and doc.get("sec_item_spans"):
    for sp in doc["sec_item_spans"]:
        segments.append((sp["start_char"], sp["end_char"]))
else:
    segments.append((0, len(text) - 1))
```

**Chunking Process**:
1. Extracts H2/H3 heading positions (`chunk_documents.py:35-45`)
2. For SEC docs, adds Item start positions as boundary candidates (lines 170-172)
3. Plans chunk slices with boundary snapping (lines 67-102)
4. Builds chunks with title boost (lines 125-160)
5. Merges tiny residuals (lines 205-223)

**Title Boost** (lines 126-138):
- Prepends document title + first H1 to every chunk
- Ensures chunks retain context when retrieved in isolation

**Chunk ID Format**: `{doc_id}::chunk{seq_no:04d}`
- Example: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0004`

**Output** (at `data/interim/chunks/<doc_id>.chunks.jsonl`):
```json
{
  "chunk_id": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0004",
  "doc_id": "crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2",
  "seq_no": 4,
  "text": "crm-20250131\n\nItem 7. Management's Discussion...",
  "word_count": 687,
  "token_count": 1453,
  "start_char": 9624,
  "end_char": 13397,
  "local_heads": ["H2: Financial Performance", "H3: Revenue Analysis"],
  "metadata_snapshot": {
    "company": "Salesforce",
    "doctype": "10-K",
    "date": "2025-03-05",
    "url": "https://www.sec.gov/...",
    "title": "crm-20250131"
  }
}
```

**Observed Chunk Sizes**:
- 10-K: 133 chunks (average ~800 tokens each)
- 10-Q: 91 chunks
- 8-K: 2-3 chunks (short documents)

---

### 5. Embedding Generation (OpenAI Ada-002)

**Entry Point**: `scripts/qa_step01_embeddings.py:115`

**Critical Discovery**: The system uses **OpenAI ada-002** (1536 dimensions), NOT hashlex-v1 (768 dimensions) as documented in CLAUDE.md.

**Configuration** (`configs/vector.indexing.yaml:1-5`):
```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20
  notes: OpenAI text-embedding-ada-002 with caching and retry logic
```

**Embedding Process** (`embedding_utils.py:86-133`):
1. Validates dimension matches config (must be 1536 at lines 97-101)
2. Checks SHA256-based cache at `data/cache/embeddings/{hash}.json` (lines 109-111)
3. If cache miss, calls OpenAI API with 3-attempt retry (lines 70-84)
4. Returns 1536-dimensional float32 vector
5. Saves to cache (line 131)

**Batch Processing** (`embedding_utils.py:152-212`):
- Groups texts into batches of 20 (per config)
- Processes uncached texts only
- Maintains original input order

**API Configuration**:
- Model: `text-embedding-ada-002`
- Retry logic: 3 attempts, exponential backoff 4-10 seconds
- Retries on: `APIError`, `APIConnectionError`, `RateLimitError`
- Cost estimation: $0.0001 per 1K tokens

**Output Parquet** (`data/vector/embeddings/embeddings.parquet`):
```
Schema:
  chunk_id: string
  doc_id: string
  seq_no: int32
  token_count: int32
  l2_norm: float32
  vector: list<float32>[1536]
```

**Quality Gates** (Gate-1):
- G1-01: Row count matches baseline chunks
- G1-02: Vector dimension == 1536
- G1-03a: Zero vectors == 0
- G1-03b: NaN vectors == 0
- G1-04: Outlier percentage <= 0.5%

---

### 6. Vector Indexing (FAISS & Simulated Weaviate/Pinecone)

**Entry Point**: `scripts/qa_step02_indexes.py:274`

#### FAISS Index Building (lines 80-164)

**HNSW Parameters** (`configs/vector.indexing.yaml:7-12`):
```yaml
faiss:
  type: HNSW
  metric: L2
  M: 32                 # Connections per node
  efConstruction: 200   # Build-time search depth
  efSearch: 128         # Query-time search depth
```

**Construction** (lines 121-133):
```python
idx = faiss.IndexHNSWFlat(dim, M, metric)
idx.hnsw.efConstruction = efConstruction
idx.hnsw.efSearch = efSearch
xb = np.array(vecs, dtype=np.float32)
idx.add(xb)
```

**Output Files** (at `data/vector/faiss/`):
- `index.faiss` - Binary HNSW index (FAISS native format)
- `idmap.parquet` - Maps FAISS int IDs → (chunk_id, doc_id, seq_no)
- `faiss_manifest.json` - Config and integrity metrics

**Integrity Check** (lines 136-149):
- Samples 100 random vectors
- Queries index for top-1 nearest neighbor
- Computes L2 reconstruction error
- Threshold: `max_err <= 0.001` (Gate-2 check G2-05)

#### Weaviate & Pinecone (Simulated)

**Status**: Both backends are **manifest-only** (no actual network calls)

**Weaviate Schema** (lines 294-308):
- Class name: `"Doc"`
- Properties: `[doc_id, text, doctype, date, section, topic, url, title, company, persona_tags, source_domain]`
- Output: `data/vector/weaviate/schema_applied.json`

**Pinecone Manifest** (lines 282-292):
- Index name: `demo-index`
- Metric: `cosine`
- Output: `data/vector/pinecone/index_manifest.json`

**Sanity Search Validation** (lines 212-271):
- Tests 3 hardcoded queries: `"latest earnings results"`, `"agentforce product announcement"`, `"remaining performance obligation definition"`
- Validates non-empty results (G2-06: >= 3 results per query)
- Validates keyword hits (G2-07: >= 1 keyword match per query)

---

### 7. Query Processing and Routing

**Entry Point**: `scripts/router_core.py:72-100`

#### Backend Selection (First Match Wins)

**Rule-based Routing** (`configs/router.heuristics.yaml:19-40`):
1. **SEC/Financial queries** → Pinecone (keywords: `results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k`)
2. **Developer/API queries** → Weaviate (keywords: `api, apis, endpoint, schema, developer, example`)
3. **Definitional queries** → FAISS (keywords: `definition, what is, overview`)

**Persona Bias** (lines 7-10):
- `vp_sales_ops` → `pinecone`
- `solutions_architect` → `weaviate`

**Default Heuristics** (lines 96-100):
- Short queries (≤4 words) → `faiss`
- Definitional queries → `faiss`
- All others → `weaviate`

**Fallback Order** (line 41): `[faiss, weaviate, pinecone]`

#### Query Embedding

**Critical**: Queries are embedded using the **same `embed_text()` function** from `embedding_utils.py:86` that was used for document embeddings in Gate-1.

**Process** (`qa_step03_mcp.py:104`):
1. Calls `embed_text(query, dim=1536)`
2. Checks cache first (SHA256-based)
3. If cache miss, calls OpenAI ada-002 API
4. Returns 1536-dimensional vector

**Vector Space Consistency**: Both documents and queries use OpenAI ada-002 embeddings at 1536 dimensions, ensuring they exist in the same vector space.

---

### 8. Chunk Retrieval and Ranking

**Entry Point**: `scripts/qa_step03_mcp.py:82` (MCP kb.search handler)

#### Stage 1: Vector Similarity Search (lines 102-119)

**Distance Computation**:
```python
qv = embed_query(q, dim)  # Returns 1536-dim OpenAI ada-002 vector
dists = ((xb - qv)**2).sum(axis=1)  # L2 distance
```

**Candidate Retrieval**:
- Fetches `cand_k = max(top_k, 100)` candidates (line 108)
- Oversampling enables effective reranking without information loss

**Vector Score Normalization** (line 117):
```python
_vec_sim = 1.0 / (1.0 + dist)  # Range: (0, 1]
```

#### Stage 2: Lexical Reranking (lines 120-144)

**Attempted Process** (if `tokenize` available):
1. Tokenize query: `qset = set(_tok(q))`
2. Tokenize snippet: `sset = set(_tok(snippet))`
3. Compute lexical boost: `len(qset & sset) / len(qset)`
4. Combine scores: `0.7 * vec_sim + 0.3 * lex_boost`

**Current Status**: The `tokenize` function is not found in current `embedding_utils.py` (uses OpenAI ada-002, no custom tokenization). System falls back to vector-only scoring on import failure (lines 145-155).

#### Stage 3: Optional Recency/Diversity Reranking (`router_core.py:113-183`)

**Weighted Scoring** (lines 158):
```python
final = (0.5 * similarity) + (0.3 * recency) + (0.2 * diversity)
```

**Recency Score** (lines 148-153):
```python
days = (today - publish_date).days
recency = max(0.0, 1.0 - (days / 730.0))  # Linear decay over 2 years
```

**Diversity Score** (lines 155-156):
- First document from a domain: +0.1 bonus
- Subsequent documents from same domain: 0.0

**Domain Cap** (lines 176-180):
- Enforces maximum 2 documents per domain in top-k
- Prevents single source from monopolizing results

---

### 9. Gate-7 Evaluation Methodology

**Entry Point**: `scripts/qa_step07_retrieval_eval.py`

#### Ground Truth Establishment (lines 51-59)

**Evaluation Seed** (`data/interim/eval/salesforce_eval_seed.jsonl`):
```json
{
  "eval_id": "10q_q1_revenue",
  "persona": "cfo",
  "query_text": "What was Salesforce's total revenue for Q1 FY26?",
  "expected_chunk_id": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0004",
  "expected_doc_id": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866",
  "source_type": "10-Q"
}
```

**Document ID Derivation** (line 429):
```python
exp_doc_id = "::".join(exp_cid.split("::")[:-1])  # Remove ::chunk#### suffix
```

#### Chunk-Level Recall Calculation (lines 406-427)

**Process**:
1. Route query to backend via `decide_backend()`
2. Retrieve top-k results (default k=10)
3. Extract retrieved chunk IDs: `ranks = [r.get("chunk_id") for r in res]`
4. Find position of expected chunk: `rank = ranks.index(exp_cid) + 1`
5. If `rank <= 10`: chunk hit, else chunk miss

**Chunk Recall@10** (line 584):
```python
recall10 = chunk_hits / max(1, total_queries)
```

**Chunk nDCG@5** (lines 419-421, 585):
```python
dcg = (1.0 / math.log2(rank + 1)) if (rank and rank <= 5) else 0.0
ndcg5 = sum(dcg5_vals) / max(1, len(dcg5_vals))
```

#### Document-Level Recall Calculation (lines 428-445)

**Process**:
1. Extract retrieved document IDs: `doc_ranks = [r.get("doc_id") for r in res]`
2. Find position of expected document: `doc_rank = doc_ranks.index(exp_doc_id) + 1`
3. If `doc_rank <= 10`: document hit, else document miss

**Document Recall@10** (line 587):
```python
doc_recall10 = doc_hits / max(1, total_queries)
```

**Document nDCG@5** (lines 441-442, 588):
```python
doc_dcg = (1.0 / math.log2(doc_rank + 1)) if (doc_rank and doc_rank <= 5) else 0.0
doc_ndcg5 = sum(doc_dcg5_vals) / max(1, len(doc_dcg5_vals))
```

#### Near-Miss Detection (lines 447-479)

**Definition**: Expected chunk not in top-k BUT parent document IS in top-k AND a chunk from same document exists within `near_seq_tol` sequence positions (default: 1).

**Process**:
```python
if (not chunk_rank) and exp_doc_id:
    for r in results:
        if r.get("doc_id") != exp_doc_id:
            continue
        cand_meta = chunk_meta.get(r.get("chunk_id", "")) or {}
        dseq = abs(int(cand_meta.get("seq_no") or 0) - exp_seq)
        if dseq <= near_seq_tol:  # Default: 1
            near_hit = True
            soft_hits += 1
            break
```

**Near-Miss Rate** (line 590):
```python
near_miss_rate = doc_recall10 - recall10
```

This measures the gap between document-level and chunk-level recall.

#### Failure Classification (lines 551-576)

**Three Categories**:
1. **`chunk_miss_doc_miss`**: Both chunk and document not in top-k (complete miss)
2. **`chunk_miss_doc_hit_near`**: Chunk missed but doc found, with adjacent chunk within tolerance
3. **`chunk_miss_doc_hit_far`**: Chunk missed but doc found, no adjacent chunks

**Failure Log** (`reports/eval/retrieval_failures.jsonl`):
```json
{
  "eval_id": "10q_share_repurchase",
  "query_text": "What is the status of Salesforce's share repurchase program?",
  "classification": "chunk_miss_doc_hit_far",
  "expected_chunk_id": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0032",
  "expected_doc_id": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866",
  "nearest_same_doc": {
    "rank": 3,
    "chunk_id": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0089",
    "seq_no": 89,
    "delta_seq": 57,
    "delta_start": 185763
  },
  "topk": [ /* retrieved chunks with scores */ ]
}
```

#### Quality Gates (Gate-7)

**Checks** (lines 731-789):
- **G7-01**: Chunk recall@10 >= 80% (currently 65.22% - FAIL)
- **G7-02**: Chunk nDCG@5 >= 60% (currently 34.41% - FAIL)
- **G7-03**: Coverage unique domains mean >= threshold
- **G7-04**: Freshness mean age <= 540 days
- **G7-05**: Latency budgets (p50, p95) within limits

**Status Logic**:
- **RED**: G7-01 or G7-02 fails, or multiple failures
- **AMBER**: Only G7-03 or G7-04 fails
- **GREEN**: All checks pass

---

### 10. Issue005 Context: 0% Chunk Recall for SEC Filings

**Observation from Issue004** (`thoughts/shared/issues/issue004.md:26-28`):
```
Document Type Performance (chunk-level hits):
  - 10-Q filings: 0% (0/6) ⚠️
  - 10-K filings: 0% (0/3) ⚠️
  - 8-K: 100% (1/1)
```

**Document-Level Performance**:
- 10-Q: 50% doc-level recall (3/6 queries find the document)
- 10-K: 33% doc-level recall (1/3 queries find the document)

**Interpretation**:
- The router successfully routes SEC queries to the correct backend (typically Pinecone via keyword rules)
- Vector search successfully retrieves chunks from the correct SEC documents
- **However**, the retrieved chunks do not contain the expected answers
- This is a **ranking/relevance issue**, not a routing or indexing failure

**Near-Miss Analysis**:
- 17.39% near-miss rate overall (from Issue004:11)
- Relevant chunks exist in the same document but are ranked too low
- Expected chunk at position > 10, while wrong chunk from same document appears in top-10

---

## Code References

### Collection & Normalization
- `scripts/fetch_sec_filings.py:27-64` - Hardcoded SEC filing URLs
- `scripts/fetch_sec_filings.py:67-134` - Download and deduplication logic
- `scripts/normalize_html.py:83-137` - HTML normalization function
- `scripts/normalize_html.py:194-327` - Document processing orchestration
- `scripts/parse_sec_structures.py:13-19` - SEC Item regex patterns
- `scripts/parse_sec_structures.py:29-76` - Item span extraction

### Chunking & Embedding
- `scripts/chunk_documents.py:105-228` - Core chunking logic
- `scripts/chunk_documents.py:170-188` - SEC-specific segmentation by Item spans
- `scripts/embedding_utils.py:86-133` - OpenAI ada-002 embedding function
- `scripts/embedding_utils.py:152-212` - Batch embedding with caching
- `scripts/qa_step01_embeddings.py:115-200` - Gate-1 embedding generation

### Indexing
- `scripts/qa_step02_indexes.py:80-164` - FAISS HNSW index building
- `scripts/qa_step02_indexes.py:167-181` - ID mapping creation
- `scripts/qa_step02_indexes.py:212-271` - Sanity search validation

### Routing & Retrieval
- `scripts/router_core.py:72-100` - Backend selection logic
- `scripts/router_core.py:113-183` - Recency/diversity reranking
- `scripts/qa_step03_mcp.py:82-156` - MCP kb.search handler with L2 distance search
- `scripts/qa_step03_mcp.py:120-144` - Lexical reranking (attempted)

### Evaluation
- `scripts/qa_step07_retrieval_eval.py:51-59` - Ground truth loading
- `scripts/qa_step07_retrieval_eval.py:406-427` - Chunk-level recall calculation
- `scripts/qa_step07_retrieval_eval.py:428-445` - Document-level recall calculation
- `scripts/qa_step07_retrieval_eval.py:447-479` - Near-miss detection
- `scripts/qa_step07_retrieval_eval.py:551-576` - Failure classification and logging

### Configuration
- `configs/vector.indexing.yaml:1-12` - Embedding model (openai-ada-002, dim=1536), FAISS HNSW params
- `configs/chunking.config.json:1-7` - Chunking parameters (target=800, overlap=120)
- `configs/router.heuristics.yaml:1-42` - Routing rules, weights, fallback order
- `configs/normalization.rules.yaml:1-27` - HTML normalization rules

---

## Architecture Documentation

### Data Pipeline Flow

```
1. Collection (fetch_sec_filings.py)
   └─> data/raw/sec/{doc_id}.raw.html + .meta.json

2. Normalization (normalize_html.py)
   ├─> BeautifulSoup HTML parsing
   ├─> Heading markers (H1:/H2:/H3:)
   ├─> Whitespace collapse
   └─> data/interim/normalized/{doc_id}.json

3. SEC Structure Parsing (parse_sec_structures.py)
   ├─> Regex Item detection
   ├─> Character offset spans
   └─> Updates normalized JSON with sec_item_spans

4. Chunking (chunk_documents.py)
   ├─> SEC segmentation by Item spans
   ├─> Sliding window with boundary snapping
   ├─> Title boost prepending
   └─> data/interim/chunks/{doc_id}.chunks.jsonl

5. Embedding (qa_step01_embeddings.py + embedding_utils.py)
   ├─> OpenAI ada-002 API calls (1536-dim)
   ├─> SHA256-based caching
   ├─> Batch processing (20 texts/call)
   └─> data/vector/embeddings/embeddings.parquet

6. Indexing (qa_step02_indexes.py)
   ├─> FAISS HNSW index (M=32, efConstruction=200, efSearch=128)
   ├─> ID mapping (FAISS int → chunk metadata)
   ├─> Weaviate/Pinecone manifests (simulated)
   └─> data/vector/faiss/index.faiss + idmap.parquet

7. Query Processing (router_core.py + qa_step03_mcp.py)
   ├─> Keyword-based backend selection
   ├─> Query embedding (same embed_text() as documents)
   ├─> L2 distance vector search
   ├─> Lexical reranking (attempted, fallback to vector-only)
   └─> Returns top-k {chunk_id, doc_id, score, snippet}

8. Evaluation (qa_step07_retrieval_eval.py)
   ├─> Ground truth: expected_chunk_id, expected_doc_id
   ├─> Chunk-level recall: exact chunk match in top-k
   ├─> Document-level recall: parent doc match in top-k
   ├─> Near-miss detection: adjacent chunks within tolerance
   └─> Classification: chunk_miss_doc_hit_near/far vs chunk_miss_doc_miss
```

### Critical Consistency Points

**Vector Space Alignment**:
1. Document embedding: `qa_step01_embeddings.py:173` calls `embed_batch()` → `embedding_utils.py:152` → internally uses `embed_text()` at line 181
2. Query embedding: `qa_step03_mcp.py:104` calls `embed_text()` directly
3. Both use: `embedding_utils.py:86` → OpenAI ada-002 API at 1536 dimensions

**Configuration Cascade**:
- `configs/vector.indexing.yaml:2-3` defines `model: openai-ada-002`, `dim: 1536`
- Gate-1 reads dimension at `qa_step01_embeddings.py:28-39`
- MCP stub loads same config at `qa_step03_mcp.py:20-21`
- Mismatched dimensions would cause `ValueError` at `embedding_utils.py:97-101`

**SEC-Specific Handling**:
1. **Structure parsing**: `parse_sec_structures.py` adds `sec_item_spans` to normalized JSON
2. **Segmentation**: `chunk_documents.py:170` checks `doctype in ("10-k", "10-q", "8-k")` and loads `sec_item_spans`
3. **Routing**: `router.heuristics.yaml:22-25` routes SEC keywords to Pinecone

---

## Historical Context (from thoughts/)

### Related Research Documents

1. **`thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md`**
   - Investigated document-level vs chunk-level recall gap
   - Identified SEC filing 0% chunk recall
   - Documented OpenAI ada-002 migration (not hashlex-v1)

2. **`thoughts/shared/research/2025-10-07-issue003-retrieval-recall-investigation.md`**
   - Covered OpenAI ada-002 vs hashlex-v1 discrepancy
   - Analyzed SEC Item structure parsing failures
   - Documented XBRL metadata pollution issues

3. **`thoughts/shared/research/2025-10-07-issue004-retrieval-recall-investigation.md`**
   - Gate-7 RED status investigation
   - Per-doctype performance breakdown (10-Q: 0%, 10-K: 0%, 8-K: 100%)
   - Backend performance differences (FAISS 80%, Weaviate 65%, Pinecone 50%)

4. **`thoughts/shared/research/issue001:2025-10-06-embedding-model-architecture.md`**
   - Original hashlex-v1 embedding model documentation
   - Dimension parameter architecture
   - **Note**: System has since migrated to OpenAI ada-002

### Migration Context

**`thoughts/shared/plans/OpenAI Ada-002 Migration Plan v2 (Unified).md`**:
- Documents planned migration from hashlex-v1 to OpenAI ada-002
- Migration has been completed (confirmed by code inspection)
- CLAUDE.md documentation remains outdated (still references hashlex-v1)

---

## Open Questions

1. **Why 0% chunk-level recall for 10-K/10-Q but 44.4% document-level recall?**
   - Document retrieval works (router finds correct SEC filings)
   - Chunk ranking fails (wrong chunks from correct document surface in top-10)
   - Possible causes: chunking strategy issues, embedding quality for financial text, lack of effective reranking

2. **Is lexical reranking actually functional?**
   - Code attempts to import `tokenize` from `embedding_utils` (line 122)
   - Current `embedding_utils.py` uses OpenAI ada-002, no custom tokenization found
   - System falls back to vector-only scoring on import failure (lines 145-155)
   - This may explain poor ranking quality (no lexical boost applied)

3. **How does table linearization affect SEC filing embeddings?**
   - Financial statements lose column alignment during normalization
   - Numeric relationships become ambiguous
   - Ada-002 may struggle to understand linearized financial tables

4. **Are SEC Item spans effectively used during retrieval?**
   - Item spans segment documents during chunking (lines 170-188)
   - No evidence of Item-aware filtering or boosting during retrieval
   - Expected chunk might be in "Item 7" but retriever returns chunk from "Item 1A"

5. **What is the actual near-miss rate for SEC filings specifically?**
   - Overall near-miss rate: 17.39% (from Issue004:11)
   - SEC-specific breakdown not provided in current reports
   - Would help distinguish "wrong section entirely" vs "adjacent chunk"

---

## Related Research

- `thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md` - Root cause analysis
- `thoughts/shared/research/2025-10-07-issue003-retrieval-recall-investigation.md` - SEC structure parsing
- `thoughts/shared/research/2025-10-07-issue004-retrieval-recall-investigation.md` - Backend performance
- `thoughts/shared/research/issue001:2025-10-06-embedding-model-architecture.md` - Hashlex-v1 (outdated)
- `thoughts/shared/issues/issue004.md` - Gate-7 RED status issue
- `thoughts/shared/issues/issue005.md` - This investigation's source issue

---

## Evidence Files

**Configuration**:
- `configs/vector.indexing.yaml` - Embedding model and FAISS parameters
- `configs/chunking.config.json` - Chunking parameters
- `configs/router.heuristics.yaml` - Routing rules and weights
- `configs/normalization.rules.yaml` - HTML normalization rules

**Data Artifacts**:
- `data/raw/sec/` - Raw SEC filing HTML + metadata
- `data/interim/normalized/` - Normalized documents with SEC Item spans
- `data/interim/chunks/` - Chunked documents (JSONL per document)
- `data/vector/embeddings/embeddings.parquet` - OpenAI ada-002 vectors (1536-dim)
- `data/vector/faiss/` - FAISS index + idmap
- `data/interim/eval/salesforce_eval_seed.jsonl` - Ground truth queries

**Reports**:
- `reports/qa/step07_retrieval_eval.json` - Gate-7 evaluation results
- `reports/eval/retrieval_failures.jsonl` - Failed query diagnostics
- `reports/router/step07_retrieval_trace.jsonl` - Per-query routing traces

**Cache**:
- `data/cache/embeddings/` - SHA256-keyed OpenAI API response cache
