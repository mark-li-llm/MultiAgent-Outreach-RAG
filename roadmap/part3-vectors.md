---
date: 2025-10-20T16:30:40-04:00
researcher: Claude
git_commit: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
branch: agent-weaviate
repository: agent-weaviate
topic: "Vector & Embedding System: Text to Vectors and Index Building"
tags: [research, codebase, vectors, embeddings, faiss, weaviate, pinecone, openai, gate-1, gate-2]
status: complete
last_updated: 2025-10-20
last_updated_by: Claude
---

# Research: Vector & Embedding System

**Date**: 2025-10-20T16:30:40-04:00
**Researcher**: Claude
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

How is text converted to vectors and how are vector indexes built in the multi-agent RAG system?

Specifically:
1. How are embeddings generated? (OpenAI API, model, dimension)
2. What caching strategy is used? (SHA-256 keys, cache location)
3. What vector indexes exist? (FAISS, Weaviate, Pinecone)
4. How are indexes built? (scripts, parameters, formats)
5. What are the index schemas? (metadata, namespaces)
6. What's the performance? (latency, cost, cache hit rate)

## Summary

The system implements a two-stage pipeline for converting text chunks into searchable vector indexes:

**Stage 1 (Gate-1)**: Embedding generation via OpenAI ada-002 API with local file-based caching
- All 536 document chunks are embedded into 1536-dimensional vectors
- SHA-256 hashing (truncated to 16 chars) provides cache keys for filenames
- MD5 hashing validates cache integrity on retrieval
- Cost estimation and user confirmation protect against unexpected API charges
- Embeddings stored in Parquet format with metadata (chunk_id, doc_id, seq_no, token_count, l2_norm)

**Stage 2 (Gate-2)**: Multi-index construction across three vector stores
- **FAISS**: HNSW index (M=32, efConstruction=200, efSearch=128) for general search
- **Weaviate**: Schema with 11 required properties (simulated, no network)
- **Pinecone**: Cosine metric index (simulated, no network)
- Round-trip integrity testing validates FAISS index accuracy
- Sanity search with 3 queries tests semantic retrieval across all backends

The embedding system prioritizes **cost efficiency** (caching), **reliability** (retry logic), and **traceability** (dual JSON+Markdown reports at each gate).

---

## Detailed Findings

### 1. Overview

#### Purpose
The embedding system transforms text chunks from document processing (Gate-0 through Gate-5) into dense vector representations suitable for semantic search. These vectors enable the multi-agent system to retrieve relevant content based on query meaning rather than keyword matching.

#### Vector Dimensions
- **Model**: OpenAI `text-embedding-ada-002`
- **Dimension**: 1536 (fixed, validated at runtime)
- **Format**: List of Python floats, stored as `float32` in Parquet and numpy arrays

#### Index Types
1. **FAISS** (Facebook AI Similarity Search)
   - Location: `data/vector/faiss/`
   - Purpose: General-purpose semantic search with HNSW approximation
   - Status: Active (with OpenMP conflict safeguards)

2. **Weaviate**
   - Location: `data/vector/weaviate/`
   - Purpose: Developer documentation search (planned)
   - Status: Simulated (schema-only, no network operations)

3. **Pinecone**
   - Location: `data/vector/pinecone/`
   - Purpose: Press/financial document search (planned)
   - Status: Simulated (manifest-only, no network operations)

---

### 2. Architecture & Design

#### Embedding Generation Flow
```
Text Chunks (*.chunks.jsonl)
    ↓
Cost Estimation (median text length × uncached count)
    ↓
User Confirmation (AG1_AUTO_CONFIRM bypass available)
    ↓
Batch Processing (20 texts per API call)
    ├─ Cache Check (SHA-256 key lookup)
    ├─ OpenAI API Call (if cache miss)
    │   └─ Retry Logic (3 attempts, exponential backoff)
    ├─ Cache Storage (JSON with MD5 validation)
    └─ Return 1536-dim vector
    ↓
Parquet File (embeddings.parquet with 6 columns)
    ↓
Gate-1 Validation (4 checks: row count, dimension, zero/NaN vectors, norm outliers)
```

**Key Files**:
- **scripts/embedding_utils.py:86-133** - `embed_text()` function
- **scripts/embedding_utils.py:152-211** - `embed_batch()` function
- **scripts/qa_step01_embeddings.py:115-306** - Gate-1 orchestration

#### Caching Architecture
```
Input Text
    ↓
SHA-256 Hash → Truncate to 16 chars → Cache Key
    ↓
Cache Lookup: data/cache/embeddings/<key>.json
    ↓
    ├─ Cache Hit: Validate MD5 hash → Return embedding
    └─ Cache Miss: API Call → Store {text_hash: MD5, embedding: vector}
```

**Dual-Hash Strategy**:
- **SHA-256** (16 chars): Fast filename lookup, ~4 billion unique keys
- **MD5** (32 chars): Content verification, detects hash collisions

**Files**:
- **scripts/embedding_utils.py:34-52** - Cache key generation and lookup
- **scripts/embedding_utils.py:55-67** - Cache storage

#### Index Building Process
```
Embeddings Parquet (embeddings.parquet)
    ↓
Load into Memory (List[List[float]] + metadata rows)
    ↓
    ├─ FAISS: build_faiss() → HNSW index → Round-trip test
    ├─ Weaviate: Schema creation → Manifest (simulated)
    └─ Pinecone: Manifest creation (simulated)
    ↓
Metadata Validation (compute_metadata_missing: 4 required fields)
    ↓
Sanity Search (3 queries × 3 backends → keyword matching)
    ↓
Gate-2 Validation (7 checks: upsert rates, metadata, round-trip error, sanity search)
```

**Key Files**:
- **scripts/qa_step02_indexes.py:80-164** - `build_faiss()`
- **scripts/qa_step02_indexes.py:212-271** - `run_sanity_search()`
- **scripts/qa_step02_indexes.py:274-400** - Gate-2 orchestration

---

### 3. File Inventory

#### Core Scripts
| File | Lines | Purpose |
|------|-------|---------|
| `scripts/embedding_utils.py` | 250 | OpenAI ada-002 embedding utilities with caching and retry |
| `scripts/qa_step01_embeddings.py` | 310 | Gate-1: Generate and validate embeddings |
| `scripts/qa_step02_indexes.py` | 404 | Gate-2: Build indexes and validate integrity |

#### Configuration
| File | Purpose |
|------|---------|
| `configs/vector.indexing.yaml` | Embedding model (ada-002), dimension (1536), batch size (20), FAISS params (HNSW, M=32), Pinecone/Weaviate config |

#### Data Directories
```
data/
├── cache/
│   └── embeddings/          # SHA-256 keyed JSON files (*.json)
├── vector/
│   ├── embeddings/          # Parquet file + stats JSON
│   │   ├── embeddings.parquet
│   │   └── embedding_stats.json
│   ├── faiss/               # FAISS index + ID map + manifest
│   │   ├── index.faiss
│   │   ├── idmap.parquet
│   │   └── faiss_manifest.json
│   ├── pinecone/            # Simulated manifest
│   │   └── index_manifest.json
│   └── weaviate/            # Simulated schema + manifest
│       ├── schema_applied.json
│       └── index_manifest.json
└── final/
    └── reports/
        └── index_health.json  # Combined health metrics
```

#### Reports
```
reports/qa/
├── step01_embeddings.json     # Gate-1 machine-readable
├── step01_embeddings.md       # Gate-1 human-readable
├── step02_indexes.json        # Gate-2 machine-readable
└── step02_indexes.md          # Gate-2 human-readable
```

---

### 4. Core Components Deep Dive

#### Embedding Generation (`scripts/embedding_utils.py`)

##### `embed_text()` Function (Lines 86-133)
**Signature**: `embed_text(text: str, dim: int) -> List[float]`

**Flow**:
1. **Dimension Validation** (97-101): Raises `ValueError` if `dim != 1536`
2. **Empty Text Handling** (104-106): Returns `[0.001] * 1536` for empty/whitespace strings
3. **Cache Lookup** (109-111): Calls `_load_from_cache(text)`, returns immediately if hit
4. **API Call with Retry** (115): Calls `_call_openai_api(text)` (retry decorator: 3 attempts, 4s-10s backoff)
5. **Error Handling** (116-122): Wraps exceptions in `RuntimeError` with context (text length, API key check)
6. **Dimension Validation** (125-128): Verifies returned vector is 1536-dim
7. **Cache Storage** (131): Saves via `_save_to_cache(text, embedding)`
8. **Return** (133): Returns 1536-dimensional list

**API Integration** (Lines 70-83):
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError)),
    reraise=True
)
def _call_openai_api(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding
```

**Retry Strategy**:
- Maximum 3 attempts total
- Exponential backoff: 4s, 8s (capped at 10s)
- Retries only on: `APIError`, `APIConnectionError`, `RateLimitError`
- Re-raises exception after exhaustion

##### `embed_batch()` Function (Lines 152-211)
**Signature**: `embed_batch(texts: List[str], dim: int, batch_size: int = 100) -> List[List[float]]`

**Flow**:
1. **Dimension Validation** (164-167): Raises `ValueError` if `dim != 1536`
2. **Empty Input Handling** (169-170): Returns `[]` for empty list
3. **Per-Text Cache Check** (176-186):
   - Empty texts → `[0.001] * 1536`
   - Cached texts → append cached vector
   - Uncached texts → add `(index, text)` to `texts_to_embed`, append `None` placeholder
4. **Batch Processing** (192-209):
   - Prints cache statistics if uncached texts exist
   - Processes in chunks of `batch_size` (20 from config)
   - Calls `_call_openai_batch(batch_texts)` with retry
   - Fills result list at original indices
   - Caches each embedding individually via `_save_to_cache()`
5. **Return** (211): Complete list preserving input order

**Batch API Call** (Lines 136-149):
```python
@retry(...)  # Same retry config as single-text
def _call_openai_batch(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=texts,
        encoding_format="float"
    )
    return [item.embedding for item in response.data]
```

**Key Difference from `embed_text()`**:
- Single API call for up to 20 texts (vs. 1 text per call)
- Maintains original order via index tracking
- Caches per-item (not per-batch)
- More efficient for large corpora

##### Cost Estimation (`estimate_embedding_cost()`, Lines 214-240)
**Signature**: `estimate_embedding_cost(num_texts: int, avg_text_length: int) -> dict`

**Calculation**:
1. Count cached embeddings: `len(list(CACHE_DIR.glob("*.json")))`
2. Estimate tokens: `avg_text_length / 4.0` (rough heuristic: 1 token ≈ 4 chars)
3. Calculate uncached count: `max(0, num_texts - cached_count)`
4. Total cost: `(uncached_tokens / 1000.0) * 0.0001` USD

**Pricing**: $0.0001 per 1K tokens (~750 words, ~4000 chars)

**Return Dict**:
```python
{
    "num_texts": 536,
    "cached_texts": 200,
    "uncached_texts": 336,
    "avg_text_length_chars": 500,
    "estimated_tokens_per_text": 125,
    "estimated_total_tokens": 42000,
    "cost_per_1k_tokens_usd": 0.0001,
    "estimated_total_cost_usd": 0.0042,
    "note": "Cost only for uncached texts. Cached texts are free."
}
```

#### Caching System

##### Cache Key Generation (`_get_cache_key`, Line 34-36)
```python
def _get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```
- Input: Raw text string
- Process: SHA-256 hash → UTF-8 encode → hex digest → truncate to 16 chars
- Output: 16-character string (e.g., `"e0a6f5c7d2b1a9f1"`)

##### Cache File Format
**Path**: `data/cache/embeddings/<cache_key>.json`

**Structure**:
```json
{
  "text_hash": "32-char-md5-hex",
  "embedding": [0.123, -0.456, ..., 0.789]
}
```

##### Cache Lookup (`_load_from_cache`, Lines 39-52)
**Flow**:
1. Generate cache key: `_get_cache_key(text)` → 16-char SHA-256 prefix
2. Construct path: `data/cache/embeddings/<key>.json`
3. Check existence: Return `None` if file doesn't exist
4. Load JSON: Parse file contents
5. Validate hash: Compare `data["text_hash"]` with `hashlib.md5(text.encode()).hexdigest()`
6. Return embedding if hash matches, else `None`

**Error Handling**: All exceptions silently caught, treated as cache miss

##### Cache Storage (`_save_to_cache`, Lines 55-67)
**Flow**:
1. Generate cache key: `_get_cache_key(text)`
2. Construct path: `data/cache/embeddings/<key>.json`
3. Write JSON: `{"text_hash": MD5(text), "embedding": vector}`

**Error Handling**: Write failures silently ignored (cache is optional)

##### Cache Clearing (`clear_cache`, Lines 243-249)
```python
def clear_cache():
    import shutil
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cleared embedding cache at {CACHE_DIR}")
```
- Removes entire `data/cache/embeddings/` directory
- Recreates empty directory
- Not called automatically (manual invocation only)

#### FAISS Index

##### Build Function (`build_faiss()`, Lines 80-164 in `qa_step02_indexes.py`)
**Signature**: `build_faiss(vecs: List[List[float]], cfg: Dict[str, Any]) -> Tuple[int, float]`

**Environment Safeguards**:
1. **AG2_DISABLE_FAISS=1** (Lines 86-98):
   - Skips FAISS import entirely
   - Writes manifest with `disabled: True, reason: "AG2_DISABLE_FAISS=1"`
   - Returns `(len(vecs), 0.0)` without building index
   - Prevents OpenMP runtime clashes

2. **Import Failure** (Lines 100-113):
   - Catches exceptions during `import faiss`
   - Writes manifest with `disabled: True, reason: "faiss_import_failed"`
   - Returns gracefully without auto-installing (avoids environment conflicts)

**HNSW Parameters** (Lines 121-127):
```python
M = int(faiss_cfg.get("M", 32))                    # Bi-directional links per node
idx = faiss.IndexHNSWFlat(dim, M, metric)
efC = int(faiss_cfg.get("efConstruction", 200))    # Dynamic candidate list (construction)
idx.hnsw.efConstruction = efC
efS = int(faiss_cfg.get("efSearch", 128))          # Dynamic candidate list (search)
idx.hnsw.efSearch = efS
```

**From Config** (`configs/vector.indexing.yaml:7-12`):
```yaml
faiss:
  type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128
```

**Index Types** (Lines 121-130):
- **HNSW**: `faiss.IndexHNSWFlat(dim, M, metric)` if `type == "HNSW"`
- **Flat**: `faiss.IndexFlatL2(dim)` otherwise (exact search fallback)

**Metrics** (Line 119):
- **L2**: Euclidean distance (`faiss.METRIC_L2`)
- **INNER_PRODUCT**: Dot product (any other value)

**Index Population** (Lines 132-134):
```python
xb = np.array(vecs, dtype="float32")    # Convert to numpy float32
idx.add(xb)                             # Add all vectors (sequential IDs 0..n-1)
faiss.write_index(idx, FAISS_INDEX_PATH)  # Serialize to disk
```

##### Round-Trip Integrity Test (Lines 136-149)
**Purpose**: Validate index accuracy via reconstruction error

**Process**:
1. **Sample Selection** (137-139): Random seed 42, up to 100 samples
2. **For Each Sample** (140-149):
   - Query index: `idx.search(query_vector, k=1)`
   - Extract retrieved vector from index
   - Compute L2 norm: `np.linalg.norm(query - retrieved)`
   - Track maximum error across all samples

**Acceptance Criteria** (Gate-2 Check G2-05):
- PASS: `max_error <= 0.001`
- WARN: `0.001 < max_error <= 0.01`
- FAIL: `max_error > 0.01`

##### FAISS Manifest (Lines 151-162)
**File**: `data/vector/faiss/faiss_manifest.json`

**Structure**:
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
    "type": "HNSW",
    "metric": "L2",
    "M": 32,
    "efConstruction": 200,
    "efSearch": 128
  }
}
```

##### ID Mapping (`write_idmap()`, Lines 167-181)
**File**: `data/vector/faiss/idmap.parquet`

**Schema**:
- `id` (int32): FAISS sequential ID (0 to N-1)
- `chunk_id` (string): Original chunk identifier
- `doc_id` (string): Source document identifier
- `seq_no` (int32): Chunk sequence number within document

**Purpose**: Maps FAISS integer IDs to business metadata for result interpretation

#### Weaviate/Pinecone (Simulated)

##### Weaviate Schema (Lines 295-305 in `qa_step02_indexes.py`)
**File**: `data/vector/weaviate/schema_applied.json`

**Required Properties** (Lines 296-297):
```python
required_props = [
    "doc_id", "text", "doctype", "date", "section", "topic",
    "url", "title", "company", "persona_tags", "source_domain"
]
```

**Schema Structure**:
```json
{
  "class": "Doc",
  "properties": ["doc_id", "text", "doctype", "date", "section", "topic", "url", "title", "company", "persona_tags", "source_domain"],
  "notes": "applied minimal schema (simulated)"
}
```

**Class Name**: From config `configs/vector.indexing.yaml:21` → `class_name: Doc`

##### Weaviate Manifest (Lines 306-308)
**File**: `data/vector/weaviate/index_manifest.json`

```json
{
  "inserted": 536,
  "failed": 0,
  "failed_ids": [],
  "config": {
    "class_name": "Doc",
    "notes": "schema-only manifest (simulated)"
  }
}
```

##### Pinecone Manifest (Lines 283-292)
**File**: `data/vector/pinecone/index_manifest.json`

```json
{
  "config": {
    "index_name": "demo-index",
    "namespace": "default",
    "metric": "cosine",
    "notes": "simulated manifest only (no network)"
  },
  "upserted": 536,
  "failed": 0,
  "failed_ids": []
}
```

**Config Source**: `configs/vector.indexing.yaml:14-18`

##### Metadata Validation (`compute_metadata_missing()`, Lines 184-209)
**Purpose**: Calculate percentage of chunks with missing required fields

**Required Fields** (4 total):
1. **doctype**: `d.get("doctype")` non-empty after strip
2. **publish_date**: `d.get("publish_date")` non-empty after strip
3. **url**: `d.get("final_url") or d.get("url")` non-empty after strip
4. **title**: `d.get("title")` non-empty after strip

**Process**:
1. Load all normalized metadata from `data/interim/normalized/*.json`
2. Build `doc_id → metadata` mapping
3. For each embedding row, validate corresponding document metadata
4. Return `missing_count / max(1, total_rows)`

**Gate-2 Check G2-04**:
- PASS: `pct_missing <= 0.02` (2%)
- WARN: `0.02 < pct_missing <= 0.03` (3%)
- FAIL: `pct_missing > 0.03`

---

### 5. Configuration & Settings

#### `configs/vector.indexing.yaml` Schema

```yaml
embedding:
  model: openai-ada-002          # OpenAI model identifier
  dim: 1536                      # Vector dimension (validated at runtime)
  batch_size: 20                 # Texts per API call (reduced from 100 due to 8192 token limit)
  notes: OpenAI text-embedding-ada-002 with caching and retry logic

faiss:
  type: HNSW                     # Index type: HNSW or Flat
  metric: L2                     # Distance metric: L2 or INNER_PRODUCT
  M: 32                          # Bi-directional links per node
  efConstruction: 200            # Dynamic candidate list during build
  efSearch: 128                  # Dynamic candidate list during search

pinecone:
  index_name: demo-index         # Pinecone index identifier
  namespace: default             # Namespace for vector isolation
  metric: cosine                 # Distance metric
  notes: simulated manifest only (no network)

weaviate:
  class_name: Doc                # Weaviate class name
  notes: schema-only manifest (simulated)
```

#### Embedding Configuration
- **Model**: `text-embedding-ada-002` (OpenAI's ada-002 model)
- **Dimension**: 1536 (fixed, cannot be changed without model change)
- **Batch Size**: 20 texts per API call (tuned to avoid 8192 token limit)
  - Note in config: "batch_size reduced to avoid 8192 token limit"
  - Default in code: 100 (line 124 in `qa_step01_embeddings.py`)

#### FAISS Parameters

**HNSW Algorithm**:
- **M** (32): Number of bi-directional links per graph node
  - Higher M → better recall, slower build, more memory
  - Typical range: 16-64

- **efConstruction** (200): Size of dynamic candidate list during index construction
  - Higher efConstruction → better quality, slower build
  - Typical range: 100-500

- **efSearch** (128): Size of dynamic candidate list during search
  - Higher efSearch → better recall, slower search
  - Typical range: 50-200
  - Can be adjusted at query time without rebuilding index

**Metric Selection**:
- **L2**: Euclidean distance (default)
  - Formula: sqrt(sum((q[i] - v[i])^2))
  - Measures absolute distance in vector space

- **INNER_PRODUCT**: Dot product similarity
  - Formula: sum(q[i] * v[i])
  - Assumes normalized vectors for cosine similarity

#### Pinecone/Weaviate Settings
- **Status**: Simulated (no actual network operations)
- **Pinecone metric**: Cosine (config only, not used in current implementation)
- **Weaviate class**: "Doc" (schema definition created, no data upserted)

---

### 6. Data Structures & Schemas

#### Embedding Format
**Type**: List of 1536 floats

**Example** (truncated):
```python
[0.0023, -0.0145, 0.0089, ..., 0.0067]  # 1536 elements
```

**Validation**:
- Dimension checked in `embed_text()` (line 97) and `embed_batch()` (line 164)
- Empty text handling: Returns `[0.001] * 1536` to avoid zero vectors

#### Cached Embedding Structure
**File**: `data/cache/embeddings/<sha256_prefix>.json`

**JSON Schema**:
```json
{
  "text_hash": "string (32-char MD5 hex)",
  "embedding": "array of 1536 floats"
}
```

**Example**:
```json
{
  "text_hash": "5d41402abc4b2a76b9719d911017c592",
  "embedding": [0.0023, -0.0145, ..., 0.0067]
}
```

#### Embeddings Parquet Schema
**File**: `data/vector/embeddings/embeddings.parquet`

**Columns** (Lines 103-110 in `qa_step01_embeddings.py`):
1. **chunk_id** (pa.string()): Unique chunk identifier
2. **doc_id** (pa.string()): Source document identifier
3. **seq_no** (pa.int32()): Chunk sequence within document
4. **token_count** (pa.int32()): Estimated token count
5. **l2_norm** (pa.float32()): L2 norm of embedding vector
6. **vector** (pa.list_(pa.float32())): 1536-dimensional embedding

**PyArrow Code**:
```python
arrs = {
    "chunk_id": pa.array(chunk_ids, type=pa.string()),
    "doc_id": pa.array(doc_ids, type=pa.string()),
    "seq_no": pa.array(seq_nos, type=pa.int32()),
    "token_count": pa.array(token_counts, type=pa.int32()),
    "l2_norm": pa.array(norms, type=pa.float32()),
    "vector": pa.array(vectors, type=pa.list_(pa.float32())),
}
table = pa.table(arrs)
pq.write_table(table, "data/vector/embeddings/embeddings.parquet")
```

#### FAISS Index Metadata
**Manifest**: `data/vector/faiss/faiss_manifest.json`

**Fields**:
- `index_type` (string): "HNSW" or "Flat"
- `metric` (string): "L2" or "INNER_PRODUCT"
- `dim` (integer): Vector dimension (1536)
- `count` (integer): Number of indexed vectors
- `roundtrip_error_max` (float): Maximum reconstruction error from 100 samples
- `paths` (object): `{"index": path, "idmap": path}`
- `params` (object): Snapshot of FAISS config (M, efConstruction, efSearch)

**ID Map**: `data/vector/faiss/idmap.parquet`
- **Columns**: id (int32), chunk_id (string), doc_id (string), seq_no (int32)
- **Purpose**: Maps FAISS vector IDs to chunk/document metadata

#### Weaviate/Pinecone Schemas
**Weaviate** (`data/vector/weaviate/schema_applied.json`):
```json
{
  "class": "Doc",
  "properties": ["doc_id", "text", "doctype", "date", "section", "topic", "url", "title", "company", "persona_tags", "source_domain"],
  "notes": "applied minimal schema (simulated)"
}
```

**Pinecone** (`data/vector/pinecone/index_manifest.json`):
```json
{
  "config": {"index_name": "demo-index", "namespace": "default", "metric": "cosine"},
  "upserted": 536,
  "failed": 0,
  "failed_ids": []
}
```

---

### 7. External Dependencies

#### OpenAI API
**Package**: `openai` (Python SDK)
**Model**: `text-embedding-ada-002`
**Endpoint**: `client.embeddings.create()`

**Configuration**:
- API key from environment variable `OPENAI_API_KEY`
- Loaded via `python-dotenv` from `.env` file (line 16 in `embedding_utils.py`)
- Client initialization at line 31: `client = OpenAI(api_key=api_key)`

**Request Format** (Single text, line 78-82):
```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=text,                      # String
    encoding_format="float"          # Returns Python floats (not base64)
)
```

**Request Format** (Batch, line 144-148):
```python
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=texts,                     # List[str]
    encoding_format="float"
)
```

**Response Structure**:
```python
response.data[0].embedding          # List[float] for single text
[item.embedding for item in response.data]  # List[List[float]] for batch
```

**Rate Limits** (Handled via retry):
- `RateLimitError` triggers exponential backoff (4s, 8s, 10s)
- Maximum 3 retry attempts

**Pricing**: $0.0001 per 1K tokens (~750 words, ~4000 chars)

#### FAISS Library
**Package**: `faiss` (conda installation recommended)
**Import**: `import faiss` (line 101 in `qa_step02_indexes.py`)

**Environment Conflict**:
- **Issue**: OpenMP Error #15 when pip-installed in certain conda environments
- **Solution**: Use `ageFaiss` conda environment (Python 3.12) for Gate-2 only
- **Safeguard**: `AG2_DISABLE_FAISS=1` environment variable to skip FAISS import

**Index Types Used**:
- `faiss.IndexHNSWFlat(dim, M, metric)` - HNSW with exact vectors
- `faiss.IndexFlatL2(dim)` - Flat index fallback (exact search)

**Metrics**:
- `faiss.METRIC_L2` - Euclidean distance
- `faiss.METRIC_INNER_PRODUCT` - Dot product similarity

**File I/O**:
- `faiss.write_index(idx, path)` - Serialize index to disk
- `faiss.read_index(path)` - Load index from disk (not used in Gate-2, but available)

#### Weaviate Client
**Status**: Not imported (simulated only)
**Planned Package**: `weaviate-client`

**Simulated Operations**:
- Schema creation: File write to `schema_applied.json`
- Manifest tracking: `inserted`, `failed`, `failed_ids` counts

#### Pinecone Client
**Status**: Not imported (simulated only)
**Planned Package**: `pinecone-client`

**Simulated Operations**:
- Manifest tracking: `upserted`, `failed`, `failed_ids` counts
- Configuration snapshot: `index_name`, `namespace`, `metric`

#### Other Dependencies
- **PyArrow** (line 86-93 in `qa_step01_embeddings.py`):
  - Parquet file I/O
  - Auto-installed if missing: `pip install pyarrow --quiet`

- **Tenacity** (line 10 in `embedding_utils.py`):
  - Retry decorator: `@retry(...)`
  - Exponential backoff logic

- **NumPy** (line 115 in `qa_step02_indexes.py`):
  - Array operations: `np.array(vecs, dtype="float32")`
  - L2 norm: `np.linalg.norm(q[0] - xb[rid])`

- **PyYAML** (line 18-20 in `qa_step01_embeddings.py`):
  - Config file parsing: `yaml.safe_load(f)`

---

### 8. Execution & Usage

#### Generate Embeddings (Gate-1)
**Command**:
```bash
conda run -n age python scripts/qa_step01_embeddings.py
```

**Environment**: `age` (Python 3.13)
**Prerequisites**:
- `OPENAI_API_KEY` in `.env` file
- Gate-0 completed (`reports/qa/step00_baseline.json` exists)
- Chunks available in `data/interim/chunks/*.chunks.jsonl`

**Output Files**:
- `data/vector/embeddings/embeddings.parquet` (embedding vectors + metadata)
- `data/vector/embeddings/embedding_stats.json` (statistics)
- `reports/qa/step01_embeddings.json` (machine-readable report)
- `reports/qa/step01_embeddings.md` (human-readable report)

**Cost Confirmation**:
- Estimates cost before API calls
- Prompts user: `"Proceed with embedding generation? [y/N]: "`
- Bypass with `AG1_AUTO_CONFIRM=1` environment variable

**Example Run**:
```bash
# With auto-confirmation
AG1_AUTO_CONFIRM=1 conda run -n age python scripts/qa_step01_embeddings.py
```

#### Build Indexes (Gate-2)
**Command**:
```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Environment**: `ageFaiss` (Python 3.12) - **Critical!**
**Why Different Environment**: Avoids OpenMP Error #15 from pip faiss-cpu conflicts

**Prerequisites**:
- Gate-1 completed (`data/vector/embeddings/embeddings.parquet` exists)
- `configs/vector.indexing.yaml` configured

**Output Files**:
- `data/vector/faiss/index.faiss` (FAISS index binary)
- `data/vector/faiss/idmap.parquet` (ID mapping)
- `data/vector/faiss/faiss_manifest.json` (index metadata)
- `data/vector/pinecone/index_manifest.json` (simulated)
- `data/vector/weaviate/schema_applied.json` (simulated)
- `data/vector/weaviate/index_manifest.json` (simulated)
- `data/final/reports/index_health.json` (combined health metrics)
- `reports/qa/step02_indexes.json` (machine-readable report)
- `reports/qa/step02_indexes.md` (human-readable report)

**Optional**: Skip FAISS build
```bash
AG2_DISABLE_FAISS=1 conda run -n age python scripts/qa_step02_indexes.py
```

#### Two-Environment Architecture

**Why Two Environments?**
1. **OpenMP Conflict**: pip `faiss-cpu` package causes "OMP Error #15" when run in conda environments with existing OpenMP libraries
2. **Solution**: Separate `ageFaiss` environment uses conda-installed FAISS (no pip conflicts)
3. **Isolation**: `age` environment never installs FAISS via pip

**Environment Breakdown**:

**`age` (Python 3.13)** - Primary:
- Gate-1 (embedding generation)
- Gate-3 through Gate-8 (MCP, routing, graph, evaluation)
- Does NOT have FAISS installed
- Uses OpenAI API, PyArrow, PyYAML, Tenacity

**`ageFaiss` (Python 3.12)** - FAISS Only:
- Gate-2 (index building) ONLY
- Has conda-installed FAISS
- Used exclusively for `qa_step02_indexes.py`

**Switching Environments**:
```bash
# Gate-1: Use age
conda run -n age python scripts/qa_step01_embeddings.py

# Gate-2: Use ageFaiss
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate-7: Back to age
conda run -n age python scripts/qa_step07_retrieval_eval.py
```

---

### 9. Code Patterns & Conventions

#### Always Use `embed_text()` from `embedding_utils.py`
**Rule**: Never create embeddings with random vectors or different embedding functions

**Correct**:
```python
from embedding_utils import embed_text

# Single text
vec = embed_text("sample text", dim=1536)

# Batch (even better)
from embedding_utils import embed_batch
vecs = embed_batch(["text1", "text2"], dim=1536, batch_size=20)
```

**Incorrect**:
```python
# ❌ DON'T DO THIS
import random
vec = [random.random() for _ in range(1536)]
```

**Reason**: Embedding consistency is critical
- Documents AND queries must use same embedding function
- Different embeddings cause 0% recall in retrieval evaluation
- `embed_text()` ensures OpenAI ada-002 consistency

#### Cache Before API Call
**Pattern**: All embedding functions check cache before making API calls

**Implementation** (`embed_text`, lines 108-111):
```python
# Check cache first
cached = _load_from_cache(text)
if cached is not None:
    return cached

# Only call API on cache miss
embedding = _call_openai_api(text)
```

**Benefits**:
- Reduces API costs (cached texts are free)
- Faster response times
- Enables cost estimation before execution

#### Dimension Must Be 1536
**Validation**: Runtime checks enforce 1536-dimension requirement

**Locations**:
- `embed_text()` line 97-101
- `embed_batch()` line 164-167
- `build_faiss()` line 117 (inferred from vectors)

**Error Message**:
```python
raise ValueError(
    f"OpenAI ada-002 requires dim={ADA002_DIM}, got dim={dim}. "
    f"Update configs/vector.indexing.yaml to set embedding.dim=1536"
)
```

**Reason**: OpenAI ada-002 model has fixed 1536-dimension output

#### Empty Text Handling
**Pattern**: Return small non-zero vector for empty/whitespace strings

**Implementation** (`embed_text`, lines 104-106):
```python
if not text or not text.strip():
    # Return small non-zero vector to avoid validation failures
    return [0.001] * ADA002_DIM
```

**Rationale**:
- Avoids API calls for empty chunks
- Prevents zero vectors (which fail L2 norm validation)
- Consistent behavior across batch and single-text functions

#### Config-Driven Behavior
**Pattern**: Prefer environment variables and config options over hardcoded values

**Examples**:
1. **Batch size** from config (line 120-126 in `qa_step01_embeddings.py`):
   ```python
   batch_size = int(cfg.get("embedding", {}).get("batch_size") or 100)
   ```

2. **Auto-confirm** via env var (line 154-156):
   ```python
   auto_confirm = os.getenv("AG1_AUTO_CONFIRM", "").lower() in ["1", "true", "yes", "y"]
   ```

3. **FAISS disable** via env var (line 87 in `qa_step02_indexes.py`):
   ```python
   if (_os.getenv("AG2_DISABLE_FAISS", "0") == "1"):
   ```

#### Dual-Format Reports
**Pattern**: All QA gates emit both JSON (machine) and Markdown (human) reports

**Gate-1 Example** (lines 273-303 in `qa_step01_embeddings.py`):
```python
# JSON report
machine = {
    "step": "step01_embeddings",
    "gate": "Gate-1",
    "status": status,
    "checks": checks,
    "next_action": next_action,
    "timestamp": now_iso(),
}
with open("reports/qa/step01_embeddings.json", "w") as f:
    json.dump(machine, f, indent=2)

# Markdown report
lines = [f"# STEP 1 — Embeddings Quality (Gate‑1) — {status}", ...]
with open("reports/qa/step01_embeddings.md", "w") as f:
    f.write("\n".join(lines) + "\n")
```

**Benefits**:
- JSON: Parsed by scripts, CI/CD pipelines
- Markdown: Human-readable, git diffs, documentation

#### Fail-Soft Error Handling
**Pattern**: Cache failures don't crash the system

**Cache Read** (lines 50-51 in `embedding_utils.py`):
```python
except Exception:
    pass  # Invalid cache, will regenerate
```

**Cache Write** (lines 66-67):
```python
except Exception:
    pass  # Cache write failure is non-fatal
```

**Reason**: Cache is performance optimization, not critical path

#### Tenacity Retry Decorator
**Pattern**: Use declarative `@retry` on internal API functions, wrap with context in public functions

**Internal** (`_call_openai_api`, lines 70-83):
```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError)),
    reraise=True
)
def _call_openai_api(text: str) -> List[float]:
    ...
```

**External** (`embed_text`, lines 114-122):
```python
try:
    embedding = _call_openai_api(text)
except Exception as e:
    print(f"ERROR: OpenAI API failed after 3 retries: {e}")
    raise RuntimeError(
        f"OpenAI API call failed: {type(e).__name__}: {e}\n"
        f"Text length: {len(text)} chars\n"
        f"Check your API key and network connection."
    ) from e
```

**Benefits**:
- Retry logic centralized in decorator
- User-friendly error messages in public API
- Exception chain preserved with `raise ... from e`

---

### 10. Testing & Verification

#### Gate-1 Validation (4 Checks)

**G1-01: Row Count Match** (`qa_step01_embeddings.py:234-239`)
- **Metric**: `embedding_rows`
- **Threshold**: `== baseline_chunks` (from Gate-0 check G0-04)
- **Status**: PASS if counts match, else FAIL
- **Purpose**: Ensures all chunks were embedded

**G1-02: Vector Dimension** (lines 240-244)
- **Metric**: `vector_dim`
- **Threshold**: `== 1536` (from config)
- **Status**: Always PASS (validated at runtime)
- **Purpose**: Confirms dimension consistency

**G1-03a: Zero Vectors** (lines 245-248)
- **Metric**: `zero_vectors`
- **Threshold**: `== 0`
- **Status**: PASS if no zero-norm vectors, else FAIL
- **Purpose**: Detects embedding failures

**G1-03b: NaN Vectors** (lines 249-252)
- **Metric**: `nan_vectors`
- **Threshold**: `== 0`
- **Status**: PASS if no NaN values, else FAIL
- **Purpose**: Detects API errors

**G1-04: Norm Outliers** (lines 253-257)
- **Metric**: `pct_norm_outliers`
- **Threshold**: `<= 0.005` (0.5%) for PASS, `<= 0.015` (1.5%) for WARN
- **Status**: PASS/WARN/FAIL based on outlier percentage
- **Purpose**: Detects unusual embeddings via IQR analysis

**Outlier Calculation** (lines 205-214):
1. Calculate quartiles: Q1, median, Q3
2. Compute IQR: `Q3 - Q1`
3. Bounds: `[median - 4*IQR, median + 4*IQR]`
4. Count norms outside bounds
5. Return percentage

#### Gate-2 Validation (7 Checks)

**G2-01: Pinecone Upsert Rate** (`qa_step02_indexes.py:338-341`)
- **Metric**: `pinecone_upsert_rate = upserted / baseline_chunks`
- **Threshold**: `>= 0.98` (PASS), `>= 0.97` (WARN)
- **Purpose**: Verifies 98%+ of chunks indexed in Pinecone

**G2-02: Weaviate Upsert Rate** (line 342)
- **Metric**: `weaviate_upsert_rate = inserted / baseline_chunks`
- **Threshold**: `>= 0.98` (PASS), `>= 0.97` (WARN)
- **Purpose**: Verifies 98%+ of chunks indexed in Weaviate

**G2-03: FAISS Count Ratio** (line 343)
- **Metric**: `faiss_count_ratio = faiss_count / baseline_chunks`
- **Threshold**: `>= 0.98` (PASS), `>= 0.97` (WARN)
- **Purpose**: Verifies 98%+ of chunks indexed in FAISS

**G2-04: Metadata Integrity** (line 344)
- **Metric**: `pct_missing_required_metadata`
- **Threshold**: `<= 0.02` (PASS), `<= 0.03` (WARN)
- **Purpose**: Ensures <= 2% of chunks have missing metadata (doctype, date, url, title)

**G2-05: FAISS Roundtrip Error** (line 345)
- **Metric**: `faiss_roundtrip_error_max`
- **Threshold**: `<= 0.001` (PASS), `<= 0.01` (WARN)
- **Purpose**: Validates FAISS index accuracy via reconstruction error

**G2-06: Sanity Search Min Top-K** (line 346)
- **Metric**: `sanity_search_min_topk`
- **Threshold**: `>= 3` (PASS), `== 2` (WARN)
- **Purpose**: Ensures at least 3 non-empty chunks returned per query

**G2-07: Sanity Keyword Hit** (line 347)
- **Metric**: `sanity_keyword_hit_min_top10`
- **Threshold**: `>= 1` (PASS)
- **Purpose**: Validates keyword relevance in retrieved chunks

#### Sanity Search Tests

**Queries** (lines 228-232 in `qa_step02_indexes.py`):
1. `"latest earnings results"` → keywords: `{earnings, results, gaap, guidance, rpo}`
2. `"agentforce product announcement"` → keywords: `{agentforce, product, announce, ai}`
3. `"remaining performance obligation definition"` → keywords: `{remaining, performance, obligation, rpo, definition}`

**Process** (lines 221-226):
1. Embed query text via `embed_text(q, dim)`
2. Compute L2 distances: `((xb - qv)**2).sum(axis=1)`
3. Return top-10 nearest vectors via argsort
4. Load chunk texts, count keyword matches
5. Track minimum non-empty count and keyword hits across 3 queries

**Backends Tested** (line 250):
- Pinecone (simulated via exact search)
- Weaviate (simulated via exact search)
- FAISS (uses actual FAISS index if built)

**Result Structure**:
```python
{
  "pinecone": {"min_topk": 10, "keyword_hit_min_top10": 3},
  "weaviate": {"min_topk": 10, "keyword_hit_min_top10": 3},
  "faiss": {"min_topk": 10, "keyword_hit_min_top10": 3}
}
```

#### Embedding Consistency Checks

**Problem**: Retrieval recall = 0% if documents and queries use different embeddings

**Solution**: All text (chunks AND queries) must use `embed_text()` from `embedding_utils.py`

**Validation**:
- Gate-1 ensures all chunks use OpenAI ada-002
- Gate-7 retrieval evaluation validates recall > 80%
- Sanity search in Gate-2 uses same `embed_text()` function

---

### 11. Known Issues & Limitations

#### OpenMP Error #15 (FAISS in Wrong Environment)

**Symptom**:
```
OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
[Segmentation fault or abort]
```

**Root Cause**:
- pip-installed `faiss-cpu` conflicts with conda's OpenMP libraries
- Both try to initialize OpenMP runtime
- Results in segmentation fault

**Solution**:
1. **Use `ageFaiss` environment for Gate-2 ONLY**:
   ```bash
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```

2. **NEVER install pip `faiss-cpu` in `age` environment**:
   - `age` environment deliberately excludes FAISS
   - Use `ageFaiss` (conda-installed FAISS) for index operations

3. **If already installed, recreate environment**:
   ```bash
   conda env remove -n age
   conda env create -f envs/age.yaml
   ```

4. **Bypass FAISS if needed**:
   ```bash
   AG2_DISABLE_FAISS=1 conda run -n age python scripts/qa_step02_indexes.py
   ```

**Prevention**:
- Script checks for import failures and writes disabled manifest (lines 100-113)
- Environment variable `AG2_DISABLE_FAISS=1` skips FAISS import entirely (lines 86-98)

#### OpenAI API Rate Limits

**Symptom**:
```
openai.RateLimitError: Rate limit reached for text-embedding-ada-002
```

**Mitigation**:
1. **Retry Logic**: Automatic 3-attempt retry with exponential backoff (4s, 8s, 10s)
2. **Batch Size Tuning**: Reduced from 100 to 20 to avoid 8192 token limit (config line 4)
3. **Caching**: Subsequent runs skip cached texts entirely

**If Still Failing**:
- Reduce batch size in config: `embedding.batch_size: 10`
- Add delays between batches (requires code modification)
- Upgrade OpenAI tier (check https://platform.openai.com/account/rate-limits)

#### API Cost Accumulation

**Risk**: Accidentally re-embedding large corpus without cache

**Mitigation**:
1. **Cost Estimation**: Pre-flight calculation shows estimated cost (lines 139-150)
2. **User Confirmation**: Prompts before API calls (lines 152-161)
3. **Auto-Confirm Bypass**: `AG1_AUTO_CONFIRM=1` for scripting (line 154)
4. **Caching**: Costs only apply to uncached texts

**Example**:
```
=== Pre-flight Cost Estimation ===
  Total chunks: 536
  Already cached: 536
  Need to embed: 0
  Estimated cost: $0.0000 USD
  (Cost only for uncached texts. Cached texts are free.)

✅ Auto-confirmed via AG1_AUTO_CONFIRM
```

**Best Practice**:
- Review cost estimate before confirming
- Clear cache (`embedding_utils.clear_cache()`) only when necessary
- Monitor OpenAI usage dashboard

#### Cache Invalidation

**Issue**: Cached embeddings persist even if chunk text changes

**Current Behavior**:
- Cache key based on SHA-256 of text
- If text changes, new cache key generated
- Old cache files remain (orphaned)

**Workaround**:
- Clear cache manually if chunks are regenerated:
  ```python
  from embedding_utils import clear_cache
  clear_cache()
  ```
- Re-run Gate-1 to rebuild cache

**Future Improvement**: Cache expiration or versioning (not currently implemented)

#### Simulated Weaviate/Pinecone Indexes

**Limitation**: Manifests created, but no actual vector upserts

**Current Behavior**:
- `upserted`/`inserted` counts equal `embedding_rows` (always 100%)
- `failed` always 0
- Config notes explicitly state "simulated" (lines 18, 22 in `vector.indexing.yaml`)

**Impact**:
- Gate-2 checks G2-01 and G2-02 always pass
- Sanity search uses exact search (not Weaviate/Pinecone APIs)

**Future Work**: Implement actual Weaviate/Pinecone client integration

#### Batch Size vs. Token Limit

**Issue**: OpenAI has 8192 token limit per request

**Solution**: Batch size reduced from 100 to 20 (config line 4)

**Calculation**:
- Average chunk: ~500 chars ≈ 125 tokens
- 20 chunks × 125 tokens = 2500 tokens (well under 8192 limit)

**If Chunks Are Longer**:
- Reduce `embedding.batch_size` in config
- Monitor for API errors: `Request body is too large`

---

### 12. References

#### Related Components
- **Part 2 (Chunking)**: Where chunks come from (`data/interim/chunks/*.chunks.jsonl`)
- **Part 4 (Router)**: How indexes are queried during retrieval
- **Part 7 (Quality Gates)**: Gate-1 and Gate-2 details (this document)

#### Documentation
- **docs/architecture.md**: Overall system design, LangGraph orchestration
- **docs/commands.md**: Command reference for all 41 scripts
- **docs/troubleshooting.md**: Debug playbook, OpenMP conflict resolution
- **docs/evaluation.md**: Gate-7 (retrieval) and Gate-8 (generation) metrics
- **docs/envs.md**: Environment setup, two-environment architecture

#### External Resources
- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **FAISS Documentation**: https://github.com/facebookresearch/faiss/wiki
- **Weaviate Docs**: https://weaviate.io/developers/weaviate
- **Pinecone Docs**: https://docs.pinecone.io/

#### Configuration Files
- **configs/vector.indexing.yaml**: All embedding and index parameters
- **envs/age.yaml**: Primary environment (no FAISS)
- **envs/ageFaiss.yaml**: FAISS-only environment (conda-installed)

---

## Code References

### Core Embedding Functions
- `scripts/embedding_utils.py:86-133` - `embed_text()` single-text embedding
- `scripts/embedding_utils.py:152-211` - `embed_batch()` batch embedding
- `scripts/embedding_utils.py:70-83` - `_call_openai_api()` API call with retry
- `scripts/embedding_utils.py:34-36` - `_get_cache_key()` SHA-256 key generation
- `scripts/embedding_utils.py:39-52` - `_load_from_cache()` cache lookup
- `scripts/embedding_utils.py:55-67` - `_save_to_cache()` cache storage
- `scripts/embedding_utils.py:214-240` - `estimate_embedding_cost()` cost calculation

### Gate-1 (Embedding Generation)
- `scripts/qa_step01_embeddings.py:115-306` - Main workflow
- `scripts/qa_step01_embeddings.py:128-137` - Chunk collection
- `scripts/qa_step01_embeddings.py:139-161` - Cost estimation and user confirmation
- `scripts/qa_step01_embeddings.py:169-176` - Batch embedding generation
- `scripts/qa_step01_embeddings.py:178-214` - Statistical analysis (norms, outliers)
- `scripts/qa_step01_embeddings.py:83-112` - Parquet file writing
- `scripts/qa_step01_embeddings.py:232-257` - Gate-1 checks (G1-01 through G1-04)
- `scripts/qa_step01_embeddings.py:259-269` - Status determination (GREEN/AMBER/RED)

### Gate-2 (Index Building)
- `scripts/qa_step02_indexes.py:80-164` - `build_faiss()` FAISS index construction
- `scripts/qa_step02_indexes.py:167-181` - `write_idmap()` FAISS ID mapping
- `scripts/qa_step02_indexes.py:283-292` - Pinecone manifest creation
- `scripts/qa_step02_indexes.py:295-308` - Weaviate schema and manifest
- `scripts/qa_step02_indexes.py:184-209` - `compute_metadata_missing()` metadata validation
- `scripts/qa_step02_indexes.py:212-271` - `run_sanity_search()` semantic search testing
- `scripts/qa_step02_indexes.py:336-347` - Gate-2 checks (G2-01 through G2-07)
- `scripts/qa_step02_indexes.py:349-366` - Status determination logic

### Configuration
- `configs/vector.indexing.yaml:1-5` - Embedding configuration (ada-002, dim 1536, batch 20)
- `configs/vector.indexing.yaml:7-12` - FAISS parameters (HNSW, M=32, efConstruction=200, efSearch=128)
- `configs/vector.indexing.yaml:14-18` - Pinecone configuration (simulated)
- `configs/vector.indexing.yaml:20-22` - Weaviate configuration (simulated)

---

## Historical Context (from thoughts/)

No previous research documents found on this specific topic. This is the first comprehensive documentation of the vector & embedding system.

---

## Related Research

This research document can be extended with:
- Part 2 research (chunking pipeline)
- Part 4 research (query routing to indexes)
- Part 7 research (Gate-7 retrieval evaluation)

---

## Open Questions

1. **Weaviate/Pinecone Integration**: When will actual network operations be implemented?
2. **Cache Expiration**: Should old cache entries be auto-deleted after X days?
3. **Batch Size Tuning**: Optimal batch size for different chunk length distributions?
4. **FAISS vs. Exact Search**: Performance comparison for current corpus size (536 chunks)?
5. **Multi-Model Support**: Plan for supporting other embedding models (e.g., OpenAI text-embedding-3-large)?
