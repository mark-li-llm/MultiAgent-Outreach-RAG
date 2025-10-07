---
date: 2025-10-06T16:46:20-04:00
researcher: Momo23569
git_commit: cbf660d4aabcfcf4fc879c5c037a3ac73abb0008
branch: agent-faiss
repository: agent-faiss
topic: "Embedding Model Architecture and Dimension Dependencies"
tags: [research, codebase, embeddings, hashlex-v1, dimensions, gate-1, gate-2, gate-7, mcp]
status: complete
last_updated: 2025-10-06
last_updated_by: Momo23569
---

# Research: Embedding Model Architecture and Dimension Dependencies

**Date**: 2025-10-06T16:46:20-04:00
**Researcher**: Momo23569
**Git Commit**: cbf660d4aabcfcf4fc879c5c037a3ac73abb0008
**Branch**: agent-faiss
**Repository**: agent-faiss

## Research Question

Document the current embedding model implementation (hashlex-v1), understand where embeddings are generated and used throughout the system, and map all dimension dependencies to understand what components rely on the 768-dimensional vector space.

## Summary

The system currently uses a deterministic, hash-based embedding model called `hashlex-v1` that generates 768-dimensional vectors. The embedding dimension is configured in `configs/vector.indexing.yaml` and is used consistently across the entire pipeline:

- **Gate-1** (`qa_step01_embeddings.py`): Generates embeddings for document chunks
- **Gate-2** (`qa_step02_indexes.py`): Builds FAISS indexes using the embedded vectors
- **Gate-7** (`qa_step07_retrieval_eval.py`): Generates query embeddings for retrieval evaluation
- **MCP Service** (`qa_step03_mcp.py`): Embeds queries for the knowledge base search tool

All components use the same `embed_text(text, dim)` function from `scripts/embedding_utils.py`, ensuring vector space compatibility. The dimension value (768) flows from configuration through the entire pipeline.

## Detailed Findings

### Core Embedding Implementation

#### hashlex-v1 Model (`scripts/embedding_utils.py`)

The `embed_text(text: str, dim: int) -> List[float]` function at line 65-66 is the primary entry point for all embedding operations:

```python
def embed_text(text: str, dim: int) -> List[float]:
    return hashlex_embed(tokenize(text), dim)
```

**Process**:
1. **Text Normalization** (`normalize_text()` at lines 8-19):
   - Lowercase conversion
   - Unicode to ASCII normalization
   - Hyphen/slash separation
   - Digit collapsing to "0"
   - Whitespace normalization

2. **Tokenization** (`tokenize()` at lines 22-30):
   - Extracts alphanumeric tokens (2-20 characters)
   - Generates bigrams for local context (e.g., "machine_learning")
   - Returns combined list of unigrams + bigrams with "bg:" prefix

3. **Feature Hashing** (`hashlex_embed()` at lines 42-62):
   - Uses FNV-1a 64-bit hash with seed `0x9E3779B9` (golden ratio constant)
   - Maps tokens to dimension indices via modulo: `idx = hash(token) % dim`
   - Applies signed feature hashing (sign determined by least significant bit)
   - Accumulates signed contributions at each dimension
   - L2-normalizes final vector to unit length

**Key Properties**:
- **Deterministic**: Same input always produces same output
- **Stateless**: No training, no model files, no external dependencies
- **Dimension-parameterized**: Output vector length controlled by `dim` parameter
- **Symmetric**: Documents and queries use identical embedding process

#### The `dim` Parameter

The `dim` parameter controls:
- Output vector dimensionality (number of floats in the returned list)
- Hash table size for feature mapping
- Collision rate (higher dim → fewer collisions → more distinct features)

**Current Value**: 768 dimensions (specified in `configs/vector.indexing.yaml:3`)

**Critical Invariant**: All embeddings in the same vector space MUST use identical `dim` values. Mismatched dimensions result in incompatible vectors and zero retrieval recall.

### Configuration Architecture

#### Primary Configuration File

**`configs/vector.indexing.yaml`** (lines 1-5):
```yaml
embedding:
  model: hashlex-v1
  dim: 768
  batch_size: 256
  notes: deterministic hash-based embedding for QA
```

**Fields**:
- `model`: Embedding model identifier (currently `hashlex-v1`)
- `dim`: Vector dimensionality (currently `768`)
- `batch_size`: Batch size for embedding generation (currently `256`)

This configuration is the single source of truth for embedding parameters.

#### Configuration Loading Points

| File | Line | How `dim` is Determined | Value |
|------|------|------------------------|-------|
| `qa_step01_embeddings.py` | 115 | `read_yaml_dim(CONF)` reads from config | 768 |
| `qa_step02_indexes.py` | 117 | `len(vecs[0])` inferred from vectors, config fallback | 768 |
| `qa_step03_mcp.py` | 72 | `xb.shape[1]` inferred from parquet matrix | 768 |
| `qa_step07_retrieval_eval.py` | 209 | `load_yaml(...).get("embedding").get("dim")` with fallback to 768 | 768 |
| `run_graph.py` | 297 | `load_yaml(...).get("embedding").get("dim")` with fallback to 768 | 768 |
| `qa_step04_router.py` | 214, 221 | Hardcoded | 768 |

**Patterns**:
- **Configuration-driven**: Most scripts read from YAML config
- **Data-driven**: Some scripts infer from existing vector shapes
- **Fallback defaults**: Scripts default to 768 if config is missing

### Pipeline Integration

#### Gate-1: Embedding Generation (`scripts/qa_step01_embeddings.py`)

**Purpose**: Generate embeddings for all document chunks and save to Parquet.

**Configuration Reading** (line 29-37):
```python
def read_yaml_dim(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim = int(cfg.get("embedding", {}).get("dim") or 0)
    if not dim:
        raise ValueError("embedding.dim missing or invalid...")
    return dim
```

- Reads `configs/vector.indexing.yaml`
- Extracts `embedding.dim` field (768)
- Raises error if missing or zero
- Called at line 115: `dim = read_yaml_dim(CONF)`

**Embedding Generation** (line 138):
```python
v = embed_text(text, dim)
```

- Imports `embed_text` from `embedding_utils.py` at line 13
- Calls for each chunk's text content
- Passes dimension value from config (768)

**Output** (line 110):
- Writes to `data/vector/embeddings/embeddings.parquet`
- Schema includes `vector` column with list of float32
- Each vector has exactly `dim` elements (768)

**Validation Checks**:
- G1-01: Row count matches baseline chunks
- G1-02: Dimension matches config
- G1-03a: No zero-magnitude vectors
- G1-03b: No NaN values
- G1-04: Outlier percentage ≤0.5%

#### Gate-2: FAISS Index Building (`scripts/qa_step02_indexes.py`)

**Purpose**: Build FAISS HNSW index from generated embeddings.

**Dimension Extraction** (line 117):
```python
dim = len(vecs[0]) if vecs else int(cfg.get("embedding", {}).get("dim") or 0)
```

- **Primary source**: Infers dimension from first vector's length
- **Fallback**: Reads from config if no vectors exist
- **No validation**: Doesn't check that all vectors have same length
- **No cross-check**: Doesn't verify inferred dimension matches config

**FAISS Index Creation** (line 123):
```python
idx = faiss.IndexHNSWFlat(dim, M, metric)
```

- Creates HNSW index with:
  - `dim`: Vector dimensionality (768 from vector shape)
  - `M`: Graph connectivity (32 from config)
  - `metric`: Distance metric (L2 from config)
- Sets `efConstruction` (200) and `efSearch` (128) from config

**Vector Addition** (line 133):
```python
idx.add(xb)
```

- Converts Python list to NumPy float32 array at line 132
- Adds all 768-dimensional vectors to index
- FAISS assigns sequential IDs: 0, 1, 2, ..., n-1

**Outputs**:
- `data/vector/faiss/index.faiss`: Binary FAISS index file
- `data/vector/faiss/idmap.parquet`: Mapping from FAISS IDs to chunk_id/doc_id
- `data/vector/faiss/faiss_manifest.json`: Metadata including dimension value

**Round-Trip Validation** (lines 136-149):
- Randomly samples 100 vectors
- Searches index for each vector
- Computes L2 distance error between query and retrieved vector
- Check G2-05: `faiss_roundtrip_error_max <= 0.001`

#### Gate-7: Retrieval Evaluation (`scripts/qa_step07_retrieval_eval.py`)

**Purpose**: Evaluate end-to-end retrieval quality with query embeddings.

**Configuration Loading** (line 209):
```python
dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)
```

- Reads from `configs/vector.indexing.yaml`
- Extracts `embedding.dim` field
- Defaults to 768 if config missing or malformed

**Offline Mode Query Embedding** (lines 233-235):
```python
from embedding_utils import embed_text as _embed_text
def embed_query(q: str, d: int) -> List[float]:
    return _embed_text(q, d)
```

- Imports same `embed_text` function used in Gate-1
- Wraps in local function for convenience
- Uses configured dimension (768)

**Query Execution** (line 386):
```python
qv = embed_query(q, dim)
```

- Embeds query text using same `embed_text()` function
- Uses same dimension as document embeddings (768)
- Ensures query and document vectors exist in same vector space

**L2 Distance Computation** (line 390):
```python
d = sum((x - y) * (x - y) for x, y in zip(qv, v))
```

- Computes squared L2 distance between query vector and document vectors
- Both vectors are 768-dimensional
- Python `zip()` would truncate silently if dimensions mismatched

**Evaluation Metrics**:
- recall@10: Proportion of relevant documents in top 10 results
- nDCG@5: Ranking quality metric
- Coverage: Whether all indexed documents are reachable
- Latency: Response time budgets (median, p95, p99)

#### MCP Service: Knowledge Base Search (`scripts/qa_step03_mcp.py`)

**Purpose**: Provide kb.search tool for agent-to-agent interactions.

**Embedding Matrix Loading** (line 45-53):
```python
vecs = []
for path in sorted(glob.glob(os.path.join("data", "interim", "chunks", "*.chunks.jsonl"))):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            v = embed_text(j.get("text") or "", dim)
            vecs.append(v)
xb = np.array(vecs, dtype="float32")
```

- Loads document chunks
- Embeds each chunk's text
- Stores in NumPy matrix with shape `(n_chunks, dim)`

**Dimension Inference** (line 72):
```python
dim = xb.shape[1]
```

- Extracts dimension from loaded embedding matrix
- Matrix has 768 columns (from Gate-1 embeddings)
- Uses this dimension for query embedding

**Query Embedding Function** (line 71-76):
```python
from embedding_utils import embed_text
dim = xb.shape[1]
v = embed_text(q, dim)
return np.array(v, dtype="float32").reshape(1, -1)
```

- Imports same `embed_text` function
- Uses dimension from embedding matrix (768)
- Ensures query vector matches document vectors

**Search Handler** (line 104-106):
```python
qv = state["embed_query"](q)
dists = ((xb - qv)**2).sum(axis=1)
```

- Embeds query using matrix-derived dimension
- Computes L2 distance via NumPy broadcasting
- NumPy would raise `ValueError` if shapes incompatible

### Dimension Dependencies Summary

#### Files That Import `embed_text()`

1. **`scripts/qa_step01_embeddings.py:13`**
   - Imports: `from embedding_utils import embed_text`
   - Usage: Embedding document chunks at line 138
   - Dimension source: Config via `read_yaml_dim()` at line 115

2. **`scripts/qa_step02_indexes.py:11`**
   - Imports: `from embedding_utils import embed_text`
   - Usage: Query embedding for index verification at line 218
   - Dimension source: Inferred from vector shape at line 217

3. **`scripts/qa_step03_mcp.py:71`**
   - Imports: `from embedding_utils import embed_text` (within function)
   - Usage: Query embedding for kb.search at line 73
   - Dimension source: Inferred from matrix shape at line 72

4. **`scripts/qa_step07_retrieval_eval.py:233`**
   - Imports: `from embedding_utils import embed_text as _embed_text`
   - Usage: Query embedding for retrieval evaluation at line 386
   - Dimension source: Config with fallback to 768 at line 209

#### Files That Use Dimension But DON'T Import `embed_text()`

1. **`scripts/run_graph.py:297`**
   - Reads `embedding.dim` from config with fallback to 768
   - Uses custom `hash_vec()` and `embed_query()` functions (lines 300-322)
   - Does NOT use `embed_text()` from embedding_utils
   - Custom embedding logic differs from hashlex-v1

2. **`scripts/qa_step04_router.py:214, 221`**
   - Hardcodes `dim = 768` in two locations
   - Does NOT import or use `embed_text()`
   - No actual embedding generation in this script

#### Critical Consistency Points

All components that perform actual embedding must:
1. Use `embed_text()` from `scripts/embedding_utils.py`
2. Pass the same `dim` parameter value (768)
3. Use the same configuration source or data-derived dimension

**Exception**: `run_graph.py` uses custom embedding functions instead of `embed_text()`, creating a potential inconsistency if those functions don't match hashlex-v1 behavior.

### Data Flow: End-to-End Embedding Pipeline

```
Configuration
    ↓
configs/vector.indexing.yaml (embedding.dim: 768)
    ↓
    ├─→ Gate-1 (qa_step01_embeddings.py)
    │       ↓
    │   Read config → dim=768
    │       ↓
    │   Load chunks from data/interim/chunks/*.chunks.jsonl
    │       ↓
    │   For each chunk.text: embed_text(text, 768)
    │       ↓
    │   Write data/vector/embeddings/embeddings.parquet
    │       ↓
    ├─→ Gate-2 (qa_step02_indexes.py)
    │       ↓
    │   Load embeddings.parquet
    │       ↓
    │   Infer dim=len(vecs[0]) → 768
    │       ↓
    │   Build FAISS index: IndexHNSWFlat(768, M=32, L2)
    │       ↓
    │   Write data/vector/faiss/index.faiss
    │       ↓
    ├─→ MCP Service (qa_step03_mcp.py)
    │       ↓
    │   Load embeddings from chunks
    │       ↓
    │   Build matrix xb with shape (n, 768)
    │       ↓
    │   Infer dim=xb.shape[1] → 768
    │       ↓
    │   For each query: embed_text(query, 768)
    │       ↓
    │   Compute L2 distances and return top-k
    │       ↓
    └─→ Gate-7 (qa_step07_retrieval_eval.py)
            ↓
        Read config → dim=768
            ↓
        For each eval query: embed_text(query, 768)
            ↓
        Compute retrieval metrics (recall@10, nDCG@5)
            ↓
        Write reports/qa/step07_retrieval_eval.{json,md}
```

## Code References

### Core Implementation
- `scripts/embedding_utils.py:65-66` - Main `embed_text()` function
- `scripts/embedding_utils.py:8-19` - Text normalization
- `scripts/embedding_utils.py:22-30` - Tokenization (unigrams + bigrams)
- `scripts/embedding_utils.py:42-62` - Feature hashing and L2 normalization
- `scripts/embedding_utils.py:33-39` - FNV-1a hash function

### Configuration
- `configs/vector.indexing.yaml:1-5` - Embedding configuration
- `configs/vector.indexing.yaml:3` - Dimension specification (dim: 768)
- `configs/vector.indexing.yaml:7-12` - FAISS parameters

### Gate-1 (Embedding Generation)
- `scripts/qa_step01_embeddings.py:29-37` - Configuration reading
- `scripts/qa_step01_embeddings.py:13` - Import embed_text
- `scripts/qa_step01_embeddings.py:115` - Load dimension from config
- `scripts/qa_step01_embeddings.py:138` - Call embed_text(text, dim)
- `scripts/qa_step01_embeddings.py:81-110` - Parquet writing

### Gate-2 (FAISS Indexing)
- `scripts/qa_step02_indexes.py:63-77` - Load embeddings from Parquet
- `scripts/qa_step02_indexes.py:117` - Infer dimension from vectors
- `scripts/qa_step02_indexes.py:123` - Create FAISS HNSW index
- `scripts/qa_step02_indexes.py:133` - Add vectors to index
- `scripts/qa_step02_indexes.py:136-149` - Round-trip validation

### Gate-7 (Retrieval Evaluation)
- `scripts/qa_step07_retrieval_eval.py:209` - Load dimension from config
- `scripts/qa_step07_retrieval_eval.py:233` - Import embed_text
- `scripts/qa_step07_retrieval_eval.py:386` - Query embedding
- `scripts/qa_step07_retrieval_eval.py:390` - L2 distance computation

### MCP Service
- `scripts/qa_step03_mcp.py:45-53` - Load and embed document chunks
- `scripts/qa_step03_mcp.py:71-76` - Query embedding function
- `scripts/qa_step03_mcp.py:72` - Infer dimension from matrix
- `scripts/qa_step03_mcp.py:104-106` - Search handler

## Architecture Documentation

### Vector Space Consistency

The system maintains vector space consistency through three mechanisms:

1. **Shared Embedding Function**: All components use `embed_text()` from `embedding_utils.py`
2. **Configuration-Driven Dimension**: Primary scripts read from `configs/vector.indexing.yaml`
3. **Data-Derived Dimension**: Downstream scripts infer dimension from existing vectors

### Dimension Flow Patterns

**Pattern 1: Configuration-First (Gate-1, Gate-7)**
- Read dimension from YAML config
- Use for all embedding operations
- Ensures consistency with configuration intent

**Pattern 2: Data-First (Gate-2, MCP Service)**
- Infer dimension from loaded vectors or matrices
- Use for query embedding
- Ensures consistency with existing data

**Pattern 3: Hardcoded (Gate-4 Router)**
- Hardcodes dimension value (768)
- No embedding generation in this script
- Only used for diagnostic purposes

### Error Handling

**Missing Configuration**:
- Gate-1: Raises `ValueError` if dimension missing
- Gate-7: Defaults to 768 if config load fails
- Gate-2: Falls back to config if no vectors exist

**Dimension Mismatch**:
- No explicit validation that vectors have consistent dimensions
- No cross-validation between config and inferred dimensions
- Python `zip()` silently truncates if query/document dimensions mismatch (offline mode)
- NumPy broadcasting raises `ValueError` if shapes incompatible (online mode)

**Empty Input**:
- Empty text produces uniform vector `[1/sqrt(dim)] * dim`
- Never returns zero-length or malformed vectors

### Design Characteristics

**Deterministic**: Same input text always produces same 768-dimensional vector (no randomness)

**Stateless**: No training required, no model files, no external API calls

**Lightweight**: Only uses Python standard library (math, re, unicodedata, typing)

**Portable**: No GPU required, no large dependencies, works on any platform

**Transparent**: Feature hashing with FNV-1a is fully specified and reproducible

**Configurable**: Dimension can be changed by updating single config value (though see Dependencies section)

## Historical Context (from thoughts/)

No prior research documents found in thoughts/shared/research/ related to embedding model architecture or dimension dependencies.

## Related Research

This is the first research document on embedding architecture in this repository.

## Open Questions

1. **Dimension Validation**: Should there be explicit checks that all vectors have the same dimension?
2. **Config vs Data**: Should dimension be read from config or inferred from data? Current system uses both.
3. **run_graph.py Consistency**: Why does `run_graph.py` use custom embedding functions instead of `embed_text()`?
4. **Hardcoded Dimensions**: Should `qa_step04_router.py` read dimension from config instead of hardcoding?
5. **Error Handling**: Should dimension mismatches raise explicit errors instead of silent truncation/broadcasting errors?
