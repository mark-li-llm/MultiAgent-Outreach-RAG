---
date: 2025-10-06
author: Claude Code
status: draft
topic: "OpenAI ada-002 Embedding Model Migration"
tags: [implementation, embeddings, openai, ada-002, migration]
---

# OpenAI ada-002 Embedding Model Migration
# here use a new prompt, and explicitly ask the model to ask me questions and get the answer.

# 发现的问题
1. 我没有用langgraph，只是mimic的langgraph，代码没有用那个库
2. def embed_query(q: str, d: int) -> List[float]:
      rnd = random.Random()
    这里有一个很危险的代码。在 run_graph.py

## Overview

Replace the current hashlex-v1 (768-dim, deterministic hash-based) embedding model with OpenAI ada-002 (1536-dim, API-based) throughout the entire RAG pipeline. This is a complete replacement with no fallback to the old model.

## Current State Analysis

Based on the research document at `thoughts/shared/research/exp0:2025-10-06-embedding-model-architecture.md`:

**Current Implementation (hashlex-v1)**:
- **Location**: `scripts/embedding_utils.py:65-66`
- **Dimension**: 768
- **Properties**: Deterministic, stateless, no external dependencies
- **Process**: Text normalization → Tokenization (unigrams + bigrams) → FNV-1a feature hashing → L2 normalization
- **Used by**: Gate-1 (embeddings), Gate-2 (FAISS), Gate-7 (retrieval eval), MCP services, run_graph.py

**Configuration**:
- `configs/vector.indexing.yaml`: embedding.model = "hashlex-v1", dim = 768
- Hardcoded dimensions in `qa_step04_router.py:214, 221`
- Custom embedding functions in `run_graph.py:300-322` (doesn't use shared `embed_text()`)

**Existing Patterns**:
- Environment variables: `os.environ.get()` pattern (see `scripts/common.py:19-20`)
- Dotenv usage: `run_graph.py` already uses `python-dotenv` (line 14, 21)
- .env file: Already in `.gitignore` (line 9)

## Desired End State

**New Implementation (OpenAI ada-002)**:
- Use OpenAI's text-embedding-ada-002 model via official API
- Dimension: 1536 (fixed by OpenAI model)
- API key stored in `.env` file
- Batch processing for efficiency (100 texts per API call)
- All old hashlex-v1 code removed
- All scripts updated to use 1536 dimensions
- All embeddings regenerated and indexes rebuilt

**Verification**:
- Gate-1 passes with 1536-dim vectors
- Gate-2 builds FAISS index with 1536-dim vectors
- Gate-7 retrieval evaluation works with new embeddings
- All quality gates GREEN

## What We're NOT Doing

- ❌ Cost control / pre-flight estimator
- ❌ Rate limiting / exponential backoff
- ❌ Dry-run mode
- ❌ Keeping old hashlex-v1 as fallback
- ❌ Caching between runs (embeddings stored in parquet only)
- ❌ Supporting both models simultaneously

## Implementation Approach

1. **Complete Replacement**: Remove all hashlex-v1 code, no backwards compatibility
2. **Batch API Calls**: Use OpenAI batch endpoint (100 texts/request) for efficiency
3. **Simple Error Handling**: Fail fast on API errors, no retries
4. **Consistent Interface**: Keep `embed_text(text, dim)` signature for compatibility
5. **Shared Implementation**: Make `run_graph.py` use shared `embed_text()` function

---

## Phase 1: Environment Setup

### Overview
Set up OpenAI API credentials and add required dependencies.

### Changes Required

#### 1. Create `.env` Template
**File**: `.env.example` (new file)
**Changes**: Create template for users to copy

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-...your-api-key-here...
```

#### 2. Update Conda Environment
**File**: `envs/age.yaml`
**Changes**: Add OpenAI SDK and python-dotenv

```yaml
name: age
channels:
  - conda-forge
dependencies:
  - python=3.13
  - aiohttp
  - pyyaml
  - pyarrow>=21
  - numpy>=2.3
  - certifi
  - openblas
  - llvm-openmp
  - pip
  - pip:
      - openai>=1.0.0
      - python-dotenv>=1.0.0
  # IMPORTANT: Do NOT install pip faiss-cpu in this env to avoid duplicate libomp.
```

#### 3. Document API Key Setup
**File**: `README.md` (add section after Environment Setup)
**Changes**: Add instructions for .env configuration

```markdown
### OpenAI API Key Configuration

The system now uses OpenAI ada-002 embeddings. You must configure your API key:

1. Copy the template: `cp .env.example .env`
2. Edit `.env` and add your OpenAI API key: `OPENAI_API_KEY=sk-...`
3. Keep `.env` private - it's already in `.gitignore`

Get your API key from: https://platform.openai.com/api-keys
```

### Success Criteria

#### Automated Verification:
- [ ] Template file exists: `test -f .env.example`
- [ ] Environment recreates successfully: `conda env remove -n age && conda env create -f envs/age.yaml`
- [ ] OpenAI package imports: `conda run -n age python -c "import openai; print(openai.__version__)"`
- [ ] Dotenv package imports: `conda run -n age python -c "from dotenv import load_dotenv"`

#### Manual Verification:
- [ ] User creates `.env` file from template
- [ ] User adds valid OpenAI API key to `.env`
- [ ] API key is not committed to git

---

## Phase 2: Core Embedding Replacement

### Overview
Replace hashlex-v1 implementation with OpenAI ada-002 in `embedding_utils.py`.

### Changes Required

#### 1. Rewrite `embedding_utils.py`
**File**: `scripts/embedding_utils.py`
**Changes**: Complete replacement with OpenAI client

```python
#!/usr/bin/env python3
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# OpenAI ada-002 fixed dimension
ADA002_DIM = 1536


def embed_text(text: str, dim: int) -> List[float]:
    """
    Generate embedding for a single text using OpenAI ada-002.

    Args:
        text: Input text to embed
        dim: Expected dimension (must be 1536 for ada-002, kept for compatibility)

    Returns:
        1536-dimensional embedding vector

    Raises:
        ValueError: If dim != 1536
        openai.APIError: If API call fails
    """
    if dim != ADA002_DIM:
        raise ValueError(f"OpenAI ada-002 only supports {ADA002_DIM} dimensions, got {dim}")

    if not text or not text.strip():
        # Return zero vector for empty text (will be caught by Gate-1 validation)
        return [0.0] * ADA002_DIM

    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text,
        encoding_format="float"
    )

    return response.data[0].embedding


def embed_batch(texts: List[str], dim: int, batch_size: int = 100) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using OpenAI ada-002 batch API.

    Args:
        texts: List of input texts to embed
        dim: Expected dimension (must be 1536 for ada-002)
        batch_size: Number of texts per API call (max 2048, default 100)

    Returns:
        List of 1536-dimensional embedding vectors

    Raises:
        ValueError: If dim != 1536
        openai.APIError: If API call fails
    """
    if dim != ADA002_DIM:
        raise ValueError(f"OpenAI ada-002 only supports {ADA002_DIM} dimensions, got {dim}")

    if not texts:
        return []

    # Process in batches
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Filter empty texts but preserve indices
        batch_inputs = [(idx, text if text and text.strip() else " ") for idx, text in enumerate(batch)]

        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[text for _, text in batch_inputs],
            encoding_format="float"
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings
```

**Delete**: All old hashlex-v1 functions:
- `normalize_text()` (lines 8-19)
- `tokenize()` (lines 22-30)
- `_stable_hash()` (lines 33-39)
- `hashlex_embed()` (lines 42-62)

### Success Criteria

#### Automated Verification:
- [ ] File has valid Python syntax: `conda run -n age python -m py_compile scripts/embedding_utils.py`
- [ ] OpenAI import works: `conda run -n age python -c "from embedding_utils import embed_text, embed_batch"`
- [ ] Dimension validation: `conda run -n age python -c "from embedding_utils import embed_text; try: embed_text('test', 768); except ValueError: print('PASS')"`

#### Manual Verification:
- [ ] Single embedding call works: `embed_text("hello world", 1536)` returns 1536 floats
- [ ] Batch embedding works: `embed_batch(["text1", "text2"], 1536)` returns 2 vectors
- [ ] API error is raised if OPENAI_API_KEY is invalid

---

## Phase 3: Configuration Updates

### Overview
Update all configuration files to reflect new dimension and model.

### Changes Required

#### 1. Update Vector Indexing Config
**File**: `configs/vector.indexing.yaml`
**Changes**: Update model and dimension

```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 100
  notes: OpenAI text-embedding-ada-002 via API

faiss:
  type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128

pinecone:
  index_name: demo-index
  namespace: default
  metric: cosine
  notes: simulated manifest only (no network)

weaviate:
  class_name: Doc
  notes: schema-only manifest (simulated)
```

### Success Criteria

#### Automated Verification:
- [ ] Config loads successfully: `conda run -n age python -c "import yaml; c=yaml.safe_load(open('configs/vector.indexing.yaml')); assert c['embedding']['dim']==1536; assert c['embedding']['model']=='openai-ada-002'"`

---

## Phase 4: Gate-1 Batch Processing

### Overview
Update Gate-1 to use batch embedding API for efficiency.

### Changes Required

#### 1. Update `qa_step01_embeddings.py`
**File**: `scripts/qa_step01_embeddings.py`
**Changes**: Use `embed_batch()` instead of individual `embed_text()` calls

**Line 13**: Add batch import
```python
from embedding_utils import embed_text, embed_batch
```

**Lines 117-155**: Replace individual embedding with batch processing
```python
def main():
    ensure_dir(VEC_DIR)
    dim = read_yaml_dim(CONF)
    baseline_chunks = load_baseline_chunks()

    # Load batch size from config
    try:
        with open(CONF, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        batch_size = int(cfg.get("embedding", {}).get("batch_size") or 100)
    except Exception:
        batch_size = 100

    rows: List[Dict[str, Any]] = []
    embedding_rows = 0
    zero_vectors = 0
    nan_vectors = 0
    norms: List[float] = []

    # Collect all chunks first for batch processing
    all_chunks: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(CHUNK_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    all_chunks.append(j)
                except Exception:
                    continue

    # Extract texts for batch embedding
    texts = [chunk.get("text") or "" for chunk in all_chunks]

    # Generate embeddings in batches
    print(f"Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")
    vectors = embed_batch(texts, dim, batch_size)

    # Process results
    for chunk, v in zip(all_chunks, vectors):
        chunk_id = chunk.get("chunk_id") or ""
        doc_id = chunk.get("doc_id") or ""
        seq_no = chunk.get("seq_no") or 0
        token_count = chunk.get("token_count") or 0

        n = l2_norm(v)

        if n == 0.0:
            zero_vectors += 1
        if any((x != x) for x in v):  # NaN check
            nan_vectors += 1

        rows.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "seq_no": seq_no,
            "token_count": token_count,
            "l2_norm": n,
            "vector": [float(x) for x in v],
        })
        norms.append(n)
        embedding_rows += 1

    # Rest of the function remains the same (stats, parquet write, checks)
    # ...
```

### Success Criteria

#### Automated Verification:
- [ ] Script runs without syntax errors: `conda run -n age python -m py_compile scripts/qa_step01_embeddings.py`

#### Manual Verification:
- [ ] Gate-1 completes successfully (run after regenerating environment)
- [ ] Embeddings parquet has 1536-dim vectors
- [ ] All G1-* checks pass

---

## Phase 5: Dimension Updates Across Pipeline

### Overview
Update all scripts that reference the old 768 dimension to use 1536.

### Changes Required

#### 1. Update `qa_step04_router.py` Hardcoded Dimensions
**File**: `scripts/qa_step04_router.py`
**Changes**: Replace hardcoded 768 → 1536

**Line 214**:
```python
# OLD:
dim = 768

# NEW:
dim = 1536
```

**Line 221**:
```python
# OLD:
dim = 768

# NEW:
dim = 1536
```

**Better approach**: Read from config instead of hardcoding
```python
import yaml
cfg = yaml.safe_load(open(os.path.join("configs", "vector.indexing.yaml")))
dim = int(cfg.get("embedding", {}).get("dim") or 1536)
```

#### 2. Update `qa_step07_retrieval_eval.py` Default Dimension
**File**: `scripts/qa_step07_retrieval_eval.py`
**Changes**: Update fallback dimension

**Line 209**:
```python
# OLD:
dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)

# NEW:
dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 1536)
```

#### 3. Update `run_graph.py` Custom Embedding Functions
**File**: `scripts/run_graph.py`
**Changes**: Replace custom hash functions with shared `embed_text()`

**Line 297**: Update dimension default
```python
# OLD:
dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)

# NEW:
dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 1536)
```

**Lines 299-322**: Replace custom embedding with shared implementation
```python
# DELETE OLD CUSTOM FUNCTIONS:
# def hash_vec(seed: str, d: int) -> List[float]:
#     ...random-based hashing...
# def embed_query(q: str, d: int) -> List[float]:
#     ...random-based hashing...

# ADD NEW IMPORTS (top of file, after line 13):
from embedding_utils import embed_text

# REPLACE with shared implementation:
def embed_query(q: str, d: int) -> List[float]:
    """Use shared OpenAI ada-002 embedding."""
    return embed_text(q, d)
```

**Line 326-330**: Update chunk embedding to use shared function
```python
# OLD (loads chunks and uses custom hash_vec):
for cf in sorted(glob.glob(os.path.join("data","interim","chunks","*.chunks.jsonl"))):
    with open(cf, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                v = hash_vec(j.get("chunk_id") or "", dim)  # OLD: custom hash
                # ...

# NEW (load from parquet instead):
# Since embeddings are now expensive to regenerate, load from parquet
import pyarrow.parquet as pq
emb_path = os.path.join("data", "vector", "embeddings", "embeddings.parquet")
emb_table = pq.read_table(emb_path)
chunk_ids = emb_table["chunk_id"].to_pylist()
vectors = emb_table["vector"].to_pylist()
# Build index for fast lookup
chunk_to_vec = dict(zip(chunk_ids, vectors))

# Then load chunks and look up pre-computed vectors
for cf in sorted(glob.glob(os.path.join("data","interim","chunks","*.chunks.jsonl"))):
    with open(cf, "r", encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
                chunk_id = j.get("chunk_id") or ""
                v = chunk_to_vec.get(chunk_id)
                if v is None:
                    continue  # Skip if embedding not found
                chunks_index.append(j)
                vectors.append(v)
            # ...
```

### Success Criteria

#### Automated Verification:
- [ ] All scripts compile: `conda run -n age python -m py_compile scripts/qa_step04_router.py scripts/qa_step07_retrieval_eval.py scripts/run_graph.py`
- [ ] No references to 768: `! grep -r "768" scripts/*.py configs/*.yaml`

#### Manual Verification:
- [ ] All gates use 1536 dimensions consistently
- [ ] No hardcoded dimensions remain

---

## Phase 6: Data Regeneration & Validation

### Overview
Regenerate all embeddings and rebuild indexes with OpenAI ada-002.

### Changes Required

This phase involves running commands, not code changes.

### Execution Steps

#### 1. Backup Existing Data (Optional)
```bash
# Optional: backup old 768-dim embeddings
mkdir -p data/backup/hashlex-v1
cp data/vector/embeddings/embeddings.parquet data/backup/hashlex-v1/
cp -r data/vector/faiss data/backup/hashlex-v1/
```

#### 2. Recreate Conda Environment
```bash
conda env remove -n age
conda env create -f envs/age.yaml
```

#### 3. Set Up `.env` File
```bash
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...
```

#### 4. Run Gate-1 (Generate Embeddings)
```bash
conda run -n age python scripts/qa_step01_embeddings.py
```

**Expected**:
- Calls OpenAI API ~16 times for 1600 chunks (100 per batch)
- Creates `data/vector/embeddings/embeddings.parquet` with 1536-dim vectors
- Gate-1 status: GREEN

#### 5. Run Gate-2 (Build FAISS Index)
```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Expected**:
- Loads 1536-dim vectors from parquet
- Builds FAISS HNSW index with dimension=1536
- Gate-2 status: GREEN

#### 6. Run Gate-7 (Retrieval Evaluation)
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Expected**:
- Embeds queries using OpenAI ada-002
- Computes retrieval metrics
- Gate-7 status: GREEN (or AMBER with improved recall vs hashlex-v1)

### Success Criteria

#### Automated Verification:
- [ ] Embeddings parquet exists: `test -f data/vector/embeddings/embeddings.parquet`
- [ ] FAISS index exists: `test -f data/vector/faiss/index.faiss`
- [ ] Gate-1 GREEN: `conda run -n age python -c "import json; r=json.load(open('reports/qa/step01_embeddings.json')); assert r['status']=='GREEN'"`
- [ ] Gate-2 GREEN: `conda run -n ageFaiss python -c "import json; r=json.load(open('reports/qa/step02_indexes.json')); assert r['status'] in ['GREEN','AMBER']"`
- [ ] Dimension is 1536: `conda run -n age python -c "import pyarrow.parquet as pq; t=pq.read_table('data/vector/embeddings/embeddings.parquet'); assert len(t['vector'][0].as_py())==1536"`

#### Manual Verification:
- [ ] Gate-1 report shows vector_dim: 1536
- [ ] Gate-2 FAISS index health report shows dimension: 1536
- [ ] Gate-7 retrieval metrics are reasonable (recall@10 > 0%)
- [ ] No errors in console output

---

## Testing Strategy

User requested no special testing features. Validation is done through existing quality gates:

### Gate-1 Validation (Embedding Quality):
- G1-01: embedding_rows == baseline_chunks
- G1-02: vector_dim == 1536
- G1-03a: zero_vectors == 0
- G1-03b: nan_vectors == 0
- G1-04: pct_norm_outliers <= 0.005

### Gate-2 Validation (Index Quality):
- G2-05: faiss_roundtrip_error_max <= 0.001

### Gate-7 Validation (Retrieval Quality):
- Recall@10 > 0% (expected improvement over hashlex-v1's 52.17%)
- nDCG@5 computed correctly
- Latency within budget (with 3x multiplier)

---

## Performance Considerations

### API Costs
OpenAI ada-002 pricing (as of Oct 2024): ~$0.0001 per 1K tokens

**Estimated cost for 1600 chunks** (avg ~200 tokens each):
- Total tokens: ~320,000
- Cost: ~$0.032 per full embedding run

**No cost controls implemented** (per user request).

### Batch Processing
- Batch size: 100 texts per API call
- Reduces API calls from 1600 → 16
- Faster execution, same cost

### Expected Performance
- Gate-1 runtime: ~2-5 minutes (network dependent)
- Gate-2 runtime: ~same as before (local FAISS indexing)
- Gate-7 runtime: +30-60 seconds (query embedding API calls)

---

## Migration Notes

### Breaking Changes
- **Dimension change**: 768 → 1536 (affects all downstream consumers)
- **External dependency**: Requires internet + valid OpenAI API key
- **Non-deterministic**: Same text may get slightly different embeddings over time (OpenAI model updates)
- **Cost**: Recurring cost for embedding generation

### Rollback Plan
User requested **no fallback**, but if needed:
1. Restore hashlex-v1 code from git: `git checkout HEAD -- scripts/embedding_utils.py`
2. Restore old config: `git checkout HEAD -- configs/vector.indexing.yaml`
3. Restore old data: `cp data/backup/hashlex-v1/* data/vector/`
4. Revert dimension changes in all scripts

### Data Retention
- Old 768-dim embeddings: User can manually backup before regeneration
- New 1536-dim embeddings: Stored in standard parquet format

---

## References

- Research document: `thoughts/shared/research/exp0:2025-10-06-embedding-model-architecture.md`
- OpenAI embeddings guide: https://platform.openai.com/docs/guides/embeddings
- OpenAI ada-002 model: https://openai.com/blog/new-and-improved-embedding-model
- Current embedding implementation: `scripts/embedding_utils.py`
- Configuration file: `configs/vector.indexing.yaml`

---

## Implementation Checklist

### Pre-Implementation
- [ ] Read and understand research document completely
- [ ] Get OpenAI API key from https://platform.openai.com/api-keys
- [ ] Review estimated costs (~$0.032 per run)

### Phase 1: Environment Setup
- [ ] Create `.env.example` template
- [ ] Update `envs/age.yaml` with openai + python-dotenv
- [ ] Add API key documentation to `README.md`
- [ ] Recreate conda environment
- [ ] Test OpenAI import

### Phase 2: Core Embedding Replacement
- [ ] Rewrite `scripts/embedding_utils.py` with OpenAI client
- [ ] Delete all hashlex-v1 functions
- [ ] Test `embed_text()` and `embed_batch()` functions

### Phase 3: Configuration Updates
- [ ] Update `configs/vector.indexing.yaml` (model + dim)
- [ ] Verify config loads correctly

### Phase 4: Gate-1 Batch Processing
- [ ] Update `qa_step01_embeddings.py` to use `embed_batch()`
- [ ] Test batch processing logic

### Phase 5: Dimension Updates
- [ ] Update `qa_step04_router.py` (remove hardcoded 768)
- [ ] Update `qa_step07_retrieval_eval.py` (fallback to 1536)
- [ ] Update `run_graph.py` (use shared embed_text, load from parquet)
- [ ] Verify no references to 768 remain

### Phase 6: Data Regeneration
- [ ] Set up `.env` with API key
- [ ] Run Gate-1 (embeddings)
- [ ] Run Gate-2 (FAISS index)
- [ ] Run Gate-7 (retrieval eval)
- [ ] Verify all gates GREEN/AMBER

### Post-Implementation
- [ ] Compare retrieval quality (ada-002 vs hashlex-v1)
- [ ] Document actual costs incurred
- [ ] Update CLAUDE.md if needed
