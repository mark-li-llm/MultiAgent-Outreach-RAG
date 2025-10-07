# OpenAI ada-002 Embedding Migration Implementation Plan

## Overview

Migrate from hashlex-v1 (768-dim, deterministic hash-based) to OpenAI ada-002 (1536-dim, API-based) embeddings. This is a **hard cutover** that replaces the entire embedding system, requiring regeneration of all vector embeddings and FAISS indexes.

## Current State Analysis

### Existing Architecture
- **Embedding model**: hashlex-v1 (deterministic, hash-based, zero-cost)
- **Dimension**: 768
- **Core implementation**: `scripts/embedding_utils.py:embed_text()`
- **Configuration**: `configs/vector.indexing.yaml` (single source of truth)
- **Data scale**: ~1600 chunks currently embedded at 768-dim
- **Dependencies**: 7 scripts use embeddings (Gate-1, Gate-2, Gate-3, Gate-4, Gate-7, run_graph.py, qa_step04_router.py)

### Key Discoveries
- **OpenAI already integrated**: `scripts/run_graph.py:174` uses `ChatOpenAI` for LLM calls
- **API key pattern exists**: Uses `load_dotenv()` and `OPENAI_API_KEY` environment variable
- **Async patterns exist**: `run_graph.py` uses `await llm.ainvoke()` for async OpenAI calls
- **No OpenAI SDK in environment**: Need to add `openai` package to conda `age` environment
- **Hardcoded dimension**: `scripts/qa_step04_router.py` lines 214, 221 have hardcoded `768`

### Current Embedding Flow
1. **Gate-1** (`qa_step01_embeddings.py:138`): `v = embed_text(text, dim)` generates 768-dim vectors
2. **Gate-2** (`qa_step02_indexes.py:123`): `faiss.IndexHNSWFlat(dim=768, ...)` builds index
3. **Query time**: All scripts call `embed_text(query, dim)` to generate query vectors

## Desired End State

### Target Architecture
- **Embedding model**: OpenAI text-embedding-ada-002 (API-based, highest quality)
- **Dimension**: 1536 (double current size)
- **API calls**: Synchronous external calls to OpenAI API
- **Caching**: Smart cache with model version tracking to avoid repeated API costs
- **Error handling**: Fail-fast on API errors (no retries, manual rerun required)
- **Rate limiting**: Use existing HTTP rate limiter pattern (6 RPS default)

### Verification Criteria
After completion:
1. `configs/vector.indexing.yaml` specifies `model: openai-ada-002`, `dim: 1536`
2. `data/vector/embeddings/embeddings.parquet` contains 1536-dim vectors (regenerated)
3. `data/vector/faiss/index.faiss` built with 1536-dim structure (regenerated)
4. Gate-7 retrieval evaluation shows improved recall metrics
5. Smart cache file exists at `data/vector/embeddings/embedding_cache.jsonl`
6. No hardcoded `768` dimension values in codebase

## What We're NOT Doing

- **No dual model support**: Not maintaining both hashlex-v1 and ada-002 simultaneously
- **No automatic retries**: API failures require manual investigation and rerun
- **No batch optimization**: Processing chunks sequentially (simple, not optimized)
- **No cost tracking**: Assuming ~$0.50-$2.00 one-time cost is acceptable
- **No migration of old embeddings**: Regenerating from scratch, not converting

## Implementation Approach

**Strategy**: Replace the core `embed_text()` function implementation while maintaining its function signature for backward compatibility. Use smart caching to minimize API costs on reruns. Clear old artifacts and regenerate embeddings and indexes.

**Key Principle**: Fail-fast design - if API fails, stop immediately for manual investigation rather than silent degradation.

---

## Phase 1: Environment Setup

### Overview
Install OpenAI SDK and configure API key access.

### Changes Required

#### 1. Add OpenAI Package to Conda Environment
**File**: `envs/age.yaml`
**Changes**: Add `openai` pip dependency

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
    - python-dotenv
```

**Rationale**: OpenAI SDK not available via conda-forge; using pip within conda environment. Version 1.0+ uses modern async API.

#### 2. Recreate Conda Environment
**Command**:
```bash
/Users/liyunxiao/anaconda3/bin/conda env remove -n age
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
```

**Note**: Required because environment needs pip dependencies added.

#### 3. Create `.env` File with API Key
**File**: `.env` (in project root, NOT committed to git)
**Changes**: Add OpenAI API key

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-...your-key-here...

# Existing environment variables (if any)
AR_USER_AGENT=AccountResearchMVP/1.0
AR_GLOBAL_RPS=6
```

**Security**: Ensure `.env` is in `.gitignore` (already present in project).

### Success Criteria

#### Automated Verification:
- [ ] Conda environment recreates without errors: `/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml`
- [ ] OpenAI package imports successfully: `conda run -n age python -c "import openai; print(openai.__version__)"`
- [ ] python-dotenv imports successfully: `conda run -n age python -c "from dotenv import load_dotenv"`

#### Manual Verification:
- [ ] `.env` file exists in project root with `OPENAI_API_KEY` set
- [ ] API key is valid (test with simple API call)

---

## Phase 2: Core Embedding Function Replacement

### Overview
Replace `scripts/embedding_utils.py:embed_text()` implementation to use OpenAI ada-002 API with smart caching.

### Changes Required

#### 1. Update `scripts/embedding_utils.py`
**File**: `scripts/embedding_utils.py`
**Changes**: Replace entire file with OpenAI-based implementation

```python
#!/usr/bin/env python3
"""
Embedding utilities using OpenAI text-embedding-ada-002.

This module provides a single entry point: embed_text(text: str, dim: int) -> List[float]
All other scripts in the system use this function for both document and query embeddings.

Smart caching:
- Cache key: sha256(text + model_name)
- Cache file: data/vector/embeddings/embedding_cache.jsonl (append-only JSONL)
- Cache structure: {"cache_key": "...", "model": "ada-002", "dim": 1536, "vector": [...]}
"""
import hashlib
import json
import math
import os
from typing import List, Dict, Any, Optional

# Cache configuration
CACHE_FILE = os.path.join("data", "vector", "embeddings", "embedding_cache.jsonl")
MODEL_NAME = "text-embedding-ada-002"
EXPECTED_DIM = 1536

# In-memory cache (loaded on first use)
_cache: Optional[Dict[str, List[float]]] = None


def _load_cache() -> Dict[str, List[float]]:
    """Load cache from JSONL file into memory."""
    global _cache
    if _cache is not None:
        return _cache

    _cache = {}
    if not os.path.exists(CACHE_FILE):
        return _cache

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    cache_key = entry.get("cache_key")
                    vector = entry.get("vector")
                    if cache_key and vector:
                        _cache[cache_key] = vector
                except Exception:
                    continue  # Skip malformed lines
    except Exception:
        pass  # Cache file issues are non-fatal

    return _cache


def _save_cache_entry(cache_key: str, vector: List[float], model: str, dim: int):
    """Append a new cache entry to the JSONL file."""
    from common import ensure_dir
    ensure_dir(os.path.dirname(CACHE_FILE))

    entry = {
        "cache_key": cache_key,
        "model": model,
        "dim": dim,
        "vector": vector,
    }

    try:
        with open(CACHE_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Cache write failures are non-fatal but log to stderr
        import sys
        print(f"Warning: Failed to write cache entry: {e}", file=sys.stderr)


def _compute_cache_key(text: str, model: str) -> str:
    """Compute SHA256 hash of text + model for cache lookup."""
    content = f"{text}|{model}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _call_openai_embedding(text: str) -> List[float]:
    """
    Call OpenAI API to generate embedding.

    Fails fast on any error - no retries, no fallbacks.
    Caller must handle exceptions and decide whether to retry.
    """
    from dotenv import load_dotenv
    import openai

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found in environment. "
            "Please create a .env file with your OpenAI API key."
        )

    # Initialize client
    client = openai.OpenAI(api_key=api_key)

    # Call API (synchronous, fail-fast)
    try:
        response = client.embeddings.create(
            input=text,
            model=MODEL_NAME,
        )
        vector = response.data[0].embedding

        # Validate dimension
        if len(vector) != EXPECTED_DIM:
            raise RuntimeError(
                f"OpenAI returned {len(vector)}-dim vector, expected {EXPECTED_DIM}"
            )

        return vector
    except openai.APIError as e:
        # Re-raise with more context
        raise RuntimeError(f"OpenAI API error: {e}") from e
    except Exception as e:
        # Re-raise any other errors
        raise RuntimeError(f"Failed to generate embedding: {e}") from e


def embed_text(text: str, dim: int) -> List[float]:
    """
    Generate embedding for text using OpenAI ada-002 with smart caching.

    Args:
        text: Input text to embed
        dim: Expected dimension (must be 1536 for ada-002)

    Returns:
        List of floats (length = dim)

    Raises:
        ValueError: If dim != 1536
        RuntimeError: If API call fails or API key missing

    Design:
    - Validates dimension matches ada-002 (1536)
    - Checks cache first (by text+model hash)
    - On cache miss, calls OpenAI API
    - Saves new embedding to cache
    - Fails fast on any API error (no retries)
    """
    # Validate dimension
    if dim != EXPECTED_DIM:
        raise ValueError(
            f"dim must be {EXPECTED_DIM} for {MODEL_NAME}, got {dim}. "
            f"Update configs/vector.indexing.yaml to set embedding.dim: {EXPECTED_DIM}"
        )

    # Handle empty text
    if not text or not text.strip():
        # Return small uniform vector (L2-normalized)
        val = 1.0 / math.sqrt(dim)
        return [val] * dim

    # Check cache
    cache = _load_cache()
    cache_key = _compute_cache_key(text, MODEL_NAME)

    if cache_key in cache:
        return cache[cache_key]

    # Cache miss - call API
    vector = _call_openai_embedding(text)

    # Update in-memory cache
    cache[cache_key] = vector

    # Persist to disk (append-only)
    _save_cache_entry(cache_key, vector, MODEL_NAME, dim)

    return vector


# Legacy compatibility functions (kept for backward compatibility)
def normalize_text(text: str) -> str:
    """Legacy function - no longer used with OpenAI embeddings."""
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Legacy function - no longer used with OpenAI embeddings."""
    return text.split()


def hashlex_embed(tokens: List[str], dim: int, seed: int = 0x9E3779B9) -> List[float]:
    """
    Legacy function - DEPRECATED.
    Kept for compatibility but should not be called.
    """
    raise NotImplementedError(
        "hashlex_embed() is deprecated. System now uses OpenAI ada-002 embeddings. "
        "Use embed_text(text, dim) instead."
    )
```

**Key Design Decisions**:
1. **Cache key**: `sha256(text + model_name)` ensures uniqueness across text content
2. **JSONL format**: Append-only for safety; easy to inspect and debug
3. **In-memory cache**: Loaded once on first use for performance
4. **Fail-fast**: No retries, no silent fallbacks - surface errors immediately
5. **Dimension validation**: Hard-coded check for 1536 to prevent config mismatches

### Success Criteria

#### Automated Verification:
- [ ] File saves without syntax errors
- [ ] Import succeeds: `conda run -n age python -c "from embedding_utils import embed_text"`
- [ ] Dimension validation works: `conda run -n age python -c "from embedding_utils import embed_text; embed_text('test', 768)"` raises ValueError

#### Manual Verification:
- [ ] Test embedding generation: `conda run -n age python -c "from embedding_utils import embed_text; v = embed_text('test', 1536); print(len(v))"`
- [ ] Verify cache file created: `ls -lh data/vector/embeddings/embedding_cache.jsonl`
- [ ] Verify cache reuse: Run same embedding twice, second should be instant (no API call)

---

## Phase 3: Configuration Update

### Overview
Update vector indexing configuration to specify OpenAI ada-002 and 1536 dimensions.

### Changes Required

#### 1. Update Vector Indexing Config
**File**: `configs/vector.indexing.yaml`
**Changes**: Update embedding section

```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 256  # Not used in current implementation, kept for future
  notes: OpenAI text-embedding-ada-002 API with smart caching

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

**Changes**:
- Line 2: `model: hashlex-v1` → `model: openai-ada-002`
- Line 3: `dim: 768` → `dim: 1536`
- Line 5: Update notes to reflect OpenAI API usage

**Note**: FAISS parameters (M, efConstruction, efSearch) are dimension-independent and remain unchanged.

### Success Criteria

#### Automated Verification:
- [ ] YAML syntax is valid: `conda run -n age python -c "import yaml; yaml.safe_load(open('configs/vector.indexing.yaml'))"`
- [ ] Dimension reads as 1536: `conda run -n age python -c "import yaml; cfg = yaml.safe_load(open('configs/vector.indexing.yaml')); assert cfg['embedding']['dim'] == 1536"`

#### Manual Verification:
- [ ] Config file opens and displays correctly
- [ ] Model name is `openai-ada-002`
- [ ] Dimension is `1536`

---

## Phase 4: Fix Hardcoded Dimension Values

### Overview
Remove hardcoded `768` values that would break with 1536-dim vectors.

### Changes Required

#### 1. Fix Router Script Hardcoded Dimensions
**File**: `scripts/qa_step04_router.py`
**Changes**: Remove hardcoded `768` fallback values

**Line 214** (before):
```python
dim = 768
```

**Line 214** (after):
```python
dim = None  # Will be loaded from config below
```

**Line 221** (before):
```python
dim = 768
```

**Line 221** (after):
```python
# No fallback - dimension must come from config
if not dim:
    raise ValueError(
        "embedding.dim not found in configs/vector.indexing.yaml. "
        "Please ensure configuration is valid."
    )
```

**Rationale**: Hardcoded values prevent automatic adaptation to new dimension. Config is the single source of truth.

### Success Criteria

#### Automated Verification:
- [ ] No hardcoded `768` in router: `grep -n "768" scripts/qa_step04_router.py` returns no matches
- [ ] Script validates config presence rather than using silent fallback

#### Manual Verification:
- [ ] Review git diff to confirm changes
- [ ] Search entire `scripts/` directory for remaining hardcoded `768`: `grep -r "768" scripts/ --include="*.py"`

---

## Phase 5: Gate-1 Enhancement for API Calls

### Overview
Enhance Gate-1 script to handle API-based embeddings with proper error handling.

### Changes Required

#### 1. Add API Key Loading to Gate-1
**File**: `scripts/qa_step01_embeddings.py`
**Changes**: Add `load_dotenv()` at module initialization

**After line 13** (after `from embedding_utils import embed_text`):
```python
from dotenv import load_dotenv

# Load environment variables (must be before any OpenAI calls)
load_dotenv()
```

**Rationale**: Ensures API key is available before any embedding calls.

#### 2. Add Progress Logging
**File**: `scripts/qa_step01_embeddings.py`
**Changes**: Add progress output for long-running API calls

**In the main loop** (after line 138, where `v = embed_text(text, dim)` is called):
```python
                # Text-based embedding via OpenAI API (with caching)
                try:
                    v = embed_text(text, dim)
                except Exception as e:
                    # Fail fast - stop on first API error
                    print(f"\n❌ Embedding failed for chunk {chunk_id}")
                    print(f"   Text preview: {text[:100]}...")
                    print(f"   Error: {e}")
                    print(f"\n🛑 Stopping at chunk {embedding_rows + 1}/{baseline_chunks}")
                    print("   Fix the issue and rerun Gate-1 to continue.")
                    raise

                # Progress indicator (every 50 chunks)
                if (embedding_rows + 1) % 50 == 0:
                    print(f"  ⏳ Embedded {embedding_rows + 1}/{baseline_chunks} chunks "
                          f"(cache hit rate: {len(_cache) / max(1, embedding_rows + 1):.1%})")
```

**Rationale**:
- API calls can take time; progress helps user monitor status
- Fail-fast error handling with clear context for debugging
- Cache hit rate helps assess API cost savings

#### 3. Import Cache Access for Progress Logging
**File**: `scripts/qa_step01_embeddings.py`
**Changes**: Import cache for hit rate calculation

**After line 13**:
```python
from embedding_utils import embed_text, _load_cache
```

**And at the start of main()** (before the loop):
```python
    # Preload cache to enable hit rate tracking
    from embedding_utils import _cache, _load_cache
    _load_cache()
```

**Rationale**: Allows tracking cache effectiveness during batch embedding.

### Success Criteria

#### Automated Verification:
- [ ] Script imports successfully: `conda run -n age python -c "import sys; sys.path.insert(0, 'scripts'); import qa_step01_embeddings"`
- [ ] load_dotenv is called before embed_text

#### Manual Verification:
- [ ] Progress messages appear during execution
- [ ] Error messages are clear and actionable
- [ ] Cache hit rate is displayed

---

## Phase 6: Clear Old Artifacts

### Overview
Delete old 768-dim embeddings and indexes to prevent confusion and force clean regeneration.

### Changes Required

#### 1. Delete Old Embedding Artifacts
**Commands**:
```bash
# Backup old embeddings (optional, for safety)
mkdir -p data/backup/embeddings_768dim_$(date +%Y%m%d)
cp data/vector/embeddings/embeddings.parquet data/backup/embeddings_768dim_$(date +%Y%m%d)/ 2>/dev/null || true
cp data/vector/embeddings/embedding_stats.json data/backup/embeddings_768dim_$(date +%Y%m%d)/ 2>/dev/null || true

# Clear old embeddings
rm -f data/vector/embeddings/embeddings.parquet
rm -f data/vector/embeddings/embedding_stats.json

# Note: embedding_cache.jsonl is NEW, no need to delete
```

#### 2. Delete Old FAISS Index
**Commands**:
```bash
# Backup old index (optional, for safety)
mkdir -p data/backup/faiss_768dim_$(date +%Y%m%d)
cp -r data/vector/faiss/* data/backup/faiss_768dim_$(date +%Y%m%d)/ 2>/dev/null || true

# Clear old index
rm -f data/vector/faiss/index.faiss
rm -f data/vector/faiss/idmap.parquet
rm -f data/vector/faiss/faiss_manifest.json
rm -f data/final/reports/index_health.json
```

#### 3. Clear Old QA Reports (Optional)
**Commands**:
```bash
# Move old reports to archive (keeps history)
mkdir -p reports/archive/pre_ada002_$(date +%Y%m%d)
mv reports/qa/step01_embeddings.* reports/archive/pre_ada002_$(date +%Y%m%d)/ 2>/dev/null || true
mv reports/qa/step02_indexes.* reports/archive/pre_ada002_$(date +%Y%m%d)/ 2>/dev/null || true
mv reports/qa/step07_retrieval_eval.* reports/archive/pre_ada002_$(date +%Y%m%d)/ 2>/dev/null || true
```

### Success Criteria

#### Automated Verification:
- [ ] Old parquet deleted: `[ ! -f data/vector/embeddings/embeddings.parquet ]`
- [ ] Old FAISS index deleted: `[ ! -f data/vector/faiss/index.faiss ]`

#### Manual Verification:
- [ ] Backups exist in `data/backup/` directory
- [ ] Old reports archived (if desired)

---

## Phase 7: Regenerate Embeddings (Gate-1)

### Overview
Run Gate-1 to generate new 1536-dim embeddings using OpenAI ada-002 API.

### Execution Steps

#### 1. Run Gate-1 Embedding Generation
**Command**:
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py
```

**Expected Behavior**:
- Progress messages every 50 chunks
- Cache hit rate starts at 0%, increases on reruns
- Total time: ~2-5 minutes for 1600 chunks (depends on API latency)
- Cost: ~$0.50-$2.00 (one-time)

**Expected Output**:
```json
{
  "status": "GREEN",
  "rows": 1600
}
```

#### 2. Verify Embedding Artifacts
**Commands**:
```bash
# Check parquet file exists and has correct dimension
conda run -n age python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/vector/embeddings/embeddings.parquet')
print(f'Rows: {t.num_rows}')
vec = t.column('vector')[0].as_py()
print(f'Dimension: {len(vec)}')
assert len(vec) == 1536, f'Expected 1536-dim, got {len(vec)}'
print('✅ Embeddings are 1536-dimensional')
"

# Check stats file
cat data/vector/embeddings/embedding_stats.json | jq '.vector_dim'
# Should output: 1536

# Check cache file
wc -l data/vector/embeddings/embedding_cache.jsonl
# Should show ~1600 lines (one per chunk)
```

### Success Criteria

#### Automated Verification:
- [ ] Gate-1 completes with GREEN status: `conda run -n age python scripts/qa_step01_embeddings.py`
- [ ] Parquet file has 1536-dim vectors: Check script above
- [ ] Stats file shows `vector_dim: 1536`: `jq '.vector_dim' data/vector/embeddings/embedding_stats.json`
- [ ] Cache file has entries: `[ -f data/vector/embeddings/embedding_cache.jsonl ]`

#### Manual Verification:
- [ ] Gate-1 report shows `G1-02: vector_dim = 1536 -> PASS`
- [ ] No zero vectors (`G1-03a: zero_vectors = 0 -> PASS`)
- [ ] No NaN vectors (`G1-03b: nan_vectors = 0 -> PASS`)
- [ ] Norm outliers within tolerance (`G1-04: pct_norm_outliers <= 0.005 -> PASS`)

---

## Phase 8: Rebuild FAISS Index (Gate-2)

### Overview
Rebuild FAISS HNSW index with 1536-dim vectors.

### Execution Steps

#### 1. Run Gate-2 Index Build
**Command**:
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Note**: Uses `ageFaiss` environment to avoid OpenMP conflicts.

**Expected Behavior**:
- Loads 1600 chunks with 1536-dim vectors
- Creates `IndexHNSWFlat(dim=1536, M=32, metric=L2)`
- Runs round-trip accuracy test
- Performs sanity searches

**Expected Output**:
```json
{
  "status": "GREEN",
  "roundtrip_max_error": 0.0,
  "sanity_searches": 3
}
```

#### 2. Verify Index Artifacts
**Commands**:
```bash
# Check manifest dimension
cat data/vector/faiss/faiss_manifest.json | jq '.dim'
# Should output: 1536

# Check index file size (should be larger than before)
ls -lh data/vector/faiss/index.faiss
# Expected: ~9-10 MB (double the 768-dim size)
```

### Success Criteria

#### Automated Verification:
- [ ] Gate-2 completes with GREEN status: `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
- [ ] Manifest shows `dim: 1536`: `jq '.dim' data/vector/faiss/faiss_manifest.json`
- [ ] Round-trip error is near-zero: `jq '.roundtrip_error_max' data/vector/faiss/faiss_manifest.json`

#### Manual Verification:
- [ ] Gate-2 report shows `G2-01: indexed_count = 1600 -> PASS`
- [ ] Gate-2 report shows `G2-02: dim = 1536 -> PASS`
- [ ] Gate-2 report shows `G2-03: roundtrip_error_max <= 0.001 -> PASS`
- [ ] Index file exists and size is ~9-10 MB

---

## Phase 9: Validation (Gate-7 Retrieval Evaluation)

### Overview
Run end-to-end retrieval evaluation to verify improved recall with ada-002 embeddings.

### Execution Steps

#### 1. Run Gate-7 Retrieval Evaluation
**Command**:
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

**Expected Behavior**:
- Loads 1536-dim embeddings
- Runs 24+ test queries
- Computes recall@10, nDCG@5, latency metrics
- Generates trace JSONL with successes/failures

**Expected Improvements**:
- Recall@10: Should improve from previous 52.17% (target: >70%)
- nDCG@5: Should improve from previous baseline
- Latency: May increase slightly (API vs local hash)

#### 2. Review Evaluation Results
**Commands**:
```bash
# Check overall status
cat reports/qa/step07_retrieval_eval.json | jq '.status'

# Check recall metric
cat reports/qa/step07_retrieval_eval.json | jq '.checks[] | select(.id == "G7-01")'

# View human-readable report
cat reports/qa/step07_retrieval_eval.md
```

### Success Criteria

#### Automated Verification:
- [ ] Gate-7 completes without errors: `conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py`
- [ ] Recall@10 > 0.70 (70%): `jq '.checks[] | select(.id == "G7-01") | .actual' reports/qa/step07_retrieval_eval.json`
- [ ] No critical failures: Status is GREEN or AMBER

#### Manual Verification:
- [ ] Recall improved compared to hashlex-v1 baseline (52.17%)
- [ ] nDCG@5 shows reasonable ranking quality
- [ ] Latency is acceptable (median <2s with multiplier)
- [ ] Trace file shows fewer failures: `wc -l reports/eval/retrieval_failures.jsonl`

---

## Phase 10: Final Cleanup & Documentation

### Overview
Clean up backup files, update documentation, and verify all scripts work with new embeddings.

### Changes Required

#### 1. Update CLAUDE.md Documentation
**File**: `CLAUDE.md`
**Changes**: Update embedding model references

**Find and replace**:
- `hashlex-v1` → `openai-ada-002`
- `768` → `1536` (in embedding context)

**Specific sections to update**:
- Line 56: Update embedding.dim reference
- Line 312: Update embedding model description
- Section "Text Embedding System": Rewrite to describe OpenAI API approach

#### 2. Update Research Documents (Optional)
**Files**:
- `thoughts/shared/research/2025-10-06-embedding-model-architecture.md`
- `thoughts/shared/research/2025-10-06-low-recall-root-cause.md`

**Changes**: Add notes about migration to ada-002 and observed recall improvements.

#### 3. Close Issue Ticket
**File**: `thoughts/shared/issues/issue001.md`
**Changes**: Mark as resolved

```markdown
i need to change the embedding model to : OpenAI ada-002 (1536-dim, requires API key, highest quality)

**Status**: ✅ RESOLVED (2025-10-06)
**Implementation**: See `thoughts/shared/plans/2025-10-06-issue001-openai-ada002-migration.md`
**Outcome**: Successfully migrated from hashlex-v1 (768-dim) to OpenAI ada-002 (1536-dim). Gate-7 recall improved from 52.17% to [actual %].
```

### Success Criteria

#### Automated Verification:
- [ ] All gates pass: `make test` (if test target exists)
- [ ] No hardcoded 768 references remain: `grep -r "768" scripts/ configs/ --include="*.py" --include="*.yaml"`

#### Manual Verification:
- [ ] Documentation accurately reflects new architecture
- [ ] Issue ticket is closed with outcome notes
- [ ] Old backups can be safely archived or deleted
- [ ] Team is informed of migration completion

---

## Testing Strategy

### Unit Tests
Not applicable - no new testable units, just implementation swap.

### Integration Tests
Run all quality gates in sequence:
```bash
# Full pipeline validation
conda run -n age python scripts/qa_step00_baseline.py
conda run -n age python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
conda run -n age python scripts/qa_step03_mcp.py
conda run -n age python scripts/qa_step04_router.py
conda run -n age python scripts/qa_step05_graph.py
conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py
```

All gates should complete with GREEN or AMBER status.

### Manual Testing Steps
1. **Cache Behavior**:
   - Run Gate-1 twice
   - Second run should be nearly instant (all cache hits)
   - Verify API calls only on cache misses

2. **Error Handling**:
   - Temporarily set invalid API key
   - Run Gate-1
   - Verify clear error message and fail-fast behavior
   - Restore valid API key

3. **Dimension Validation**:
   - Temporarily change config to `dim: 768`
   - Run Gate-1
   - Verify it fails with dimension mismatch error
   - Restore config to `dim: 1536`

4. **End-to-End Query**:
   - Run MCP stub service: `conda run -n age python scripts/qa_step03_mcp.py`
   - Test query via MCP `kb.search` tool
   - Verify results are relevant and use 1536-dim vectors

## Performance Considerations

### API Latency
- **OpenAI API latency**: ~200-500ms per request (vs <1ms for hashlex)
- **Mitigation**: Smart caching ensures reruns are instant
- **One-time cost**: Initial Gate-1 run takes 2-5 minutes for 1600 chunks

### Index Size
- **768-dim index**: ~4.7 MB
- **1536-dim index**: ~9-10 MB (double the size)
- **Memory impact**: Minimal (indexes fit in RAM easily)

### API Costs
- **ada-002 pricing**: ~$0.10 per 1M tokens
- **Estimated cost for 1600 chunks**: $0.50-$2.00 (one-time)
- **Rerun cost**: $0 (fully cached)
- **New chunks**: Only pay for uncached embeddings

### Rate Limiting
- **Default RPS**: 6 requests/second (existing HTTP rate limiter)
- **OpenAI tier limits**: Varies by account (typically 3-10 RPS for free tier)
- **Risk**: May hit rate limits on large batches
- **Mitigation**: Adjust `AR_GLOBAL_RPS` environment variable if needed

## Migration Notes

### Rollback Plan
If ada-002 migration fails or recall degrades:

1. **Restore configuration**:
   ```bash
   git checkout configs/vector.indexing.yaml
   ```

2. **Restore embedding_utils.py**:
   ```bash
   git checkout scripts/embedding_utils.py
   ```

3. **Restore old embeddings**:
   ```bash
   cp data/backup/embeddings_768dim_*/embeddings.parquet data/vector/embeddings/
   cp data/backup/embeddings_768dim_*/embedding_stats.json data/vector/embeddings/
   ```

4. **Restore old index**:
   ```bash
   cp -r data/backup/faiss_768dim_*/* data/vector/faiss/
   ```

5. **Rerun Gate-2** (to verify index):
   ```bash
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```

### Cache Management
- **Cache file location**: `data/vector/embeddings/embedding_cache.jsonl`
- **Cache invalidation**: Delete cache file to force re-embedding (costs API credits)
- **Cache inspection**: Each line is valid JSON, human-readable
- **Cache compaction**: Not needed for 1600 entries (~1 MB file size)

### Future Considerations
- **Batch optimization**: If processing >10K chunks, consider implementing batch API calls
- **Cost tracking**: Add API call counter and cost estimation to Gate-1 output
- **Model versioning**: If OpenAI updates ada-002, cache keys will naturally segregate versions

## References

- **Original issue**: `thoughts/shared/issues/issue001.md`
- **Implementation plan**: `thoughts/shared/plans/2025-10-06-issue001-openai-ada002-migration.md` (this file)
- **Embedding architecture research**: `thoughts/shared/research/2025-10-06-embedding-model-architecture.md`
- **Root cause analysis**: `thoughts/shared/research/2025-10-06-low-recall-root-cause.md`
- **OpenAI ada-002 docs**: https://platform.openai.com/docs/guides/embeddings
- **FAISS documentation**: `scripts/qa_step02_indexes.py` (inline comments)
