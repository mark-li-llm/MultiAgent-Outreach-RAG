# OpenAI ada-002 Embedding Migration Implementation Plan

## Overview

Migrate the RAG system from the current deterministic hashlex-v1 embedding model (768 dimensions) to OpenAI's ada-002 embedding model (1536 dimensions) to improve retrieval quality. The current system shows 52.17% recall, partially due to offline mode using random embeddings instead of proper text embeddings. This migration will also fix those critical bugs.

## Current State Analysis

### Embedding Architecture (hashlex-v1)

**Implementation**: `scripts/embedding_utils.py:65-66`
- **Algorithm**: Deterministic feature hashing using FNV-1a
- **Process**: Text normalization → Tokenization (unigrams + bigrams) → Signed feature hashing → L2 normalization
- **Dimensions**: 768 (configured in `configs/vector.indexing.yaml:3`)
- **Dependencies**: None (pure Python, no external API calls)
- **Quality**: Lexical overlap-based similarity, no semantic understanding

### Current Usage Locations

| File | Usage | Dimension Source | Line |
|------|-------|------------------|------|
| `qa_step01_embeddings.py` | Document chunk embedding (batch) | Config (strict validation) | 138 |
| `qa_step02_indexes.py` | Query embedding for validation | Inferred from vectors | 218 |
| `qa_step03_mcp.py` | Runtime query embedding for kb.search | Inferred from matrix | 73 |
| `qa_step07_retrieval_eval.py` | Eval query embedding | Config (fallback to 768) | 386 |

### Critical Issues Discovered

**Issue 1: Random Embeddings in Offline Mode**
- **Files**: `scripts/run_graph.py:300-322`, `scripts/qa_step04_router.py:224-240`
- **Problem**: Custom `hash_vec()` and `embed_query()` functions generate pseudo-random vectors instead of semantic embeddings
- **Impact**: Document vectors based on metadata (`chunk_id::length::tokens`), not content
- **Result**: ~52% retrieval recall (essentially random matching)
- **Root Cause**: Development stubs that were never replaced with proper embeddings

**Issue 2: Hardcoded Dimensions**
- **File**: `scripts/qa_step04_router.py:214,221`
- **Problem**: Hardcoded `dim = 768` with fallback to 768 if config fails
- **Impact**: Won't automatically update when config changes

**Issue 3: No Semantic Understanding**
- hashlex-v1 only captures lexical overlap (exact token/bigram matches)
- Cannot understand synonyms, paraphrases, or semantic similarity
- Limits recall on queries that use different wording

### Key Discoveries

From `thoughts/shared/research/2025-10-06-embedding-model-architecture.md`:

1. **Vector Space Consistency**: All components using `embed_text()` from `embedding_utils.py` maintain consistency
2. **Configuration Flow**: `configs/vector.indexing.yaml` is single source of truth for embedding parameters
3. **Dimension Dependencies**:
   - Configuration files: 1 file to update
   - Scripts with fallbacks: 3 files (`qa_step04_router.py`, `qa_step07_retrieval_eval.py`, `run_graph.py`)
   - Data artifacts: All must be regenerated (embeddings.parquet, FAISS indexes, manifests)
4. **Offline Mode**: Used when MCP service unavailable, currently broken due to random embedding bug

## Desired End State

### Target Architecture (ada-002)

**Model**: OpenAI text-embedding-ada-002
- **Dimensions**: 1536 (vs. 768 for hashlex-v1)
- **Quality**: State-of-the-art semantic embeddings
- **API**: Requires OpenAI API key and network access
- **Cost**: ~$0.0001 per 1K tokens (~$0.16 for current corpus at ~1.6k chunks)
- **Rate Limits**: 3,500 RPM for tier 1 accounts

### Configuration-Driven Model Selection

**Updated `configs/vector.indexing.yaml`**:
```yaml
embedding:
  model: ada-002  # Changed from: hashlex-v1
  dim: 1536       # Changed from: 768
  batch_size: 256
  api_provider: openai  # New field
  notes: OpenAI ada-002 text embeddings for semantic retrieval
```

**Backward Compatibility**: Support both models via config, allowing:
- Development/testing with hashlex-v1 (no API costs)
- Production with ada-002 (high quality)
- Easy A/B testing and rollback

### Fixed Offline Mode

**Options**:
1. **Disable offline mode** for ada-002 (requires pre-generated embeddings from Gate-1)
2. **Fall back to hashlex-v1** in offline mode (development/testing only)
3. **Load embeddings from Parquet** instead of regenerating (recommended)

**Recommended approach**: Load pre-generated embeddings from `data/vector/embeddings/embeddings.parquet` in offline mode, eliminating the need for runtime embedding generation.

### Verification Criteria

**Automated Verification**:
- [ ] Config validation passes: `dim` matches selected model (768 for hashlex-v1, 1536 for ada-002)
- [ ] Gate-1 completes successfully with ada-002: `make -C . gate1`
- [ ] Gate-2 builds FAISS index with 1536-dim vectors: `make -C . gate2`
- [ ] All unit tests pass: `pytest scripts/test_*.py` (if they exist)
- [ ] No import errors when OPENAI_API_KEY missing (graceful degradation)
- [ ] Embedding generation succeeds for sample queries
- [ ] API retry logic triggers on transient errors

**Manual Verification**:
- [ ] Gate-7 retrieval recall improves significantly (target: >80% vs. current 52.17%)
- [ ] Offline mode no longer uses random embeddings
- [ ] Error messages are clear when API key missing
- [ ] Rate limiting prevents quota exhaustion
- [ ] Generated embeddings have correct dimensionality (1536)
- [ ] Vector L2 norms are reasonable (close to 1.0 after normalization)

## What We're NOT Doing

1. **NOT migrating to other embedding providers** (Cohere, Voyage, etc.) - only OpenAI ada-002
2. **NOT changing FAISS index parameters** (M, efConstruction, efSearch stay the same)
3. **NOT modifying Weaviate/Pinecone manifests** (simulated backends remain simulated)
4. **NOT adding embedding caching layer** (future optimization)
5. **NOT implementing embedding fine-tuning** (using pre-trained ada-002 as-is)
6. **NOT changing chunking strategy** (chunk sizes and overlap remain the same)
7. **NOT updating environment.yaml files** - `openai` package must be installed manually via pip

## Implementation Approach

### High-Level Strategy

1. **Enhance `embedding_utils.py`** with multi-model support (both hashlex-v1 and ada-002)
2. **Update configuration** to specify model and dimension
3. **Add API key management** following existing dotenv pattern
4. **Implement robust error handling** (retries, rate limiting, quotas)
5. **Fix offline mode bugs** in run_graph.py and qa_step04_router.py
6. **Regenerate all embeddings** using new model (Gate-1)
7. **Rebuild all indexes** with new dimensions (Gate-2)
8. **Validate end-to-end** with retrieval evaluation (Gate-7, Gate-8)

### Backward Compatibility

- Preserve `embed_text(text, dim)` signature for hashlex-v1
- Add new `embed_text_with_config(text, config)` for model selection
- Fallback to hashlex-v1 if OpenAI API unavailable
- Allow config to specify model explicitly

---

## Phase 1: Enhanced Embedding Module

### Overview
Add OpenAI ada-002 support to `scripts/embedding_utils.py` while preserving backward compatibility with hashlex-v1.

### Changes Required

#### 1. `scripts/embedding_utils.py`

**File**: `scripts/embedding_utils.py`

**Changes**: Add multi-model support with graceful degradation

```python
#!/usr/bin/env python3
import math
import os
import re
import time
import unicodedata
from typing import List, Dict, Any, Optional

# Preserve existing hashlex-v1 functions (lines 8-66)
# ... [existing normalize_text, tokenize, _stable_hash, hashlex_embed, embed_text]

# New: OpenAI API integration
def _get_openai_client():
    """
    Lazy-load OpenAI client to avoid import errors if package not installed.
    Returns None if openai package unavailable or API key missing.
    """
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None

def embed_text_ada002(text: str, retry_count: int = 3, backoff_base: float = 1.0) -> Optional[List[float]]:
    """
    Generate OpenAI ada-002 embedding for text.

    Args:
        text: Input text to embed
        retry_count: Number of retries on transient errors (default: 3)
        backoff_base: Base backoff in seconds (default: 1.0, exponential: 1, 2, 4)

    Returns:
        List of 1536 floats (ada-002 dimension), or None on error

    Raises:
        RuntimeError: If openai package not installed
        ValueError: If OPENAI_API_KEY not set
        Exception: On persistent API errors after retries
    """
    client = _get_openai_client()
    if client is None:
        # Check which error to raise
        try:
            import openai
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or create a .env file.")
        except ImportError:
            raise RuntimeError("openai package not installed. Install with: pip install openai>=1.54.3")

    # Retry logic with exponential backoff
    for attempt in range(retry_count):
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            # Extract embedding vector
            embedding = response.data[0].embedding
            # Verify dimension
            if len(embedding) != 1536:
                raise ValueError(f"Expected 1536-dim embedding, got {len(embedding)}")
            return embedding

        except Exception as e:
            error_type = type(e).__name__
            is_transient = any(x in str(e).lower() for x in ["rate limit", "timeout", "connection", "503", "429"])

            if attempt < retry_count - 1 and is_transient:
                # Exponential backoff: 1s, 2s, 4s
                sleep_time = backoff_base * (2 ** attempt)
                print(f"[embedding_utils] Transient error ({error_type}), retry {attempt+1}/{retry_count} after {sleep_time}s: {e}")
                time.sleep(sleep_time)
                continue
            else:
                # Non-transient or final retry
                raise Exception(f"OpenAI API error after {attempt+1} attempts: {e}")

    return None  # Should not reach here

def embed_text_with_config(text: str, config: Dict[str, Any]) -> List[float]:
    """
    Embed text using model specified in config.

    Args:
        text: Input text to embed
        config: Configuration dict with 'embedding.model' and 'embedding.dim'

    Returns:
        Embedding vector (list of floats)

    Raises:
        ValueError: If model unknown or config invalid
    """
    embedding_cfg = config.get("embedding", {})
    model = embedding_cfg.get("model", "hashlex-v1")
    dim = int(embedding_cfg.get("dim") or 0)

    if model == "hashlex-v1":
        if dim <= 0:
            raise ValueError("embedding.dim must be positive for hashlex-v1")
        return embed_text(text, dim)

    elif model == "ada-002":
        if dim != 1536:
            raise ValueError(f"ada-002 requires dim=1536, got {dim}")
        result = embed_text_ada002(text)
        if result is None:
            raise RuntimeError("Failed to generate ada-002 embedding")
        return result

    else:
        raise ValueError(f"Unknown embedding model: {model}. Supported: hashlex-v1, ada-002")

# Backward compatibility: preserve original embed_text() signature
# (no changes to lines 65-66)
```

**Rationale**:
- Lazy-load OpenAI client to avoid breaking systems without the package
- Exponential backoff matches `scripts/common.py:238-245` pattern
- Clear error messages distinguish missing package vs. missing API key
- `embed_text()` unchanged for hashlex-v1 backward compatibility
- New `embed_text_with_config()` for model-agnostic usage

### Success Criteria

#### Automated Verification:
- [ ] Script imports successfully with and without openai package: `python -c "import embedding_utils"`
- [ ] hashlex-v1 still works: `python -c "from embedding_utils import embed_text; v = embed_text('test', 768); assert len(v) == 768"`
- [ ] ada-002 fails gracefully without API key: `python -c "from embedding_utils import embed_text_ada002; embed_text_ada002('test')"` (should raise ValueError)
- [ ] Type checking passes (if using mypy): `mypy scripts/embedding_utils.py`

#### Manual Verification:
- [ ] ada-002 works with valid OPENAI_API_KEY: Set key, call `embed_text_ada002('test')`, verify 1536-dim vector
- [ ] Retry logic triggers on simulated rate limit (requires mocking or actual rate limit)
- [ ] Error messages are helpful and actionable

---

## Phase 2: Configuration Update

### Overview
Update configuration file to support model selection and new dimension.

### Changes Required

#### 1. `configs/vector.indexing.yaml`

**File**: `configs/vector.indexing.yaml`

**Changes**: Update embedding section

```yaml
embedding:
  model: ada-002             # Changed from: hashlex-v1
  dim: 1536                  # Changed from: 768
  batch_size: 256            # Unchanged (can adjust for API rate limits)
  api_provider: openai       # New: for future multi-provider support
  notes: OpenAI ada-002 text embeddings for high-quality semantic retrieval

  # Fallback for development/testing
  # To use hashlex-v1: change model to "hashlex-v1" and dim to 768

faiss:
  type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128
  # Note: FAISS index dimension is inferred from embedding dim (1536)

pinecone:
  index_name: demo-index
  namespace: default
  metric: cosine
  notes: simulated manifest only (no network)

weaviate:
  class_name: Doc
  notes: schema-only manifest (simulated)
```

**Rationale**:
- Explicit `api_provider` field for future extensibility
- Comments guide users on how to switch back to hashlex-v1
- FAISS parameters unchanged (dimension inferred from embeddings)

### Success Criteria

#### Automated Verification:
- [ ] YAML parses successfully: `python -c "import yaml; yaml.safe_load(open('configs/vector.indexing.yaml'))"`
- [ ] Dimension extraction works: `python -c "import yaml; cfg = yaml.safe_load(open('configs/vector.indexing.yaml')); assert cfg['embedding']['dim'] == 1536"`
- [ ] Model extraction works: `python -c "import yaml; cfg = yaml.safe_load(open('configs/vector.indexing.yaml')); assert cfg['embedding']['model'] == 'ada-002'"`

#### Manual Verification:
- [ ] Configuration comments are clear and actionable
- [ ] Switching to hashlex-v1 works (change model + dim, run Gate-1)

---

## Phase 3: Update Gate Scripts

### Overview
Update all scripts that use embeddings to support multi-model configuration.

### Changes Required

#### 1. `scripts/qa_step01_embeddings.py` (Gate-1)

**File**: `scripts/qa_step01_embeddings.py`

**Changes**: Use `embed_text_with_config()` instead of `embed_text()`

```python
# Line 13: Update import
from embedding_utils import embed_text, embed_text_with_config

# Line 115: Read full config instead of just dim
def main():
    ensure_dir(VEC_DIR)
    # OLD: dim = read_yaml_dim(CONF)
    # NEW: Load full config
    with open(CONF, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Validate config
    embedding_cfg = config.get("embedding", {})
    model = embedding_cfg.get("model")
    dim = int(embedding_cfg.get("dim") or 0)
    if not model or dim <= 0:
        raise ValueError(f"Invalid embedding config: model={model}, dim={dim}")

    print(f"[qa_step01] Using embedding model: {model}, dim: {dim}")

    baseline_chunks = load_baseline_chunks()
    # ... existing chunk loading code ...

    # Line 138: Update embedding call
    # OLD: v = embed_text(text, dim)
    # NEW: Use config-based embedding
    v = embed_text_with_config(text, config)

    # ... rest unchanged ...
```

**Validation Updates** (lines 159-183):
- G1-02 dimension check now uses config dim (1536 for ada-002)
- Add check for correct model in output metadata

#### 2. `scripts/qa_step02_indexes.py` (Gate-2)

**File**: `scripts/qa_step02_indexes.py`

**Changes**: Support 1536-dim vectors, update validation

```python
# Line 117: Dimension inference (no change needed - infers from data)
dim = len(vecs[0]) if vecs else int(cfg.get("embedding", {}).get("dim") or 0)

# Line 123: FAISS index creation (no change - dim is dynamic)
idx = faiss.IndexHNSWFlat(dim, M, metric)

# Validation checks (lines 191-206): Update expected dimension based on config
# Check that inferred dim matches config
cfg_dim = int(cfg.get("embedding", {}).get("dim") or 0)
if dim != cfg_dim:
    print(f"[WARNING] Dimension mismatch: vectors={dim}, config={cfg_dim}")
```

**Rationale**: Gate-2 already infers dimension from data, so minimal changes needed. Add warning if mismatch detected.

#### 3. `scripts/qa_step04_router.py` (Gate-4)

**File**: `scripts/qa_step04_router.py`

**Changes**: Remove hardcoded dim, fix offline mode

```python
# Lines 214-221: Remove hardcoded dimension
# OLD:
#   dim = 768
#   try:
#       cfg = load_yaml(os.path.join("configs", "vector.indexing.yaml"))
#       dim = int(((cfg or {}).get("embedding") or {}).get("dim") or 768)
#   except Exception:
#       dim = 768

# NEW:
cfg = load_yaml(os.path.join("configs", "vector.indexing.yaml"))
dim = int(((cfg or {}).get("embedding") or {}).get("dim") or 768)
if dim <= 0:
    raise ValueError("Invalid embedding dimension in config")

# Lines 224-240: Replace random hash_vec with proper embedding
# REMOVE custom hash_vec() and embed_query() functions entirely

# NEW: Use embeddings from Parquet file (pre-generated by Gate-1)
if use_offline:
    import pyarrow.parquet as pq
    # Load pre-generated embeddings from Gate-1
    emb_path = os.path.join("data", "vector", "embeddings", "embeddings.parquet")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Embeddings not found: {emb_path}. Run Gate-1 first.")

    t = pq.read_table(emb_path)
    vectors = [row["vector"] for row in t.to_pylist()]
    dim = len(vectors[0]) if vectors else dim
    print(f"[qa_step04] Loaded {len(vectors)} pre-generated embeddings (dim={dim})")

    # Load chunk metadata for mapping
    # ... existing chunk loading code ...

    # For queries, use embed_text_with_config
    from embedding_utils import embed_text_with_config
    def embed_query(q: str) -> List[float]:
        return embed_text_with_config(q, cfg)
```

**Rationale**:
- Eliminates the root cause of 52.17% recall (random embeddings)
- Uses pre-generated embeddings from Gate-1 (same as production)
- Falls back to config-based embedding for queries

#### 4. `scripts/qa_step07_retrieval_eval.py` (Gate-7)

**File**: `scripts/qa_step07_retrieval_eval.py`

**Changes**: Update offline mode to use proper embeddings

```python
# Line 209: Dimension loading (keep fallback for safety)
dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)

# Lines 232-248: Update offline embedding
if use_offline:
    # Load config for model selection
    cfg = load_yaml(os.path.join("configs", "vector.indexing.yaml"))

    # OLD: from embedding_utils import embed_text as _embed_text
    # NEW:
    from embedding_utils import embed_text_with_config as _embed_text_cfg

    def embed_query(q: str, d: int) -> List[float]:
        # Use config-based embedding instead of fixed hashlex
        return _embed_text_cfg(q, cfg)

    # ... rest of offline mode setup ...

    # Line 248: Update chunk embedding
    # OLD: vectors.append(_embed_text(j.get('text') or '', dim))
    # NEW:
    vectors.append(_embed_text_cfg(j.get('text') or '', cfg))
```

**Rationale**: Ensures offline mode uses the same embedding model as online mode.

#### 5. `scripts/run_graph.py` (LangGraph Execution)

**File**: `scripts/run_graph.py`

**Changes**: Fix offline mode to use proper embeddings

```python
# Line 297: Load dimension and config
cfg_path = os.path.join("configs", "vector.indexing.yaml")
cfg = load_yaml(cfg_path)
dim = int(((cfg or {}).get("embedding") or {}).get("dim") or 768)

# Lines 300-322: REMOVE custom hash_vec() and embed_query() functions

# Lines 294-337: Replace offline mode with Parquet loading
if use_offline:
    import pyarrow.parquet as pq
    import numpy as np

    # Load pre-generated embeddings from Gate-1
    emb_path = os.path.join("data", "vector", "embeddings", "embeddings.parquet")
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Embeddings file not found: {emb_path}\n"
            f"Please run Gate-1 first: conda run -n age python scripts/qa_step01_embeddings.py"
        )

    t = pq.read_table(emb_path)
    rows_emb = t.to_pylist()

    # Build vectors matrix
    vectors = [row["vector"] for row in rows_emb]
    xb = np.array(vectors, dtype="float32")
    dim_actual = xb.shape[1]

    if dim_actual != dim:
        print(f"[WARNING] Dimension mismatch: embeddings={dim_actual}, config={dim}")
        dim = dim_actual

    print(f"[run_graph] Loaded {len(vectors)} pre-generated embeddings (dim={dim})")

    # Map chunk_id to index
    chunk_to_idx = {row["chunk_id"]: i for i, row in enumerate(rows_emb)}

    # For queries, use embed_text_with_config
    from embedding_utils import embed_text_with_config

    def embed_query(q: str, d: int) -> np.ndarray:
        v = embed_text_with_config(q, cfg)
        return np.array(v, dtype="float32").reshape(1, -1)

    # ... rest of offline mode setup ...
```

**Rationale**:
- Eliminates random embeddings (root cause of low recall)
- Uses Gate-1 outputs (consistent with production)
- Clear error message if embeddings missing
- Dimension validation warns on mismatches

### Success Criteria

#### Automated Verification:
- [ ] Gate-1 completes without errors: `conda run -n age python scripts/qa_step01_embeddings.py`
- [ ] Gate-2 builds FAISS index successfully: `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
- [ ] Gate-4 runs without errors: `conda run -n age python scripts/qa_step04_router.py`
- [ ] Gate-7 evaluation runs: `conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py`
- [ ] run_graph.py offline mode loads embeddings: `conda run -n age python scripts/run_graph.py --offline --persona sales --company Acme`
- [ ] No hardcoded "768" remains in scripts: `grep -n "dim = 768" scripts/*.py` (should be empty)

#### Manual Verification:
- [ ] Embeddings.parquet has 1536-dim vectors
- [ ] FAISS index dimension is 1536
- [ ] Offline mode no longer uses random vectors
- [ ] Error messages guide users to run Gate-1 if embeddings missing

---

## Phase 4: API Key Management & Documentation

### Overview
Set up environment variable handling and document the API key requirement.

### Changes Required

#### 1. Create `.env.example` Template

**File**: `.env.example` (new file in repo root)

**Content**:
```bash
# OpenAI API Key (required for ada-002 embeddings)
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-...

# Optional: Override HTTP request settings
# AR_USER_AGENT=AccountResearchMVP/1.0
# AR_GLOBAL_RPS=6
```

**Rationale**: Standard practice to provide example .env file (actual .env is gitignored)

#### 2. Update CLAUDE.md Documentation

**File**: `CLAUDE.md`

**Changes**: Add section on OpenAI API key

```markdown
## Environment Setup

### API Keys

**OpenAI API Key** (required for ada-002 embeddings):

Create a `.env` file in the repository root:
```bash
OPENAI_API_KEY=sk-your-key-here
```

Or export as environment variable:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

Get your API key from: https://platform.openai.com/api-keys

**Note**: The `age` conda environment does not include the `openai` Python package by default. Install it manually:
```bash
conda run -n age pip install openai>=1.54.3
```

To revert to hashlex-v1 (no API key required):
- Edit `configs/vector.indexing.yaml`
- Change `model: ada-002` to `model: hashlex-v1`
- Change `dim: 1536` to `dim: 768`
- Re-run Gate-1 and Gate-2
```

#### 3. Update Environment Variable Section in CLAUDE.md

**File**: `CLAUDE.md` (lines 348-365)

**Changes**: Add OPENAI_API_KEY to the list

```markdown
## Important Environment Variables

### OpenAI API Configuration
- `OPENAI_API_KEY`: OpenAI API key for ada-002 embeddings (required if using ada-002 model)

### Gate-7 (Retrieval Evaluation)
- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating (recommended for initial runs)
...
```

#### 4. Update Quick Start in CLAUDE.md

**File**: `CLAUDE.md` (Quick Start section)

**Changes**: Add API key setup step

```markdown
## Quick Start

1. **Create environments**:
   ```bash
   conda env create -f envs/age.yaml
   conda env create -f envs/ageFaiss.yaml
   ```

2. **Install OpenAI package** (for ada-002 embeddings):
   ```bash
   conda run -n age pip install openai>=1.54.3
   ```

3. **Set up API key**:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

4. **Build embeddings and indexes**:
   ```bash
   conda run -n age python scripts/qa_step01_embeddings.py
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```
...
```

### Success Criteria

#### Automated Verification:
- [ ] .env.example file exists and has correct format
- [ ] Documentation builds/renders correctly (if using doc generator)

#### Manual Verification:
- [ ] Following Quick Start from scratch works
- [ ] Error message when OPENAI_API_KEY missing is helpful
- [ ] Instructions for reverting to hashlex-v1 are clear

---

## Phase 5: Data Regeneration

### Overview
Regenerate all embedding-dependent artifacts with ada-002 (1536 dimensions).

### Changes Required

**WARNING**: This phase will **overwrite** existing data. Back up first:
```bash
tar -czf data_backup_$(date +%Y%m%d_%H%M%S).tar.gz data/
```

#### 1. Clear Existing Embedding Artifacts

**Commands**:
```bash
# Remove old 768-dim embeddings
rm -f data/vector/embeddings/embeddings.parquet
rm -f data/vector/embeddings/embedding_stats.json

# Remove old FAISS indexes
rm -f data/vector/faiss/index.faiss
rm -f data/vector/faiss/idmap.parquet
rm -f data/vector/faiss/faiss_manifest.json
rm -f data/final/reports/index_health.json

# Remove old gate reports
rm -f reports/qa/step01_embeddings.*
rm -f reports/qa/step02_indexes.*
```

#### 2. Regenerate Embeddings (Gate-1)

**Command**:
```bash
conda run -n age python scripts/qa_step01_embeddings.py
```

**Expected Output**:
- `data/vector/embeddings/embeddings.parquet` with 1536-dim vectors
- `reports/qa/step01_embeddings.json` with G1-02 check showing dim=1536

**Validation**:
```python
import pyarrow.parquet as pq
t = pq.read_table("data/vector/embeddings/embeddings.parquet")
sample_vector = t.to_pylist()[0]["vector"]
assert len(sample_vector) == 1536, f"Expected 1536-dim, got {len(sample_vector)}"
print(f"✓ Embeddings regenerated: {t.num_rows} chunks, dim={len(sample_vector)}")
```

**Cost Estimate**: ~$0.16 for ~1600 chunks (~400K tokens at $0.0001/1K tokens)

#### 3. Rebuild FAISS Index (Gate-2)

**Command**:
```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Expected Output**:
- `data/vector/faiss/index.faiss` with 1536-dim index
- `data/vector/faiss/faiss_manifest.json` with dim=1536

**Validation**:
```python
import faiss
import json

# Check FAISS index dimension
idx = faiss.read_index("data/vector/faiss/index.faiss")
assert idx.d == 1536, f"Expected 1536-dim index, got {idx.d}"
print(f"✓ FAISS index rebuilt: {idx.ntotal} vectors, dim={idx.d}")

# Check manifest
with open("data/vector/faiss/faiss_manifest.json") as f:
    manifest = json.load(f)
    assert manifest["dimension"] == 1536
```

### Success Criteria

#### Automated Verification:
- [ ] Embeddings.parquet exists and has correct schema: `ls -lh data/vector/embeddings/embeddings.parquet`
- [ ] FAISS index exists: `ls -lh data/vector/faiss/index.faiss`
- [ ] Gate-1 report shows dim=1536: `jq '.checks[] | select(.id=="G1-02")' reports/qa/step01_embeddings.json`
- [ ] Gate-2 report shows dim=1536: `jq '.index_stats.dimension' reports/qa/step02_indexes.json`
- [ ] No zero vectors: `jq '.checks[] | select(.id=="G1-03a")' reports/qa/step01_embeddings.json` (should pass)
- [ ] No NaN values: `jq '.checks[] | select(.id=="G1-03b")' reports/qa/step01_embeddings.json` (should pass)

#### Manual Verification:
- [ ] Embedding generation time is reasonable (~5-10 minutes for 1600 chunks with rate limiting)
- [ ] OpenAI API usage shows expected token count (~400K tokens)
- [ ] Backup of old data exists and can be restored if needed

---

## Phase 6: End-to-End Validation

### Overview
Run full quality gate pipeline to validate the migration.

### Changes Required

No code changes needed - just run the gates in sequence.

#### 1. Validate MCP Service (Gate-3)

**Command**:
```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**Expected**:
- All MCP tools respond (kb.search, web.fetch, link.resolve, crm.lookup, safety.check)
- kb.search uses 1536-dim vectors from embeddings.parquet

**Validation Check**: MCP service should load embeddings successfully and respond to queries.

#### 2. Validate Router (Gate-4)

**Command**:
```bash
conda run -n age python scripts/qa_step04_router.py
```

**Expected**:
- Router loads 1536-dim embeddings from Parquet (no random vectors)
- Keyword routing works correctly
- Offline mode uses proper embeddings

#### 3. Validate LangGraph (Gate-5)

**Command**:
```bash
conda run -n age python scripts/qa_step05_graph.py
```

**Expected**: Graph execution completes without dimension errors.

#### 4. Validate A2A Compliance (Gate-6)

**Command**:
```bash
conda run -n age python scripts/qa_step06_a2a.py --session-id test_ada002
```

**Expected**: Agent-to-agent handoffs work with new embeddings.

#### 5. Validate Retrieval Quality (Gate-7)

**Command**:
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Expected Improvement**:
- **Baseline (hashlex-v1 with random offline mode)**: 52.17% recall
- **Target (ada-002 with proper embeddings)**: >80% recall
- **Metric**: recall@10, nDCG@5

**Key Result**: This is the ultimate validation that the migration succeeded.

#### 6. Validate Generation Quality (Gate-8)

**Command**:
```bash
conda run -n age python scripts/qa_step08_generation_eval.py
```

**Expected**:
- All structural checks pass (G8-01, G8-02)
- Length/readability passes (G8-03)
- Persona keyword hits improve (G8-04)

**Validation**: Check `reports/qa/step08_generation_eval.md` for GREEN status.

### Success Criteria

#### Automated Verification:
- [ ] Gate-3 passes: All MCP tools respond
- [ ] Gate-4 passes: Router heuristics work correctly
- [ ] Gate-5 passes: LangGraph completes without errors
- [ ] Gate-6 passes: A2A compliance checks pass
- [ ] Gate-7 shows improved recall: recall@10 > 0.80 (vs. 0.5217 baseline)
- [ ] Gate-8 passes: All thresholds met (G8-01, G8-02, G8-03, G8-04)

#### Manual Verification:
- [ ] Retrieval failures.jsonl shows semantically relevant misses (not random)
- [ ] Generated emails use insights from correct documents
- [ ] No dimension mismatch warnings in logs
- [ ] Offline mode in run_graph.py works correctly
- [ ] System feels "smarter" - retrieves semantically similar documents even with different wording

---

## Testing Strategy

### Unit Tests

**New Test File**: `scripts/test_embedding_utils.py`

```python
import pytest
import os
from embedding_utils import (
    embed_text,
    embed_text_ada002,
    embed_text_with_config,
    hashlex_embed,
    tokenize
)

def test_hashlex_v1_unchanged():
    """Verify hashlex-v1 still works exactly as before."""
    text = "salesforce agentforce platform"
    v1 = embed_text(text, 768)
    assert len(v1) == 768
    assert all(isinstance(x, float) for x in v1)
    # Deterministic
    v2 = embed_text(text, 768)
    assert v1 == v2

def test_ada002_dimension():
    """Verify ada-002 returns 1536-dim vectors."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    text = "salesforce agentforce platform"
    v = embed_text_ada002(text)
    assert len(v) == 1536
    assert all(isinstance(x, float) for x in v)

def test_ada002_missing_key():
    """Verify clear error when API key missing."""
    old_key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            embed_text_ada002("test")
    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key

def test_config_based_hashlex():
    """Verify config-based embedding with hashlex-v1."""
    config = {"embedding": {"model": "hashlex-v1", "dim": 768}}
    v = embed_text_with_config("test", config)
    assert len(v) == 768

def test_config_based_ada002():
    """Verify config-based embedding with ada-002."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    config = {"embedding": {"model": "ada-002", "dim": 1536}}
    v = embed_text_with_config("test", config)
    assert len(v) == 1536

def test_unknown_model():
    """Verify error on unknown model."""
    config = {"embedding": {"model": "unknown", "dim": 768}}
    with pytest.raises(ValueError, match="Unknown embedding model"):
        embed_text_with_config("test", config)
```

**Run Tests**:
```bash
conda run -n age pytest scripts/test_embedding_utils.py -v
```

### Integration Tests

**Manual Test Scenarios**:

1. **End-to-End Pipeline**:
   ```bash
   # Clean slate
   rm -rf data/vector/embeddings/* data/vector/faiss/*
   # Run pipeline
   conda run -n age python scripts/qa_step01_embeddings.py
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py
   # Check recall improved
   jq '.metrics.recall' reports/qa/step07_retrieval_eval.json
   ```

2. **Offline Mode Test**:
   ```bash
   # Run graph in offline mode
   conda run -n age python scripts/run_graph.py --offline --persona sales --company Acme
   # Verify uses proper embeddings (not random)
   grep "Loaded.*pre-generated embeddings" logs/*.log
   ```

3. **Model Switching Test**:
   ```bash
   # Switch back to hashlex-v1
   sed -i '' 's/model: ada-002/model: hashlex-v1/' configs/vector.indexing.yaml
   sed -i '' 's/dim: 1536/dim: 768/' configs/vector.indexing.yaml
   # Regenerate
   conda run -n age python scripts/qa_step01_embeddings.py
   # Verify dim=768
   python -c "import pyarrow.parquet as pq; t = pq.read_table('data/vector/embeddings/embeddings.parquet'); print(f'dim={len(t.to_pylist()[0][\"vector\"])}')"
   ```

### Manual Testing Steps

1. **API Key Validation**:
   - Unset OPENAI_API_KEY → expect clear error message
   - Set invalid key → expect authentication error with retry
   - Set valid key → expect successful embedding

2. **Rate Limiting**:
   - Run Gate-1 with 1600 chunks → observe retry messages on rate limit (429 errors)
   - Verify exponential backoff (1s, 2s, 4s delays)

3. **Semantic Similarity**:
   - Query: "agentic AI capabilities"
   - Expected results: Documents about Agentforce, AI agents, autonomous systems
   - Compare with hashlex-v1: Verify ada-002 finds more relevant results

4. **Dimension Consistency**:
   - Check all vectors in embeddings.parquet have 1536 dimensions
   - Check FAISS index has dimension 1536
   - Check no dimension mismatch warnings in Gate-7

## Performance Considerations

### API Cost Estimation

**OpenAI Pricing** (as of October 2025):
- ada-002: $0.0001 per 1K tokens
- Average chunk size: ~250 tokens
- Current corpus: ~1600 chunks

**Total Cost**:
- Embedding generation (one-time): 1600 chunks × 250 tokens = 400K tokens = **$0.04**
- Gate-7 evaluation queries: 23 queries × ~10 tokens = 230 tokens = **$0.00002**
- **Total for full migration**: ~$0.04

**Ongoing Costs**:
- Per new document: ~$0.00002 per chunk
- Per query: ~$0.000001 per query
- Negligible for evaluation/testing workloads

### Rate Limiting

**OpenAI Rate Limits** (Tier 1):
- 3,500 requests per minute (RPM)
- 200,000 tokens per minute (TPM)

**Gate-1 Throughput**:
- Batch size: 256 chunks (from config)
- Sequential processing (current implementation)
- Expected time: ~3-5 minutes for 1600 chunks (well under rate limits)

**Optimization Opportunity** (future):
- Implement batch embedding API calls (up to 2048 inputs per request)
- Reduce API calls from 1600 to ~1 (massive speedup)
- Current implementation is conservative (one request per chunk)

### Latency Comparison

| Model | Embedding Time (per chunk) | Total Time (1600 chunks) |
|-------|---------------------------|-------------------------|
| hashlex-v1 | ~1ms (local) | ~2 seconds |
| ada-002 (sequential) | ~100ms (API latency) | ~160 seconds (~3 min) |
| ada-002 (batched, future) | ~100ms (batch of 256) | ~1 second |

**Impact on Gates**:
- Gate-1: 2s → 3min (one-time regeneration, acceptable)
- Gate-7 queries: 23 queries × 100ms = 2.3s (acceptable)
- Online retrieval: Negligible if using pre-generated embeddings

## Migration Notes

### Rollback Strategy

If the migration fails or ada-002 doesn't improve recall:

1. **Restore Config**:
   ```bash
   git checkout configs/vector.indexing.yaml  # Revert to hashlex-v1
   ```

2. **Restore Data** (if backed up):
   ```bash
   tar -xzf data_backup_YYYYMMDD_HHMMSS.tar.gz
   ```

3. **Or Regenerate with hashlex-v1**:
   ```bash
   # Update config manually
   sed -i '' 's/model: ada-002/model: hashlex-v1/' configs/vector.indexing.yaml
   sed -i '' 's/dim: 1536/dim: 768/' configs/vector.indexing.yaml

   # Regenerate
   conda run -n age python scripts/qa_step01_embeddings.py
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```

### Data Preservation

**What to back up before migration**:
- `data/vector/embeddings/` - Current 768-dim embeddings
- `data/vector/faiss/` - Current FAISS indexes
- `reports/qa/step0*.json` - Baseline gate reports

**Backup command**:
```bash
tar -czf data_backup_hashlex_v1_$(date +%Y%m%d_%H%M%S).tar.gz \
    data/vector/embeddings/ \
    data/vector/faiss/ \
    reports/qa/
```

### Version Control

**Recommended Git workflow**:

1. **Create feature branch**:
   ```bash
   git checkout -b feature/ada-002-embeddings
   ```

2. **Commit by phase**:
   ```bash
   git commit -m "Phase 1: Add ada-002 support to embedding_utils.py"
   git commit -m "Phase 2: Update config for ada-002"
   git commit -m "Phase 3: Update gate scripts for multi-model support"
   # etc.
   ```

3. **Tag before data regeneration**:
   ```bash
   git tag -a pre-ada002-migration -m "Checkpoint before data regeneration"
   ```

4. **Merge after validation**:
   ```bash
   # After Gate-7 shows >80% recall
   git checkout main
   git merge feature/ada-002-embeddings
   ```

## References

### Original Context
- Research document: `thoughts/shared/research/2025-10-06-embedding-model-architecture.md`
- Issue: `thoughts/shared/issues/issue001.md`
- User request: "change the embedding model to: OpenAI ada-002 (1536-dim, requires API key, highest quality)"

### Related Documentation
- OpenAI Embeddings API: https://platform.openai.com/docs/guides/embeddings
- ada-002 Model Card: https://platform.openai.com/docs/models/embeddings
- FAISS Documentation: https://github.com/facebookresearch/faiss
- LangChain OpenAI Integration: https://python.langchain.com/docs/integrations/text_embedding/openai

### Codebase Documentation
- Main docs: `README.md`
- Agent guidelines: `AGENTS.md`
- Environment setup: `docs/envs.md`
- This file: Claude Code guidance for implementation

## Notes for Implementation

### Critical Success Factors

1. **Don't break hashlex-v1**: Preserve backward compatibility for testing/development
2. **Fix offline mode bugs**: This is as important as the embedding upgrade
3. **Clear error messages**: Guide users when API key missing or quota exceeded
4. **Validate at each phase**: Don't proceed to next phase until current phase passes
5. **Measure recall improvement**: Gate-7 is the ultimate success metric

### Common Pitfalls to Avoid

1. **Forgetting to install openai package**: Document clearly, provide helpful error
2. **Rate limit exhaustion**: Implement retry with backoff, don't hammer API
3. **Dimension mismatches**: Validate config vs. data at every gate
4. **Offline mode still using random vectors**: Load from Parquet, not re-generate
5. **Missing API key in CI/CD**: Make ada-002 optional, fall back to hashlex-v1

### Expected Outcomes

**Before Migration** (hashlex-v1 with broken offline mode):
- recall@10: 52.17% (12/23 queries)
- Offline mode: Random embeddings (broken)
- Semantic queries: Poor (lexical overlap only)

**After Migration** (ada-002 with fixed offline mode):
- recall@10: **>80%** (target: 19+/23 queries)
- Offline mode: Proper embeddings from Parquet (fixed)
- Semantic queries: Excellent (semantic similarity)

**Key Improvements**:
1. **+28% absolute recall improvement** (52% → 80%)
2. **Fixed offline mode bug** (random → semantic)
3. **Semantic understanding** (synonyms, paraphrases work)
4. **Future-proof architecture** (easy to add more models)
