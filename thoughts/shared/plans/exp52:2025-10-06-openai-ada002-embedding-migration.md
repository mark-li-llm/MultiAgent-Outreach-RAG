---
date: 2025-10-06T17:30:00-04:00
author: Claude Code
topic: "OpenAI ada-002 Embedding Migration"
tags: [implementation, embeddings, openai, ada-002, migration, gate-1, gate-2]
status: ready_for_review
related_research: thoughts/shared/research/2025-10-06-embedding-model-architecture.md
related_issue: thoughts/shared/issues/issue001.md
---
# this is same to exp5 and with ultrathink


# OpenAI ada-002 Embedding Migration Implementation Plan

## Overview

Migrate the system from hashlex-v1 (768-dim deterministic hash-based embeddings) to OpenAI ada-002 (1536-dim API-based embeddings) to improve retrieval quality. This is a complete replacement - we will remove hashlex-v1 and use only OpenAI ada-002.

**Goal**: Improve recall@10 from current ~52% to target e80% by using production-quality embeddings.

## Current State Analysis

### Existing Implementation

**Embedding Model**: hashlex-v1 (deterministic, stateless, 768-dim)
- Location: `scripts/embedding_utils.py`
- Process: Text normalization ’ Tokenization (unigrams + bigrams) ’ Feature hashing (FNV-1a) ’ L2 normalization
- Properties: No API calls, no dependencies, deterministic, but limited semantic understanding

**Current Dimension**: 768 (configured in `configs/vector.indexing.yaml:3`)

**Files Using `embed_text()`**:
1. `scripts/qa_step01_embeddings.py:13,138` - Document chunk embedding (Gate-1)
2. `scripts/qa_step02_indexes.py:11,218` - Query embedding for index verification (Gate-2)
3. `scripts/qa_step03_mcp.py:71,73` - Query embedding for kb.search (Gate-3/MCP)
4. `scripts/qa_step07_retrieval_eval.py:233,386` - Query embedding for retrieval evaluation (Gate-7)

**Data Scale**: ~1.6k chunks indexed in FAISS

**Problem**: Low recall (52.17%) due to limited semantic understanding of hash-based embeddings

### Key Discoveries

From research document (`thoughts/shared/research/2025-10-06-embedding-model-architecture.md`):
- All components use same `embed_text(text, dim)` interface
- Dimension flows from config through entire pipeline
- Critical invariant: All embeddings in same vector space MUST use identical dim values
- FAISS dimension inferred from vector shape at Gate-2 (line 117)

## Desired End State

**Embedding Model**: OpenAI ada-002
- API-based, 1536-dimensional vectors
- Production-quality semantic embeddings
- Requires `OPENAI_API_KEY` environment variable

**Updated Dimension**: 1536 (configured in `configs/vector.indexing.yaml:3`)

**Interface Preserved**: `embed_text(text, dim)` remains unchanged for all consumers

**Validation Criteria**:
- All 4 gates pass (Gate-1, Gate-2, Gate-3, Gate-7)
- Recall@10 e 80% (up from ~52%)
- All ~1.6k chunks successfully re-embedded
- FAISS index rebuilt with 1536-dim vectors

## What We're NOT Doing

- NOT keeping hashlex-v1 as fallback (complete replacement)
- NOT implementing retry logic (fail fast on API errors)
- NOT modifying embedding interface (keep `embed_text(text, dim)`)
- NOT changing chunking, deduplication, or other pipeline stages
- NOT migrating to different embedding models (e.g., Cohere, Hugging Face)
- NOT implementing streaming embeddings (batch mode only)
- NOT modifying FAISS index parameters (M, efConstruction, efSearch)

## Implementation Approach

**Strategy**: Replace embedding implementation in-place while preserving the same interface, then re-run Gates 1-7 sequentially.

**Key Principles**:
1. **Fail Fast**: No retry logic - fail immediately on API errors with clear diagnostics
2. **Cost Transparency**: Show estimated cost before making API calls
3. **Progress Visibility**: Display progress during long embedding runs
4. **Vector Space Consistency**: Ensure all embeddings use identical dimension (1536)
5. **Validation at Each Gate**: Verify success before proceeding to next gate

---

## Phase 1: Environment Setup

### Overview
Add OpenAI Python library to the `age` conda environment and verify API key access.

### Changes Required

#### 1. Update Environment Definition
**File**: `envs/age.yaml`

Add `openai` package to dependencies:

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
  # IMPORTANT: Do NOT install pip faiss-cpu in this env to avoid duplicate libomp.
```

**Rationale**: OpenAI library not available via conda-forge, must use pip within conda environment.

#### 2. Recreate Environment
```bash
# Remove existing age environment
/Users/liyunxiao/anaconda3/bin/conda env remove -n age

# Recreate from updated YAML
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
```

#### 3. Set API Key
```bash
# Add to your shell profile (~/.zshrc or ~/.bashrc)
export OPENAI_API_KEY="sk-..."

# Or set temporarily for this session
export OPENAI_API_KEY="sk-..."
```

### Success Criteria

#### Automated Verification:
- [ ] Environment recreates successfully: `/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml`
- [ ] OpenAI package installed: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import openai; print(openai.__version__)"`
- [ ] API key accessible: `echo $OPENAI_API_KEY | grep -q "sk-" && echo "PASS" || echo "FAIL"`

#### Manual Verification:
- [ ] OpenAI client can be instantiated without errors
- [ ] API key is valid and has credits available

---

## Phase 2: Update Embedding Implementation

### Overview
Replace hashlex-v1 implementation in `scripts/embedding_utils.py` with OpenAI ada-002 API calls, preserving the same interface.

### Changes Required

#### 1. Replace Embedding Function
**File**: `scripts/embedding_utils.py`

**Complete File Replacement**:

```python
#!/usr/bin/env python3
"""
Embedding utilities for OpenAI ada-002 model.

This module provides text embedding via OpenAI's text-embedding-ada-002 model,
which generates 1536-dimensional vectors optimized for semantic similarity.

Critical: All documents and queries MUST use this same embed_text() function
to ensure they exist in the same vector space.
"""
import os
import sys
from typing import List


def embed_text(text: str, dim: int) -> List[float]:
    """
    Generate text embedding using OpenAI ada-002 model.

    Args:
        text: Input text to embed
        dim: Expected dimension (must be 1536 for ada-002)

    Returns:
        List of float32 values representing the embedding vector

    Raises:
        ValueError: If dim != 1536 or API key missing
        RuntimeError: If OpenAI API call fails
    """
    # Validate dimension
    if dim != 1536:
        raise ValueError(
            f"OpenAI ada-002 requires dim=1536, got dim={dim}. "
            f"Update configs/vector.indexing.yaml to set embedding.dim=1536"
        )

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Set it with: export OPENAI_API_KEY='sk-...'"
        )

    # Import OpenAI library
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError(
            "OpenAI library not installed. "
            "Install with: conda run -n age pip install openai>=1.0.0"
        )

    # Initialize client
    client = OpenAI(api_key=api_key)

    # Handle empty input
    if not (text or "").strip():
        # Return zero vector for empty text (will be caught by validation)
        return [0.0] * dim

    # Call OpenAI API (no retry - fail fast)
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
            encoding_format="float"
        )
        embedding = response.data[0].embedding

        # Validate dimension
        if len(embedding) != dim:
            raise RuntimeError(
                f"OpenAI returned {len(embedding)}-dim vector, expected {dim}"
            )

        return embedding

    except Exception as e:
        # Fail fast with clear error message
        raise RuntimeError(
            f"OpenAI API call failed: {type(e).__name__}: {e}\n"
            f"Text length: {len(text)} chars\n"
            f"This is a FATAL error - no retry logic. Fix the issue and re-run."
        ) from e


def estimate_embedding_cost(num_texts: int, avg_text_length: int) -> dict:
    """
    Estimate cost of embedding a corpus.

    OpenAI ada-002 pricing: $0.0001 per 1K tokens (~750 words, ~4000 chars)

    Args:
        num_texts: Number of texts to embed
        avg_text_length: Average text length in characters

    Returns:
        Dict with cost breakdown
    """
    # Rough estimate: 1 token H 4 characters
    tokens_per_text = avg_text_length / 4.0
    total_tokens = num_texts * tokens_per_text
    total_cost_usd = (total_tokens / 1000.0) * 0.0001

    return {
        "num_texts": num_texts,
        "avg_text_length_chars": avg_text_length,
        "estimated_tokens_per_text": int(tokens_per_text),
        "estimated_total_tokens": int(total_tokens),
        "cost_per_1k_tokens_usd": 0.0001,
        "estimated_total_cost_usd": round(total_cost_usd, 4),
        "note": "Estimate only - actual cost may vary based on tokenization"
    }


# Deprecated functions (removed)
# - normalize_text() - no longer needed
# - tokenize() - no longer needed
# - hashlex_embed() - no longer needed
# - _stable_hash() - no longer needed
```

**Rationale**:
- Preserves `embed_text(text, dim)` interface for all consumers
- Validates dim=1536 to prevent configuration errors
- Fails fast on API errors (no retry)
- Provides clear error messages for debugging
- Adds cost estimation utility for transparency

### Success Criteria

#### Automated Verification:
- [ ] Python syntax valid: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -m py_compile scripts/embedding_utils.py`
- [ ] Import succeeds: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from embedding_utils import embed_text, estimate_embedding_cost"`
- [ ] Dimension validation works: Test that `embed_text("test", 768)` raises ValueError
- [ ] Empty text handled: Test that `embed_text("", 1536)` returns 1536-length zero vector

#### Manual Verification:
- [ ] API key validation: Calling without `OPENAI_API_KEY` set raises clear error
- [ ] Successful embedding: `embed_text("Salesforce Agentforce", 1536)` returns 1536-length vector
- [ ] Cost estimator: `estimate_embedding_cost(1600, 500)` returns reasonable cost estimate

---

## Phase 3: Update Configuration

### Overview
Update vector indexing configuration to specify OpenAI ada-002 model with 1536 dimensions.

### Changes Required

#### 1. Update Vector Configuration
**File**: `configs/vector.indexing.yaml`

**Changes**:
```yaml
embedding:
  model: openai-ada-002  # Changed from: hashlex-v1
  dim: 1536              # Changed from: 768
  batch_size: 100        # Changed from: 256 (OpenAI rate limits)
  notes: OpenAI ada-002 text embeddings (requires OPENAI_API_KEY)
  api:
    cost_per_1k_tokens: 0.0001
    max_retries: 0       # Fail fast - no retries
    timeout_seconds: 30

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

**Rationale**:
- Model name documents the change for future reference
- Dimension increased to 1536 per ada-002 specification
- Batch size reduced to respect OpenAI rate limits
- Added API configuration section for transparency

### Success Criteria

#### Automated Verification:
- [ ] YAML syntax valid: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import yaml; yaml.safe_load(open('configs/vector.indexing.yaml'))"`
- [ ] Dimension readable: `grep "dim: 1536" configs/vector.indexing.yaml`
- [ ] Model name updated: `grep "model: openai-ada-002" configs/vector.indexing.yaml`

#### Manual Verification:
- [ ] Configuration loads correctly in Gate-1 script
- [ ] All YAML fields parse without errors

---

## Phase 4: Update Gate-1 (Embedding Generation)

### Overview
Add cost estimation and progress tracking to Gate-1 embedding generation script.

### Changes Required

#### 1. Add Cost Estimation and Progress Tracking
**File**: `scripts/qa_step01_embeddings.py`

**Changes** (add before main embedding loop at line ~115):

```python
def main():
    ensure_dir(VEC_DIR)
    dim = read_yaml_dim(CONF)
    baseline_chunks = load_baseline_chunks()

    # NEW: Pre-flight cost estimation
    print(f"=== Pre-flight Cost Estimation ===")
    # Sample 100 chunks to estimate average length
    sample_lengths = []
    sample_count = 0
    for path in sorted(glob.glob(CHUNK_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    text_len = len(j.get("text") or "")
                    sample_lengths.append(text_len)
                    sample_count += 1
                    if sample_count >= 100:
                        break
                except Exception:
                    continue
        if sample_count >= 100:
            break

    if sample_lengths:
        from embedding_utils import estimate_embedding_cost
        from statistics import median
        avg_len = int(median(sample_lengths))
        cost_estimate = estimate_embedding_cost(baseline_chunks, avg_len)
        print(f"  Total chunks: {baseline_chunks}")
        print(f"  Avg text length: {avg_len} chars")
        print(f"  Estimated tokens: {cost_estimate['estimated_total_tokens']:,}")
        print(f"  Estimated cost: ${cost_estimate['estimated_total_cost_usd']:.4f} USD")
        print(f"  (Note: {cost_estimate['note']})")

        # Confirm before proceeding
        response = input(f"\nProceed with embedding generation? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted by user.")
            sys.exit(0)

    print(f"\n=== Starting Embedding Generation ===")
    print(f"  Model: openai-ada-002")
    print(f"  Dimension: {dim}")
    print(f"  Total chunks: {baseline_chunks}")
    print("")

    rows: List[Dict[str, Any]] = []
    embedding_rows = 0
    zero_vectors = 0
    nan_vectors = 0
    norms: List[float] = []

    # NEW: Progress tracking
    import time
    start_time = time.time()
    last_report = start_time

    for path in sorted(glob.glob(CHUNK_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                chunk_id = j.get("chunk_id") or ""
                doc_id = j.get("doc_id") or ""
                seq_no = j.get("seq_no") or 0
                token_count = j.get("token_count") or 0
                text = j.get("text") or ""

                # Call OpenAI API (will fail fast on error)
                try:
                    v = embed_text(text, dim)
                except Exception as e:
                    print(f"\n FATAL ERROR at chunk {embedding_rows + 1}/{baseline_chunks}")
                    print(f"  Chunk ID: {chunk_id}")
                    print(f"  Error: {e}")
                    print(f"\nEmbedding generation FAILED. Fix the issue and re-run.")
                    sys.exit(1)

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

                # NEW: Progress reporting every 10 chunks or 5 seconds
                now = time.time()
                if embedding_rows % 10 == 0 or (now - last_report) >= 5.0:
                    elapsed = now - start_time
                    rate = embedding_rows / elapsed if elapsed > 0 else 0
                    remaining = (baseline_chunks - embedding_rows) / rate if rate > 0 else 0
                    pct = (embedding_rows / baseline_chunks) * 100 if baseline_chunks > 0 else 0
                    print(f"  Progress: {embedding_rows}/{baseline_chunks} ({pct:.1f}%) | "
                          f"Rate: {rate:.1f} chunks/sec | "
                          f"ETA: {int(remaining)}s", end="\r")
                    last_report = now

    print(f"\n Embedding generation complete: {embedding_rows} chunks")

    # Continue with stats and validation (existing code)...
```

**Rationale**:
- Cost estimation prevents surprise API bills
- User confirmation adds safety gate before API calls
- Progress tracking provides visibility during long runs
- Fail-fast error handling stops immediately on API errors
- Clear error messages aid debugging

### Success Criteria

#### Automated Verification:
- [ ] Script runs without Python syntax errors: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -m py_compile scripts/qa_step01_embeddings.py`
- [ ] Cost estimator called correctly: Check that `estimate_embedding_cost` is imported and used

#### Manual Verification:
- [ ] Cost estimation displays before embedding starts
- [ ] User can abort before API calls if cost too high
- [ ] Progress bar updates every 10 chunks or 5 seconds
- [ ] Final success message shows total chunks embedded

---

## Phase 5: Re-generate Embeddings (Gate-1)

### Overview
Run Gate-1 to generate new OpenAI ada-002 embeddings for all ~1.6k chunks.

### Execution Steps

```bash
# Set API key if not already set
export OPENAI_API_KEY="sk-..."

# Run Gate-1 embedding generation
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py
```

**Expected Output**:
```
=== Pre-flight Cost Estimation ===
  Total chunks: 1600
  Avg text length: 500 chars
  Estimated tokens: 200,000
  Estimated cost: $0.0200 USD
  (Note: Estimate only - actual cost may vary based on tokenization)

Proceed with embedding generation? [y/N]: y

=== Starting Embedding Generation ===
  Model: openai-ada-002
  Dimension: 1536
  Total chunks: 1600

  Progress: 1600/1600 (100.0%) | Rate: 2.5 chunks/sec | ETA: 0s
 Embedding generation complete: 1600 chunks

{
  "status": "GREEN",
  "rows": 1600
}
```

### Success Criteria

#### Automated Verification:
- [ ] Gate-1 exits with status code 0: `echo $?`
- [ ] Parquet file created: `ls -lh data/vector/embeddings/embeddings.parquet`
- [ ] JSON report shows GREEN: `jq -r '.status' reports/qa/step01_embeddings.json`
- [ ] Check G1-01: Row count matches baseline: `jq '.checks[] | select(.id=="G1-01")' reports/qa/step01_embeddings.json`
- [ ] Check G1-02: Dimension is 1536: `jq '.checks[] | select(.id=="G1-02") | .actual' reports/qa/step01_embeddings.json`
- [ ] Check G1-03a: No zero vectors: `jq '.checks[] | select(.id=="G1-03a") | .actual' reports/qa/step01_embeddings.json`
- [ ] Check G1-03b: No NaN vectors: `jq '.checks[] | select(.id=="G1-03b") | .actual' reports/qa/step01_embeddings.json`

#### Manual Verification:
- [ ] Cost estimate appears and is reasonable (< $1.00 for ~1.6k chunks)
- [ ] Progress bar updates smoothly during generation
- [ ] All chunks embedded successfully (no FATAL errors)
- [ ] Markdown report shows GREEN status: `cat reports/qa/step01_embeddings.md`

---

## Phase 6: Rebuild FAISS Index (Gate-2)

### Overview
Rebuild FAISS HNSW index using new 1536-dimensional embeddings.

### Execution Steps

```bash
# Run Gate-2 index build in ageFaiss environment
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Expected Behavior**:
- Loads `data/vector/embeddings/embeddings.parquet`
- Infers dimension from vector shape: `dim = len(vecs[0])` ’ 1536
- Creates FAISS IndexHNSWFlat(1536, M=32, metric=L2)
- Writes `data/vector/faiss/index.faiss` with 1536-dim vectors

### Success Criteria

#### Automated Verification:
- [ ] Gate-2 exits with status code 0: `echo $?`
- [ ] FAISS index created: `ls -lh data/vector/faiss/index.faiss`
- [ ] Manifest shows dim=1536: `jq -r '.dim' data/vector/faiss/faiss_manifest.json`
- [ ] Manifest shows count matches: `jq -r '.count' data/vector/faiss/faiss_manifest.json`
- [ ] JSON report shows GREEN: `jq -r '.status' reports/qa/step02_indexes.json`
- [ ] Check G2-03: FAISS count ratio e 0.98: `jq '.checks[] | select(.id=="G2-03")' reports/qa/step02_indexes.json`
- [ ] Check G2-05: Roundtrip error d 0.001: `jq '.checks[] | select(.id=="G2-05")' reports/qa/step02_indexes.json`

#### Manual Verification:
- [ ] Index rebuild completes without errors
- [ ] Round-trip validation passes (error < 0.001)
- [ ] Markdown report shows GREEN: `cat reports/qa/step02_indexes.md`

---

## Phase 7: Validate MCP Service (Gate-3)

### Overview
Verify that MCP kb.search service works correctly with new 1536-dim embeddings.

### Execution Steps

```bash
# Run Gate-3 MCP health checks
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py
```

**Expected Behavior**:
- Loads chunks and embeds them using new `embed_text()` (1536-dim)
- Matrix shape: `xb.shape = (n_chunks, 1536)`
- Infers dimension: `dim = xb.shape[1]` ’ 1536
- Query embeddings use same dimension

### Success Criteria

#### Automated Verification:
- [ ] Gate-3 exits with status code 0: `echo $?`
- [ ] JSON report shows GREEN: `jq -r '.status' reports/qa/step03_mcp.json`
- [ ] Check G3-01: All 5 health endpoints OK: `jq '.checks[] | select(.id=="G3-01")' reports/qa/step03_mcp.json`
- [ ] Check G3-02: Contract conformance: `jq '.checks[] | select(.id | startswith("G3-02"))' reports/qa/step03_mcp.json`
- [ ] Check G3-04: No timeouts: `jq '.checks[] | select(.id=="G3-04")' reports/qa/step03_mcp.json`

#### Manual Verification:
- [ ] All MCP tools respond to health checks
- [ ] kb.search returns results for test queries
- [ ] Latency within expected bounds
- [ ] Markdown report shows GREEN: `cat reports/qa/step03_mcp.md`

---

## Phase 8: Run Retrieval Evaluation (Gate-7)

### Overview
Run end-to-end retrieval evaluation to measure recall improvement with OpenAI embeddings.

### Execution Steps

```bash
# Run Gate-7 retrieval evaluation (with relaxed latency and no coverage check)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

**Expected Improvement**:
- **Before**: recall@10 H 52% (with hashlex-v1)
- **After**: recall@10 e 80% (target with ada-002)

### Success Criteria

#### Automated Verification:
- [ ] Gate-7 exits with status code 0: `echo $?`
- [ ] JSON report created: `ls -lh reports/qa/step07_retrieval_eval.json`
- [ ] Status is GREEN or AMBER: `jq -r '.status' reports/qa/step07_retrieval_eval.json`
- [ ] Check G7-01: recall@10 e 0.80: `jq '.checks[] | select(.id=="G7-01") | .actual' reports/qa/step07_retrieval_eval.json`
- [ ] Check G7-02: nDCG@5 e 0.60: `jq '.checks[] | select(.id=="G7-02") | .actual' reports/qa/step07_retrieval_eval.json`
- [ ] Retrieval failures logged: `wc -l reports/eval/retrieval_failures.jsonl`
- [ ] Trace created: `wc -l reports/router/step07_retrieval_trace.jsonl`

#### Manual Verification:
- [ ] Recall@10 significantly improved from baseline (~52% ’ e80%)
- [ ] nDCG@5 improved (ranking quality better)
- [ ] Review failure cases in `reports/eval/retrieval_failures.jsonl`
- [ ] Markdown report shows metrics: `cat reports/qa/step07_retrieval_eval.md`

---

## Phase 9: Update run_graph.py (Optional but Recommended)

### Overview
Update `scripts/run_graph.py` to use shared `embed_text()` instead of custom embedding logic.

### Current Issue

**File**: `scripts/run_graph.py` (lines 300-322)

The file currently uses custom `hash_vec()` and `embed_query()` functions that differ from `embed_text()`:

```python
def hash_vec(seed: str, d: int) -> List[float]:
    rnd = random.Random()
    h = 0
    for ch in seed:
        h = (h * 1315423911) ^ ord(ch)
        h &= 0xFFFFFFFFFFFFFFFF
    rnd.seed(h)
    vals = [rnd.uniform(-1.0, 1.0) for _ in range(d)]
    s2 = sum(v*v for v in vals) or 1.0
    inv = 1.0 / math.sqrt(s2)
    return [v*inv for v in vals]
```

This creates **inconsistent embeddings** - documents use ada-002 but queries in graph use random hashing!

### Changes Required

**File**: `scripts/run_graph.py`

**Replace custom embedding functions** (lines 300-322):

```python
    # Offline index setup
    chunks_index: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)
    if use_offline:
        # Import shared embedding function
        from embedding_utils import embed_text as _embed_text

        def embed_query(q: str, d: int) -> List[float]:
            """Use shared embedding function for consistency."""
            return _embed_text(q, d)

        def l2(a: List[float], b: List[float]) -> float:
            return sum((x-y)*(x-y) for x,y in zip(a,b))

        # Load chunks
        for cf in sorted(glob.glob(os.path.join("data","interim","chunks","*.chunks.jsonl"))):
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                        if not j.get("chunk_id"):
                            continue
                        chunks_index.append(j)
                        # Use same embedding function as Gate-1
                        vectors.append(_embed_text(j.get('text') or '', dim))
                    except Exception:
                        continue
```

**Rationale**: Ensures query embeddings in graph execution use same OpenAI ada-002 model as indexed documents.

### Success Criteria

#### Automated Verification:
- [ ] Script compiles: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -m py_compile scripts/run_graph.py`
- [ ] Custom `hash_vec()` removed: `! grep -q "hash_vec" scripts/run_graph.py`
- [ ] Imports `embed_text`: `grep "from embedding_utils import embed_text" scripts/run_graph.py`

#### Manual Verification:
- [ ] Graph execution uses OpenAI embeddings (watch for API calls)
- [ ] No embedding inconsistency errors
- [ ] Graph completes successfully with improved retrieval

---

## Testing Strategy

### Unit Tests

**Not required** - This is a migration, not new functionality. Existing validation in gates is sufficient.

### Integration Tests

Run all quality gates in sequence to validate end-to-end pipeline:

```bash
# Sequence: Gate-1 ’ Gate-2 ’ Gate-3 ’ Gate-7
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py
/Users/liyunxiao/anaconda3/bin/conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Expected**: All gates GREEN or AMBER, recall@10 e 80%

### Manual Testing Steps

1. **Verify cost estimation**:
   - Run Gate-1, observe cost estimate
   - Confirm cost is reasonable (< $1 for ~1.6k chunks)
   - Verify user can abort before API calls

2. **Monitor progress tracking**:
   - Watch progress bar during Gate-1 execution
   - Confirm rate and ETA update correctly
   - Verify completion message displays

3. **Check recall improvement**:
   - Compare Gate-7 metrics before/after migration
   - Target: recall@10 improvement from ~52% ’ e80%
   - Review specific query improvements in trace

4. **Validate dimension consistency**:
   ```bash
   # Check parquet dimension
   python -c "import pyarrow.parquet as pq; t = pq.read_table('data/vector/embeddings/embeddings.parquet'); print(f'Vector dim: {len(t.column(\"vector\")[0].as_py())}')"

   # Check FAISS manifest dimension
   jq -r '.dim' data/vector/faiss/faiss_manifest.json

   # Both should show: 1536
   ```

5. **Test error handling**:
   - Unset `OPENAI_API_KEY` and run Gate-1 ’ should fail with clear error
   - Set invalid API key ’ should fail with clear error
   - Pass `dim=768` ’ should fail with "requires dim=1536" error

---

## Performance Considerations

### API Rate Limits

**OpenAI ada-002 Rate Limits** (as of 2024, check current docs):
- Free tier: ~3 RPM (requests per minute), ~40K TPM (tokens per minute)
- Paid tier: Higher limits based on usage tier

**Impact**:
- ~1.6k chunks at 2-3 requests/sec ’ ~10-15 minutes total
- Batch size reduced to 100 (from 256) to respect rate limits
- No retry logic - fails immediately on rate limit errors

**Mitigation**:
- Progress tracking shows ETA
- Cost estimation prevents surprise bills
- User can abort before API calls

### Latency Comparison

| Stage | hashlex-v1 | OpenAI ada-002 | Delta |
|-------|-----------|----------------|-------|
| Gate-1 (1.6k chunks) | ~1 second | ~10-15 minutes | +900-1000x |
| Gate-7 (query) | ~5ms | ~100-200ms | +20-40x |

**Rationale**: Quality improvement justifies latency increase. This is a one-time cost for Gate-1 (re-embedding corpus).

### Storage Requirements

| Artifact | hashlex-v1 (768-dim) | OpenAI ada-002 (1536-dim) | Delta |
|----------|---------------------|---------------------------|-------|
| Parquet embeddings | ~5 MB | ~10 MB | +100% |
| FAISS index | ~10 MB | ~20 MB | +100% |

**Impact**: Negligible - modern systems handle this easily.

---

## Migration Notes

### Backing Up Old Embeddings

Before running Phase 5 (re-generate embeddings), consider backing up existing data:

```bash
# Backup old embeddings and indexes
mkdir -p data/backup/hashlex_v1_$(date +%Y%m%d)
cp -r data/vector/embeddings data/backup/hashlex_v1_$(date +%Y%m%d)/
cp -r data/vector/faiss data/backup/hashlex_v1_$(date +%Y%m%d)/
cp reports/qa/step01_embeddings.json data/backup/hashlex_v1_$(date +%Y%m%d)/
cp reports/qa/step02_indexes.json data/backup/hashlex_v1_$(date +%Y%m%d)/
cp reports/qa/step07_retrieval_eval.json data/backup/hashlex_v1_$(date +%Y%m%d)/
```

**Rationale**: Allows rollback if migration fails or results are unsatisfactory.

### Rollback Procedure

If migration fails or recall doesn't improve:

1. **Restore old `embedding_utils.py`**:
   ```bash
   git checkout HEAD -- scripts/embedding_utils.py
   ```

2. **Restore old config**:
   ```bash
   git checkout HEAD -- configs/vector.indexing.yaml
   ```

3. **Restore old embeddings and indexes**:
   ```bash
   cp -r data/backup/hashlex_v1_YYYYMMDD/embeddings data/vector/
   cp -r data/backup/hashlex_v1_YYYYMMDD/faiss data/vector/
   ```

4. **Verify rollback**:
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step07_retrieval_eval.py
   ```

---

## References

- **Original Research**: `thoughts/shared/research/2025-10-06-embedding-model-architecture.md`
- **Related Issue**: `thoughts/shared/issues/issue001.md`
- **OpenAI Embeddings Guide**: https://platform.openai.com/docs/guides/embeddings
- **ada-002 Model Card**: https://platform.openai.com/docs/models/embeddings
- **Pricing**: https://openai.com/api/pricing/

---

## Appendix: Key Files and Line Numbers

### Files Modified

1. `envs/age.yaml` - Add openai package
2. `scripts/embedding_utils.py` - Complete rewrite for ada-002
3. `configs/vector.indexing.yaml` - Update model and dimension
4. `scripts/qa_step01_embeddings.py` - Add cost estimation and progress tracking
5. `scripts/run_graph.py` - Update to use shared embed_text() (optional)

### Files That Will Change (Data Artifacts)

1. `data/vector/embeddings/embeddings.parquet` - New 1536-dim vectors
2. `data/vector/faiss/index.faiss` - Rebuilt with 1536-dim
3. `data/vector/faiss/faiss_manifest.json` - Updated dimension metadata
4. `reports/qa/step01_embeddings.{json,md}` - New Gate-1 reports
5. `reports/qa/step02_indexes.{json,md}` - New Gate-2 reports
6. `reports/qa/step07_retrieval_eval.{json,md}` - New Gate-7 reports

### Files Using embed_text() (No Changes Needed)

1. `scripts/qa_step01_embeddings.py:13,138` - Imports and calls embed_text()
2. `scripts/qa_step02_indexes.py:11,218` - Imports and calls embed_text()
3. `scripts/qa_step03_mcp.py:71,73` - Imports and calls embed_text()
4. `scripts/qa_step07_retrieval_eval.py:233,386` - Imports and calls embed_text()

All these files will automatically use OpenAI ada-002 after Phase 2 changes, because they import the shared `embed_text()` function.

---

## Success Metrics

### Primary Goals (Must Achieve)

- [ ] All gates pass (Gate-1, Gate-2, Gate-3, Gate-7)
- [ ] Recall@10 e 80% (from ~52% baseline)
- [ ] All ~1.6k chunks successfully embedded
- [ ] FAISS index rebuilt with 1536-dim vectors
- [ ] Dimension consistency verified across pipeline

### Secondary Goals (Nice to Have)

- [ ] nDCG@5 e 0.70 (from ~0.60 baseline)
- [ ] Cost < $1.00 USD for full corpus embedding
- [ ] Gate-1 completes in < 20 minutes
- [ ] Clear error messages on API failures
- [ ] Progress tracking shows accurate ETA

### Validation Checklist

Run this checklist after completing all phases:

```bash
# 1. Check environment
/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import openai; print(openai.__version__)"

# 2. Check configuration
grep "model: openai-ada-002" configs/vector.indexing.yaml
grep "dim: 1536" configs/vector.indexing.yaml

# 3. Check embedding function
/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from embedding_utils import embed_text; v = embed_text('test', 1536); print(f'Dimension: {len(v)}')"

# 4. Check parquet dimension
/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import pyarrow.parquet as pq; t = pq.read_table('data/vector/embeddings/embeddings.parquet'); print(f'Vector dim: {len(t.column(\"vector\")[0].as_py())}')"

# 5. Check FAISS manifest dimension
jq -r '.dim' data/vector/faiss/faiss_manifest.json

# 6. Check Gate-1 status
jq -r '.status' reports/qa/step01_embeddings.json

# 7. Check Gate-2 status
jq -r '.status' reports/qa/step02_indexes.json

# 8. Check Gate-7 recall
jq -r '.summary.recall@10' reports/qa/step07_retrieval_eval.json

# Expected: All checks return expected values (1536-dim, GREEN status, recall e 0.80)
```

---

## Notes for Implementation

1. **Fail Fast Philosophy**: All API errors are fatal - no retry logic. This is intentional to surface issues immediately rather than masking them with retries.

2. **Cost Transparency**: Cost estimation before API calls ensures no surprise bills. User can abort if cost exceeds expectations.

3. **Progress Visibility**: Progress bar every 10 chunks (or 5 seconds) provides reassurance during long runs.

4. **Vector Space Consistency**: All consumers use shared `embed_text()` function - no custom embedding logic allowed.

5. **Dimension Validation**: The new `embed_text()` enforces dim=1536, preventing configuration drift.

6. **Backward Compatibility**: Interface `embed_text(text, dim)` unchanged - all consumers work without modification.

7. **Testing Strategy**: Quality gates provide comprehensive validation - no additional unit tests needed.

8. **Rollback Plan**: Backup old embeddings before migration allows easy rollback if needed.

---

**Last Updated**: 2025-10-06T17:30:00-04:00
**Status**: Ready for Review
**Next Action**: Review plan, then proceed with Phase 1 (Environment Setup)
