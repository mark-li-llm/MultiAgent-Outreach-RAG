---
date: 2025-10-06
author: Claude Code
topic: "Migration from hashlex-v1 to OpenAI ada-002 Embeddings"
tags: [implementation, embeddings, openai, ada-002, gate-1, gate-2, migration]
status: draft
---
# this one i forget the ultrathink. so it might be bad.

# prompt

 /create_plan is running… i need to change the embedding model to : OpenAI ada-002
(1536-dim, requires API key, highest quality) it means i do not want the old one, just
 the openai one would be good enough.  ultrathink. i also has a research md
that you should read full at first to have a better understanding of my problem
/Users/liyunxiao/repo/ag3/worktrees/agent-faiss/thoughts/shared/research/2025-10-06-em
bedding-model-architecture.md

# OpenAI ada-002 Embedding Migration Implementation Plan

## Overview

Migrate the RAG system from the deterministic hash-based `hashlex-v1` embedding model (768-dim) to OpenAI's `text-embedding-ada-002` model (1536-dim) to achieve higher retrieval quality and semantic understanding.

## Current State Analysis

### Existing Architecture

**hashlex-v1 Implementation** (`scripts/embedding_utils.py:65-66`):
- Deterministic feature hashing (FNV-1a based)
- 768 dimensions (configured in `configs/vector.indexing.yaml:3`)
- Zero external dependencies, stateless
- No API calls, no cost, instant execution
- Used by 4 core scripts:
  - `scripts/qa_step01_embeddings.py:13,138` - Document embedding generation
  - `scripts/qa_step02_indexes.py:11,218` - Index verification queries
  - `scripts/qa_step03_mcp.py:71,73` - MCP kb.search query embedding
  - `scripts/qa_step07_retrieval_eval.py:233,386` - Retrieval evaluation

**Current Scale**:
- ~1,600 chunks across 100+ documents
- Embedding generation: instant (hash-based)
- Index size: 768-dim FAISS HNSW (M=32)
- No API costs or rate limits

**Key Dependencies**:
- Configuration: `configs/vector.indexing.yaml` (embedding.dim: 768)
- Data: `data/vector/embeddings/embeddings.parquet`
- Indexes: `data/vector/faiss/index.faiss`
- Gate checks: G1-02 (dim validation), G2-05 (round-trip integrity)

## Desired End State

### OpenAI ada-002 Architecture

**Model Specifications**:
- Model: `text-embedding-ada-002`
- Dimensions: 1536 (non-configurable)
- API-based: requires `OPENAI_API_KEY`
- Cost: ~$0.0001 per 1K tokens
- Rate limits: 3,500 RPM (Tier 1), 1M TPM

**Implementation via LangChain**:
- Library: `langchain_openai.OpenAIEmbeddings` (already available)
- Batch support: `.embed_documents(texts: List[str]) -> List[List[float]]`
- Single query: `.embed_query(text: str) -> List[float]`
- Automatic retry and error handling built-in

**Success Criteria**:

#### Automated Verification:
- [ ] Configuration updated: `configs/vector.indexing.yaml` shows `model: ada-002, dim: 1536`
- [ ] Dependencies installed: `conda run -n age python -c "from langchain_openai import OpenAIEmbeddings; print('OK')"`
- [ ] API key validation: `conda run -n age python -c "import os; assert os.getenv('OPENAI_API_KEY'), 'Missing API key'"`
- [ ] Gate-1 passes: `conda run -n age python scripts/qa_step01_embeddings.py` returns GREEN
- [ ] Gate-2 passes: `conda run -n ageFaiss python scripts/qa_step02_indexes.py` returns GREEN
- [ ] Gate-7 passes: `conda run -n age python scripts/qa_step07_retrieval_eval.py` returns GREEN or AMBER
- [ ] Embedding dimension: `data/vector/embeddings/embeddings.parquet` contains 1536-dim vectors
- [ ] FAISS index dimension: `data/vector/faiss/faiss_manifest.json` shows `"dim": 1536`
- [ ] No hashlex imports remain: `grep -r "hashlex_embed" scripts/ | wc -l` returns 0

#### Manual Verification:
- [ ] Cost estimation runs before embedding and shows reasonable estimate (~$0.05-0.20 for 1.6k chunks)
- [ ] Progress tracking displays during long-running embedding operations
- [ ] API failures cause immediate stop with clear error message (no silent failures)
- [ ] Retrieval quality improved compared to hashlex-v1 baseline (subjective assessment via Gate-7 reports)

## What We're NOT Doing

- **NOT** keeping hashlex-v1 as a fallback (clean replacement only)
- **NOT** implementing retry logic for API failures (fail-fast approach per requirements)
- **NOT** migrating to other embedding models (e.g., ada-001, Cohere, local models)
- **NOT** changing FAISS parameters (M, efConstruction, efSearch remain same)
- **NOT** modifying the router, reranker, or MCP stub logic
- **NOT** changing the evaluation seed or metrics thresholds

## Implementation Approach

**Strategy**: Incremental replacement with validation at each gate

1. Replace `embed_text()` function with OpenAI API wrapper
2. Update configuration and add cost estimation
3. Re-run Gate-1 to generate new 1536-dim embeddings
4. Re-run Gate-2 to rebuild FAISS index with new dimensions
5. Validate MCP stubs and retrieval evaluation (Gate-3, Gate-7)

**Risk Mitigation**:
- Backup existing embeddings and indexes before migration
- Test API key validation before expensive operations
- Add cost estimator with user confirmation
- Preserve original `embedding_utils.py` as `embedding_utils_hashlex_backup.py`

---

## Phase 1: Environment Setup & Dependencies

### Overview
Set up OpenAI API credentials and install required Python packages.

### Changes Required:

#### 1. Conda Environment (`envs/age.yaml`)
**File**: `envs/age.yaml`
**Changes**: Add `langchain-openai` and `python-dotenv` to dependencies

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
    - langchain-openai>=0.1.0
    - python-dotenv>=1.0.0
    - openai>=1.0.0
  # IMPORTANT: Do NOT install pip faiss-cpu in this env to avoid duplicate libomp.
```

**Rationale**:
- `langchain-openai` provides `OpenAIEmbeddings` class
- `python-dotenv` enables `.env` file support for API keys
- `openai` is a dependency of `langchain-openai`

#### 2. Environment File (`.env`)
**File**: `.env` (create in repository root)
**Changes**: Add OpenAI API key

```bash
# OpenAI API Configuration
OPENAI_API_KEY=sk-proj-...your-key-here...

# Optional: Override default model (defaults to text-embedding-ada-002)
# OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
```

**Security Notes**:
- Add `.env` to `.gitignore` (if not already present)
- Document required environment variables in `README.md`
- For production: use secret management (AWS Secrets Manager, etc.)

#### 3. Update `.gitignore`
**File**: `.gitignore`
**Changes**: Ensure `.env` is excluded

```gitignore
# Environment variables
.env
.env.local
.env.*.local
```

#### 4. Rebuild Conda Environment

**Commands**:
```bash
# Backup existing environment
/Users/liyunxiao/anaconda3/bin/conda env export -n age > envs/age_backup_$(date +%Y%m%d).yaml

# Update environment with new dependencies
/Users/liyunxiao/anaconda3/bin/conda env update -n age -f envs/age.yaml --prune

# Verify installation
/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from langchain_openai import OpenAIEmbeddings; from dotenv import load_dotenv; print('✓ Dependencies installed')"
```

### Success Criteria:

#### Automated Verification:
- [ ] Environment updated successfully: `/Users/liyunxiao/anaconda3/bin/conda env update -n age -f envs/age.yaml --prune` exits with code 0
- [ ] Import test passes: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from langchain_openai import OpenAIEmbeddings; print('OK')"` prints "OK"
- [ ] Dotenv available: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from dotenv import load_dotenv; print('OK')"` prints "OK"
- [ ] `.gitignore` updated: `grep -q '^\.env$' .gitignore` exits with code 0

#### Manual Verification:
- [ ] `.env` file exists with valid `OPENAI_API_KEY`
- [ ] API key works: Test with `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"` returns 200 OK

---

## Phase 2: Replace Core Embedding Function

### Overview
Replace `hashlex_embed()` implementation with OpenAI API wrapper while preserving the same function signature.

### Changes Required:

#### 1. Backup Original Implementation
**File**: `scripts/embedding_utils.py`
**Action**: Create backup before modifications

```bash
cp scripts/embedding_utils.py scripts/embedding_utils_hashlex_backup.py
git add scripts/embedding_utils_hashlex_backup.py
git commit -m "backup: preserve hashlex-v1 implementation before OpenAI migration"
```

#### 2. Rewrite `scripts/embedding_utils.py`
**File**: `scripts/embedding_utils.py`
**Changes**: Complete replacement with OpenAI integration

```python
#!/usr/bin/env python3
"""
OpenAI ada-002 Embedding Utilities

Replaced hashlex-v1 (deterministic hash-based) with OpenAI text-embedding-ada-002.
Original implementation preserved in embedding_utils_hashlex_backup.py.
"""
import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Global embedding client (lazy initialization)
_embedding_client: OpenAIEmbeddings | None = None


def _get_embedding_client() -> OpenAIEmbeddings:
    """Get or create the singleton OpenAI embedding client."""
    global _embedding_client
    if _embedding_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please add it to your .env file or export it in your shell."
            )

        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        _embedding_client = OpenAIEmbeddings(
            model=model,
            openai_api_key=api_key,
            # Disable retries for fail-fast behavior (per requirements)
            max_retries=0,
        )
    return _embedding_client


def embed_text(text: str, dim: int) -> List[float]:
    """
    Generate embedding for a single text string using OpenAI ada-002.

    Args:
        text: Input text to embed
        dim: Expected output dimension (must be 1536 for ada-002)

    Returns:
        List of floats representing the 1536-dimensional embedding vector

    Raises:
        ValueError: If dim != 1536 (ada-002 is fixed at 1536 dimensions)
        RuntimeError: If OpenAI API call fails (network, auth, rate limit, etc.)
    """
    if dim != 1536:
        raise ValueError(
            f"OpenAI ada-002 produces 1536-dimensional embeddings, but dim={dim} was requested. "
            "Update configs/vector.indexing.yaml to set embedding.dim=1536"
        )

    client = _get_embedding_client()

    try:
        # Use embed_query for single text (optimized call)
        vector = client.embed_query(text)
    except Exception as e:
        # Fail fast: no retries, propagate error immediately
        raise RuntimeError(
            f"OpenAI embedding API call failed: {type(e).__name__}: {e}"
        ) from e

    # Validate dimension
    if len(vector) != 1536:
        raise RuntimeError(
            f"OpenAI returned unexpected dimension: {len(vector)} (expected 1536)"
        )

    return vector


def embed_batch(texts: List[str], dim: int) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts (more efficient than individual calls).

    Args:
        texts: List of text strings to embed
        dim: Expected output dimension (must be 1536 for ada-002)

    Returns:
        List of embedding vectors (each vector is a list of 1536 floats)

    Raises:
        ValueError: If dim != 1536
        RuntimeError: If OpenAI API call fails
    """
    if dim != 1536:
        raise ValueError(
            f"OpenAI ada-002 produces 1536-dimensional embeddings, but dim={dim} was requested."
        )

    if not texts:
        return []

    client = _get_embedding_client()

    try:
        vectors = client.embed_documents(texts)
    except Exception as e:
        raise RuntimeError(
            f"OpenAI batch embedding API call failed: {type(e).__name__}: {e}"
        ) from e

    # Validate all dimensions
    for i, vec in enumerate(vectors):
        if len(vec) != 1536:
            raise RuntimeError(
                f"OpenAI returned unexpected dimension for text {i}: {len(vec)} (expected 1536)"
            )

    return vectors


# Backward compatibility: preserve old function names (now deprecated)
def normalize_text(text: str) -> str:
    """DEPRECATED: Text normalization is now handled by OpenAI's tokenizer."""
    return text.lower().strip()


def tokenize(text: str) -> List[str]:
    """DEPRECATED: Tokenization is now handled by OpenAI's model."""
    return text.split()
```

**Key Design Decisions**:
- **Lazy initialization**: Client created on first use (avoids import-time API key validation)
- **Fail-fast**: `max_retries=0` ensures immediate failure (per requirements)
- **Dimension validation**: Explicit checks to catch config mismatches early
- **Batch support**: `embed_batch()` function for efficient bulk operations
- **Backward compatibility**: Deprecated functions remain to avoid breaking imports

#### 3. Add Cost Estimation Utility
**File**: `scripts/embedding_cost_estimator.py` (new file)
**Purpose**: Pre-flight cost estimation before running Gate-1

```python
#!/usr/bin/env python3
"""
Cost estimation for OpenAI ada-002 embedding generation.

Usage:
    python scripts/embedding_cost_estimator.py

Output:
    Estimated tokens, API calls, and total cost for embedding all chunks.
"""
import glob
import json
import os
from typing import Tuple


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using simple heuristic (1 token ≈ 4 characters).

    OpenAI's actual tokenizer is more complex, but this is conservative.
    For precise estimates, use tiktoken library (not adding dependency for now).
    """
    return max(1, len(text) // 4)


def load_chunks() -> Tuple[int, int]:
    """
    Load all chunks and compute total token count.

    Returns:
        (chunk_count, total_tokens)
    """
    chunk_count = 0
    total_tokens = 0

    chunk_glob = os.path.join("data", "interim", "chunks", "*.chunks.jsonl")
    for path in sorted(glob.glob(chunk_glob)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue

                text = chunk.get("text") or ""
                chunk_count += 1
                total_tokens += estimate_tokens(text)

    return chunk_count, total_tokens


def estimate_cost(total_tokens: int) -> float:
    """
    Estimate cost in USD for ada-002 embeddings.

    Pricing: $0.0001 per 1K tokens (as of 2024)
    """
    return (total_tokens / 1000.0) * 0.0001


def main():
    chunk_count, total_tokens = load_chunks()
    cost_usd = estimate_cost(total_tokens)

    # Compute API calls (batches of 100 chunks)
    batch_size = 100
    api_calls = (chunk_count + batch_size - 1) // batch_size

    print("=" * 60)
    print("OpenAI ada-002 Embedding Cost Estimation")
    print("=" * 60)
    print(f"Chunks to embed:      {chunk_count:,}")
    print(f"Estimated tokens:     {total_tokens:,}")
    print(f"API calls (batches):  {api_calls:,} (batch_size={batch_size})")
    print(f"Estimated cost:       ${cost_usd:.4f} USD")
    print("=" * 60)
    print()
    print("Note: This is an estimate using a 4-char/token heuristic.")
    print("Actual cost may vary by ±20% depending on tokenization.")
    print()

    # Sanity check: warn if cost seems unreasonably high
    if cost_usd > 5.0:
        print("⚠️  WARNING: Estimated cost exceeds $5.00!")
        print("   Please verify chunk count and pricing before proceeding.")
        print()


if __name__ == "__main__":
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Backup created: `test -f scripts/embedding_utils_hashlex_backup.py` exits with code 0
- [ ] New implementation syntax valid: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -m py_compile scripts/embedding_utils.py` exits with code 0
- [ ] Import test passes: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from embedding_utils import embed_text; print('OK')"` prints "OK"
- [ ] API key validation: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "from embedding_utils import _get_embedding_client; _get_embedding_client(); print('OK')"` prints "OK" (requires valid API key in .env)
- [ ] Cost estimator runs: `/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/embedding_cost_estimator.py` exits with code 0

#### Manual Verification:
- [ ] Cost estimate is reasonable (~$0.05-0.20 for 1.6k chunks)
- [ ] Test embedding call works: Run `python -c "from embedding_utils import embed_text; v = embed_text('test', 1536); print(f'✓ Generated {len(v)}-dim vector')"` and verify 1536-dim output

---

## Phase 3: Update Configuration

### Overview
Update YAML configuration to specify ada-002 model and 1536 dimensions.

### Changes Required:

#### 1. Update `configs/vector.indexing.yaml`
**File**: `configs/vector.indexing.yaml`
**Changes**: Modify embedding section

```yaml
embedding:
  model: text-embedding-ada-002
  dim: 1536
  batch_size: 100
  notes: OpenAI ada-002 embeddings (API-based, 1536-dim)

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
- `embedding.model`: `hashlex-v1` → `text-embedding-ada-002`
- `embedding.dim`: `768` → `1536`
- `embedding.batch_size`: `256` → `100` (OpenAI recommended batch size)
- `embedding.notes`: Updated description

#### 2. Update Hardcoded Dimension in `qa_step04_router.py`
**File**: `scripts/qa_step04_router.py`
**Changes**: Replace hardcoded 768 with config-driven dimension

**Before** (lines 214, 221):
```python
dim = 768  # hardcoded
```

**After**:
```python
# Read dimension from config to match embedding model
import yaml
cfg_path = os.path.join("configs", "vector.indexing.yaml")
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
dim = int(cfg.get("embedding", {}).get("dim") or 1536)
```

#### 3. Fix `run_graph.py` Custom Embedding (Optional but Recommended)
**File**: `scripts/run_graph.py` (lines 300-322)
**Issue**: Uses custom `hash_vec()` and `embed_query()` that differ from `embed_text()`

**Recommendation**: Replace custom functions with `embed_text()` for consistency

**Before** (lines 300-322):
```python
def hash_vec(seed: str, d: int) -> List[float]:
    # Custom hashing logic...

def embed_query(q: str, d: int) -> List[float]:
    # Custom hashing logic...
```

**After**:
```python
# Use consistent embedding function from embedding_utils
from embedding_utils import embed_text as embed_query

# Remove hash_vec() and custom embed_query() definitions
```

**Note**: This change ensures all embedding operations use the same OpenAI model, preventing vector space inconsistencies.

### Success Criteria:

#### Automated Verification:
- [ ] Config validation: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import yaml; cfg = yaml.safe_load(open('configs/vector.indexing.yaml')); assert cfg['embedding']['dim'] == 1536; assert cfg['embedding']['model'] == 'text-embedding-ada-002'; print('OK')"` prints "OK"
- [ ] No hardcoded 768: `grep -n "768" scripts/qa_step04_router.py | wc -l` returns 0
- [ ] Syntax valid: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -m py_compile scripts/qa_step04_router.py` exits with code 0

#### Manual Verification:
- [ ] Review `configs/vector.indexing.yaml` shows correct model and dimension
- [ ] Review `scripts/run_graph.py` (if modified) uses `embed_text()` instead of custom hashing

---

## Phase 4: Re-generate Embeddings (Gate-1)

### Overview
Run Gate-1 to generate new 1536-dimensional embeddings using OpenAI ada-002.

### Changes Required:

#### 1. Backup Existing Embeddings
**Commands**:
```bash
# Backup existing 768-dim embeddings
mkdir -p data/backup/embeddings_hashlex_768d
cp data/vector/embeddings/embeddings.parquet data/backup/embeddings_hashlex_768d/
cp data/vector/embeddings/embedding_stats.json data/backup/embeddings_hashlex_768d/
echo "$(date): Backed up hashlex-v1 embeddings" >> data/backup/migration.log
```

#### 2. Modify `qa_step01_embeddings.py` for Progress Tracking
**File**: `scripts/qa_step01_embeddings.py`
**Changes**: Add progress display and batch processing

Insert after line 116 (`dim = read_yaml_dim(CONF)`):

```python
# Cost estimation and confirmation
print(f"Embedding {embedding_rows} chunks with OpenAI ada-002 (dim={dim})...")
print(f"This will make approximately {(embedding_rows + 99) // 100} API calls.")
print()

# Import batch embedding function
from embedding_utils import embed_batch

# Progress tracking
import sys
total_chunks = embedding_rows
processed = 0
batch_size = int(cfg.get("embedding", {}).get("batch_size") or 100)
```

Replace the single-embedding loop (lines 124-155) with batch processing:

```python
# Batch processing with progress tracking
batch_texts = []
batch_metadata = []

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

            batch_texts.append(text)
            batch_metadata.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "seq_no": seq_no,
                "token_count": token_count,
            })

            # Process batch when full
            if len(batch_texts) >= batch_size:
                try:
                    vectors = embed_batch(batch_texts, dim)
                except Exception as e:
                    print(f"\n✗ Embedding API call failed: {e}", file=sys.stderr)
                    print(f"  Failed at chunk {processed + 1}/{total_chunks}", file=sys.stderr)
                    sys.exit(1)

                for meta, v in zip(batch_metadata, vectors):
                    n = l2_norm(v)
                    if n == 0.0:
                        zero_vectors += 1
                    if any((x != x) for x in v):
                        nan_vectors += 1

                    rows.append({
                        "chunk_id": meta["chunk_id"],
                        "doc_id": meta["doc_id"],
                        "seq_no": meta["seq_no"],
                        "token_count": meta["token_count"],
                        "l2_norm": n,
                        "vector": [float(x) for x in v],
                    })
                    norms.append(n)

                processed += len(batch_texts)
                pct = (processed / total_chunks) * 100
                print(f"\r[{processed}/{total_chunks}] {pct:.1f}% complete", end="", flush=True)

                batch_texts = []
                batch_metadata = []

# Process remaining batch
if batch_texts:
    try:
        vectors = embed_batch(batch_texts, dim)
    except Exception as e:
        print(f"\n✗ Embedding API call failed: {e}", file=sys.stderr)
        print(f"  Failed at chunk {processed + 1}/{total_chunks}", file=sys.stderr)
        sys.exit(1)

    for meta, v in zip(batch_metadata, vectors):
        n = l2_norm(v)
        if n == 0.0:
            zero_vectors += 1
        if any((x != x) for x in v):
            nan_vectors += 1

        rows.append({
            "chunk_id": meta["chunk_id"],
            "doc_id": meta["doc_id"],
            "seq_no": meta["seq_no"],
            "token_count": meta["token_count"],
            "l2_norm": n,
            "vector": [float(x) for x in v],
        })
        norms.append(n)

    processed += len(batch_texts)

print(f"\n✓ Embedded {processed} chunks")
embedding_rows = len(rows)
```

#### 3. Run Gate-1
**Commands**:
```bash
# Show cost estimate first
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/embedding_cost_estimator.py

# Run Gate-1 (will take 1-3 minutes for ~1.6k chunks)
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py
```

**Expected Output**:
```
Embedding 1600 chunks with OpenAI ada-002 (dim=1536)...
This will make approximately 16 API calls.

[1600/1600] 100.0% complete
✓ Embedded 1600 chunks
{
  "status": "GREEN",
  "rows": 1600
}
```

### Success Criteria:

#### Automated Verification:
- [ ] Backup exists: `test -f data/backup/embeddings_hashlex_768d/embeddings.parquet` exits with code 0
- [ ] Gate-1 passes: `/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py` exits with code 0 and outputs `"status": "GREEN"`
- [ ] Parquet dimension: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import pyarrow.parquet as pq; t = pq.read_table('data/vector/embeddings/embeddings.parquet'); v = t.column('vector')[0].as_py(); assert len(v) == 1536; print('OK')"` prints "OK"
- [ ] Row count matches: Check `reports/qa/step01_embeddings.json` has `G1-01` with status PASS

#### Manual Verification:
- [ ] Progress tracking displayed during execution
- [ ] No API errors in output
- [ ] Embedding stats look reasonable (no zero vectors, no NaNs)
- [ ] Time to complete is 1-5 minutes (depending on network and rate limits)

---

## Phase 5: Rebuild FAISS Index (Gate-2)

### Overview
Rebuild FAISS HNSW index with new 1536-dimensional embeddings.

### Changes Required:

#### 1. Backup Existing Index
**Commands**:
```bash
# Backup 768-dim FAISS index
mkdir -p data/backup/faiss_hashlex_768d
cp data/vector/faiss/index.faiss data/backup/faiss_hashlex_768d/
cp data/vector/faiss/idmap.parquet data/backup/faiss_hashlex_768d/
cp data/vector/faiss/faiss_manifest.json data/backup/faiss_hashlex_768d/
echo "$(date): Backed up hashlex-v1 FAISS index" >> data/backup/migration.log
```

#### 2. Run Gate-2
**Commands**:
```bash
# Rebuild FAISS index (uses ageFaiss environment)
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**What This Does**:
- Loads 1536-dim embeddings from `data/vector/embeddings/embeddings.parquet`
- Infers dimension from first vector: `dim = len(vecs[0])` → 1536
- Creates FAISS `IndexHNSWFlat(1536, M=32, metric=L2)`
- Writes new index to `data/vector/faiss/index.faiss`
- Runs round-trip validation (G2-05 check)
- Updates manifests for Pinecone and Weaviate (simulated)

**No Code Changes Required**: Gate-2 automatically infers dimension from vector shape.

### Success Criteria:

#### Automated Verification:
- [ ] Backup exists: `test -f data/backup/faiss_hashlex_768d/index.faiss` exits with code 0
- [ ] Gate-2 passes: `/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py` exits with code 0 and outputs `"status": "GREEN"` or `"AMBER"`
- [ ] Index dimension: `/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "import json; m = json.load(open('data/vector/faiss/faiss_manifest.json')); assert m['dim'] == 1536; print('OK')"` prints "OK"
- [ ] Round-trip error low: Check `reports/qa/step02_indexes.json` has `G2-05` with `actual <= 0.001`

#### Manual Verification:
- [ ] Index file size increased (1536-dim vectors are 2x larger than 768-dim)
- [ ] Sanity search returns relevant results (check `G2-06` and `G2-07` in report)

---

## Phase 6: Validate MCP & Retrieval (Gates 3, 7)

### Overview
Validate that MCP stub services and retrieval evaluation work with new embeddings.

### Changes Required:

**No code changes required** - both gates automatically adapt to new dimensions:
- Gate-3: Infers `dim = xb.shape[1]` from loaded embeddings (line 72)
- Gate-7: Reads `dim` from config with fallback to 768 (line 209) - will read 1536

#### 1. Run Gate-3 (MCP Health Check)
**Commands**:
```bash
# Validate MCP stub services
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py
```

**Expected**: GREEN status (all health checks and contracts pass)

#### 2. Run Gate-7 (Retrieval Evaluation)
**Commands**:
```bash
# Run with relaxed latency (OpenAI API may be slower than hash-based)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

**What to Expect**:
- **Recall@10**: Should improve from 52.17% (hashlex baseline) to 70-85% (OpenAI semantic)
- **nDCG@5**: Should improve from 0.37 to 0.55-0.70
- **Latency**: Will increase due to API calls (hence `AG7_LATENCY_MULTIPLIER=3.0`)

**Note**: If Gate-7 is in **offline mode** (no MCP service), it will re-embed queries using `embed_text()` at line 233-235, which will now call OpenAI API.

### Success Criteria:

#### Automated Verification:
- [ ] Gate-3 passes: `/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py` outputs `"status": "GREEN"`
- [ ] Gate-7 runs: `/Users/liyunxiao/anaconda3/bin/conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py` exits with code 0
- [ ] Recall improved: `jq '.summary."recall@10"' reports/qa/step07_retrieval_eval.json` shows value ≥ 0.70

#### Manual Verification:
- [ ] Review `reports/qa/step07_retrieval_eval.md` and compare to hashlex baseline
- [ ] Check `reports/eval/retrieval_failures.jsonl` for reduced failure count
- [ ] Verify query embedding calls OpenAI API (watch for progress/latency)

---

## Phase 7: Clean Up & Documentation

### Overview
Remove deprecated code and update documentation.

### Changes Required:

#### 1. Remove Hashlex References (Optional)
**Files to Clean**:
- Keep `scripts/embedding_utils_hashlex_backup.py` for reference
- No other cleanup needed (all imports point to `embed_text()` which now calls OpenAI)

#### 2. Update Documentation
**File**: `CLAUDE.md`
**Changes**: Update embedding model description

**Before** (lines 39-62):
```markdown
### Text Embedding System (hashlex-v1)

**Location**: `scripts/embedding_utils.py`

**Process**:
1. Normalize text (lowercase, ASCII, collapse digits, whitespace)
2. Tokenize (extract words + bigrams for local context)
3. Signed feature hashing (FNV-1a based, deterministic)
4. L2 normalization

**Critical**: Both documents and queries MUST use the same `embed_text(text, dim)` function...
```

**After**:
```markdown
### Text Embedding System (OpenAI ada-002)

**Location**: `scripts/embedding_utils.py`

**Model**: OpenAI `text-embedding-ada-002` (1536 dimensions)

**Process**:
1. API-based semantic embeddings via LangChain `OpenAIEmbeddings`
2. Requires `OPENAI_API_KEY` environment variable
3. L2-normalized vectors returned by OpenAI

**Critical**: Both documents and queries MUST use the same `embed_text(text, dim)` function to ensure they exist in the same vector space. Dimension must be 1536 (fixed for ada-002).

**Configuration**: `configs/vector.indexing.yaml` specifies:
- `embedding.model: text-embedding-ada-002`
- `embedding.dim: 1536`
- `embedding.batch_size: 100`

**Cost**: ~$0.0001 per 1K tokens. Run `python scripts/embedding_cost_estimator.py` for estimates.

**Original Implementation**: Deterministic hashlex-v1 (768-dim) preserved in `scripts/embedding_utils_hashlex_backup.py`.
```

**File**: `README.md`
**Changes**: Update environment variables section

Add after environment setup section:

```markdown
### OpenAI API Configuration

The system uses OpenAI's `text-embedding-ada-002` for document embeddings. You must provide an API key:

1. Create a `.env` file in the repository root:
   ```bash
   echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env
   ```

2. Verify API key works:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $(grep OPENAI_API_KEY .env | cut -d= -f2)"
   ```

3. Estimate embedding costs before running Gate-1:
   ```bash
   conda run -n age python scripts/embedding_cost_estimator.py
   ```

**Security**: Never commit `.env` to version control. The file is excluded via `.gitignore`.
```

#### 3. Update Migration Log
**File**: `data/backup/migration.log` (create if doesn't exist)
**Purpose**: Document migration for future reference

```bash
# Record migration completion
cat >> data/backup/migration.log <<EOF
$(date): Migration from hashlex-v1 (768d) to OpenAI ada-002 (1536d) completed
- Gate-1 embedding generation: PASS
- Gate-2 FAISS index rebuild: PASS
- Gate-3 MCP validation: PASS
- Gate-7 retrieval evaluation: PASS (recall@10: XX.XX%)
- Original hashlex implementation: scripts/embedding_utils_hashlex_backup.py
- Backups: data/backup/embeddings_hashlex_768d/, data/backup/faiss_hashlex_768d/
EOF
```

### Success Criteria:

#### Automated Verification:
- [ ] Documentation updated: `grep -q "text-embedding-ada-002" CLAUDE.md` exits with code 0
- [ ] README updated: `grep -q "OPENAI_API_KEY" README.md` exits with code 0
- [ ] Migration log exists: `test -f data/backup/migration.log` exits with code 0
- [ ] Backup verified: `test -f scripts/embedding_utils_hashlex_backup.py` exits with code 0

#### Manual Verification:
- [ ] Review `CLAUDE.md` for accuracy
- [ ] Review `README.md` for clarity
- [ ] Confirm all backups are in place

---

## Testing Strategy

### Unit Tests
No dedicated unit tests (system uses integration tests via quality gates).

**Smoke Test** (manual):
```bash
# Test single embedding
/Users/liyunxiao/anaconda3/bin/conda run -n age python -c "
from embedding_utils import embed_text
v = embed_text('Salesforce Agentforce', 1536)
print(f'✓ Generated {len(v)}-dimensional embedding')
assert len(v) == 1536
assert all(isinstance(x, float) for x in v)
print('✓ All values are floats')
"
```

### Integration Tests
Run all quality gates in sequence:

```bash
# Gate-0: Baseline
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step00_baseline.py

# Gate-1: Embeddings
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py

# Gate-2: Indexes
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate-3: MCP Tools
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py

# Gate-7: Retrieval Evaluation
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

**Success**: All gates GREEN or AMBER.

### Manual Testing Steps

1. **API Key Validation**:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[0].id'
   ```
   Expected: Returns model ID (e.g., `"babbage"`).

2. **Cost Estimation**:
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/embedding_cost_estimator.py
   ```
   Expected: Cost < $0.50 for 1.6k chunks.

3. **Retrieval Quality Comparison**:
   - Before: Check `thoughts/shared/research/2025-10-06-embedding-model-architecture.md` (recall@10: 52.17%)
   - After: Check `reports/qa/step07_retrieval_eval.md` (expect ≥70%)

4. **Query Test**:
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda run -n age python -c "
   from qa_step03_mcp import start_stub_servers, stop_stub_servers
   import asyncio, aiohttp, json

   async def test():
       state = {}
       await start_stub_servers(state, {'tools': {'kb.search': {'host': '127.0.0.1', 'port': 7801, 'timeout_ms': 2000}}})

       async with aiohttp.ClientSession() as session:
           resp = await session.post('http://127.0.0.1:7801/invoke', json={
               'method': 'search',
               'params': {'query': 'Agentforce pricing', 'backend': 'faiss', 'top_k': 3}
           })
           results = await resp.json()
           print(json.dumps(results, indent=2))

       await stop_stub_servers(state)

   asyncio.run(test())
   "
   ```
   Expected: Returns 3 relevant chunks about Agentforce pricing.

---

## Performance Considerations

### Latency Impact

**Before (hashlex-v1)**:
- Gate-1: ~1-2 seconds (hash computation, local)
- Gate-7 query embedding: ~1ms per query

**After (OpenAI ada-002)**:
- Gate-1: ~1-3 minutes (API calls, network latency)
- Gate-7 query embedding: ~100-300ms per query (API overhead)

**Mitigation**:
- Batch embedding reduces API calls (100 texts/request)
- Cached embeddings in Parquet avoid re-generation
- `AG7_LATENCY_MULTIPLIER=3.0` relaxes evaluation budgets

### Cost Optimization

**Current Scale** (1.6k chunks):
- Estimated tokens: ~40,000-80,000
- Estimated cost: $0.004-$0.008 USD (one-time)
- Negligible for development/testing

**Future Scale** (10k chunks):
- Estimated cost: ~$0.05 USD per full re-embedding
- Query cost: ~$0.00001 per query (100 queries = $0.001)

**Best Practices**:
- Cache embeddings in Parquet (already implemented)
- Only re-run Gate-1 when documents change
- Use `embedding_cost_estimator.py` before large runs

---

## Migration Notes

### Rollback Procedure

If migration fails or ada-002 quality is worse than expected:

```bash
# 1. Restore hashlex-v1 implementation
cp scripts/embedding_utils_hashlex_backup.py scripts/embedding_utils.py

# 2. Restore configuration
cat > configs/vector.indexing.yaml <<EOF
embedding:
  model: hashlex-v1
  dim: 768
  batch_size: 256
  notes: deterministic hash-based embedding for QA
# ... (rest of config)
EOF

# 3. Restore embeddings and indexes
cp data/backup/embeddings_hashlex_768d/embeddings.parquet data/vector/embeddings/
cp data/backup/embeddings_hashlex_768d/embedding_stats.json data/vector/embeddings/
cp data/backup/faiss_hashlex_768d/index.faiss data/vector/faiss/
cp data/backup/faiss_hashlex_768d/idmap.parquet data/vector/faiss/
cp data/backup/faiss_hashlex_768d/faiss_manifest.json data/vector/faiss/

# 4. Re-run gates to verify rollback
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

### Data Compatibility

**Parquet Schema** (unchanged):
- `chunk_id: string`
- `doc_id: string`
- `seq_no: int32`
- `token_count: int32`
- `l2_norm: float32`
- `vector: list<float32>` (length increases from 768 → 1536)

**FAISS Index** (incompatible):
- 768-dim index cannot load 1536-dim vectors (requires rebuild)
- No migration path - must delete and recreate

---

## References

- Original research: `thoughts/shared/research/2025-10-06-embedding-model-architecture.md`
- LangChain OpenAI docs: https://python.langchain.com/docs/integrations/text_embedding/openai
- OpenAI ada-002 docs: https://platform.openai.com/docs/guides/embeddings
- FAISS documentation: https://github.com/facebookresearch/faiss/wiki
- Gate-1 script: `scripts/qa_step01_embeddings.py`
- Gate-2 script: `scripts/qa_step02_indexes.py`
- Gate-7 script: `scripts/qa_step07_retrieval_eval.py`

---

## Next Steps After Implementation

1. **Commit Changes**:
   ```bash
   git add -A
   git commit -m "feat: migrate from hashlex-v1 to OpenAI ada-002 embeddings (768d→1536d)"
   ```

2. **Run Full Pipeline**:
   ```bash
   # Execute all gates in sequence
   make -C . gate-0 gate-1 gate-2 gate-3 gate-7
   ```

3. **Compare Quality Metrics**:
   - Generate side-by-side comparison of hashlex vs ada-002 recall rates
   - Document findings in `thoughts/shared/research/`

4. **Optimize Query Performance**:
   - Profile Gate-7 latency with OpenAI API
   - Consider caching frequent queries if needed

5. **Production Deployment**:
   - Set up API key in production secret manager
   - Monitor OpenAI API costs and rate limits
   - Set up alerts for embedding failures
