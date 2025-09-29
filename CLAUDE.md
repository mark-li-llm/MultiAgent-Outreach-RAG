# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses two conda environments to avoid OpenMP runtime conflicts:

- `age` (Python 3.13): Main environment for most tasks including embeddings, routing, and retrieval evaluation
- `ageFaiss` (Python 3.12): Dedicated environment for FAISS index builds and health checks

Create environments from provided YAMLs:
```bash
conda env create -f envs/age.yaml
conda env create -f envs/ageFaiss.yaml
```

**Critical**: Never install pip `faiss-cpu` in the `age` environment as it bundles libomp and causes duplicate OpenMP runtime crashes (OMP Error #15).

## Key Commands

### Quality Gates (QA Scripts)
- Gate-1 (Embeddings): `conda run -n age python scripts/qa_step01_embeddings.py`
- Gate-2 (FAISS Index): `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
- Gate-7 (Retrieval Eval): `conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py`

### MCP Service
- Start local stub services: `conda run -n age python scripts/qa_step03_mcp.py`
- Services run on localhost ports 7801-7805 (configured in `configs/mcp.tools.yaml`)

### Other Scripts
- Build inventory: `python3 scripts/build_inventory_csv.py`
- Day-1 verification: `python3 scripts/verify_day1_milestones.py`
- Link health check: `python3 scripts/link_health_check.py`

## Architecture Overview

### Text Embedding System
- **Core utility**: `scripts/embedding_utils.py` implements `hashlex-v1` embedding model
- **Process**: normalize → tokenize (words + bigrams) → signed feature hashing → L2 normalization
- **Function**: Use `embed_text(text, dim)` for both documents and queries
- **Dimensions**: 768 (configured in `configs/vector.indexing.yaml`)

### Data Pipeline
1. **Collection**: Fetch and normalize documents from various sources
2. **Embedding**: Generate text embeddings using hashlex-v1 model (`scripts/qa_step01_embeddings.py`)
3. **Indexing**: Build FAISS HNSW index for vector search (`scripts/qa_step02_indexes.py`)
4. **Routing**: Query routing and reranking logic (`scripts/router_core.py`)
5. **Evaluation**: Retrieval quality assessment with recall@10, nDCG@5 metrics

### Key Directories
- `configs/`: YAML configuration files for various components
- `scripts/`: Main processing scripts and QA gates
- `data/`: Processed data artifacts (embeddings, indexes, reports)
- `reports/qa/`: Machine and human-readable QA reports
- `logs/`: Runtime logs organized by component

### Configuration Files
- `configs/vector.indexing.yaml`: Embedding and index settings
- `configs/router.heuristics.yaml`: Query routing logic
- `configs/mcp.tools.yaml`: MCP service endpoints and ports
- `configs/metadata.dictionary.yaml`: Metadata extraction rules

## Important Environment Variables
- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating in Gate-7
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets for retrieval eval
- `AR_USER_AGENT`: Custom user agent for web requests
- `AR_GLOBAL_RPS`: Rate limiting for HTTP requests

## Troubleshooting

### OpenMP Conflicts
- **Symptom**: OMP Error #15 or segfault during FAISS operations
- **Cause**: Mixing pip `faiss-cpu` with conda OpenBLAS+OpenMP
- **Fix**: Always run Gate-2 in `ageFaiss` environment, never install pip FAISS in `age`

### Recall Issues
- **Symptom**: recall=0 in Gate-7 evaluation
- **Cause**: Mismatched embeddings between documents and queries
- **Fix**: Ensure both use the same `embed_text()` function from `embedding_utils.py`

### Port Conflicts
- **Symptom**: "Port busy" errors for MCP services
- **Fix**: Stop existing stub services or check running processes on ports 7801-7805

## Code Conventions
- Use existing embedding utilities (`scripts/embedding_utils.py`) for all text vectorization
- Follow configuration patterns in `configs/` directory
- Output QA reports to `reports/qa/` with both JSON and Markdown formats
- Preserve report schemas used by quality gates
- Use environment variables for optional behavior changes rather than code modifications