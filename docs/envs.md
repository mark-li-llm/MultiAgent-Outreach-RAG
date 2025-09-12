# Environments

This repo uses two Python environments to avoid OpenMP conflicts while keeping FAISS available for index builds:

- `age` — main runtime for collection, normalization, chunking, routing, Gate‑7 eval, etc.
- `ageFaiss` — dedicated environment for FAISS index build and Gate‑2 checks.

## Create environments

You can create them from the provided YAMLs (recommended):

```
conda env create -f envs/age.yaml
conda env create -f envs/ageFaiss.yaml
```

Or create them manually:

```
# Main env (already present in your setup)
conda create -n age -y -c conda-forge python=3.13 aiohttp pyyaml pyarrow numpy openblas llvm-openmp certifi

# FAISS env (Python 3.12 for conda‑forge FAISS bindings)
conda create -n ageFaiss -y -c conda-forge \
  python=3.12 faiss-cpu numpy scipy pyarrow openblas llvm-openmp aiohttp pyyaml certifi
```

## Usage

- Gate‑1 (embeddings) and Gate‑7 (retrieval eval):
  `conda run -n age python scripts/qa_step01_embeddings.py`
  `conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py`

- Gate‑2 (index build + health) — use FAISS env:
  `conda run -n ageFaiss python scripts/qa_step02_indexes.py`

Rationale: conda‑forge FAISS wheels are currently published for Python ≤ 3.12. Installing pip FAISS into the main env can bundle a separate libomp and cause duplicate OpenMP runtime crashes. Using `ageFaiss` isolates FAISS with a consistent OpenMP/OpenBLAS stack.
