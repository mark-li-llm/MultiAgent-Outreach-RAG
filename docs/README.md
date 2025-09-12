# Project Environments and Runners

We run different gates in different Python environments to avoid OpenMP runtime conflicts while keeping FAISS available.

- Main env: `age` (Python 3.13) — all steps except FAISS index build
- FAISS env: `ageFaiss` (Python 3.12) — Gate‑2 (FAISS index + health)

Quick commands:

```
# Gate‑1: Embeddings (age)
conda run -n age python scripts/qa_step01_embeddings.py

# Gate‑2: Index build & integrity (ageFaiss)
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate‑7: Retrieval eval (age)
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

See `docs/envs.md` for environment creation details.

