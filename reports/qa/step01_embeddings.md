# STEP 1 — Embeddings Quality (Gate‑1) — GREEN

Checks:
- G1-01: embedding_rows = 1566 (threshold == baseline_chunks (1566)) -> PASS
- G1-02: vector_dim = 768 (threshold == 768 (from config)) -> PASS
- G1-03a: zero_vectors = 0 (threshold ==0) -> PASS
- G1-03b: nan_vectors = 0 (threshold ==0) -> PASS
- G1-04: pct_norm_outliers = 0.0 (threshold <=0.005) -> PASS

Stats:
- embedding_rows: 1566
- vector_dim: 768
- zero_vectors: 0
- nan_vectors: 0
- median_norm: 1.0
- iqr: 0.0
- pct_norm_outliers: 0.0

Go/No-Go: Go
