# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

**Service Mode**: internal_stub (fallback mode: default)

**Checks:**
- G7-01: recall@10 = 0.6522 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.3441 (threshold >=0.60) -> FAIL
- G7-04: freshness_mean_age_days = 341.22 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 199.18, 'p95': 329.43, 'budget_p95': 89.09700000000001}, 'weaviate': {'p50': 320.27, 'p95': 933.6, 'budget_p95': 238.72199999999998}, 'pinecone': {'p50': 368.54, 'p95': 839.48, 'budget_p95': 594.924}} (threshold p50,p95 <= budget_p95 per backend) -> FAIL

Diagnostics (not gating):
- recall@k: {'@1': 0.1957, '@3': 0.3913, '@5': 0.5, '@10': 0.6522}
- doc_recall@k: {'@1': 0.6087, '@3': 0.7609, '@5': 0.7609, '@10': 0.8261}
- doc_recall@10: 0.8261
- soft_recall@10: 0.0435
- doc_nDCG@5: 0.6933
- near_miss_rate: 0.1739
- rank_stats.chunk: {'count': 30, 'p50': 3, 'p75': 5, 'p90': 7, 'max': 10}
- rank_stats.doc: {'count': 38, 'p50': 1, 'p75': 1, 'p90': 3, 'max': 7}
- by_doctype (total/chunk_hit/doc_hit/soft_hit):
  - 10-K: 3/0/1/0
  - 10-Q: 6/0/3/0
  - 8-K: 1/1/1/0
  - dev_docs: 1/1/1/0
  - help_docs: 1/1/1/0
  - press: 26/20/23/2
  - product: 6/6/6/0
  - wiki: 2/1/2/0

Latency by backend:
- faiss: p50=199.18 p95=329.43 budget_p95=89.09700000000001 -> FAIL
- weaviate: p50=320.27 p95=933.6 budget_p95=238.72199999999998 -> FAIL
- pinecone: p50=368.54 p95=839.48 budget_p95=594.924 -> FAIL

Per-backend quality (queries, recall@10, doc_recall@10, nDCG@5, doc_nDCG@5):
- faiss: 10, 0.8, 1.0, 0.5131, 0.95
- weaviate: 26, 0.6538, 0.7308, 0.3922, 0.6204
- pinecone: 10, 0.5, 0.9, 0.05, 0.6262
