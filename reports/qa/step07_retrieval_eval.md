# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

**Service Mode**: internal_stub (fallback mode: default)

**Checks:**
- G7-01: recall@10 = 0.7174 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.3591 (threshold >=0.60) -> FAIL
- G7-04: freshness_mean_age_days = 337.54 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 11.51, 'p95': 13.54, 'budget_p95': 89.09700000000001}, 'weaviate': {'p50': 59.1, 'p95': 82.81, 'budget_p95': 238.72199999999998}, 'pinecone': {'p50': 118.74, 'p95': 158.82, 'budget_p95': 594.924}} (threshold p50,p95 <= budget_p95 per backend) -> PASS

Diagnostics (not gating):
- recall@k: {'@1': 0.1739, '@3': 0.413, '@5': 0.5435, '@10': 0.7174}
- doc_recall@k: {'@1': 0.587, '@3': 0.7391, '@5': 0.7826, '@10': 0.8478}
- doc_recall@10: 0.8478
- soft_recall@10: 0.0652
- doc_nDCG@5: 0.6922
- near_miss_rate: 0.1304
- rank_stats.chunk: {'count': 33, 'p50': 3, 'p75': 5, 'p90': 7, 'max': 10}
- rank_stats.doc: {'count': 39, 'p50': 1, 'p75': 2, 'p90': 4, 'max': 7}
- by_doctype (total/chunk_hit/doc_hit/soft_hit):
  - 10-K: 3/0/1/0
  - 10-Q: 6/2/3/1
  - 8-K: 1/1/1/0
  - dev_docs: 1/1/1/0
  - help_docs: 1/1/1/0
  - press: 26/21/24/2
  - product: 6/6/6/0
  - wiki: 2/1/2/0

Latency by backend:
- faiss: p50=11.51 p95=13.54 budget_p95=89.09700000000001 -> PASS
- weaviate: p50=59.1 p95=82.81 budget_p95=238.72199999999998 -> PASS
- pinecone: p50=118.74 p95=158.82 budget_p95=594.924 -> PASS

Per-backend quality (queries, recall@10, doc_recall@10, nDCG@5, doc_nDCG@5):
- faiss: 10, 0.8, 1.0, 0.5131, 0.95
- weaviate: 26, 0.7692, 0.7692, 0.4188, 0.6228
- pinecone: 10, 0.5, 0.9, 0.05, 0.6149
