# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

- G7-01: recall@10 = 0.1556 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.0655 (threshold >=0.60) -> FAIL
- G7-04: freshness_mean_age_days = 185.39 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 15.03, 'p95': 16.95, 'budget_p95': 15.662}, 'weaviate': {'p50': 73.56, 'p95': 79.52, 'budget_p95': 81.997}, 'pinecone': {'p50': 129.85, 'p95': 158.88, 'budget_p95': 154.778}} (threshold p50,p95 <= budget_p95 per backend) -> FAIL

Diagnostics (not gating):
- recall@k: {'@1': 0.0222, '@3': 0.0667, '@5': 0.1111, '@10': 0.1556, '@20': 0.1556}
- doc_recall@k: {'@1': 0.1111, '@3': 0.2667, '@5': 0.4444, '@10': 0.5778, '@20': 0.5778}
- doc_recall@10: 0.5778
- soft_recall@10: 0.0889
- doc_nDCG@5: 0.2703
- near_miss_rate: 0.4222
- rank_stats.chunk: {'count': 7, 'p50': 4, 'p75': 5, 'p90': 7, 'max': 8}
- rank_stats.doc: {'count': 26, 'p50': 5, 'p75': 5, 'p90': 7, 'max': 9}
- by_doctype (total/chunk_hit/doc_hit/soft_hit):
  - 10-Q: 1/0/1/0
  - ars_pdf: 9/0/9/0
  - dev_docs: 2/0/0/0
  - press: 25/6/12/2
  - product: 6/0/2/2
  - wiki: 2/1/2/0

Latency by backend:
- faiss: p50=15.03 p95=16.95 budget_p95=15.662 -> FAIL
- weaviate: p50=73.56 p95=79.52 budget_p95=81.997 -> PASS
- pinecone: p50=129.85 p95=158.88 budget_p95=154.778 -> FAIL

Per-backend quality (queries, recall@10, doc_recall@10, nDCG@5, doc_nDCG@5):
- faiss: 15, 0.1333, 0.6667, 0.0287, 0.2224
- weaviate: 15, 0.2, 0.5333, 0.1345, 0.344
- pinecone: 15, 0.1333, 0.5333, 0.0333, 0.2444
