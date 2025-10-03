# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

**Service Mode**: internal_stub (fallback mode: strict)

**Checks:**
- G7-01: recall@10 = 0.5217 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.2888 (threshold >=0.60) -> FAIL
- G7-03: coverage_unique_domains_top10_mean = 2.739 (threshold >=3.0) -> FAIL
- G7-04: freshness_mean_age_days = 276.74 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 13.58, 'p95': 16.94, 'budget_p95': 29.699}, 'weaviate': {'p50': 60.62, 'p95': 81.2, 'budget_p95': 79.574}, 'pinecone': {'p50': 128.78, 'p95': 153.29, 'budget_p95': 198.308}} (threshold p50,p95 <= budget_p95 per backend) -> FAIL

Diagnostics (not gating):
- recall@k: {'@1': 0.1739, '@3': 0.3696, '@5': 0.3696, '@10': 0.5217}
- doc_recall@k: {'@1': 0.413, '@3': 0.5652, '@5': 0.587, '@10': 0.7174}
- doc_recall@10: 0.7174
- soft_recall@10: 0.1087
- doc_nDCG@5: 0.5061
- near_miss_rate: 0.1957
- rank_stats.chunk: {'count': 24, 'p50': 2, 'p75': 6, 'p90': 8, 'max': 10}
- rank_stats.doc: {'count': 33, 'p50': 1, 'p75': 3, 'p90': 6, 'max': 10}
- by_doctype (total/chunk_hit/doc_hit/soft_hit):
  - 10-K: 3/1/1/0
  - 10-Q: 6/1/5/2
  - 8-K: 1/0/0/0
  - dev_docs: 1/1/1/0
  - help_docs: 1/1/1/0
  - press: 26/16/20/3
  - product: 6/4/5/0
  - wiki: 2/0/0/0

Latency by backend:
- faiss: p50=13.58 p95=16.94 budget_p95=29.699 -> PASS
- weaviate: p50=60.62 p95=81.2 budget_p95=79.574 -> FAIL
- pinecone: p50=128.78 p95=153.29 budget_p95=198.308 -> PASS

Per-backend quality (queries, recall@10, doc_recall@10, nDCG@5, doc_nDCG@5):
- faiss: 10, 0.7, 0.7, 0.4393, 0.6387
- weaviate: 26, 0.5769, 0.8077, 0.342, 0.5485
- pinecone: 10, 0.2, 0.5, 0.0, 0.2631
