# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

- G7-01: recall@10 = 0.0444 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.014 (threshold >=0.60) -> FAIL
- G7-03: coverage_unique_domains_top10_mean = 2.333 (threshold >=3.0) -> FAIL
- G7-04: freshness_mean_age_days = 123.8 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 45.7, 'p95': 47.21, 'budget_p95': 15.662}, 'weaviate': {'p50': 45.6, 'p95': 46.83, 'budget_p95': 81.997}, 'pinecone': {'p50': 45.79, 'p95': 48.55, 'budget_p95': 154.778}} (threshold p50,p95 <= budget_p95 per backend) -> FAIL

Diagnostics (not gating):
- doc_recall@10: 0.2667
- soft_recall@10: 0.0222
- doc_nDCG@5: 0.0807
- near_miss_rate: 0.2222
- by_doctype (total/chunk_hit/doc_hit/soft_hit):
  - 10-Q: 1/0/1/0
  - ars_pdf: 9/0/3/0
  - dev_docs: 2/0/0/0
  - press: 25/1/4/0
  - product: 6/0/2/1
  - wiki: 2/1/2/0

Latency by backend:
- faiss: p50=45.7 p95=47.21 budget_p95=15.662 -> FAIL
- weaviate: p50=45.6 p95=46.83 budget_p95=81.997 -> PASS
- pinecone: p50=45.79 p95=48.55 budget_p95=154.778 -> PASS
