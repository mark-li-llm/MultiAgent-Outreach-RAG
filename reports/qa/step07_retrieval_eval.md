# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

- G7-01: recall@10 = 0.0 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.0 (threshold >=0.60) -> FAIL
- G7-03: coverage_unique_domains_top10_mean = 2.422 (threshold >=3.0) -> FAIL
- G7-04: freshness_mean_age_days = 69.18 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 12.1, 'p95': 14.61, 'budget_p95': 15.662}, 'weaviate': {'p50': 67.02, 'p95': 82.92, 'budget_p95': 81.997}, 'pinecone': {'p50': 125.04, 'p95': 161.59, 'budget_p95': 154.778}} (threshold p50,p95 <= budget_p95 per backend) -> FAIL

Latency by backend:
- faiss: p50=12.1 p95=14.61 budget_p95=15.662 -> PASS
- weaviate: p50=67.02 p95=82.92 budget_p95=81.997 -> FAIL
- pinecone: p50=125.04 p95=161.59 budget_p95=154.778 -> FAIL
