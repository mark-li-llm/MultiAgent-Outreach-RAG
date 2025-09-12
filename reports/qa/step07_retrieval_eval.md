# STEP 7 — Retrieval Evaluation (Gate‑7) — RED

- G7-01: recall@10 = 0.1556 (threshold >=0.80) -> FAIL
- G7-02: nDCG@5 = 0.0655 (threshold >=0.60) -> FAIL
- G7-04: freshness_mean_age_days = 171.39 (threshold <=540) -> PASS
- G7-05: latency_budgets = {'faiss': {'p50': 17.36, 'p95': 20.45, 'budget_p95': 46.986000000000004}, 'weaviate': {'p50': 73.6, 'p95': 86.64, 'budget_p95': 245.99099999999999}, 'pinecone': {'p50': 132.41, 'p95': 171.55, 'budget_p95': 464.33399999999995}} (threshold p50,p95 <= budget_p95 per backend) -> PASS

Latency by backend:
- faiss: p50=17.36 p95=20.45 budget_p95=46.986000000000004 -> PASS
- weaviate: p50=73.6 p95=86.64 budget_p95=245.99099999999999 -> PASS
- pinecone: p50=132.41 p95=171.55 budget_p95=464.33399999999995 -> PASS
