# STEP 4 — Router Heuristics & Coverage (Gate‑4) — RED

- COV-pinecone: pinecone_route_share = 0.3333 (threshold >=0.10 or >=1 route) -> PASS
- COV-weaviate: weaviate_route_share = 0.3333 (threshold >=0.10 or >=1 route) -> PASS
- COV-faiss: faiss_route_share = 0.3333 (threshold >=0.10 or >=1 route) -> PASS
- EMP-001: empty_result_rate = 0.0 (threshold <=0.02) -> PASS
- EMP-002: auto_retry_success_rate = 1.0 (threshold >=0.95 (if any empty)) -> PASS
- FRS-001: avg_doc_age_days = 69.18 (threshold <=365) -> PASS
- DIV-001: mean_unique_domains_top10 = 2.422 (threshold >=3.0) -> FAIL

Summary:
- total_queries: 45
- route_counts: {'faiss': 15, 'weaviate': 15, 'pinecone': 15}
- empty_result_rate: 0.0
- auto_retry_success_rate: 1.0
- avg_doc_age_days: 69.18
- mean_unique_domains_top10: 2.422

Go/No-Go: No-Go
