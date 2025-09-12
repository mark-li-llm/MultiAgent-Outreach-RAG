# STEP 2 — Index Build & Integrity (Gate‑2) — RED

Checks:
- G2-01: pinecone_upsert_rate = 1.0 (threshold >=0.98) -> PASS
- G2-02: weaviate_upsert_rate = 1.0 (threshold >=0.98) -> PASS
- G2-03: faiss_count_ratio = 1.0 (threshold >=0.98) -> PASS
- G2-04: pct_missing_required_metadata = 0.0 (threshold <=0.02) -> PASS
- G2-05: faiss_roundtrip_error_max = 0.0 (threshold <=0.001) -> PASS
- G2-06: sanity_search_min_topk = 10 (threshold >=3) -> PASS
- G2-07: sanity_keyword_hit_min_top10 = 0 (threshold >=1) -> FAIL

Summary:
- pinecone_upserted: 1566 failed: 0
- weaviate_inserted: 1566 failed: 0
- faiss_count: 1566 roundtrip_error_max: 0.0
- pct_missing_required_metadata: 0.0
- sanity_search_min_topk: 10

Go/No-Go: No-Go
