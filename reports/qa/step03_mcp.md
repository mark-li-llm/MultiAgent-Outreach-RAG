# STEP 3 — MCP Tool Health & Contract Conformance (Gate‑3) — GREEN

Checks:
- G3-01: health_endpoints_ok = 5 (threshold ==5 tools) -> PASS
- G3-02: contract_ok_rate_kb.search = 1.0 (threshold ==1.0) -> PASS
- G3-02-web_fetch: contract_ok_rate_web.fetch = 1.0 (threshold ==1.0) -> PASS
- G3-02-link_resolve: contract_ok_rate_link.resolve = 1.0 (threshold ==1.0) -> PASS
- G3-02-crm_lookup: contract_ok_rate_crm.lookup = 1.0 (threshold ==1.0) -> PASS
- G3-02-safety_check: contract_ok_rate_safety.check = 1.0 (threshold ==1.0) -> PASS
- G3-03-pinecone: pinecone_latency_budget = {'p50': 127.515, 'p95': 167.525, 'budget_p95': 201.03} (threshold p50,p95 <= budget) -> PASS
- G3-03-weaviate: weaviate_latency_budget = {'p50': 56.909, 'p95': 82.993, 'budget_p95': 99.592} (threshold p50,p95 <= budget) -> PASS
- G3-03-faiss: faiss_latency_budget = {'p50': 11.673, 'p95': 13.455, 'budget_p95': 16.146} (threshold p50,p95 <= budget) -> PASS
- G3-04: timeout_rate = 0.0 (threshold ==0.0) -> PASS

Go/No-Go: Go
