# STEP 3 — MCP Tool Health & Contract Conformance (Gate‑3) — GREEN

Checks:
- G3-01: health_endpoints_ok = 5 (threshold ==5 tools) -> PASS
- G3-02: contract_ok_rate_kb.search = 1.0 (threshold ==1.0) -> PASS
- G3-02-web_fetch: contract_ok_rate_web.fetch = 1.0 (threshold ==1.0) -> PASS
- G3-02-link_resolve: contract_ok_rate_link.resolve = 1.0 (threshold ==1.0) -> PASS
- G3-02-crm_lookup: contract_ok_rate_crm.lookup = 1.0 (threshold ==1.0) -> PASS
- G3-02-safety_check: contract_ok_rate_safety.check = 1.0 (threshold ==1.0) -> PASS
- G3-03-pinecone: pinecone_latency_budget = {'p50': 126.777, 'p95': 165.257, 'budget_p95': 198.308} (threshold p50,p95 <= budget) -> PASS
- G3-03-weaviate: weaviate_latency_budget = {'p50': 63.305, 'p95': 66.312, 'budget_p95': 79.574} (threshold p50,p95 <= budget) -> PASS
- G3-03-faiss: faiss_latency_budget = {'p50': 19.813, 'p95': 24.749, 'budget_p95': 29.699} (threshold p50,p95 <= budget) -> PASS
- G3-04: timeout_rate = 0.0 (threshold ==0.0) -> PASS

Go/No-Go: Go
