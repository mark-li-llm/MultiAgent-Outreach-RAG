# STEP 0 — Baseline Snapshot (Gate‑0) — RED

Inputs:
- Inventory: data/final/inventory/salesforce_inventory.csv
- Chunks: data/interim/chunks/*.chunks.jsonl
- Eval seed: data/interim/eval/salesforce_eval_seed.jsonl

Counts:
- baseline_docs: 97
- publish_date_pct: 1.0
- baseline_chunks: 396
- seed_eval_size: 0
- baseline_domain_count: 7

Recency (days):
- p50: 192
- p90: 749
- buckets: <=90d=26, <=180d=48, <=365d=80, >365d=17

Tokens per chunk:
- p50: 814
- p90: 892

Checks:
- G0-01: baseline_docs = 97 (threshold >=80) -> PASS
- G0-02: publish_date_pct = 1.0 (threshold >=0.98) -> PASS
- G0-03: seed_eval_size = 0 (threshold >=40) -> FAIL
- G0-04: baseline_chunks = 396 (threshold >=baseline_docs (97)) -> PASS
- G0-05: baseline_domain_count = 7 (threshold >=3) -> PASS

Gate-0 status: RED — next_action: stop
Timestamp: 2025-09-29T19:25:15.437158+00:00

