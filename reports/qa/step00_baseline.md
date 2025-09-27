# STEP 0 — Baseline Snapshot (Gate‑0) — GREEN

Inputs:
- Inventory: data/final/inventory/salesforce_inventory.csv
- Chunks: data/interim/chunks/*.chunks.jsonl
- Eval seed: data/interim/eval/salesforce_eval_seed.jsonl

Counts:
- baseline_docs: 97
- publish_date_pct: 1.0
- baseline_chunks: 1566
- seed_eval_size: 45
- baseline_domain_count: 7

Recency (days):
- p50: 171
- p90: 728
- buckets: <=90d=31, <=180d=49, <=365d=81, >365d=16

Tokens per chunk:
- p50: 810
- p90: 848

Checks:
- G0-01: baseline_docs = 97 (threshold >=80) -> PASS
- G0-02: publish_date_pct = 1.0 (threshold >=0.98) -> PASS
- G0-03: seed_eval_size = 45 (threshold >=40) -> PASS
- G0-04: baseline_chunks = 1566 (threshold >=baseline_docs (97)) -> PASS
- G0-05: baseline_domain_count = 7 (threshold >=3) -> PASS

Gate-0 status: GREEN — next_action: continue
Timestamp: 2025-09-08T01:13:24.595058+00:00

