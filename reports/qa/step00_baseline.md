# STEP 0 — Baseline Snapshot (Gate‑0) — GREEN

Inputs:
- Inventory: data/final/inventory/salesforce_inventory.csv
- Chunks: data/interim/chunks/*.chunks.jsonl
- Eval seed: data/interim/eval/salesforce_eval_seed.jsonl

Counts:
- baseline_docs: 97
- publish_date_pct: 1.0
- baseline_chunks: 536
- seed_eval_size: 46
- baseline_domain_count: 7

Recency (days):
- p50: 203
- p90: 760
- buckets: <=90d=24, <=180d=48, <=365d=80, >365d=17

Tokens per chunk:
- p50: 776
- p90: 985

Checks:
- G0-01: baseline_docs = 97 (threshold >=80) -> PASS
- G0-02: publish_date_pct = 1.0 (threshold >=0.98) -> PASS
- G0-03: seed_eval_size = 46 (threshold >=40) -> PASS
- G0-04: baseline_chunks = 536 (threshold >=baseline_docs (97)) -> PASS
- G0-05: baseline_domain_count = 7 (threshold >=3) -> PASS

Gate-0 status: GREEN — next_action: continue
Timestamp: 2025-10-10T16:26:44.691116+00:00

