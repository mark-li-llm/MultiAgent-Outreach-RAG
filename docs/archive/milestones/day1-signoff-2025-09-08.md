# Day‑1 Sign‑off (Gate G08)

This README captures the commands used, run timestamps, final gate statuses (G01–G08), and packaged artifact locations for Step 8 — Final Inventory & Day‑1 Sign‑off.

## Commands Used

- Inventory: `python3 scripts/build_inventory_csv.py`
- Day‑1 verification: `python3 scripts/verify_day1_milestones.py`
- Link health scan: `python3 scripts/link_health_check.py` (default `--concurrency=8`)
- Link health QA: `python3 scripts/qa_verify_link_health.py`
- Sign‑off QA (G08): `python3 scripts/qa_verify_day1_signoff.py`

Notes: No custom limits were set for these commands during the final pass.

## Final Gate Statuses

- G01 Collection: PASS (computed_at: 2025-09-07T21:17:56.671087+00:00)
- G02 Normalization: PASS (computed_at: 2025-09-07T21:25:32.019195+00:00)
- G03 Metadata: PASS (computed_at: 2025-09-07T21:54:27.061823+00:00)
- G04 Chunking: PASS (computed_at: 2025-09-07T22:20:54.572976+00:00)
- G05 Dedupe: PASS (computed_at: 2025-09-07T22:26:47.698859+00:00)
- G06 Link Health: PASS (computed_at: 2025-09-08T00:32:05.232616+00:00)
- G07 Eval Seed: PASS (computed_at: 2025-09-07T23:19:54.882939+00:00)
- G08 Day‑1 Sign‑off: PASS (computed_at: 2025-09-08T00:45:57.690667+00:00)

## Key Results

- Coverage: 97 docs with publish_date (>= 80)
- Cap: 97 fetched docs (<= 120)
- Duplicate ratio: 0.0583 (<= 0.15)
- Link health: 1.0 (== 1.0)
- Required fields presence: 0.9897 (>= 0.98)
- PR Recency: 73 since 2024 (>= 15), 67 in last 12mo (>= 8)

## Packaged Artifacts

- Inventory CSV: `data/final/inventory/salesforce_inventory.csv` (98 lines; 62.5 KB)
- Day‑1 verification (machine): `data/final/reports/day1_verification.json` (560 B)
- Link health (machine): `reports/qa/gate06_link_health.json` (842 B)
- Eval seed (jsonl): `data/interim/eval/salesforce_eval_seed.jsonl` (26.9 KB)
- Dedupe map: `data/interim/dedup/dedup_map.json` (39.0 KB)
- G08 Sign‑off (machine): `reports/qa/gate08_day1_signoff.json` (2.7 KB)
- G08 Sign‑off (human): `reports/qa/human_readable/gate08_day1_signoff.md`
- Config copy — dictionary: `data/final/dictionaries/metadata.dictionary.yaml` (825 B)
- Config copy — rules: `data/final/rules/normalization.rules.yaml` (343 B)

## Logs

- G06 Link health logs dir: `logs/link`
- G08 Sign‑off log: `logs/signoff/20250908_004557.log`

## Re‑run Sequence

1) `python3 scripts/build_inventory_csv.py`
2) `python3 scripts/verify_day1_milestones.py`
3) `python3 scripts/link_health_check.py` && `python3 scripts/qa_verify_link_health.py`
4) `python3 scripts/qa_verify_day1_signoff.py`

