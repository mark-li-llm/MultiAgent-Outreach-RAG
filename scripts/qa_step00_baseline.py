#!/usr/bin/env python3
import csv
import glob
import json
import math
import os
from datetime import datetime, timezone, date
from typing import List

from common import ensure_dir, now_iso


INV_PATH = "data/final/inventory/salesforce_inventory.csv"
CHUNK_GLOB = "data/interim/chunks/*.chunks.jsonl"
EVAL_PATH = "data/interim/eval/salesforce_eval_seed.jsonl"


def read_inventory_rows(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append(r)
    return rows


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    # Nearest-rank method
    k = math.ceil((p / 100) * len(sorted_vals)) - 1
    k = min(max(k, 0), len(sorted_vals) - 1)
    return float(sorted_vals[k])


def main():
    nowd = datetime.now(timezone.utc).date()

    # Inventory-driven stats
    inv = read_inventory_rows(INV_PATH)
    inv_total = len(inv)
    has_date = 0
    ages: List[int] = []
    domains = set()
    for r in inv:
        pd = (r.get("publish_date") or "").strip()
        if pd:
            has_date += 1
            try:
                d = date.fromisoformat(pd)
                ages.append((nowd - d).days)
            except Exception:
                pass
        sd = (r.get("source_domain") or "").strip()
        if sd:
            domains.add(sd)

    baseline_docs = has_date
    publish_date_pct = (has_date / max(1, inv_total)) if inv_total else 0.0

    # Chunk-driven stats
    baseline_chunks = 0
    token_counts: List[int] = []
    for path in sorted(glob.glob(CHUNK_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                baseline_chunks += 1
                try:
                    j = json.loads(line)
                    tc = int(j.get("token_count") or 0)
                    token_counts.append(tc)
                except Exception:
                    continue

    # Percentiles
    ages_sorted = sorted(ages)
    tc_sorted = sorted(token_counts)
    age_p50 = int(percentile(ages_sorted, 50)) if ages_sorted else 0
    age_p90 = int(percentile(ages_sorted, 90)) if ages_sorted else 0
    tok_p50 = int(percentile(tc_sorted, 50)) if tc_sorted else 0
    tok_p90 = int(percentile(tc_sorted, 90)) if tc_sorted else 0

    # Age buckets
    b_90 = sum(1 for a in ages if a <= 90)
    b_180 = sum(1 for a in ages if a <= 180)
    b_365 = sum(1 for a in ages if a <= 365)
    b_gt365 = sum(1 for a in ages if a > 365)

    # Seed eval size
    seed_eval_size = 0
    if os.path.exists(EVAL_PATH):
        with open(EVAL_PATH, "r", encoding="utf-8") as f:
            for _ in f:
                seed_eval_size += 1

    # Checks per DoD
    checks = []
    # G0-01 baseline_docs ≥ 80
    checks.append({
        "id": "G0-01",
        "metric": "baseline_docs",
        "actual": baseline_docs,
        "threshold": ">=80",
        "status": "PASS" if baseline_docs >= 80 else "FAIL",
        "evidence": INV_PATH,
    })
    # G0-02 publish_date_pct ≥ 0.98
    checks.append({
        "id": "G0-02",
        "metric": "publish_date_pct",
        "actual": round(publish_date_pct, 4),
        "threshold": ">=0.98",
        "status": "PASS" if publish_date_pct >= 0.98 else "FAIL",
        "evidence": INV_PATH,
    })
    # G0-03 seed_eval_size ≥ 40
    checks.append({
        "id": "G0-03",
        "metric": "seed_eval_size",
        "actual": seed_eval_size,
        "threshold": ">=40",
        "status": "PASS" if seed_eval_size >= 40 else "FAIL",
        "evidence": EVAL_PATH,
    })
    # G0-04 baseline_chunks ≥ baseline_docs
    checks.append({
        "id": "G0-04",
        "metric": "baseline_chunks",
        "actual": baseline_chunks,
        "threshold": f">=baseline_docs ({baseline_docs})",
        "status": "PASS" if baseline_chunks >= baseline_docs else "FAIL",
        "evidence": os.path.dirname(CHUNK_GLOB),
    })
    # G0-05 baseline_domain_count ≥ 3
    baseline_domain_count = len(sorted(domains))
    checks.append({
        "id": "G0-05",
        "metric": "baseline_domain_count",
        "actual": baseline_domain_count,
        "threshold": ">=3",
        "status": "PASS" if baseline_domain_count >= 3 else "FAIL",
        "evidence": INV_PATH,
    })

    # Status rules
    fails = [c for c in checks if c["status"] == "FAIL"]
    status = "GREEN"
    next_action = "continue"
    if len(fails) == 0:
        status = "GREEN"
        next_action = "continue"
    elif len(fails) == 1:
        # within 10% relative margin
        f = fails[0]
        rel_ok = False
        if f["id"] == "G0-01":
            rel_ok = (80 - baseline_docs) / 80 <= 0.10 if baseline_docs < 80 else True
        elif f["id"] == "G0-02":
            rel_ok = (0.98 - publish_date_pct) / 0.98 <= 0.10 if publish_date_pct < 0.98 else True
        elif f["id"] == "G0-03":
            rel_ok = (40 - seed_eval_size) / 40 <= 0.10 if seed_eval_size < 40 else True
        elif f["id"] == "G0-04":
            rel_ok = (baseline_docs - baseline_chunks) / max(1, baseline_docs) <= 0.10 if baseline_chunks < baseline_docs else True
        elif f["id"] == "G0-05":
            rel_ok = (3 - baseline_domain_count) / 3 <= 0.10 if baseline_domain_count < 3 else True
        status = "AMBER" if rel_ok else "RED"
        next_action = "proceed_with_caution" if rel_ok else "stop"
    else:
        status = "RED"
        next_action = "stop"

    machine = {
        "step": "step00_baseline",
        "gate": "Gate-0",
        "status": status,
        "checks": checks,
        "baseline": {
            "domains": sorted(domains),
            "age_days": {"p50": age_p50, "p90": age_p90},
            "age_buckets": {"<=90d": b_90, "<=180d": b_180, "<=365d": b_365, ">365d": b_gt365},
            "token_count": {"p50": tok_p50, "p90": tok_p90},
        },
        "next_action": next_action,
        "timestamp": now_iso(),
    }

    ensure_dir("reports/qa")
    with open("reports/qa/step00_baseline.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable report
    lines = []
    lines.append(f"# STEP 0 — Baseline Snapshot (Gate‑0) — {machine['status']}")
    lines.append("")
    lines.append("Inputs:")
    lines.append(f"- Inventory: {INV_PATH}")
    lines.append(f"- Chunks: {CHUNK_GLOB}")
    lines.append(f"- Eval seed: {EVAL_PATH}")
    lines.append("")
    lines.append("Counts:")
    lines.append(f"- baseline_docs: {baseline_docs}")
    lines.append(f"- publish_date_pct: {round(publish_date_pct,4)}")
    lines.append(f"- baseline_chunks: {baseline_chunks}")
    lines.append(f"- seed_eval_size: {seed_eval_size}")
    lines.append(f"- baseline_domain_count: {baseline_domain_count}")
    lines.append("")
    lines.append("Recency (days):")
    lines.append(f"- p50: {age_p50}")
    lines.append(f"- p90: {age_p90}")
    lines.append(f"- buckets: <=90d={b_90}, <=180d={b_180}, <=365d={b_365}, >365d={b_gt365}")
    lines.append("")
    lines.append("Tokens per chunk:")
    lines.append(f"- p50: {tok_p50}")
    lines.append(f"- p90: {tok_p90}")
    lines.append("")
    lines.append("Checks:")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}")
    lines.append("")
    lines.append(f"Gate-0 status: {machine['status']} — next_action: {machine['next_action']}")
    lines.append(f"Timestamp: {machine['timestamp']}")
    lines.append("")
    with open("reports/qa/step00_baseline.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": machine["status"]}, indent=2))


if __name__ == "__main__":
    main()

