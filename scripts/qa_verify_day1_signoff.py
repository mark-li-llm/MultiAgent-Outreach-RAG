#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone, date
from typing import Any, Dict, List

from common import ensure_dir, now_iso


REQUIRED_FIELDS = [
    "doc_id","company","doctype","title","publish_date","url","final_url","source_domain","section","topic","persona_tags","language","word_count","token_count","ingestion_ts","hash_sha256"
]

ALLOWLIST = [
    "sec.gov","investor.salesforce.com","salesforce.com","developer.salesforce.com","help.salesforce.com","wikipedia.org"
]


def root_host(u: str) -> str:
    from urllib.parse import urlparse
    try:
        return urlparse(u).netloc.lower()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser(description="Gate G08 — Day-1 Sign-off QA")
    args = ap.parse_args()

    # Load normalized docs
    docs: List[Dict[str, Any]] = []
    for p in glob.glob("data/interim/normalized/*.json"):
        try:
            docs.append(json.load(open(p, "r", encoding="utf-8")))
        except Exception:
            continue
    total_docs = len(docs)

    # Required fields presence overall
    full = 0
    for d in docs:
        ok = True
        for k in REQUIRED_FIELDS:
            v = d.get(k)
            if v is None or (isinstance(v, str) and not v.strip()):
                ok = False
                break
        if ok:
            full += 1
    req_ratio = full / max(1, total_docs)

    # Link health summary
    try:
        lh = json.load(open("data/final/reports/link_health.json", "r", encoding="utf-8"))
        ok = sum(1 for e in lh if e.get("link_ok") is True)
        link_ok_pct = ok / max(1, len(lh))
    except Exception:
        link_ok_pct = 0.0

    # Inventory CSV
    inv_path = "data/final/inventory/salesforce_inventory.csv"
    inv_rows = 0
    inv_ids = []
    if os.path.exists(inv_path):
        with open(inv_path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                inv_rows += 1
                inv_ids.append(row.get("doc_id"))
    docs_with_dates = sum(1 for d in docs if (d.get("publish_date") or "").strip())
    inv_match = (inv_rows == docs_with_dates)
    inv_unique = (len(inv_ids) == len(set(inv_ids)))

    # Dedupe ratio from dedup map
    try:
        ded = json.load(open("data/interim/dedup/dedup_map.json", "r", encoding="utf-8"))
        dup_removed = sum(len(g.get("duplicate_chunk_ids") or []) for g in ded.get("groups", []))
    except Exception:
        dup_removed = 0
    # Before/after counts
    before_total = 0
    for bf in glob.glob("data/interim/chunks/pre_dedupe/*.chunks.jsonl.bak"):
        with open(bf, "r", encoding="utf-8") as f:
            for _ in f:
                before_total += 1
    after_total = 0
    for af in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(af, "r", encoding="utf-8") as f:
            for _ in f:
                after_total += 1
    dup_ratio = (dup_removed / max(1, before_total)) if before_total else 0.0

    # Cap: fetched docs ≤ 120 (use total normalized docs as proxy)
    fetched_docs_cap = total_docs

    # PR recency: since 2024 & last 12 months
    pr_since_2024 = 0
    pr_last_12 = 0
    nowd = datetime.now(timezone.utc).date()
    for d in docs:
        if (d.get("doctype") or "").lower() == "press" and (d.get("publish_date") or ""):
            try:
                dd = date.fromisoformat(d.get("publish_date"))
                if dd >= date(2024,1,1):
                    pr_since_2024 += 1
                if (nowd - dd).days <= 365:
                    pr_last_12 += 1
            except Exception:
                pass

    # Allowlist validation for inventory
    allow_ok = True
    for p in glob.glob("data/final/inventory/salesforce_inventory.csv"):
        with open(p, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                host = root_host(row.get("final_url") or row.get("url") or "")
                if not any(host == r or host.endswith("."+r) for r in ALLOWLIST):
                    allow_ok = False
                    break

    summary = {
        "total_docs": total_docs,
        "docs_with_dates": docs_with_dates,
        "chunks_total": after_total,
        "duplicates_removed": dup_removed,
        "link_ok_pct": round(link_ok_pct, 4),
        "inventory_rows": inv_rows,
        "inventory_doc_id_unique": inv_unique,
        "pr_docs_since_2024": pr_since_2024,
        "pr_docs_last_12mo": pr_last_12,
        "required_fields_presence_overall": round(req_ratio, 4),
    }

    checks = []
    checks.append({"id": "FIN-001", "metric": "docs_with_dates", "actual": docs_with_dates, "threshold": ">=80", "status": "PASS" if docs_with_dates >= 80 else "FAIL"})
    checks.append({"id": "FIN-002", "metric": "fetched_docs_cap", "actual": fetched_docs_cap, "threshold": "<=120", "status": "PASS" if fetched_docs_cap <= 120 else "FAIL"})
    checks.append({"id": "FIN-003", "metric": "duplicate_ratio", "actual": round(dup_ratio,4), "threshold": "<=0.15", "status": "PASS" if dup_ratio <= 0.15 else "FAIL"})
    checks.append({"id": "FIN-004", "metric": "link_ok_pct", "actual": round(link_ok_pct,4), "threshold": "==1.0", "status": "PASS" if link_ok_pct == 1.0 else "FAIL"})
    checks.append({"id": "FIN-005", "metric": "required_fields_presence_overall", "actual": round(req_ratio,4), "threshold": ">=0.98", "status": "PASS" if req_ratio >= 0.98 else "FAIL"})
    checks.append({"id": "FIN-006", "metric": "pr_docs_since_2024", "actual": pr_since_2024, "threshold": ">=15", "status": "PASS" if pr_since_2024 >= 15 else "FAIL"})
    checks.append({"id": "FIN-007", "metric": "pr_docs_last_12mo", "actual": pr_last_12, "threshold": ">=8", "status": "PASS" if pr_last_12 >= 8 else "FAIL"})
    checks.append({"id": "FIN-008", "metric": "inventory_row_count_match", "actual": inv_match, "threshold": "==true", "status": "PASS" if inv_match else "FAIL"})
    checks.append({"id": "FIN-009", "metric": "inventory_doc_id_unique", "actual": inv_unique, "threshold": "==true", "status": "PASS" if inv_unique else "FAIL"})

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    # Prepare machine summary
    computed = now_iso()
    machine = {
        "gate": "G08_DAY1_SIGNOFF",
        "computed_at": computed,
        "summary": summary,
        "checks": checks,
        "status": status,
        "evidence": {
            "failed_checks": [c for c in checks if c.get("status") == "FAIL"],
            "domain_breakdown": dict(Counter([ (d.get("source_domain") or "").lower() for d in docs])),
            "inventory_sample": inv_ids[:10] if (inv_ids := inv_ids if 'inv_ids' in locals() else []) else [],
            "log_path": "",
        },
    }

    # Write a timestamped sign-off log
    ensure_dir("logs/signoff")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join("logs", "signoff", f"{ts}.log")
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write(f"G08 computed_at: {computed}\n")
            lf.write(f"Status: {status}\n")
            lf.write(json.dumps({"summary": summary, "checks": checks}, ensure_ascii=False, indent=2))
            lf.write("\n")
        machine["evidence"]["log_path"] = log_path
    except Exception:
        # Fall back to directory if write fails
        machine["evidence"]["log_path"] = "logs/signoff"

    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate08_day1_signoff.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable
    lines = []
    lines.append(f"# Gate G08 — Day‑1 Sign‑off QA (Run {machine['computed_at']})")
    lines.append(f"Summary: {status}")
    lines.append("")
    get = lambda mid: next(c for c in checks if c['id']==mid)
    lines.append(f"Coverage & Cap:")
    lines.append(f"- docs_with_dates: {summary['docs_with_dates']} (>= 80) -> {get('FIN-001')['status']}")
    lines.append(f"- fetched_docs_cap: {total_docs} (<= 120) -> {get('FIN-002')['status']}")
    lines.append("")
    lines.append("Quality:")
    lines.append(f"- duplicate_ratio: {round(dup_ratio,4)} (<= 0.15) -> {get('FIN-003')['status']}")
    lines.append(f"- link_ok_pct: {round(link_ok_pct,4)} (== 1.0) -> {get('FIN-004')['status']}")
    lines.append(f"- required_fields_presence_overall: {round(req_ratio,4)} (>= 0.98) -> {get('FIN-005')['status']}")
    lines.append("")
    lines.append("PR Recency:")
    lines.append(f"- pr_docs_since_2024: {pr_since_2024} (>= 15) -> {get('FIN-006')['status']}")
    lines.append(f"- pr_docs_last_12mo: {pr_last_12} (>= 8) -> {get('FIN-007')['status']}")
    lines.append("")
    lines.append("Inventory:")
    lines.append(f"- inventory_row_count_match: {inv_match} -> {get('FIN-008')['status']}")
    lines.append(f"- inventory_doc_id_unique: {inv_unique} -> {get('FIN-009')['status']}")
    lines.append("")
    lines.append("Failures & Actions:")
    if status == 'PASS':
        lines.append("- None")
        lines.append("")
        lines.append("Proceed? (Y/N): Y")
    else:
        for c in checks:
            if c['status'] == 'FAIL':
                if c['id'] == 'FIN-001':
                    lines.append("- FIN-001: Add docs with publish_date to reach >=80; re-run Steps 2–6 as needed.")
                elif c['id'] == 'FIN-003':
                    lines.append("- FIN-003: Relax dedupe or whitelist sections; re-run G05.")
                elif c['id'] == 'FIN-004':
                    lines.append("- FIN-004: Fix broken links in G06 and re-run.")
                elif c['id'] == 'FIN-005':
                    lines.append("- FIN-005: Re-extract metadata (G03) for missing fields.")
                elif c['id'] in ('FIN-006','FIN-007'):
                    lines.append("- FIN-006/007: Add recent PRs; ensure dates correct.")
                elif c['id'] == 'FIN-008':
                    lines.append("- FIN-008: Ensure inventory row count matches docs_with_dates.")
                elif c['id'] == 'FIN-009':
                    lines.append("- FIN-009: De-duplicate doc_ids in inventory.")
        lines.append("")
        lines.append("Proceed? (Y/N): N")

    with open("reports/qa/human_readable/gate08_day1_signoff.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": status}, indent=2))


if __name__ == "__main__":
    main()
