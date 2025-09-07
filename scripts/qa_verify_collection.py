#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, load_json, now_iso


def pct95(xs: List[float]) -> float:
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = int(0.95 * (len(xs_sorted) - 1))
    return xs_sorted[k]


def get_latest_log_path() -> Optional[str]:
    ensure_dir("logs/fetch")
    files = sorted(glob.glob("logs/fetch/*.log"))
    return files[-1] if files else None


def compute_metrics(all_meta: List[Dict[str, Any]], phase: str) -> Dict[str, Any]:
    # Counts by bucket
    sec_count = sum(1 for m in all_meta if m.get("source_bucket") == "sec")
    ir_count = sum(1 for m in all_meta if m.get("source_bucket") == "investor_news")
    newsroom_total = sum(1 for m in all_meta if m.get("source_bucket") == "newsroom")
    newsroom_corporate = sum(1 for m in all_meta if m.get("source_bucket") == "newsroom" and str(m.get("notes", "")).find("feed=corporate") != -1)
    newsroom_product = sum(1 for m in all_meta if m.get("source_bucket") == "newsroom" and str(m.get("notes", "")).find("feed=product") != -1)
    product_count = sum(1 for m in all_meta if m.get("source_bucket") == "product")
    dev_docs_count = sum(1 for m in all_meta if m.get("source_bucket") == "dev_docs")
    help_docs_count = sum(1 for m in all_meta if m.get("source_bucket") == "help_docs")
    wikipedia_count = sum(1 for m in all_meta if m.get("source_bucket") == "wikipedia")

    # HTTP 200 ratio
    total_docs = len(all_meta)
    http_200 = sum(1 for m in all_meta if int(m.get("http_status") or 0) == 200)
    http_200_ratio = (http_200 / total_docs) if total_docs > 0 else 0.0

    # Duplicates by sha256_raw across different doc_ids
    by_hash: Dict[str, List[str]] = defaultdict(list)
    for m in all_meta:
        h = m.get("sha256_raw") or ""
        if not h:
            continue
        by_hash[h].append(m.get("doc_id"))
    duplicate_clusters = [ids for ids in by_hash.values() if len(ids) >= 2]
    duplicate_docs = sum(len(c) for c in duplicate_clusters)
    raw_exact_duplicate_rate = (duplicate_docs / total_docs) if total_docs > 0 else 0.0

    # PR recency ratios
    now_dt = datetime.now(timezone.utc).date()
    pr_metas = [m for m in all_meta if m.get("doctype") == "press"]
    dates = []
    for m in pr_metas:
        d = m.get("visible_date") or m.get("rss_pubdate")
        if not d:
            continue
        try:
            d_dt = datetime.fromisoformat(d).date()
            dates.append(d_dt)
        except Exception:
            continue
    if dates:
        within_12 = sum(1 for d in dates if (now_dt - d).days <= 365)
        within_24 = sum(1 for d in dates if (now_dt - d).days <= 730)
        within_12_ratio = within_12 / len(dates)
        within_24_ratio = within_24 / len(dates)
    else:
        within_12_ratio = 0.0
        within_24_ratio = 0.0
    missing_date_count = len(pr_metas) - len(dates)

    # Evidence
    failed_urls = [
        {"url": m.get("final_url") or m.get("requested_url"), "status": int(m.get("http_status") or 0)}
        for m in all_meta
        if int(m.get("http_status") or 0) != 200
    ]
    counts_by_source: Dict[str, int] = Counter([m.get("source_domain") or "unknown" for m in all_meta])

    # Checks
    checks: List[Dict[str, Any]] = []
    checks.append({
        "id": "COL-001", "metric": "sec_docs_count", "actual": sec_count, "threshold": ">=6",
        "status": "PASS" if sec_count >= 6 else "FAIL",
        "evidence": []
    })
    checks.append({
        "id": "COL-002", "metric": "ir_docs_count", "actual": ir_count, "threshold": ">=16",
        "status": "PASS" if ir_count >= 16 else "FAIL",
    })
    checks.append({
        "id": "COL-003", "metric": "newsroom_total_count", "actual": newsroom_total, "threshold": ">=24",
        "status": "PASS" if newsroom_total >= 24 else "FAIL",
    })
    feed_min_actual = min(newsroom_corporate, newsroom_product)
    checks.append({
        "id": "COL-004", "metric": "newsroom_feed_min_per_feed", "actual": feed_min_actual, "threshold": ">=10",
        "status": "PASS" if feed_min_actual >= 10 else "FAIL", "notes": "min(corporate,product)"
    })
    other_total = product_count + dev_docs_count + help_docs_count + wikipedia_count
    checks.append({
        "id": "COL-005", "metric": "product_dev_help_wiki_total", "actual": other_total, "threshold": "in[9,11]",
        "status": "PASS" if 9 <= other_total <= 11 else "FAIL",
    })
    checks.append({
        "id": "COL-006", "metric": "http_200_ratio", "actual": round(http_200_ratio, 4), "threshold": ">=0.99",
        "status": "PASS" if http_200_ratio >= 0.99 else "FAIL",
    })
    checks.append({
        "id": "COL-007", "metric": "raw_exact_duplicate_rate", "actual": round(raw_exact_duplicate_rate, 4), "threshold": "<=0.05",
        "status": "PASS" if raw_exact_duplicate_rate <= 0.05 else "FAIL",
    })
    checks.append({
        "id": "COL-008", "metric": "pr_within_24mo_ratio", "actual": round(within_24_ratio, 4), "threshold": ">=0.70",
        "status": "PASS" if within_24_ratio >= 0.70 else "FAIL",
    })
    checks.append({
        "id": "COL-009", "metric": "pr_within_12mo_ratio", "actual": round(within_12_ratio, 4), "threshold": ">=0.40",
        "status": "PASS" if within_12_ratio >= 0.40 else "FAIL",
    })

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"
    latest_log = get_latest_log_path() or ""

    machine = {
        "gate": "G01_COLLECTION",
        "phase": phase,
        "computed_at": now_iso(),
        "summary": {
            "sec_count": sec_count,
            "ir_count": ir_count,
            "newsroom_total_count": newsroom_total,
            "newsroom_corporate_count": newsroom_corporate,
            "newsroom_product_count": newsroom_product,
            "product_count": product_count,
            "dev_docs_count": dev_docs_count,
            "help_docs_count": help_docs_count,
            "wikipedia_count": wikipedia_count,
            "http_200_ratio": round(http_200_ratio, 4),
            "raw_exact_duplicate_rate": round(raw_exact_duplicate_rate, 4),
            "within_12_months_ratio": round(within_12_ratio, 4),
            "within_24_months_ratio": round(within_24_ratio, 4),
            "missing_date_count": missing_date_count,
        },
        "checks": checks,
        "status": status,
        "evidence": {
            "failed_urls": failed_urls,
            "duplicate_clusters": duplicate_clusters,
            "counts_by_source": counts_by_source,
            "log_path": latest_log,
        },
    }
    return machine


def write_reports(machine: Dict[str, Any]) -> None:
    ensure_dir("reports/qa/human_readable")
    # JSON
    with open("reports/qa/gate01_collection.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable markdown
    s = machine["summary"]
    phase = machine["phase"]
    ts = machine["computed_at"]
    checks = machine["checks"]
    status = machine["status"]
    failures = [c for c in checks if c.get("status") == "FAIL"]

    proceed = "Y" if status == "PASS" else "N"
    lines = []
    lines.append(f"# Gate G01 — Collection QA (Phase {phase}; Run {ts})")
    lines.append(f"Summary: {status}")
    lines.append("")
    lines.append("Coverage by Source (target → actual)")
    lines.append(f"- SEC: 6–7 → {s['sec_count']}")
    lines.append(f"- Investor PR: 20 (min 16) → {s['ir_count']}")
    lines.append(f"- Newsroom PR total: 30 (min 24) → {s['newsroom_total_count']}")
    lines.append(f"  - Corporate feed ≥10 → {s['newsroom_corporate_count']}")
    lines.append(f"  - Product feed ≥10 → {s['newsroom_product_count']}")
    other_total = s['product_count'] + s['dev_docs_count'] + s['help_docs_count'] + s['wikipedia_count']
    lines.append(f"- Product+Dev+Help+Wiki total: 10±1 → {other_total}")
    lines.append("")
    lines.append(f"HTTP 200 ratio: {s['http_200_ratio']} (threshold ≥0.99)")
    lines.append(f"Raw exact duplicate rate: {s['raw_exact_duplicate_rate']} (threshold ≤0.05)")
    lines.append(
        f"PR recency: within 24mo {s['within_24_months_ratio']} (≥0.70), within 12mo {s['within_12_months_ratio']} (≥0.40)"
    )
    lines.append(f"Missing PR dates excluded from ratio: {s['missing_date_count']}")
    lines.append("")
    lines.append("Failures & Actions:")
    if failures:
        for c in failures:
            metric = c.get("metric")
            actual = c.get("actual")
            threshold = c.get("threshold")
            # Basic fix guidance inline
            fix = ""
            if metric in ("ir_docs_count", "newsroom_total_count"):
                fix = "Increase --limit and verify date filtering."
            elif metric == "newsroom_feed_min_per_feed":
                fix = "Balance per-feed retrieval; ensure both feeds >= required."
            elif metric == "http_200_ratio":
                fix = "Lower concurrency, lengthen backoff, verify UA, retry."
            elif metric == "raw_exact_duplicate_rate":
                fix = "Canonicalize URLs and skip duplicates by sha during fetch."
            elif metric in ("pr_within_24mo_ratio", "pr_within_12mo_ratio"):
                fix = "Improve date parsing and prioritize newer items."
            elif metric == "sec_docs_count":
                fix = "Fetch remaining SEC filings listed."
            elif metric == "product_dev_help_wiki_total":
                fix = "Adjust which product/dev/help/wiki pages are fetched."
            lines.append(f"- {c['id']} {metric}: actual {actual}, threshold {threshold}. {fix}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append(f"Proceed? (Y/N): {proceed}")

    with open("reports/qa/human_readable/gate01_collection.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="QA verification for Gate G01 — Collection")
    ap.add_argument("--phase", required=True, choices=["A", "B"], help="Phase label for the report")
    args = ap.parse_args()

    # Gather meta files
    metas = []
    for path in glob.glob("data/raw/**/*.meta.json", recursive=True):
        try:
            metas.append(load_json(path))
        except Exception:
            continue

    machine = compute_metrics(metas, args.phase)
    write_reports(machine)
    print(json.dumps({"status": machine["status"]}, indent=2))


if __name__ == "__main__":
    main()
