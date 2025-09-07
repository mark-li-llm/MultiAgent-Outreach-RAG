#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone, date
from statistics import median
from typing import Any, Dict, List, Tuple

from common import ensure_dir, now_iso


REQUIRED_FIELDS = [
    "doc_id",
    "company",
    "doctype",
    "title",
    "publish_date",
    "url",
    "final_url",
    "source_domain",
    "section",
    "topic",
    "persona_tags",
    "language",
    "text",
    "word_count",
    "token_count",
    "ingestion_ts",
    "hash_sha256",
]


def phase_select(doc_id: str, phase: str) -> bool:
    import hashlib

    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return (int(h[-1], 16) % 2 == 0) if phase == "A" else (int(h[-1], 16) % 2 == 1)


def compute_metrics(phase: str) -> Dict[str, Any]:
    paths = sorted(glob.glob("data/interim/normalized/*.json"))
    docs = [json.load(open(p, "r", encoding="utf-8")) for p in paths]
    # Apply phase subset
    docs = [d for d in docs if phase_select(d.get("doc_id"), phase)]
    n = len(docs)

    # Required fields presence
    required_presence = 0
    missing_list: List[Dict[str, Any]] = []
    for d in docs:
        missing = []
        for k in REQUIRED_FIELDS:
            v = d.get(k)
            if v is None or (isinstance(v, str) and v.strip() == ""):
                missing.append(k)
        if not missing:
            required_presence += 1
        else:
            missing_list.append({"doc_id": d.get("doc_id"), "fields": missing})

    required_ratio = required_presence / max(1, n)

    # Date sanity and PR distribution
    invalid_dates = 0
    within12 = 0
    within24 = 0
    pr_count = 0
    nowd = datetime.now(timezone.utc).date()
    for d in docs:
        pub = d.get("publish_date")
        if pub:
            try:
                dt = date.fromisoformat(pub)
                if dt > nowd or dt < date(1999, 1, 1):
                    invalid_dates += 1
            except Exception:
                invalid_dates += 1
        if (d.get("doctype") or "").lower() == "press" and pub:
            pr_count += 1
            dt = date.fromisoformat(pub)
            days = (nowd - dt).days
            if days <= 365:
                within12 += 1
            if days <= 730:
                within24 += 1
    pr12 = within12 / max(1, pr_count)
    pr24 = within24 / max(1, pr_count)

    # Topics and personas
    topic_nonempty = sum(1 for d in docs if (d.get("topic") or "").strip() != "")
    persona_nonempty = sum(1 for d in docs if isinstance(d.get("persona_tags"), list) and len(d.get("persona_tags")) > 0)
    topic_ratio = topic_nonempty / max(1, n)
    persona_ratio = persona_nonempty / max(1, n)

    # SEC coverage
    sec_docs = [d for d in docs if (d.get("doctype") or "").lower() in ("10-k", "10-q", "8-k", "ars_pdf")]
    covs = [float(d.get("sec_item_coverage_ratio") or 0.0) for d in sec_docs]
    sec_cov_median = float(median(covs)) if covs else 0.0

    # by_doctype
    by_dt: Dict[str, Dict[str, Any]] = {}
    dts = ["10-K", "10-Q", "8-K", "ars_pdf", "press", "product", "dev_docs", "help_docs", "wiki"]
    for dt in dts:
        group = [d for d in docs if (d.get("doctype") or "").lower() == dt.lower()]
        if not group:
            by_dt[dt] = {"docs": 0, "publish_date_presence": 0.0, "title_presence": 0.0, "sec_docs": 0}
            continue
        pd_pres = sum(1 for d in group if (d.get("publish_date") or "").strip() != "") / max(1, len(group))
        title_pres = sum(1 for d in group if (d.get("title") or "").strip() != "") / max(1, len(group))
        by_dt[dt] = {"docs": len(group), "publish_date_presence": round(pd_pres, 4), "title_presence": round(title_pres, 4), "sec_docs": len(group) if dt in ("10-K","10-Q","8-K","ars_pdf") else 0}

    # Uniqueness
    doc_ids = [d.get("doc_id") for d in docs]
    doc_id_unique = (len(doc_ids) == len(set(doc_ids)))
    # final_url collisions
    by_url: Dict[str, List[str]] = defaultdict(list)
    for d in docs:
        url = d.get("final_url") or d.get("url") or ""
        if url:
            by_url[url].append(d.get("doc_id"))
    collision_sets = [ids for ids in by_url.values() if len(ids) > 1]

    # Checks with thresholds
    checks = []
    checks.append({"id": "META-001", "metric": "required_fields_presence_overall", "actual": round(required_ratio, 4), "threshold": ">=0.98", "status": "PASS" if required_ratio >= 0.98 else "FAIL"})
    # Baseline for sec coverage not available; use minimum 0.75
    checks.append({"id": "META-002", "metric": "sec_item_coverage_ratio_median", "actual": round(sec_cov_median, 4), "threshold": ">=max(0.75,baseline-0.10)", "status": "PASS" if sec_cov_median >= 0.75 else "FAIL"})
    checks.append({"id": "META-003", "metric": "topic_nonempty_ratio", "actual": round(topic_ratio, 4), "threshold": ">=0.90", "status": "PASS" if topic_ratio >= 0.90 else "FAIL"})
    checks.append({"id": "META-004", "metric": "persona_tags_ratio", "actual": round(persona_ratio, 4), "threshold": ">=max(0.60,baseline-0.10)", "status": "PASS" if persona_ratio >= 0.60 else "FAIL"})
    checks.append({"id": "META-005", "metric": "doc_id_unique", "actual": doc_id_unique, "threshold": "==true", "status": "PASS" if doc_id_unique else "FAIL"})
    checks.append({"id": "META-006", "metric": "date_invalid_count", "actual": int(invalid_dates), "threshold": "==0", "status": "PASS" if invalid_dates == 0 else "FAIL"})

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    machine = {
        "gate": "G03_METADATA",
        "phase": phase,
        "computed_at": now_iso(),
        "summary": {
            "docs_total": n,
            "required_fields_presence_overall": round(required_ratio, 4),
            "date_invalid_count": int(invalid_dates),
            "pr_within_24mo_ratio": round(pr24, 4),
            "pr_within_12mo_ratio": round(pr12, 4),
            "topic_nonempty_ratio": round(topic_ratio, 4),
            "persona_tags_ratio": round(persona_ratio, 4),
            "sec_item_coverage_ratio_median": round(sec_cov_median, 4),
            "doc_id_unique": doc_id_unique,
            "final_url_collisions": len(collision_sets),
        },
        "by_doctype": by_dt,
        "checks": checks,
        "status": status,
        "evidence": {
            "missing_required_fields": missing_list,
            "future_dates": [],
            "low_coverage_sec_docs": [{"doc_id": d.get("doc_id"), "coverage": float(d.get("sec_item_coverage_ratio") or 0.0), "items_found": len(d.get("sec_item_spans") or [])} for d in sec_docs if float(d.get("sec_item_coverage_ratio") or 0.0) < 0.75],
            "empty_topic_docs": [d.get("doc_id") for d in docs if (d.get("topic") or "").strip() == ""],
            "empty_persona_docs": [d.get("doc_id") for d in docs if not (isinstance(d.get("persona_tags"), list) and len(d.get("persona_tags")) > 0)],
            "final_url_collision_sets": collision_sets,
            "log_path": "logs/metadata",
        },
    }
    return machine


def write_reports(machine: Dict[str, Any]) -> None:
    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate03_metadata.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)
    s = machine["summary"]
    lines = []
    lines.append(f"# Gate G03 — Metadata & SEC Structure QA (Phase {machine['phase']}; Run {machine['computed_at']})")
    lines.append(f"Summary: {machine['status']}")
    lines.append("")
    lines.append(f"Required Fields (overall): {s['required_fields_presence_overall']} (>= 0.98) -> {'PASS' if s['required_fields_presence_overall']>=0.98 else 'FAIL'}")
    lines.append(f"Topic non-empty ratio: {s['topic_nonempty_ratio']} (>= 0.90) -> {'PASS' if s['topic_nonempty_ratio']>=0.90 else 'FAIL'}")
    lines.append(f"Persona tags ratio: {s['persona_tags_ratio']} (>= max(0.60, baseline-0.10)) -> {'PASS' if s['persona_tags_ratio']>=0.60 else 'FAIL'}")
    lines.append(f"Date invalid count: {s['date_invalid_count']} (== 0) -> {'PASS' if s['date_invalid_count']==0 else 'FAIL'}")
    lines.append("")
    lines.append(f"SEC Item Coverage (median): {s['sec_item_coverage_ratio_median']} (>= max(0.75, baseline-0.10)) -> {'PASS' if s['sec_item_coverage_ratio_median']>=0.75 else 'FAIL'}")
    lines.append("Breakdown by doctype:")
    for dt, v in machine.get("by_doctype", {}).items():
        lines.append(f"- {dt}: publish_date_presence {v['publish_date_presence']}, title_presence {v['title_presence']}")
    lines.append("")
    lines.append(f"Uniqueness:")
    lines.append(f"- doc_id_unique: {s['doc_id_unique']} -> {'PASS' if s['doc_id_unique'] else 'FAIL'}")
    lines.append(f"- final_url_collisions: {s['final_url_collisions']}")
    lines.append("")
    if machine['status'] == 'PASS':
        lines.append("Proceed? (Y/N): Y")
    else:
        lines.append("Proceed? (Y/N): N")
    with open("reports/qa/human_readable/gate03_metadata.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="QA for Gate G03 — Metadata & SEC Structures")
    ap.add_argument("--phase", required=True, choices=["A", "B"]) 
    args = ap.parse_args()
    m = compute_metrics(args.phase)
    write_reports(m)
    print(json.dumps({"status": m["status"]}, indent=2))


if __name__ == "__main__":
    main()

