#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime, timezone, date
from statistics import median
from typing import Any, Dict, List, Tuple

from common import ensure_dir, now_iso


def load_seed(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def main():
    ap = argparse.ArgumentParser(description="QA for Gate G07 — Evaluation Seed Set")
    args = ap.parse_args()

    seed_path = "data/interim/eval/salesforce_eval_seed.jsonl"
    items = load_seed(seed_path)

    # Load normalized docs and chunks for referential integrity and metadata
    norm: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob("data/interim/normalized/*.json"):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            norm[d.get("doc_id")] = d
        except Exception:
            continue
    chunk_map: Dict[str, Dict[str, Any]] = {}
    for cf in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ch = json.loads(line)
                    chunk_map[ch.get("chunk_id")] = ch
                except Exception:
                    continue

    total = len(items)
    # Referential integrity
    broken_refs = []
    ok_count = 0
    for it in items:
        did = it.get("expected_doc_id")
        cid = it.get("expected_chunk_id")
        if did in norm and cid in chunk_map:
            ok_count += 1
        else:
            broken_refs.append({"eval_id": it.get("eval_id"), "expected_doc_id": did, "expected_chunk_id": cid})
    ref_ratio = ok_count / max(1, total)

    # Keyphrase presence
    missing_phr = []
    present = 0
    for it in items:
        cid = it.get("expected_chunk_id")
        ch = chunk_map.get(cid) or {}
        text = (ch.get("text") or "").lower()
        kps = [k.lower() for k in it.get("expected_answer_keyphrases") or []]
        if all(k in text for k in kps):
            present += 1
        else:
            missing = [k for k in kps if k not in text]
            missing_phr.append({"eval_id": it.get("eval_id"), "chunk_id": cid, "missing": missing})
    kp_ratio = present / max(1, total)

    # Counts by source group
    cnt_sec = 0
    cnt_ir = 0
    cnt_news = 0
    cnt_pdh = 0
    cnt_wiki = 0
    within12 = 0
    within24 = 0
    pr_total = 0
    for it in items:
        did = it.get("expected_doc_id")
        d = norm.get(did) or {}
        dt = (d.get("doctype") or "").lower()
        if dt in ("10-k", "10-q", "8-k", "ars_pdf"):
            cnt_sec += 1
        elif dt == "press":
            sd = d.get("source_domain") or ""
            if sd.startswith("investor.salesforce.com"):
                cnt_ir += 1
            else:
                cnt_news += 1
            # PR recency ratios
            pub = d.get("publish_date")
            if pub:
                try:
                    dd = date.fromisoformat(pub)
                    days = (datetime.now(timezone.utc).date() - dd).days
                    pr_total += 1
                    if days <= 365:
                        within12 += 1
                    if days <= 730:
                        within24 += 1
                except Exception:
                    pass
        elif dt in ("product", "dev_docs", "help_docs"):
            cnt_pdh += 1
        elif dt == "wiki":
            cnt_wiki += 1

    within12r = (within12 / max(1, pr_total)) if pr_total else 1.0
    within24r = (within24 / max(1, pr_total)) if pr_total else 1.0

    # Persona share
    persona_counts = Counter([it.get("persona") for it in items])
    persona_share = {k: (persona_counts.get(k, 0) / max(1, total)) for k in ["vp_customer_experience", "cio", "vp_sales_ops"]}
    persona_min = min(persona_share.values()) if persona_share else 0.0

    # Difficulty distribution
    diff_counts = Counter([it.get("difficulty") for it in items])
    diff_dist = {k: diff_counts.get(k, 0) for k in ["easy", "medium", "hard"]}

    # Duplicate expected chunk ratio (unique/total)
    uniq_chunks = len(set([it.get("expected_chunk_id") for it in items]))
    dup_ratio = (uniq_chunks / max(1, total))

    # Checks
    checks = []
    checks.append({"id": "EVAL-001", "metric": "total_items", "actual": total, "threshold": ">=40", "status": "PASS" if total >= 40 else "FAIL"})
    checks.append({"id": "EVAL-002", "metric": "referential_integrity_ratio", "actual": round(ref_ratio, 4), "threshold": "==1.0", "status": "PASS" if ref_ratio == 1.0 else "FAIL"})
    checks.append({"id": "EVAL-003", "metric": "keyphrase_presence_ratio", "actual": round(kp_ratio, 4), "threshold": "==1.0", "status": "PASS" if kp_ratio == 1.0 else "FAIL"})
    checks.append({"id": "EVAL-004", "metric": "counts_SEC", "actual": cnt_sec, "threshold": ">=10", "status": "PASS" if cnt_sec >= 10 else "FAIL"})
    checks.append({"id": "EVAL-005", "metric": "counts_press_ir", "actual": cnt_ir, "threshold": ">=10", "status": "PASS" if cnt_ir >= 10 else "FAIL"})
    checks.append({"id": "EVAL-006", "metric": "counts_press_newsroom", "actual": cnt_news, "threshold": ">=10", "status": "PASS" if cnt_news >= 10 else "FAIL"})
    checks.append({"id": "EVAL-007", "metric": "counts_prod_dev_help", "actual": cnt_pdh, "threshold": ">=8", "status": "PASS" if cnt_pdh >= 8 else "FAIL"})
    checks.append({"id": "EVAL-008", "metric": "counts_wiki", "actual": cnt_wiki, "threshold": ">=2", "status": "PASS" if cnt_wiki >= 2 else "FAIL"})
    checks.append({"id": "EVAL-009", "metric": "persona_min_share", "actual": round(persona_min, 4), "threshold": ">=0.30", "status": "PASS" if persona_min >= 0.30 else "FAIL"})
    checks.append({"id": "EVAL-010", "metric": "within_24mo_ratio", "actual": round(within24r, 4), "threshold": ">=0.70", "status": "PASS" if within24r >= 0.70 else "FAIL"})
    checks.append({"id": "EVAL-011", "metric": "within_12mo_ratio", "actual": round(within12r, 4), "threshold": ">=0.40", "status": "PASS" if within12r >= 0.40 else "FAIL"})
    checks.append({"id": "EVAL-012", "metric": "duplicate_expected_chunk_ratio", "actual": round(dup_ratio, 4), "threshold": ">=0.90", "status": "PASS" if dup_ratio >= 0.90 else "FAIL"})
    # Difficulty mix: each >= 20% of total
    min_required = max(1, int(0.2 * total))
    ok_diff = (diff_dist.get("easy", 0) >= min_required and diff_dist.get("medium", 0) >= min_required and diff_dist.get("hard", 0) >= min_required)
    checks.append({"id": "EVAL-013", "metric": "difficulty_mix_min_each", "actual": min(diff_dist.get("easy",0), diff_dist.get("medium",0), diff_dist.get("hard",0)), "threshold": ">=20% of total", "status": "PASS" if ok_diff else "FAIL"})

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    machine = {
        "gate": "G07_EVAL_SEED",
        "computed_at": now_iso(),
        "summary": {
            "total_items": total,
            "referential_integrity_ratio": round(ref_ratio, 4),
            "keyphrase_presence_ratio": round(kp_ratio, 4),
            "persona_share": {
                "vp_customer_experience": round(persona_share.get("vp_customer_experience", 0.0), 4) if (persona_share := {k: (Counter([it.get("persona") for it in items]).get(k, 0) / max(1, total)) for k in ["vp_customer_experience", "cio", "vp_sales_ops"]}) else 0.0,
                "cio": round(persona_share.get("cio", 0.0), 4),
                "vp_sales_ops": round(persona_share.get("vp_sales_ops", 0.0), 4),
            },
            "counts_by_source_type": {
                "SEC": cnt_sec,
                "press_ir": cnt_ir,
                "press_newsroom": cnt_news,
                "prod_dev_help": cnt_pdh,
                "wiki": cnt_wiki,
            },
            "within_24mo_ratio": round(within24r, 4),
            "within_12mo_ratio": round(within12r, 4),
            "difficulty_dist": {"easy": diff_dist.get("easy", 0), "medium": diff_dist.get("medium", 0), "hard": diff_dist.get("hard", 0)},
            "duplicate_expected_chunk_ratio": round(dup_ratio, 4),
        },
        "checks": checks,
        "status": status,
        "evidence": {
            "broken_refs": broken_refs,
            "missing_keyphrases": missing_phr,
            "underrepresented_persona": (min(persona_share, key=persona_share.get) if items else None),
            "underfilled_buckets": ([b for b, ok in (
                ("SEC", cnt_sec >= 10),
                ("press_ir", cnt_ir >= 10),
                ("press_newsroom", cnt_news >= 10),
                ("prod_dev_help", cnt_pdh >= 8),
                ("wiki", cnt_wiki >= 2),
            ) if not ok]),
            "log_path": "logs/eval",
        },
    }

    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate07_eval_seed.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    lines = []
    s = machine["summary"]
    chks = {c['id']: c for c in machine['checks']}
    lines.append(f"# Gate G07 — Evaluation Seed QA (Run {machine['computed_at']})")
    lines.append(f"Summary: {machine['status']}")
    lines.append("")
    lines.append(f"Counts: total {s['total_items']} (>= 40) -> {chks['EVAL-001']['status']}")
    lines.append(f"Integrity: referential_integrity_ratio {s['referential_integrity_ratio']} (==1.0) -> {chks['EVAL-002']['status']}")
    lines.append(f"Keyphrase: keyphrase_presence_ratio {s['keyphrase_presence_ratio']} (==1.0) -> {chks['EVAL-003']['status']}")
    lines.append("Coverage:")
    lines.append(f"- SEC >=10: {s['counts_by_source_type']['SEC']} -> {chks['EVAL-004']['status']}")
    lines.append(f"- IR >=10: {s['counts_by_source_type']['press_ir']} -> {chks['EVAL-005']['status']}")
    lines.append(f"- Newsroom >=10: {s['counts_by_source_type']['press_newsroom']} -> {chks['EVAL-006']['status']}")
    lines.append(f"- Product/Dev/Help >=8: {s['counts_by_source_type']['prod_dev_help']} -> {chks['EVAL-007']['status']}")
    lines.append(f"- Wiki >=2: {s['counts_by_source_type']['wiki']} -> {chks['EVAL-008']['status']}")
    ps = s['persona_share']
    lines.append(f"Persona distribution: CX {ps['vp_customer_experience']}, CIO {ps['cio']}, Sales Ops {ps['vp_sales_ops']} (each >= 0.30) -> {chks['EVAL-009']['status']}")
    lines.append(f"PR recency: within 24mo {s['within_24mo_ratio']} (>=0.70), within 12mo {s['within_12mo_ratio']} (>=0.40)")
    dd = s['difficulty_dist']
    lines.append(f"Difficulty mix: easy {dd['easy']}, medium {dd['medium']}, hard {dd['hard']} (each >= 20% of total) -> {chks['EVAL-013']['status']}")
    lines.append("")
    lines.append("Failures & Actions:")
    if machine['status'] == 'PASS':
        lines.append("- None")
        lines.append("")
        lines.append("Proceed? (Y/N): Y")
    else:
        lines.append("- See evidence and adjust seed selection (add items to underfilled buckets, fix keyphrases, ensure ID references).")
        lines.append("")
        lines.append("Proceed? (Y/N): N")
    with open("reports/qa/human_readable/gate07_eval_seed.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": machine["status"]}, indent=2))


if __name__ == "__main__":
    main()
