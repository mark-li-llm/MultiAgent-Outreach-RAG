#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Tuple

from common import ensure_dir, now_iso


def normalize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def jaccard_words(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def p95(xs: List[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = int(0.95 * (len(xs) - 1))
    return xs[k]


def main():
    ap = argparse.ArgumentParser(description="QA for Gate G05 — Deduplication")
    args = ap.parse_args()

    # Load normalized docs for word_count and doctype
    doc_map: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob("data/interim/normalized/*.json"):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            doc_map[d.get("doc_id")] = d
        except Exception:
            continue

    # Chunk files after dedupe
    chunks_after: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cf in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ch = json.loads(line)
                    chunks_after[ch.get("doc_id")].append(ch)
                except Exception:
                    continue

    # Backups (before dedupe)
    chunks_before_counts: Dict[str, int] = {}
    for bf in glob.glob("data/interim/chunks/pre_dedupe/*.chunks.jsonl.bak"):
        doc_id = os.path.basename(bf).replace(".chunks.jsonl.bak", "")
        cnt = 0
        with open(bf, "r", encoding="utf-8") as f:
            for _ in f:
                cnt += 1
        chunks_before_counts[doc_id] = cnt

    # Dedup map
    try:
        dedup_map = json.load(open("data/interim/dedup/dedup_map.json", "r", encoding="utf-8"))
    except Exception:
        dedup_map = {"groups": []}
    dup_removed_total = sum(len(g.get("duplicate_chunk_ids") or []) for g in dedup_map.get("groups", []))

    docs_total = len(doc_map)
    chunks_before_total = sum(chunks_before_counts.get(did, len(chunks_after.get(did, []))) for did in doc_map.keys())
    chunks_after_total = sum(len(chunks_after.get(did, [])) for did in doc_map.keys())
    global_dup_ratio = (dup_removed_total / max(1, chunks_before_total))

    # Non-adjacent within-doc redundancy p95
    non_adj: List[float] = []
    for did, arr in chunks_after.items():
        arr.sort(key=lambda x: int(x.get("seq_no") or 0))
        for i in range(len(arr)):
            for j in range(i + 2, len(arr)):
                w1 = normalize_words(arr[i].get("text") or "")
                w2 = normalize_words(arr[j].get("text") or "")
                non_adj.append(jaccard_words(w1, w2))
    non_adj_p95 = p95(non_adj)

    # Coverage ratios
    cov_by_doc: Dict[str, float] = {}
    for did, d in doc_map.items():
        wc_doc = int(d.get("word_count") or 0)
        wc_chunks = sum(int(ch.get("word_count") or 0) for ch in chunks_after.get(did, []))
        cov_by_doc[did] = (wc_chunks / max(1, wc_doc)) if wc_doc > 0 else 1.0

    # Per-doctype medians
    by_dt: Dict[str, Dict[str, Any]] = {}
    dts = ["press", "10-K", "10-Q", "8-K", "ars_pdf", "product", "dev_docs", "help_docs", "wiki"]
    for dt in dts:
        vals = [cov_by_doc[did] for did, d in doc_map.items() if (d.get("doctype") or "").lower() == dt.lower()]
        by_dt[dt] = {"docs": len(vals), "coverage_ratio_median": float(median(vals)) if vals else 1.0}

    overall_cov_median = float(median(list(cov_by_doc.values()))) if cov_by_doc else 1.0

    # Checks
    checks = []
    checks.append({"id": "DED-001", "metric": "global_duplicate_ratio", "actual": round(global_dup_ratio, 4), "threshold": "<=0.15", "status": "PASS" if global_dup_ratio <= 0.15 else "FAIL"})
    checks.append({"id": "DED-002", "metric": "non_adjacent_jaccard_p95", "actual": round(non_adj_p95, 4), "threshold": "<=0.30", "status": "PASS" if non_adj_p95 <= 0.30 else "FAIL"})
    checks.append({"id": "DED-003", "metric": "coverage_ratio_median_overall", "actual": round(overall_cov_median, 4), "threshold": ">=0.90", "status": "PASS" if overall_cov_median >= 0.90 else "FAIL"})

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    # Evidence
    largest_groups = sorted(
        [
            {"canonical": g.get("canonical_chunk_id"), "dupes": g.get("duplicate_chunk_ids"), "size": 1 + len(g.get("duplicate_chunk_ids") or [])}
            for g in dedup_map.get("groups", [])
        ],
        key=lambda x: x["size"], reverse=True,
    )[:5]
    low_cov_docs = [{"doc_id": did, "coverage_ratio": round(cov_by_doc[did], 4)} for did in sorted(cov_by_doc, key=cov_by_doc.get) if cov_by_doc[did] < 0.90][:10]

    machine = {
        "gate": "G05_DEDUPE",
        "computed_at": now_iso(),
        "summary": {
            "docs_total": docs_total,
            "chunks_before_total": chunks_before_total,
            "chunks_after_total": chunks_after_total,
            "duplicates_removed_total": dup_removed_total,
            "global_duplicate_ratio": round(global_dup_ratio, 4),
            "non_adjacent_jaccard_p95": round(non_adj_p95, 4),
            "coverage_ratio_median_overall": round(overall_cov_median, 4),
        },
        "by_doctype": by_dt,
        "checks": checks,
        "status": status,
        "evidence": {
            "largest_duplicate_groups": largest_groups,
            "docs_with_low_coverage": low_cov_docs,
            "log_path": "logs/dedupe",
        },
    }

    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate05_dedupe.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable
    s = machine["summary"]
    lines = []
    lines.append(f"# Gate G05 — Deduplication QA (Run {machine['computed_at']})")
    lines.append(f"Summary: {machine['status']}")
    lines.append("")
    chks = {c['id']: c for c in machine['checks']}
    lines.append(f"- global_duplicate_ratio: {s['global_duplicate_ratio']} (<= 0.15) -> {chks['DED-001']['status']}")
    lines.append(f"- non_adjacent_jaccard_p95: {s['non_adjacent_jaccard_p95']} (<= 0.30) -> {chks['DED-002']['status']}")
    lines.append(f"- coverage_ratio_median_overall: {s['coverage_ratio_median_overall']} (>= 0.90) -> {chks['DED-003']['status']}")
    lines.append("By Doctype Coverage Ratios (median):")
    for dt, v in machine.get("by_doctype", {}).items():
        lines.append(f"- {dt}: {v['coverage_ratio_median']}")
    lines.append("")
    lines.append("Top duplicate clusters:")
    for g in machine["evidence"]["largest_duplicate_groups"]:
        lines.append(f"- {g['canonical']} ← {g['dupes']} (size {g['size']})")
    lines.append("")
    if machine['status'] == 'PASS':
        lines.append("Proceed? (Y/N): Y")
    else:
        lines.append("Proceed? (Y/N): N")
    with open("reports/qa/human_readable/gate05_dedupe.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": machine["status"]}, indent=2))


if __name__ == "__main__":
    main()

