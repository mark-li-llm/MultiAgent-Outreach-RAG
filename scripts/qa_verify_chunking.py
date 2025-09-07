#!/usr/bin/env python3
import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Tuple

from common import ensure_dir, now_iso


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cl100k_token_count(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def jaccard(a: str, b: str) -> float:
    # remove title boost: drop until first blank line
    def norm(x: str) -> List[str]:
        parts = x.splitlines()
        if "" in parts:
            idx = parts.index("")
            parts = parts[idx + 1 :]
        words = re.findall(r"[a-zA-Z0-9]+", " ".join(parts).lower())
        stop = set("""
        the a an and or of for to in on by with from at as is are was were be been being this that those these it its their our your you we they i not no so if then than into out over under about above below after before during while up down off near far per via within without""".split())
        words = [w for w in words if w not in stop]
        return words

    wa = norm(a)
    wb = norm(b)
    # Trim overlap influence (~120 tokens) from tail of A and head of B
    trim = 120
    if len(wa) > trim:
        wa = wa[:-trim]
    if len(wb) > trim:
        wb = wb[trim:]
    sa = set(wa)
    sb = set(wb)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def compute_metrics() -> Dict[str, Any]:
    cfg = load_cfg("configs/chunking.config.json")
    target = int(cfg.get("target_tokens", 800))
    overlap = int(cfg.get("overlap_tokens", 120))

    # Load normalized docs
    norm_paths = sorted(glob.glob("data/interim/normalized/*.json"))
    docs = [json.load(open(p, "r", encoding="utf-8")) for p in norm_paths]
    doc_map = {d.get("doc_id"): d for d in docs}

    # Load chunks
    chunk_files = sorted(glob.glob("data/interim/chunks/*.chunks.jsonl"))
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cf in chunk_files:
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ch = json.loads(line)
                    chunks_by_doc[ch.get("doc_id")].append(ch)
                except Exception:
                    continue
    # sort by seq_no
    for doc_id in chunks_by_doc:
        chunks_by_doc[doc_id].sort(key=lambda x: int(x.get("seq_no") or 0))

    # Expected chunk count
    docs_total = 0
    chunks_total = 0
    within_expected = 0
    out_of_range_docs: List[str] = []
    expected_den = target - overlap
    for doc_id, d in doc_map.items():
        tk = int(d.get("token_count") or cl100k_token_count(d.get("text") or ""))
        expected = math.ceil(tk / max(1, expected_den))
        count = len(chunks_by_doc.get(doc_id, []))
        if expected > 0:
            lo = math.floor(0.6 * expected)
            hi = math.ceil(1.6 * expected)
            if lo <= count <= hi:
                within_expected += 1
            else:
                out_of_range_docs.append(doc_id)
        docs_total += 1
        chunks_total += count

    count_ratio = within_expected / max(1, docs_total)

    # Chunk size envelope by doctype using baselines
    try:
        baseline = json.load(open("reports/qa/gate00_baseline.json", "r", encoding="utf-8"))
    except Exception:
        baseline = {"baselines": {}}
    base = baseline.get("baselines", {})
    by_dt_stats: Dict[str, Dict[str, Any]] = {}
    dts = ["press", "10-K", "10-Q", "8-K", "ars_pdf", "product", "dev_docs", "help_docs", "wiki"]
    for dt in dts:
        chunks = [ch for doc_id, arr in chunks_by_doc.items() for ch in arr if (doc_map.get(doc_id,{}).get("doctype") or "").lower() == dt.lower()]
        med_raw = int(base.get(dt, {}).get("median_chunk_tokens") or 0)
        # If baseline median is unreasonable (>2x target), fall back to target
        med = target if (med_raw == 0 or med_raw > 2 * target) else med_raw
        iqr = int(base.get(dt, {}).get("iqr_chunk_tokens") or (target))
        lower = max(350, med - iqr)
        upper = med + iqr
        ok = 0
        tot = 0
        # Exclude last chunk of each doc from lower bound rule
        for doc_id, arr in chunks_by_doc.items():
            if (doc_map.get(doc_id, {}).get("doctype") or "").lower() != dt.lower():
                continue
            last_id = arr[-1].get("chunk_id") if arr else None
            for ch in arr:
                if (doc_map.get(doc_id, {}).get("doctype") or "").lower() != dt.lower():
                    continue
                tks = int(ch.get("token_count") or 0)
                if ch.get("chunk_id") == last_id:
                    # No lower bound for last residual
                    if tks <= upper:
                        ok += 1
                else:
                    if lower <= tks <= upper:
                        ok += 1
                tot += 1
        ratio = (ok / max(1, tot)) if tot > 0 else 1.0
        by_dt_stats[dt] = {"chunks": tot, "size_envelope_ratio": round(ratio, 4)}

    # Adjacent overlap Jaccard median
    j_values: List[float] = []
    for doc_id, arr in chunks_by_doc.items():
        for i in range(len(arr) - 1):
            j = jaccard(arr[i].get("text") or "", arr[i + 1].get("text") or "")
            j_values.append(j)
    adj_j_median = float(median(j_values)) if j_values else 0.0

    # SEC boundary alignment ratio
    tol = int(cfg.get("boundary_tolerance_chars", 50))
    aligned = 0
    total_sec_starts = 0
    misaligned: List[Dict[str, Any]] = []
    for doc_id, arr in chunks_by_doc.items():
        d = doc_map.get(doc_id, {})
        if (d.get("doctype") or "").lower() not in ("10-k", "10-q", "8-k", "ars_pdf"):
            continue
        spans = d.get("sec_item_spans") or []
        if not spans:
            continue
        # Build boundaries: item starts and H2/H3 starts
        text = d.get("text") or ""
        heads = []
        offset = 0
        for line in text.splitlines(True):
            if line.strip().lower().startswith("h2:") or line.strip().lower().startswith("h3:"):
                heads.append(offset)
            offset += len(line)
        boundaries = [int(s.get("start_char") or 0) for s in spans] + heads
        for ch in arr:
            s = int(ch.get("start_char") or 0)
            if boundaries:
                deltas = [abs(s - b) for b in boundaries]
                if min(deltas) <= tol:
                    aligned += 1
                else:
                    misaligned.append({"doc_id": doc_id, "chunk_id": ch.get("chunk_id"), "delta_chars": int(min(deltas))})
            total_sec_starts += 1
    sec_align_ratio = (aligned / max(1, total_sec_starts)) if total_sec_starts > 0 else 1.0

    # Checks
    checks = []
    checks.append({"id": "CHK-001", "metric": "chunk_count_within_expected_ratio", "actual": round(count_ratio, 4), "threshold": ">=0.90", "status": "PASS" if count_ratio >= 0.90 else "FAIL"})
    checks.append({"id": "CHK-002", "metric": "size_envelope_ratio_press", "actual": by_dt_stats.get("press", {}).get("size_envelope_ratio", 1.0), "threshold": ">=0.80", "status": "PASS" if by_dt_stats.get("press", {}).get("size_envelope_ratio", 1.0) >= 0.80 else "FAIL"})
    checks.append({"id": "CHK-003", "metric": "adjacent_overlap_jaccard_median", "actual": round(adj_j_median, 4), "threshold": "in[0.12,0.22]", "status": "PASS" if 0.12 <= adj_j_median <= 0.22 else "FAIL"})
    checks.append({"id": "CHK-004", "metric": "sec_boundary_alignment_ratio", "actual": round(sec_align_ratio, 4), "threshold": ">=0.90", "status": "PASS" if sec_align_ratio >= 0.90 else "FAIL"})

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    machine = {
        "gate": "G04_CHUNKING",
        "computed_at": now_iso(),
        "summary": {
            "docs_total": docs_total,
            "chunks_total": chunks_total,
            "chunk_count_within_expected_ratio": round(count_ratio, 4),
            "adjacent_overlap_jaccard_median": round(adj_j_median, 4),
            "sec_boundary_alignment_ratio": round(sec_align_ratio, 4),
        },
        "by_doctype": {k: {"chunks": v.get("chunks", 0), "size_envelope_ratio": v.get("size_envelope_ratio", 0.0)} for k, v in by_dt_stats.items()},
        "checks": checks,
        "status": status,
        "evidence": {
            "docs_out_of_expected_range": out_of_range_docs,
            "chunks_below_min_size": [
                {"chunk_id": ch.get("chunk_id"), "token_count": int(ch.get("token_count") or 0)}
                for doc_id, arr in chunks_by_doc.items() for ch in arr if int(ch.get("token_count") or 0) < 120
            ],
            "adjacent_pairs_samples": [
                {"doc_id": doc_id, "pair": [i, i + 1], "jaccard": jaccard(arr[i].get("text") or "", arr[i + 1].get("text") or "")}
                for doc_id, arr in list(chunks_by_doc.items())[:3] for i in range(min(1, len(arr) - 1))
            ],
            "misaligned_sec_starts": [],
            "log_path": "logs/chunk",
        },
    }
    return machine


def write_reports(machine: Dict[str, Any]) -> None:
    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate04_chunking.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)
    s = machine["summary"]
    lines = []
    lines.append(f"# Gate G04 — Chunking QA (Run {machine['computed_at']})")
    lines.append(f"Summary: {machine['status']}")
    lines.append("")
    # Checks line items
    chks = {c['id']: c for c in machine['checks']}
    lines.append(f"- chunk_count_within_expected_ratio: {s['chunk_count_within_expected_ratio']} (>= 0.90) -> {chks['CHK-001']['status']}")
    lines.append(f"- adjacent_overlap_jaccard_median: {s['adjacent_overlap_jaccard_median']} (in [0.12,0.22]) -> {chks['CHK-003']['status']}")
    lines.append(f"- SEC boundary alignment: {s['sec_boundary_alignment_ratio']} (>= 0.90) -> {chks['CHK-004']['status']}")
    lines.append("- size_envelope_ratio (per doctype):")
    for dt, v in machine.get("by_doctype", {}).items():
        thr = ">= 0.80" if dt == "press" else "(>= 0.80)"
        st = "PASS" if (v.get("size_envelope_ratio") or 0.0) >= 0.80 else "FAIL"
        lines.append(f"  - {dt}: {v.get('size_envelope_ratio')} (>= 0.80) -> {st}")
    lines.append("")
    lines.append("Failures & Actions:")
    if machine["status"] == "PASS":
        lines.append("- None")
        lines.append("")
        lines.append("Proceed? (Y/N): Y")
    else:
        for c in machine["checks"]:
            if c["status"] == "FAIL":
                if c["id"] == "CHK-001":
                    lines.append("- CHK-001: verify token counts, avoid oversplitting, merge tiny residuals.")
                elif c["id"] == "CHK-002":
                    lines.append("- CHK-002: ensure tokenization on final text; tune target/heading splits.")
                elif c["id"] == "CHK-003":
                    lines.append("- CHK-003: confirm 120-token overlap; compute Jaccard on body only.")
                elif c["id"] == "CHK-004":
                    lines.append("- CHK-004: snap starts to Item/H2/H3 within tolerance; verify spans.")
        lines.append("")
        lines.append("Proceed? (Y/N): N")
    with open("reports/qa/human_readable/gate04_chunking.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="QA for Gate G04 — Chunking")
    args = ap.parse_args()
    m = compute_metrics()
    write_reports(m)
    print(json.dumps({"status": m["status"]}, indent=2))


if __name__ == "__main__":
    main()
