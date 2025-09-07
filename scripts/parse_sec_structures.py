#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from common import ensure_dir, now_iso


ITEM_PATTERNS = [
    ("Item 1", r"^\s*item\s*\xa0*\s*1\.?\s*(.*)$"),
    ("Item 1A", r"^\s*item\s*\xa0*\s*1\s*a\.?\s*(.*)$"),
    ("Item 7", r"^\s*item\s*\xa0*\s*7\.?\s*(.*)$"),
    ("Item 7A", r"^\s*item\s*\xa0*\s*7\s*a\.?\s*(.*)$"),
    ("Item 8", r"^\s*item\s*\xa0*\s*8\.?\s*(.*)$"),
]


def phase_select(doc_id: str, phase: str) -> bool:
    import hashlib

    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return (int(h[-1], 16) % 2 == 0) if phase == "A" else (int(h[-1], 16) % 2 == 1)


def find_item_spans(text: str) -> Tuple[List[Dict], float]:
    lines = text.splitlines()
    spans: List[Tuple[str, str, int]] = []  # (label, title, start_char)
    offset = 0
    for line in lines:
        lowered = line.lower()
        for label, pat in ITEM_PATTERNS:
            if re.search(pat, lowered, flags=re.IGNORECASE):
                m = re.search(pat, lowered, flags=re.IGNORECASE)
                title = m.group(1).strip() if m and m.group(1) else ""
                spans.append((label, title, offset))
                break
        offset += len(line) + 1  # +1 for newline

    if not spans:
        return [], 0.0

    # Build end_char using next start - 1
    result: List[Dict] = []
    for idx, (label, title, start) in enumerate(spans):
        end = (spans[idx + 1][2] - 1) if idx + 1 < len(spans) else (len(text) - 1)
        # Normalize a few common titles
        norm_title = title
        if label == "Item 1A" and (not norm_title or "risk" in norm_title.lower()):
            norm_title = "Risk Factors"
        elif label == "Item 1" and (not norm_title or "business" in norm_title.lower()):
            norm_title = "Business"
        elif label == "Item 7" and (not norm_title or "discussion" in norm_title.lower()):
            norm_title = "Management’s Discussion and Analysis"
        elif label == "Item 7A" and (not norm_title or "risk" in norm_title.lower()):
            norm_title = "Market Risk"
        elif label == "Item 8" and (not norm_title or "financial" in norm_title.lower()):
            norm_title = "Financial Statements"
        result.append({"label": label, "title": norm_title, "start_char": start, "end_char": end})

    covered = sum(s["end_char"] - s["start_char"] + 1 for s in result)
    coverage = (covered / max(1, len(text)))
    return result, coverage


def main():
    ap = argparse.ArgumentParser(description="Parse SEC item structures and annotate normalized docs")
    ap.add_argument("--phase", required=True, choices=["A", "B"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    ensure_dir("logs/metadata")
    log_path = os.path.join("logs", "metadata", datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")

    # Load normalized docs
    paths = sorted(glob.glob("data/interim/normalized/*.json"))
    out_count = 0
    for p in paths:
        d = json.load(open(p, "r", encoding="utf-8"))
        doctype = (d.get("doctype") or "").lower()
        if doctype not in ("10-k", "10-q", "8-k", "ars_pdf"):
            continue
        doc_id = d.get("doc_id")
        if not phase_select(doc_id, args.phase):
            continue
        text = d.get("text") or ""
        spans, cov = find_item_spans(text)
        d["sec_item_spans"] = spans
        d["sec_item_coverage_ratio"] = round(cov, 6)
        if not args.dry_run:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{doc_id},{doctype},items_found={len(spans)},coverage={round(cov,4)},spans_valid={bool(spans)}\n")
        out_count += 1
        if args.limit and out_count >= args.limit:
            break

    print(f"Annotated SEC docs: {out_count}. Log: {log_path}")


if __name__ == "__main__":
    main()

