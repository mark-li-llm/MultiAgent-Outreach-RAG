#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone, date
from typing import Any, Dict

from common import ensure_dir


def main():
    ap = argparse.ArgumentParser(description="Compute Day-1 verification metrics")
    args = ap.parse_args()

    # Load normalized docs
    docs = []
    for p in glob.glob("data/interim/normalized/*.json"):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            docs.append(d)
        except Exception:
            continue
    total_docs = len(docs)
    docs_with_dates = sum(1 for d in docs if (d.get("publish_date") or "").strip())

    # Chunks after dedupe
    chunks_total = 0
    for cf in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(cf, "r", encoding="utf-8") as f:
            for _ in f:
                chunks_total += 1

    # Duplicates removed
    try:
        ded = json.load(open("data/interim/dedup/dedup_map.json", "r", encoding="utf-8"))
        duplicates_removed = sum(len(g.get("duplicate_chunk_ids") or []) for g in ded.get("groups", []))
    except Exception:
        duplicates_removed = 0

    # Link health
    try:
        lh = json.load(open("data/final/reports/link_health.json", "r", encoding="utf-8"))
        ok = sum(1 for e in lh if e.get("link_ok") is True)
        link_ok_pct = ok / max(1, len(lh))
    except Exception:
        link_ok_pct = 0.0

    # Doctype counts
    by_dt = Counter((d.get("doctype") or "") for d in docs)

    # Source domain counts (root domains only)
    def root(host: str) -> str:
        h = (host or "").lower()
        return h.split(":")[0]
    by_domain = Counter(root(d.get("source_domain") or "") for d in docs)

    # PR docs last 12 months
    nowd = datetime.now(timezone.utc).date()
    pr_last_12 = 0
    for d in docs:
        if (d.get("doctype") or "").lower() == "press" and (d.get("publish_date") or ""):
            try:
                dd = date.fromisoformat(d.get("publish_date"))
                if (nowd - dd).days <= 365:
                    pr_last_12 += 1
            except Exception:
                pass

    out = {
        "total_docs": total_docs,
        "docs_with_dates": docs_with_dates,
        "chunks_total": chunks_total,
        "duplicates_removed": duplicates_removed,
        "link_ok_pct": round(link_ok_pct, 4),
        "by_doctype_counts": {
            "10-K": int(by_dt.get("10-K", 0)),
            "10-Q": int(by_dt.get("10-Q", 0)),
            "8-K": int(by_dt.get("8-K", 0)),
            "ars_pdf": int(by_dt.get("ars_pdf", 0)),
            "press": int(by_dt.get("press", 0)),
            "product": int(by_dt.get("product", 0)),
            "dev_docs": int(by_dt.get("dev_docs", 0)),
            "help_docs": int(by_dt.get("help_docs", 0)),
            "wiki": int(by_dt.get("wiki", 0)),
        },
        "by_source_domain_counts": {
            k: int(v) for k, v in by_domain.items()
        },
        "pr_docs_last_12mo": pr_last_12,
    }

    ensure_dir("data/final/reports")
    with open("data/final/reports/day1_verification.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Wrote data/final/reports/day1_verification.json")


if __name__ == "__main__":
    main()

