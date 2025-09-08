#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

from common import ensure_dir


FIELDS = [
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
    "word_count",
    "token_count",
    "ingestion_ts",
    "hash_sha256",
]


def main():
    ap = argparse.ArgumentParser(description="Build final inventory CSV from normalized docs")
    args = ap.parse_args()

    paths = sorted(glob.glob("data/interim/normalized/*.json"))
    rows: List[Dict[str, Any]] = []
    excluded = 0
    by_dt = Counter()
    for p in paths:
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        # Exclude if missing publish_date
        if not (d.get("publish_date") or "").strip():
            excluded += 1
            continue
        r = {}
        r["doc_id"] = d.get("doc_id")
        r["company"] = d.get("company") or "Salesforce"
        r["doctype"] = d.get("doctype")
        r["title"] = d.get("title") or ""
        r["publish_date"] = d.get("publish_date")
        r["url"] = d.get("url") or d.get("final_url") or ""
        r["final_url"] = d.get("final_url") or d.get("url") or ""
        r["source_domain"] = d.get("source_domain") or ""
        r["section"] = d.get("section") or "body"
        r["topic"] = d.get("topic") or ""
        pt = d.get("persona_tags") or []
        if isinstance(pt, list):
            r["persona_tags"] = ",".join(pt)
        else:
            r["persona_tags"] = str(pt)
        r["language"] = d.get("language") or "en"
        r["word_count"] = int(d.get("word_count") or 0)
        r["token_count"] = int(d.get("token_count") or 0)
        r["ingestion_ts"] = d.get("ingestion_ts") or ""
        r["hash_sha256"] = d.get("hash_sha256") or ""
        rows.append(r)
        by_dt[r["doctype"]] += 1

    ensure_dir("data/final/inventory")
    out_csv = "data/final/inventory/salesforce_inventory.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Inventory rows written: {len(rows)}; excluded_missing_dates: {excluded}")
    print("Doctype breakdown:", dict(by_dt))


if __name__ == "__main__":
    main()

