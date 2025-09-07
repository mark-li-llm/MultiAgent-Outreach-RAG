#!/usr/bin/env python3
import argparse
import hashlib
import os
from pathlib import Path
from typing import Optional

from common import (
    build_doc_id,
    ensure_dir,
    extract_title,
    try_parse_date_from_meta,
    strip_tracking_params,
    write_json,
    domain_of,
    now_iso,
)


def read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def sha256_hex(b: bytes) -> str:
    import hashlib as _h

    return _h.sha256(b).hexdigest()


def find_canonical_simple(html: str) -> Optional[str]:
    import re

    m = re.search(r"<link[^>]+rel=\"canonical\"[^>]+href=\"(.*?)\"", html, re.I)
    if m:
        return strip_tracking_params(m.group(1))
    m = re.search(r"<meta[^>]+property=\"og:url\"[^>]+content=\"(.*?)\"", html, re.I)
    if m:
        return strip_tracking_params(m.group(1))
    return None


def doctype_for_bucket(bucket: str) -> str:
    mapping = {
        "dev_docs": "dev_docs",
        "help_docs": "help_docs",
        "product": "product",
        "wikipedia": "wiki",
    }
    return mapping.get(bucket, bucket)


def main():
    ap = argparse.ArgumentParser(description="Ingest manually saved HTML into raw buckets (dev_docs/help_docs/product/wiki)")
    ap.add_argument("--inbox", required=True, help="Folder containing *.html saved from browser")
    ap.add_argument("--dest", required=True, help="Destination folder under data/raw/<bucket>")
    ap.add_argument("--bucket", required=True, choices=["dev_docs", "help_docs", "product", "wikipedia"], help="Target bucket name")
    args = ap.parse_args()

    inbox = Path(args.inbox)
    dest = Path(args.dest)
    ensure_dir(str(dest))

    dtype = doctype_for_bucket(args.bucket)

    html_files = sorted(inbox.glob("*.html"))
    if not html_files:
        print(f"[INFO] No HTML files found in {inbox}")
        return

    ok = skip = 0
    for fp in html_files:
        raw = read_bytes(fp)
        text = raw.decode("utf-8", errors="replace")
        url = find_canonical_simple(text) or ""
        title = extract_title(text) or None
        visible_date = try_parse_date_from_meta(text) or None
        url_norm = strip_tracking_params(url)
        doc_id = build_doc_id(dtype, visible_date, title or (url_norm or fp.stem), url_norm or ("file://" + fp.name))
        raw_path = dest / f"{doc_id}.raw.html"
        meta_path = dest / f"{doc_id}.meta.json"
        if meta_path.exists():
            skip += 1
            print(f"[SKIP] exists: {meta_path.name}")
            continue
        ensure_dir(str(dest))
        raw_path.write_bytes(raw)
        meta = {
            "doc_id": doc_id,
            "source_domain": domain_of(url_norm) or ("developer.salesforce.com" if dtype == "dev_docs" else ("help.salesforce.com" if dtype == "help_docs" else "")),
            "source_bucket": args.bucket,
            "doctype": dtype,
            "requested_url": url_norm or "",
            "final_url": url_norm or "",
            "redirect_chain": [],
            "http_status": 200,
            "content_type": "text/html",
            "content_length": len(raw),
            "fetched_at": now_iso(),
            "sha256_raw": sha256_hex(raw),
            "visible_title": title,
            "visible_date": visible_date,
            "rss_pubdate": None,
            "headline": title,
            "notes": "manual_save=1",
            "latency_ms": None,
        }
        write_json(str(meta_path), meta)
        ok += 1
        print(f"[OK]  {doc_id}")

    print(f"[SUMMARY] OK={ok} SKIP={skip}")


if __name__ == "__main__":
    main()

