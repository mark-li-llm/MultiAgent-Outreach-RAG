#!/usr/bin/env python3
import argparse
import asyncio
import os
from typing import List

import aiohttp

from common import (
    RateLimiter,
    build_logger,
    build_doc_id,
    default_session_headers,
    ensure_dir,
    extract_title,
    now_iso,
    sha256_hex,
    strip_tracking_params,
    try_parse_date_from_meta,
    write_bytes,
    write_json,
    file_exists,
)


URLS = [
    "https://www.salesforce.com/agentforce/",
    "https://www.salesforce.com/data/",
    "https://www.salesforce.com/products/what-is-salesforce/",
]


async def fetch_and_save(session, limiter, url: str, out_dir: str, dry_run: bool, logger, run_iso: str):
    from common import fetch_with_retries
    res = await fetch_with_retries(session, limiter, url, logger=logger)
    content_type = res.content_type
    raw_bytes = res.body if res.body is not None else b""
    text = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else ""
    title = extract_title(text) if text else None
    visible_date = None
    doctype = "product"
    doc_id = build_doc_id(doctype, visible_date, title or url, res.final_url or url)

    raw_path = os.path.join(out_dir, f"{doc_id}.raw.html")
    meta_path = os.path.join(out_dir, f"{doc_id}.meta.json")
    if file_exists(meta_path):
        return None

    from common import domain_of
    meta = {
        "doc_id": doc_id,
        "source_domain": domain_of(res.final_url or url) or "www.salesforce.com",
        "source_bucket": "product",
        "doctype": doctype,
        "requested_url": strip_tracking_params(url),
        "final_url": res.final_url,
        "redirect_chain": res.redirect_chain,
        "http_status": res.status,
        "content_type": content_type,
        "content_length": len(raw_bytes) if raw_bytes else None,
        "fetched_at": run_iso,
        "sha256_raw": sha256_hex(raw_bytes) if raw_bytes else "",
        "visible_title": title,
        "visible_date": visible_date,
        "rss_pubdate": None,
        "headline": title,
        "notes": None,
        "latency_ms": res.latency_ms,
    }

    if dry_run:
        logger.info(f"[DRY] Would write {raw_path} and {meta_path} (status {res.status})")
        return doc_id

    if res.status == 200 and raw_bytes:
        write_bytes(raw_path, raw_bytes)
    write_json(meta_path, meta)
    logger.info(f"Saved Product meta for {doc_id} (status {res.status})")
    return doc_id


async def main_async(args):
    logger, log_path = build_logger()
    ensure_dir("data/raw/product")
    limiter = RateLimiter()
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    headers = default_session_headers()
    run_iso = now_iso()

    pick = URLS[: args.limit]
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = [
            fetch_and_save(session, limiter, u, "data/raw/product", args.dry_run, logger, run_iso)
            for u in pick
        ]
        results = await asyncio.gather(*tasks)
        logger.info(f"Fetched Product pages attempted: {len(pick)}; saved: {len([r for r in results if r])}. Logs: {log_path}")
        return len([r for r in results if r])


def parse_args():
    p = argparse.ArgumentParser(description="Fetch Salesforce product/overview pages")
    p.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    p.add_argument("--limit", type=int, default=3, help="Max pages to fetch")
    p.add_argument("--since", type=str, default=None)
    p.add_argument("--until", type=str, default=None)
    p.add_argument("--concurrency", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
