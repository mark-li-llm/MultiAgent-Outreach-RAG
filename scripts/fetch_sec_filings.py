#!/usr/bin/env python3
import argparse
import asyncio
import os
from datetime import datetime
from typing import Dict, List, Optional

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


SEC_ITEMS = [
    {
        "url": "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000006/crm-20250131.htm",
        "doctype": "10-K",
        "slug": "fy25-form-10-k",
        "visible_date_hint": "2025-03-05",
    },
    {
        "url": "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000030/crm-20250430.htm",
        "doctype": "10-Q",
        "slug": "fy26-q1-form-10-q",
        "visible_date_hint": None,
    },
    {
        "url": "https://www.sec.gov/ix?doc=%2FArchives%2Fedgar%2Fdata%2F1108524%2F000110852425000002%2Fcrm-20250226.htm",
        "doctype": "8-K",
        "slug": "fy25-results-8-k",
        "visible_date_hint": "2025-02-26",
    },
    {
        "url": "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000027/crm-20250528.htm",
        "doctype": "8-K",
        "slug": "q1-fy26-results-8-k",
        "visible_date_hint": "2025-05-28",
    },
    {
        "url": "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000019/salesforce_fy25annualreport.pdf",
        "doctype": "ars_pdf",
        "slug": "fy25-annual-report-pdf",
        "visible_date_hint": None,
    },
    {
        "url": "https://www.sec.gov/Archives/edgar/data/1108524/000110852425000033/crm-20250605.htm",
        "doctype": "8-K",
        "slug": "proxy-meeting-results-2025-06-05",
        "visible_date_hint": "2025-06-05",
    },
]


async def process_item(session: aiohttp.ClientSession, limiter: RateLimiter, item: Dict, out_dir: str, dry_run: bool, logger, defaults: Dict):
    url = strip_tracking_params(item["url"])
    doctype = item["doctype"]
    from common import domain_of
    source_bucket = "sec"
    from common import fetch_with_retries

    res = await fetch_with_retries(session, limiter, url, logger=logger)

    visible_title = None
    visible_date = item.get("visible_date_hint")
    content_type = res.content_type
    raw_bytes = res.body if res.body is not None else b""
    content_length = len(raw_bytes) if raw_bytes else None
    sha256_raw = sha256_hex(raw_bytes) if raw_bytes else ""

    # Attempt to parse title/date from HTML
    if raw_bytes and ("html" in content_type.lower()):
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        if not visible_title:
            visible_title = extract_title(text)
        if not visible_date:
            visible_date = try_parse_date_from_meta(text)

    # Build doc id
    slug_src = item.get("slug") or (visible_title or url)
    doc_id = build_doc_id(doctype, visible_date, slug_src, res.final_url or url)

    # File extensions
    ext = ".pdf" if "pdf" in (content_type or "").lower() or url.lower().endswith(".pdf") else ".raw.html"
    raw_path = os.path.join(out_dir, f"{doc_id}{ext}")
    meta_path = os.path.join(out_dir, f"{doc_id}.meta.json")

    meta = {
        "doc_id": doc_id,
        "source_domain": domain_of(res.final_url or url) or "sec.gov",
        "source_bucket": source_bucket,
        "doctype": doctype,
        "requested_url": url,
        "final_url": res.final_url,
        "redirect_chain": res.redirect_chain,
        "http_status": res.status,
        "content_type": content_type,
        "content_length": content_length,
        "fetched_at": defaults["run_iso"],
        "sha256_raw": sha256_raw,
        "visible_title": visible_title,
        "visible_date": visible_date,
        "rss_pubdate": None,
        "headline": visible_title,
        "notes": None,
        "latency_ms": res.latency_ms,
    }

    if dry_run:
        logger.info(f"[DRY] Would write {raw_path} and {meta_path} (status {res.status})")
        return 1

    # Always write meta; write raw only on 200 with body
    if res.status == 200 and raw_bytes:
        write_bytes(raw_path, raw_bytes)
    write_json(meta_path, meta)
    logger.info(f"Saved meta for {doc_id} (status {res.status})")
    return 1


async def main_async(args):
    logger, log_path = build_logger()
    ensure_dir("data/raw/sec")

    limiter = RateLimiter()
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    headers = default_session_headers()
    defaults = {"run_iso": now_iso(), "log_path": log_path}

    # Dedup by existing metas
    picked: List[Dict] = []
    for it in SEC_ITEMS:
        if len(picked) >= args.limit:
            break
        # Determine tentative doc id to check existence using hint date/title
        tentative_doc_id = build_doc_id(it["doctype"], it.get("visible_date_hint") or "unknown", it.get("slug") or it["url"], it["url"])
        meta_path = os.path.join("data/raw/sec", f"{tentative_doc_id}.meta.json")
        if file_exists(meta_path):
            try:
                from common import load_json
                meta = load_json(meta_path)
                if int(meta.get("http_status") or 0) == 200:
                    # Already good
                    continue
            except Exception:
                pass
        picked.append(it)

    if not picked:
        logger.info("Nothing to fetch; all items present or limit=0")
        return 0

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = []
        for it in picked:
            tasks.append(process_item(session, limiter, it, "data/raw/sec", args.dry_run, logger, defaults))
        done = await asyncio.gather(*tasks)
        logger.info(f"Fetched {sum(done)} SEC items. Logs: {log_path}")
        return sum(done)


def parse_args():
    p = argparse.ArgumentParser(description="Download specific SEC filings for Salesforce (CRM)")
    p.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    p.add_argument("--limit", type=int, default=10, help="Max pages to fetch this run")
    p.add_argument("--since", type=str, default=None, help="Unused for SEC direct fetch")
    p.add_argument("--until", type=str, default=None, help="Unused for SEC direct fetch")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per host")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
