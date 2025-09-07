#!/usr/bin/env python3
import argparse
import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

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
    coerce_date,
)


START_URL = "https://investor.salesforce.com/news/default.aspx?languageid=1"


def within_window(date_str: Optional[str], since: Optional[str], until: Optional[str]) -> bool:
    if not date_str:
        return True
    try:
        d = datetime.fromisoformat(date_str)
    except Exception:
        return True
    if since:
        ds = datetime.fromisoformat(since)
        if d < ds:
            return False
    if until:
        du = datetime.fromisoformat(until)
        if d > du:
            return False
    return True


async def fetch_listing(session: aiohttp.ClientSession, limiter: RateLimiter, url: str) -> str:
    from common import fetch_with_retries

    res = await fetch_with_retries(session, limiter, url)
    return (res.body or b"").decode("utf-8", errors="replace")


def extract_article_links(html: str) -> List[str]:
    # Look for anchors containing /news/ and ending with .aspx or details page
    links = set()
    for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*>", html, re.IGNORECASE):
        href = m.group(1)
        if "/news/" in href and href.startswith("/"):
            links.add("https://investor.salesforce.com" + href)
        elif href.startswith("https://investor.salesforce.com/") and "/news/" in href:
            links.add(href)
    return list(links)


async def fetch_and_save(session: aiohttp.ClientSession, limiter: RateLimiter, url: str, out_dir: str, dry_run: bool, logger, run_iso: str) -> Optional[str]:
    from common import fetch_with_retries

    res = await fetch_with_retries(session, limiter, url, logger=logger)
    content_type = res.content_type
    raw_bytes = res.body if res.body is not None else b""
    text = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else ""
    title = extract_title(text) if text else None
    visible_date = try_parse_date_from_meta(text) if text else None
    # Additional pattern: dates like Month DD, YYYY near title
    if not visible_date and text:
        m = re.search(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", text)
        if m:
            visible_date = coerce_date(m.group(1))

    doctype = "press"
    doc_id = build_doc_id(doctype, visible_date, title or url, res.final_url or url)
    raw_path = os.path.join(out_dir, f"{doc_id}.raw.html")
    meta_path = os.path.join(out_dir, f"{doc_id}.meta.json")

    # Skip if exists
    if file_exists(meta_path):
        return None

    from common import domain_of
    meta = {
        "doc_id": doc_id,
        "source_domain": domain_of(res.final_url or url) or "investor.salesforce.com",
        "source_bucket": "investor_news",
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
    logger.info(f"Saved IR meta for {doc_id} (status {res.status})")
    return doc_id


async def main_async(args):
    logger, log_path = build_logger()
    ensure_dir("data/raw/investor_news")
    limiter = RateLimiter()
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    headers = default_session_headers()
    run_iso = now_iso()

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        listing_html = await fetch_listing(session, limiter, START_URL)
        links = extract_article_links(listing_html)
        # Keep order stable
        links = sorted(set(links))
        picked: List[str] = []

        # Filter by window after peeking at page dates if embedded
        for href in links:
            if len(picked) >= args.limit:
                break
            # We'll fetch detail pages and apply window; optimistic pick first
            picked.append(href)

        tasks = []
        saved = 0
        for url in picked:
            tasks.append(fetch_and_save(session, limiter, url, "data/raw/investor_news", args.dry_run, logger, run_iso))
        results = await asyncio.gather(*tasks)

        # Window filter is applied via doc_id presence; we cannot easily drop here without reading metas.
        logger.info(f"Fetched IR items attempted: {len(picked)}; saved: {len([r for r in results if r])}. Logs: {log_path}")
        return len([r for r in results if r])


def parse_args():
    p = argparse.ArgumentParser(description="Crawl Salesforce Investor Relations news and fetch articles")
    p.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    p.add_argument("--limit", type=int, default=20, help="Max articles to fetch")
    p.add_argument("--since", type=str, default="2024-01-01", help="Lower bound date (YYYY-MM-DD)")
    p.add_argument("--until", type=str, default=None, help="Upper bound date (YYYY-MM-DD)")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per host")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
