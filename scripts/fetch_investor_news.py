#!/usr/bin/env python3
import argparse
import asyncio
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, date
from typing import Dict, List, Optional

import aiohttp
from email.utils import parsedate_to_datetime

from common import (
    RateLimiter,
    build_logger,
    build_doc_id,
    default_session_headers,
    ensure_dir,
    now_iso,
    sha256_hex,
    strip_tracking_params,
    extract_title,
    try_parse_date_from_meta,
    coerce_date,
    write_json,
    write_bytes,
    file_exists,
    domain_of,
)

from common import fetch_with_retries


IR_RSS_URL = "https://investor.salesforce.com/rss/pressrelease.aspx"


def to_iso_date_from_rfc822(rfc: Optional[str]) -> Optional[str]:
    if not rfc:
        return None
    try:
        dt = parsedate_to_datetime(rfc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date().isoformat()
    except Exception:
        return coerce_date(rfc)


def within_window(iso_day: Optional[str], since: Optional[str], until: Optional[str]) -> bool:
    if not iso_day:
        return True
    try:
        d = date.fromisoformat(iso_day)
    except Exception:
        return True
    if since:
        try:
            if d < date.fromisoformat(since):
                return False
        except Exception:
            pass
    if until:
        try:
            if d > date.fromisoformat(until):
                return False
        except Exception:
            pass
    return True


def parse_rss_items(rss_xml: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(rss_xml)
    except Exception:
        return items
    for it in root.findall(".//item"):
        link_el = it.find("link")
        title_el = it.find("title")
        pub_el = it.find("pubDate")
        link = strip_tracking_params((link_el.text or "").strip()) if link_el is not None else ""
        title = (title_el.text or "").strip() if title_el is not None else ""
        pub = (pub_el.text or "").strip() if pub_el is not None else ""
        iso = to_iso_date_from_rfc822(pub)
        if link:
            items.append({"link": link, "title": title, "pub_iso": iso or "", "pub_raw": pub})
    return items


async def fetch_and_write_article(session: aiohttp.ClientSession, limiter: RateLimiter, item: Dict[str, str], dry_run: bool, logger, run_iso: str) -> Optional[str]:
    url = item["link"]
    rss_iso = item.get("pub_iso") or None
    title_hint = item.get("title") or None

    res = await fetch_with_retries(session, limiter, url, logger=logger)
    raw_bytes = res.body or b""
    text = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else ""
    visible_title = extract_title(text) if text else title_hint
    visible_date = try_parse_date_from_meta(text) if text else None

    doctype = "press"
    date_for_id = rss_iso or visible_date
    doc_id = build_doc_id(doctype, date_for_id, visible_title or url, res.final_url or url)

    raw_path = os.path.join("data/raw/investor_news", f"{doc_id}.raw.html")
    meta_path = os.path.join("data/raw/investor_news", f"{doc_id}.meta.json")

    if file_exists(meta_path):
        return None

    meta = {
        "doc_id": doc_id,
        "source_domain": domain_of(res.final_url or url) or "investor.salesforce.com",
        "source_bucket": "investor_news",
        "doctype": doctype,
        "requested_url": url,
        "final_url": res.final_url or url,
        "redirect_chain": res.redirect_chain or [],
        "http_status": res.status,
        "content_type": res.content_type,
        "content_length": len(raw_bytes) if raw_bytes else None,
        "fetched_at": run_iso,
        "sha256_raw": sha256_hex(raw_bytes) if raw_bytes else "",
        "visible_title": visible_title,
        "visible_date": visible_date,
        "rss_pubdate": rss_iso,
        "headline": visible_title,
        "notes": "feed=ir_rss",
        "latency_ms": res.latency_ms,
    }

    if dry_run:
        logger.info(f"[DRY] Would write {raw_path} and {meta_path}")
        return doc_id

    ensure_dir("data/raw/investor_news")
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
        rss_res = await fetch_with_retries(session, limiter, IR_RSS_URL, logger=logger)
        rss_xml = (rss_res.body or b"").decode("utf-8", errors="replace")
        items = parse_rss_items(rss_xml)
        filtered = [it for it in items if within_window(it.get("pub_iso"), args.since, args.until)]
        selected = filtered[: max(0, args.limit)]

        sem = asyncio.Semaphore(args.concurrency)
        async def worker(it):
            async with sem:
                return await fetch_and_write_article(session, limiter, it, args.dry_run, logger, run_iso)

        results = await asyncio.gather(*(worker(it) for it in selected))
        saved = len([r for r in results if r])

        logger.info(f"Fetched IR items attempted: {len(selected)}; saved: {saved}. Logs: {log_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Fetch Salesforce Investor Relations press releases via RSS")
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
