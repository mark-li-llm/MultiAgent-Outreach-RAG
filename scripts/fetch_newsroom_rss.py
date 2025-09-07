#!/usr/bin/env python3
import argparse
import asyncio
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
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


FEEDS = [
    ("corporate", "https://www.salesforce.com/news/content-types/press-releases-corporate/feed/"),
    ("product", "https://www.salesforce.com/news/content-types/press-releases-product/feed/"),
]


async def fetch_text(session: aiohttp.ClientSession, limiter: RateLimiter, url: str) -> str:
    from common import fetch_with_retries

    res = await fetch_with_retries(session, limiter, url)
    return (res.body or b"").decode("utf-8", errors="replace")


def parse_rss(xml_text: str) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return items
    # Find all <item>
    for item in root.findall(".//item"):
        link_el = item.find("link")
        title_el = item.find("title")
        pub_el = item.find("pubDate") or item.find("date")
        link = link_el.text.strip() if link_el is not None and link_el.text else None
        if not link:
            continue
        title = title_el.text.strip() if title_el is not None and title_el.text else None
        pub = pub_el.text.strip() if pub_el is not None and pub_el.text else None
        items.append({"link": strip_tracking_params(link), "title": title or "", "pubDate": pub or ""})
    return items


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


async def fetch_article(session: aiohttp.ClientSession, limiter: RateLimiter, url: str, feed_name: str, out_dir: str, dry_run: bool, logger, run_iso: str) -> Optional[str]:
    from common import fetch_with_retries
    res = await fetch_with_retries(session, limiter, url, logger=logger)
    content_type = res.content_type
    raw_bytes = res.body if res.body is not None else b""
    text = raw_bytes.decode("utf-8", errors="replace") if raw_bytes else ""
    title = extract_title(text) if text else None
    visible_date = try_parse_date_from_meta(text) if text else None

    doctype = "press"
    doc_id = build_doc_id(doctype, visible_date, title or url, res.final_url or url)
    raw_path = os.path.join(out_dir, f"{doc_id}.raw.html")
    meta_path = os.path.join(out_dir, f"{doc_id}.meta.json")

    if file_exists(meta_path):
        return None

    from common import domain_of
    final_domain = domain_of(res.final_url or url)
    # Skip off-site links (e.g., external coverage) when crawling the index
    if final_domain and not final_domain.endswith("salesforce.com"):
        logger.info(f"Skipping external domain article: {res.final_url}")
        return None
    meta = {
        "doc_id": doc_id,
        "source_domain": final_domain or "www.salesforce.com",
        "source_bucket": "newsroom",
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
        "notes": f"feed={feed_name}",
        "latency_ms": res.latency_ms,
    }

    if dry_run:
        logger.info(f"[DRY] Would write {raw_path} and {meta_path} (status {res.status})")
        return doc_id

    if res.status == 200 and raw_bytes:
        write_bytes(raw_path, raw_bytes)
    write_json(meta_path, meta)
    logger.info(f"Saved Newsroom meta for {doc_id} (status {res.status})")
    return doc_id


def extract_index_links(html: str) -> List[str]:
    links = []
    for m in re.finditer(r"<a[^>]+href=\"([^\"]+)\"[^>]*>", html, re.IGNORECASE):
        href = m.group(1)
        if href.startswith("/"):
            href = "https://www.salesforce.com" + href
        if href.startswith("https://www.salesforce.com/news/") and "/feed/" not in href:
            links.append(strip_tracking_params(href))
    # Keep unique order
    seen = set()
    ordered = []
    for u in links:
        if u not in seen:
            seen.add(u)
            ordered.append(u)
    return ordered


async def main_async(args):
    logger, log_path = build_logger()
    ensure_dir("data/raw/newsroom")
    limiter = RateLimiter()
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    headers = default_session_headers()
    run_iso = now_iso()

    # Split limit roughly equally
    total_limit = args.limit
    per_feed = max(1, total_limit // 2)

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        total_saved = 0
        for feed_name, feed_url in FEEDS:
            xml_text = await fetch_text(session, limiter, feed_url)
            items = parse_rss(xml_text)
            # Normalize and filter
            norm_items: List[Tuple[str, str, Optional[str]]] = []
            for it in items:
                pub_iso = coerce_date(it.get("pubDate", ""))
                if args.since or args.until:
                    if not within_window(pub_iso, args.since, args.until):
                        continue
                norm_items.append((it["link"], it.get("title", ""), pub_iso))
            picked = norm_items[:per_feed]

            tasks = []
            for link, _, _ in picked:
                tasks.append(fetch_article(session, limiter, link, feed_name, "data/raw/newsroom", args.dry_run, logger, run_iso))
            results = await asyncio.gather(*tasks)
            total_saved += len([r for r in results if r])
        # Optional index crawl to top up
        if args.use_index and total_saved < args.limit:
            index_url = "https://www.salesforce.com/news/all-news-press-salesforce/"
            index_html = await fetch_text(session, limiter, index_url)
            links = extract_index_links(index_html)
            need = max(0, args.limit - total_saved)
            picked = links[:need]
            tasks = []
            for link in picked:
                tasks.append(fetch_article(session, limiter, link, "index", "data/raw/newsroom", args.dry_run, logger, run_iso))
            results = await asyncio.gather(*tasks)
            total_saved += len([r for r in results if r])
        logger.info(f"Fetched Newsroom RSS: saved ~{total_saved} items. Logs: {log_path}")
        return total_saved


def parse_args():
    p = argparse.ArgumentParser(description="Parse Salesforce Newsroom RSS feeds and fetch linked articles")
    p.add_argument("--dry-run", action="store_true", help="Log actions without writing files")
    p.add_argument("--limit", type=int, default=30, help="Total items to fetch across both feeds")
    p.add_argument("--since", type=str, default="2024-01-01", help="Lower bound date (YYYY-MM-DD)")
    p.add_argument("--until", type=str, default=None, help="Upper bound date (YYYY-MM-DD)")
    p.add_argument("--concurrency", type=int, default=4, help="Concurrent requests per host")
    p.add_argument("--use-index", action="store_true", help="Also crawl All News & Press index for additional items")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
