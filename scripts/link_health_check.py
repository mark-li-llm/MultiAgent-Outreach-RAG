#!/usr/bin/env python3
import argparse
import asyncio
import glob
import json
import os
import ssl
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import certifi

from common import ensure_dir, strip_tracking_params, now_iso


ALLOWLIST = {
    "sec.gov",
    "investor.salesforce.com",
    "www.salesforce.com",
    "salesforce.com",
    "developer.salesforce.com",
    "help.salesforce.com",
    "en.wikipedia.org",
    "wikipedia.org",
}


def host_of(url: str) -> str:
    from urllib.parse import urlparse

    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def with_https(url: str) -> str:
    from urllib.parse import urlparse, urlunparse

    try:
        p = urlparse(url)
        if p.scheme == "https":
            return url
        return urlunparse(("https", p.netloc, p.path, p.params, p.query, p.fragment))
    except Exception:
        return url


def default_ssl_for_host(host: str):
    # For sec.gov, disable verification due to local CA issues noted earlier
    if host.endswith("sec.gov"):
        return False
    try:
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return None


async def resolve_url(session: aiohttp.ClientSession, url: str, timeout_s: int = 5) -> Tuple[int, str, List[str], Optional[str]]:
    """Return (status, final_url, redirect_chain, last_modified) using GET then HEAD fallback."""
    from aiohttp import ClientTimeout

    url = strip_tracking_params(url)
    host = host_of(url)
    ssl_opt = default_ssl_for_host(host)

    backoffs = [0, 1, 2, 4]
    for attempt, back in enumerate(backoffs):
        if back:
            await asyncio.sleep(back)
        try:
            async with session.get(url, allow_redirects=True, max_redirects=5, timeout=ClientTimeout(total=timeout_s), ssl=ssl_opt) as resp:
                # Read small amount to ensure headers
                await resp.read()
                history = [strip_tracking_params(str(h.url)) for h in resp.history]
                final_url = strip_tracking_params(str(resp.url))
                lm = resp.headers.get("Last-Modified")
                return resp.status, final_url, history, lm
        except Exception:
            continue
    # Fallback HEAD
    try:
        async with session.head(url, allow_redirects=True, max_redirects=5, timeout=ClientTimeout(total=timeout_s), ssl=ssl_opt) as resp:
            await resp.read()
            history = [strip_tracking_params(str(h.url)) for h in resp.history]
            final_url = strip_tracking_params(str(resp.url))
            lm = resp.headers.get("Last-Modified")
            return resp.status, final_url, history, lm
    except Exception:
        return 0, url, [], None


async def worker(doc: Dict[str, Any], session: aiohttp.ClientSession, results: List[Dict[str, Any]]):
    url = (doc.get("final_url") or doc.get("url") or "").strip()
    url = strip_tracking_params(url)
    # Prefer HTTPS where possible
    candidates = []
    if url.startswith("http://"):
        candidates = [with_https(url), url]
    else:
        candidates = [url]
    status = 0
    final_url = url
    chain: List[str] = []
    last_mod: Optional[str] = None
    for u in candidates:
        status, final_url, chain, last_mod = await resolve_url(session, u)
        if status == 200:
            break
    ok = (status == 200)
    # Update doc in place
    doc["final_url"] = final_url
    doc["link_ok"] = ok
    doc["status_code"] = status
    doc["redirect_chain"] = chain
    if last_mod:
        doc["last_modified_http"] = last_mod
    doc["link_checked_at"] = now_iso()
    results.append({
        "doc_id": doc.get("doc_id"),
        "requested_url": url,
        "final_url": final_url,
        "status_code": status,
        "link_ok": ok,
        "redirect_chain": chain,
        "last_modified": last_mod,
        "checked_at": now_iso(),
    })


async def main_async(args):
    ensure_dir("logs/link")
    ensure_dir("data/final/reports")
    # Load docs
    docs = []
    for p in sorted(glob.glob("data/interim/normalized/*.json")):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            d["__path"] = p
            docs.append(d)
        except Exception:
            continue
    if args.limit:
        docs = docs[: args.limit]

    headers = {"User-Agent": os.environ.get("AR_USER_AGENT", "AccountResearchMVP/1.0"), "Accept": "*/*"}
    connector = aiohttp.TCPConnector(limit_per_host=args.concurrency)
    results: List[Dict[str, Any]] = []
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        tasks = []
        for d in docs:
            tasks.append(worker(d, session, results))
        await asyncio.gather(*tasks)

    # Write updated docs atomically
    for d in docs:
        p = d.pop("__path", None)
        if not p:
            continue
        if args.dry_run:
            continue
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)

    # Aggregate report
    with open("data/final/reports/link_health.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Checked {len(results)} docs. Wrote data/final/reports/link_health.json")


def main():
    ap = argparse.ArgumentParser(description="Link health and canonicalization scan")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

