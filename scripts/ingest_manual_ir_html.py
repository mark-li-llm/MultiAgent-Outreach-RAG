#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest manually saved Salesforce IR HTML pages into data/raw/investor_news.

Usage:
  python3 scripts/ingest_manual_ir_html.py \
    --inbox data/manual_inbox/investor_news \
    --dest  data/raw/investor_news \
    --manifest data/manual_inbox/investor_news/manifest.csv
"""

import argparse
import csv
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urlunparse

# === project helpers (already in your repo) ===
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

IR_RSS_URL = "https://investor.salesforce.com/rss/pressrelease.aspx"


# -------------------------
# Utilities
# -------------------------

def read_bytes(p: Path) -> bytes:
    return p.read_bytes()

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def normalize_url_for_rss(u: str) -> str:
    """
    Normalize URL for robust matching with RSS 'link':
    - strip tracking params
    - force https scheme
    - remove trailing '/'
    - drop trailing '/default.aspx'
    """
    u = strip_tracking_params(u or "")
    if not u:
        return u
    pr = urlparse(u)
    scheme = "https"
    netloc = pr.netloc
    path = pr.path or ""
    # remove trailing 'default.aspx'
    if path.lower().endswith("/default.aspx"):
        path = path[: -len("/default.aspx")]
    # remove trailing slash
    if path.endswith("/") and path != "/":
        path = path[:-1]
    norm = urlunparse((scheme, netloc, path, "", "", ""))
    return norm

def alt_keys_for_url(u: str) -> Tuple[str, str]:
    """
    Return two map keys for tolerance:
      - normalized without default.aspx and no trailing slash (primary)
      - same but keep /default.aspx appended (secondary)
    """
    base = normalize_url_for_rss(u)
    if not base:
        return "", ""
    # Append '/default.aspx' for alternative lookup
    pr = urlparse(base)
    with_default = urlunparse((pr.scheme, pr.netloc, (pr.path or "") + "/default.aspx", "", "", ""))
    return base, with_default


# -------------------------
# HTML canonical extraction
# -------------------------

class _CanonParser(HTMLParser):
    """
    Robustly extract:
      - <link rel="canonical" href="...">
      - <meta property="og:url" content="...">
      - data-(canonical|share)-url="..."
    Works regardless of attribute ordering, across lines.
    """
    def __init__(self):
        super().__init__()
        self.canonical: Optional[str] = None
        self.og_url: Optional[str] = None
        self.data_url: Optional[str] = None

    def handle_starttag(self, tag, attrs):
        if self.canonical and self.og_url and self.data_url:
            return
        tag = tag.lower()
        ad = { (k.lower() if k else ""): (v or "") for k, v in attrs }

        if tag == "link":
            rel = ad.get("rel", "").lower()
            href = ad.get("href", "")
            if "canonical" in rel and href:
                self.canonical = href

        if tag == "meta":
            prop = ad.get("property", "").lower()
            content = ad.get("content", "")
            if prop == "og:url" and content:
                self.og_url = content

        # Q4 pages sometimes carry data-canonical-url / data-share-url on various tags
        for k, v in list(ad.items()):
            if k.startswith("data-") and v:
                if k in ("data-canonical-url", "data-share-url"):
                    self.data_url = v

def find_canonical(html: str) -> Optional[str]:
    p = _CanonParser()
    p.feed(html)
    url = p.canonical or p.og_url or p.data_url
    return strip_tracking_params(url) if url else None


# -------------------------
# RSS fetch & parse
# -------------------------

def rss_fetch(rss_url: str = IR_RSS_URL) -> Optional[str]:
    import urllib.request
    req = urllib.request.Request(
        rss_url,
        headers={
            "User-Agent": "AccountResearchMVP/1.0",
            "Accept": "application/rss+xml,text/xml;q=0.9,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception:
        return None

@dataclass
class RssItem:
    link: str
    iso_date: Optional[str]  # YYYY-MM-DD

def parse_rss_map(rss_xml: Optional[str]) -> Dict[str, RssItem]:
    """
    Build a map: normalized_url -> RssItem
    Also insert an alternative key with '/default.aspx' to maximize hit rate.
    """
    out: Dict[str, RssItem] = {}
    if not rss_xml:
        return out
    from xml.etree import ElementTree as ET
    from email.utils import parsedate_to_datetime

    try:
        root = ET.fromstring(rss_xml)
        for it in root.findall(".//item"):
            link = (it.findtext("link") or "").strip()
            pub = (it.findtext("pubDate") or "").strip()
            if not link:
                continue

            iso = None
            if pub:
                try:
                    dt = parsedate_to_datetime(pub)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    iso = dt.astimezone(timezone.utc).date().isoformat()
                except Exception:
                    iso = None

            item = RssItem(link=strip_tracking_params(link), iso_date=iso)

            k1, k2 = alt_keys_for_url(item.link)
            if k1:
                out[k1] = item
            if k2:
                out[k2] = item
    except Exception:
        pass

    return out


# -------------------------
# Manifest (filename -> url)
# -------------------------

def load_manifest(path: Optional[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path:
        return mapping
    p = Path(path)
    if not p.exists():
        return mapping
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = (row.get("filename") or "").strip()
            url = strip_tracking_params((row.get("url") or "").strip())
            if fn and url:
                mapping[fn] = url
    return mapping


# -------------------------
# Ingest logic
# -------------------------

def process_one_html(
    file_path: Path,
    dest_dir: Path,
    rss_map: Dict[str, RssItem],
    manifest_map: Dict[str, str],
) -> Tuple[str, str]:
    """
    Returns tuple: (status, message)
      status in {"OK", "SKIP", "WARN", "ERROR"}
    """
    raw = read_bytes(file_path)
    text = raw.decode("utf-8", errors="replace")

    # 1) Resolve URL (canonical -> manifest fallback)
    url = find_canonical(text)
    if not url:
        url = manifest_map.get(file_path.name, None)
    if not url:
        return ("WARN", f"Missing canonical URL for {file_path.name}; supply via manifest CSV.")

    url_norm = normalize_url_for_rss(url)

    # 2) Extract title & visible_date from HTML
    visible_title = extract_title(text) or None
    visible_date = try_parse_date_from_meta(text) or None

    # 3) RSS pubdate (if available)
    rss_item = rss_map.get(url_norm)
    rss_iso = rss_item.iso_date if rss_item else None

    # 4) Choose date for id
    date_for_id = rss_iso or visible_date  # QA uses visible_date || rss_pubdate

    # 5) Build doc_id and target paths
    doc_id = build_doc_id("press", date_for_id, visible_title or url, url_norm)
    raw_path = dest_dir / f"{doc_id}.raw.html"
    meta_path = dest_dir / f"{doc_id}.meta.json"

    if meta_path.exists():
        return ("SKIP", f"exists: {meta_path.name}")

    # 6) Write raw & meta
    ensure_dir(str(dest_dir))
    raw_path.write_bytes(raw)

    meta = {
        "doc_id": doc_id,
        "source_domain": domain_of(url_norm) or "investor.salesforce.com",
        "source_bucket": "investor_news",
        "doctype": "press",
        "requested_url": url_norm,
        "final_url": url_norm,
        "redirect_chain": [],
        "http_status": 200,
        "content_type": "text/html",
        "content_length": len(raw),
        "fetched_at": now_iso(),
        "sha256_raw": sha256_hex(raw),
        "visible_title": visible_title,
        "visible_date": visible_date,
        "rss_pubdate": rss_iso,
        "headline": visible_title,
        "notes": "manual_save=1" + ("; feed=ir_rss" if rss_iso else ""),
        "latency_ms": None,
    }
    write_json(str(meta_path), meta)
    return ("OK", doc_id)


def main():
    ap = argparse.ArgumentParser(description="Ingest manually saved IR HTML into data/raw/investor_news")
    ap.add_argument("--inbox", default="data/manual_inbox/investor_news", help="Folder containing *.html saved from browser")
    ap.add_argument("--dest", default="data/raw/investor_news", help="Destination folder for <doc_id>.raw.html and .meta.json")
    ap.add_argument("--manifest", default=None, help="Optional CSV mapping 'filename,url' to supply canonical when missing in HTML")
    ap.add_argument("--rss-url", default=IR_RSS_URL, help="Override IR RSS url if needed")
    args = ap.parse_args()

    inbox = Path(args.inbox)
    dest = Path(args.dest)
    ensure_dir(str(dest))

    manifest_map = load_manifest(args.manifest)

    rss_xml = rss_fetch(args.rss_url)
    rss_map = parse_rss_map(rss_xml)

    html_files = sorted(inbox.glob("*.html"))
    if not html_files:
        print(f"[INFO] No HTML files found in {inbox}")
        return

    ok = skip = warn = err = 0
    for p in html_files:
        status, msg = process_one_html(p, dest, rss_map, manifest_map)
        if status == "OK":
            ok += 1
            print(f"[OK]  {msg}")
        elif status == "SKIP":
            skip += 1
            print(f"[SKIP] {msg}")
        elif status == "WARN":
            warn += 1
            print(f"[WARN] {msg}")
        else:
            err += 1
            print(f"[ERROR] {msg}")

    print(f"[SUMMARY] OK={ok} SKIP={skip} WARN={warn} ERROR={err}")


if __name__ == "__main__":
    main()
