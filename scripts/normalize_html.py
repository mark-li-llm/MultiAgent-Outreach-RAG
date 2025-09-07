#!/usr/bin/env python3
import argparse
import concurrent.futures
import glob
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from common import ensure_dir, now_iso, sha256_hex, build_logger


def load_yaml(path: str) -> Dict:
    try:
        import yaml  # type: ignore
    except Exception:
        raise SystemExit("PyYAML required. Install with: python3 -m pip install pyyaml bs4 tiktoken langdetect pdfminer.six")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cl100k_token_count(text: str) -> int:
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def detect_lang(text: str) -> str:
    sample = text[:8000]
    sample = re.sub(r"\s+", " ", sample)
    try:
        from langdetect import detect  # type: ignore

        return detect(sample)
    except Exception:
        return "en"


def html_extract_title(html: str) -> Tuple[Optional[str], Optional[str]]:
    # returns (title, meta_published_time)
    title = None
    meta_time = None
    try:
        m = re.search(r"<meta[^>]+property=\"og:title\"[^>]+content=\"(.*?)\"", html, re.I)
        if m:
            title = m.group(1).strip()
        if not title:
            m = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            if m:
                title = re.sub(r"<[^>]+>", " ", m.group(1)).strip()
        m = re.search(r"<meta[^>]+(name|property)=\"(article:published_time|pubdate|date|dc.date)\"[^>]+content=\"(.*?)\"", html, re.I)
        if m:
            meta_time = m.group(3).strip()
    except Exception:
        pass
    return title, meta_time


def strip_tracking_params_from_links(soup: BeautifulSoup) -> None:
    from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

    for a in soup.find_all("a", href=True):
        href = a.get("href")
        try:
            p = urlparse(href)
            q = [(k, v) for k, v in parse_qsl(p.query) if not (k.startswith("utm_") or k in {"gclid", "fbclid"})]
            a["href"] = urlunparse((p.scheme, p.netloc, p.path, p.params, urlencode(q), p.fragment))
        except Exception:
            continue


def normalize_html_bytes(html_bytes: bytes, rules: Dict) -> Tuple[str, int, int, bool]:
    html = html_bytes.decode("utf-8", errors="replace")
    soup_all = BeautifulSoup(html, "html.parser")

    # Prefer preserved containers (before removal) to compute before_text relative to main content
    kept_before = None
    for sel in rules.get("preserve_selectors", []):
        nodes = soup_all.select(sel)
        if nodes:
            kept_before = BeautifulSoup("", "html.parser")
            for n in nodes:
                kept_before.append(n)
            break
    if kept_before is None:
        kept_before = soup_all

    before_text = kept_before.get_text("\n")

    # Work on a fresh soup limited to kept_before for stripping
    soup = BeautifulSoup(str(kept_before), "html.parser")

    # Remove unwanted selectors
    for sel in rules.get("remove_selectors", []):
        for node in soup.select(sel):
            node.decompose()

    kept = soup

    # Replace <br> with newline
    for br in kept.find_all(["br", "br/"]):
        br.replace_with("\n")

    # Keep headings H1-H3 as lines with explicit markers
    for lvl, tag in enumerate(rules.get("heading_levels", ["h1", "h2", "h3"])):
        for h in kept.find_all(tag):
            text = h.get_text(" ", strip=True)
            h.replace_with(kept.new_string(f"H{lvl+1}: {text}\n"))

    # Insert newlines after certain blocks
    for tag in rules.get("newline_blocks", []):
        for el in kept.find_all(tag):
            if el.string:
                el.string.replace_with(str(el.string) + "\n")

    strip_tracking_params_from_links(kept)

    text = kept.get_text("\n")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\s*\n\s*)+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    after_len = len(text)
    before_len = len(before_text)
    return text, before_len, after_len, True


def extract_pdf_text(pdf_path: str) -> Tuple[str, Optional[List[Dict]], int, int]:
    try:
        from pdfminer.high_level import extract_text  # type: ignore

        raw = extract_text(pdf_path) or ""
        text = re.sub(r"\s+", " ", raw)
        text = re.sub(r"(\s*\n\s*)+", "\n", text)
        text = text.strip()
        return text, None, len(raw), len(text)
    except Exception:
        return "", None, 0, 0


def list_eligible_raw() -> List[Tuple[str, str, Dict]]:
    # Returns list of (doc_id, raw_path, meta)
    out: List[Tuple[str, str, Dict]] = []
    for meta_path in glob.glob("data/raw/**/**/*.meta.json", recursive=True):
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
        except Exception:
            continue
        if int(meta.get("http_status") or 0) != 200:
            continue
        doc_id = meta.get("doc_id")
        base_dir = os.path.dirname(meta_path)
        # Probe for raw
        candidates = [
            os.path.join(base_dir, f"{doc_id}.raw.html"),
            os.path.join(base_dir, f"{doc_id}.pdf"),
            os.path.join(base_dir, f"{doc_id}.json"),
        ]
        for c in candidates:
            if os.path.exists(c):
                out.append((doc_id, c, meta))
                break
    # Deterministic order
    out.sort(key=lambda x: x[0])
    return out


def select_phase_subset(all_items: List[Tuple[str, str, Dict]], phase: str) -> List[Tuple[str, str, Dict]]:
    if phase.upper() == "A":
        import hashlib

        subset = []
        for item in all_items:
            doc_id = item[0]
            h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
            if int(h[-1], 16) % 2 == 0:
                subset.append(item)
        return subset
    return all_items


def normalize_one(item: Tuple[str, str, Dict], rules: Dict, out_dir: str, log: logging.Logger) -> Tuple[str, Optional[str], Optional[str], Optional[int], Optional[int], Optional[str]]:
    doc_id, raw_path, meta = item
    doctype = meta.get("doctype")
    url = meta.get("final_url") or meta.get("requested_url") or ""
    source_domain = meta.get("source_domain")
    last_modified_http = meta.get("last_modified_http")

    text = ""
    before_len = 0
    after_len = 0
    pdf_page_map = None
    html_title = None
    meta_published_time = None

    try:
        if raw_path.endswith(".pdf"):
            text, pdf_page_map, before_len, after_len = extract_pdf_text(raw_path)
        elif raw_path.endswith(".raw.html") or raw_path.endswith(".html"):
            raw_bytes = open(raw_path, "rb").read()
            html_title, meta_published_time = html_extract_title(raw_bytes.decode("utf-8", errors="replace"))
            # Domain-specific relaxation: for help/dev/newsroom pages, use lighter stripping (fewer removals)
            domain_rules = dict(rules)
            if source_domain and any(k in source_domain for k in ["help.salesforce.com", "developer.salesforce.com", "salesforce.com"]):
                domain_rules = dict(rules)
                domain_rules["remove_selectors"] = [sel for sel in rules.get("remove_selectors", []) if sel not in [".sidebar", ".breadcrumb"]]
            text, before_len, after_len, _ = normalize_html_bytes(raw_bytes, domain_rules)
        elif raw_path.endswith(".json"):
            raw_json = json.load(open(raw_path, "r", encoding="utf-8"))
            text = json.dumps(raw_json, ensure_ascii=False, indent=2)
            before_len = len(text)
            after_len = len(text)
        else:
            text = ""
    except Exception as e:
        log.error(f"ERROR {doc_id} {e}")
        return doc_id, None, None, None, None, None

    lang = detect_lang(text) if text else "en"
    # Whitelist known english domains
    if source_domain and any(source_domain.endswith(d) or d in source_domain for d in [
        "sec.gov",
        "investor.salesforce.com",
        "salesforce.com",
        "developer.salesforce.com",
        "help.salesforce.com",
        "wikipedia.org",
    ]):
        lang = "en"
    if lang != "en":
        log.info(f"DROPPED_NON_EN {doc_id} lang={lang}")
        return doc_id, "DROPPED_NON_EN", lang, before_len, after_len, None

    word_count = len(re.findall(r"\b\w+\b", text))
    token_count = cl100k_token_count(text)
    record = {
        "doc_id": doc_id,
        "company": "Salesforce",
        "doctype": doctype,
        "title": html_title or "",
        "publish_date": "",
        "url": url,
        "final_url": meta.get("final_url") or url,
        "source_domain": source_domain,
        "section": "body",
        "topic": "",
        "persona_tags": [],
        "language": lang,
        "text": text,
        "word_count": word_count,
        "token_count": token_count,
        "ingestion_ts": now_iso(),
        "hash_sha256": sha256_hex(text.encode("utf-8")),
        "html_title": html_title,
        "meta_published_time": meta_published_time,
        "last_modified_http": last_modified_http,
        "byline": None,
        "press_location": None,
        "ticker_mentions": [],
        "pdf_page_map": pdf_page_map,
    }
    out_path = os.path.join(out_dir, f"{doc_id}.json")
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    retention_ratio = (after_len / max(1, before_len)) if before_len else 1.0
    log.info(f"OK {doc_id} bytes_before={before_len} bytes_after={after_len} retention={retention_ratio:.3f} lang={lang} wc={word_count} tok={token_count}")
    return doc_id, out_path, lang, before_len, after_len, text


def main():
    ap = argparse.ArgumentParser(description="Normalize raw HTML/PDF/JSON into normalized JSON records")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    args = ap.parse_args()

    ensure_dir("logs/normalize")
    logger, log_path = build_logger()
    logger.info(f"Normalization start phase={args.phase}")

    rules = load_yaml("configs/normalization.rules.yaml")
    all_items = list_eligible_raw()
    subset = select_phase_subset(all_items, args.phase)
    if args.limit:
        subset = subset[: args.limit]

    out_dir = "data/interim/normalized"
    ensure_dir(out_dir)

    if args.dry_run:
        for doc_id, raw_path, _ in subset:
            logger.info(f"[DRY] Would normalize {doc_id} from {raw_path}")
        logger.info(f"Log: {log_path}")
        return

    def worker(item):
        return normalize_one(item, rules, out_dir, logger)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(worker, subset))

    logger.info(f"Normalization complete. Log: {log_path}")


if __name__ == "__main__":
    main()
