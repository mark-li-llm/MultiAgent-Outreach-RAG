#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, now_iso


def phase_select(doc_id: str, phase: str) -> bool:
    import hashlib

    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return (int(h[-1], 16) % 2 == 0) if phase == "A" else (int(h[-1], 16) % 2 == 1)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def coerce_date_iso(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    from common import coerce_date

    return coerce_date(s)


def extract_h1_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if line.strip().lower().startswith("h1:"):
            return line.split(":", 1)[-1].strip()
    return None


def find_sidecar(doc_id: str) -> Optional[str]:
    # Search meta sidecar under data/raw/**
    for mp in glob.glob("data/raw/**/**/*.meta.json", recursive=True):
        try:
            m = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            continue
        if m.get("doc_id") == doc_id:
            return mp
    return None


def pick_title(d: Dict[str, Any], side: Optional[Dict[str, Any]]) -> str:
    # Precedence: html_title -> existing title -> first H1
    title = (d.get("html_title") or "").strip()
    if title:
        return title
    if d.get("title"):
        return d["title"]
    h1 = extract_h1_from_text(d.get("text") or "")
    if h1:
        return h1
    vt = (side or {}).get("visible_title") if side else None
    if vt and vt.strip():
        return vt.strip()
    # Last resort: derive from doc_id slug
    doc_id = d.get("doc_id") or ""
    parts = doc_id.split("::")
    if len(parts) >= 5:
        slug = parts[3]
        return slug.replace("-", " ").title()
    return ""


def pick_publish_date(d: Dict[str, Any], side: Optional[Dict[str, Any]]) -> Optional[str]:
    doctype = (d.get("doctype") or "").lower()
    if doctype in ("10-k", "10-q", "8-k", "ars_pdf"):
        # SEC: use sidecar visible_date
        for key in ("visible_date",):
            iso = coerce_date_iso(side.get(key) if side else None)
            if iso:
                return iso
        # fallback to fetched_at date part
        fa = (side or {}).get("fetched_at")
        if fa:
            try:
                dt = datetime.fromisoformat(fa.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except Exception:
                pass
        return None
    elif doctype == "press":
        # Press: meta_published_time -> visible_date -> rss_pubdate
        for key in ("meta_published_time",):
            iso = coerce_date_iso(d.get(key))
            if iso:
                return iso
        for key in ("visible_date", "rss_pubdate"):
            iso = coerce_date_iso(side.get(key) if side else None)
            if iso:
                return iso
        return None
    elif doctype in ("product", "dev_docs", "help_docs"):
        # Product/Dev/Help: look for Updated date in text, else last_modified_http from sidecar
        text = d.get("text") or ""
        m = re.search(r"(Updated|Last\s*Updated|Last\s*modified)[:\s]+([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4}|\d{4}-\d{2}-\d{2})", text, re.I)
        if m:
            iso = coerce_date_iso(m.group(2))
            if iso:
                return iso
        iso = coerce_date_iso((side or {}).get("last_modified_http"))
        if iso:
            return iso
        fa = (side or {}).get("fetched_at")
        if fa:
            try:
                dt = datetime.fromisoformat(fa.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except Exception:
                return None
        return None
    elif doctype == "wiki":
        iso = coerce_date_iso((side or {}).get("last_modified_http"))
        return iso
    return None


TOPIC_RULES = [
    ("Agentforce", ["agentforce"]),
    ("Agent API", ["agent api"]),
    ("Data Cloud", ["data cloud", "cdp", "unified profiles", "real-time data", "connections"]),
    ("Earnings", ["revenue", "gaap", "non-gaap", "guidance", "outlook", "q1 fy", "q2 fy", "full year", "results", "quarterly", "fiscal"]),
    ("Partnership", ["announce", "expanded", "partnership", "collaborate", "work with", "integration", "google", "aws", "slack", "mulesoft", "informatica", "ferrari"]),
    ("Executive", ["appoints", "chief executive", "chief operating", "president", "board of directors"]),
    ("Security", ["security", "zero trust", "governance"]),
    ("Compliance", ["compliance", "regulatory"]),
    ("GenAI", ["gen ai", "generative", "llm", "prompt"]),
    ("Industry Solutions", ["retail", "financial services", "healthcare", "communications"]),
    ("Platform", ["platform", "apis", "developer"]),
    ("AI", ["ai ", " ai", "artificial intelligence"]),
]


def assign_topics(title: str, text: str, doctype: str) -> str:
    t = (title + "\n" + text).lower()
    tags = []
    for tag, keys in TOPIC_RULES:
        if any(k in t for k in keys):
            tags.append(tag)
    # Special case for dev_docs/help
    if doctype in ("dev_docs", "help_docs") and "agent api" in t:
        tags.append("Agent API")
    # Deduplicate, preserve order
    seen = set()
    out = []
    for x in tags:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return "|".join(out)


def assign_personas(title: str, text: str, persona_cfg: Dict[str, Any]) -> List[str]:
    t = (title + "\n" + text).lower()
    out = []
    personas = persona_cfg.get("personas", {})
    for key, kws in personas.items():
        for kw in kws:
            if kw.lower() in t:
                out.append(key)
                break
    # Deduplicate
    return sorted(list(set(out)))


def main():
    ap = argparse.ArgumentParser(description="Extract and fill doc-level metadata for normalized docs")
    ap.add_argument("--phase", required=True, choices=["A", "B"]) 
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    ensure_dir("logs/metadata")
    persona_cfg = load_yaml("configs/eval.prompts.yaml")

    paths = sorted(glob.glob("data/interim/normalized/*.json"))
    count = 0
    for p in paths:
        d = json.load(open(p, "r", encoding="utf-8"))
        doc_id = d.get("doc_id")
        if not phase_select(doc_id, args.phase):
            continue
        side_path = find_sidecar(doc_id)
        side = json.load(open(side_path, "r", encoding="utf-8")) if side_path else {}

        title = pick_title(d, side)
        pub = pick_publish_date(d, side)
        # Enforce date sanity
        if pub:
            try:
                dt = date.fromisoformat(pub)
                if dt > datetime.now(timezone.utc).date() or dt < date(1999, 1, 1):
                    pub = None
            except Exception:
                pub = None

        url = d.get("final_url") or d.get("url") or (side.get("final_url") or side.get("requested_url") or "")
        final_url = url

        topic = assign_topics(title, d.get("text") or "", (d.get("doctype") or ""))
        personas = assign_personas(title, d.get("text") or "", persona_cfg)

        # PR recency flags
        is_within_12mo = None
        is_within_24mo = None
        if (d.get("doctype") or "").lower() == "press" and pub:
            try:
                dt = date.fromisoformat(pub)
                days = (datetime.now(timezone.utc).date() - dt).days
                is_within_12mo = days <= 365
                is_within_24mo = days <= 730
            except Exception:
                pass

        mutated = False
        def setf(key, val):
            nonlocal mutated
            if val is None:
                return
            if d.get(key) != val:
                d[key] = val
                mutated = True

        setf("title", title)
        setf("publish_date", pub or d.get("publish_date"))
        setf("url", url)
        setf("final_url", final_url)
        setf("topic", topic)
        setf("persona_tags", personas)
        if is_within_12mo is not None:
            setf("is_within_12mo", is_within_12mo)
        if is_within_24mo is not None:
            setf("is_within_24mo", is_within_24mo)

        if mutated and not args.dry_run:
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)
            os.replace(tmp, p)
        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"Updated metadata for {count} docs")


if __name__ == "__main__":
    main()
