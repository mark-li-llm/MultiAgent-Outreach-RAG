#!/usr/bin/env python3
import argparse
import glob
import json
import os
import random
import re
import hashlib
from collections import defaultdict, Counter
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Tuple

from common import ensure_dir


def load_normalized() -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob("data/interim/normalized/*.json"):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            m[d.get("doc_id")] = d
        except Exception:
            continue
    return m


def load_chunks() -> Dict[str, List[Dict[str, Any]]]:
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cf in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ch = json.loads(line)
                    by_doc[ch.get("doc_id")].append(ch)
                except Exception:
                    continue
    for k in by_doc:
        by_doc[k].sort(key=lambda x: int(x.get("seq_no") or 0))
    return by_doc


STOP = set("the a an and or of for to in on by with from at as is are was were be been being this that those these it its their our your you we they i not no so if then than into out over under about above below after before during while up down off near far per via within without data salesforce press release results fiscal quarter year revenue agentforce api".split())


def pick_keyphrases(text: str, title: str) -> List[str]:
    words = re.findall(r"[a-z0-9-]{4,}", (title + "\n" + text).lower())
    kws = []
    for w in words:
        w1 = w.strip("-")
        if len(w1) < 4:
            continue
        if w1 in STOP:
            continue
        if w1 not in kws and w1 in text.lower():
            kws.append(w1)
        if len(kws) >= 4:
            break
    if len(kws) < 2:
        # Fallback: add generic words that exist
        more = [w for w in re.findall(r"[a-z0-9]{4,}", text.lower()) if w not in STOP]
        for w in more:
            if w not in kws:
                kws.append(w)
            if len(kws) >= 2:
                break
    return kws[:5]


def is_within_months(pub: str, months: int) -> bool:
    try:
        dt = date.fromisoformat(pub)
        delta_days = (datetime.now(timezone.utc).date() - dt).days
        return delta_days <= (30 * months + 5)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Build retrieval-readiness evaluation seed set")
    args = ap.parse_args()

    ensure_dir("data/interim/eval")

    norm = load_normalized()
    chunks_by_doc = load_chunks()
    # Build pools
    sec_docs = [d for d in norm.values() if (d.get("doctype") or "").lower() in ("10-k", "10-q", "8-k", "ars_pdf")]
    press_docs = [d for d in norm.values() if (d.get("doctype") or "").lower() == "press"]
    ir_docs = [d for d in press_docs if (d.get("source_domain") or "").startswith("investor.salesforce.com")]
    newsroom_docs = [d for d in press_docs if "salesforce.com" in (d.get("source_domain") or "") and "investor" not in (d.get("source_domain") or "")]
    prod_docs = [d for d in norm.values() if (d.get("doctype") or "").lower() == "product"]
    dev_docs = [d for d in norm.values() if (d.get("doctype") or "").lower() == "dev_docs"]
    help_docs = [d for d in norm.values() if (d.get("doctype") or "").lower() == "help_docs"]
    wiki_docs = [d for d in norm.values() if (d.get("doctype") or "").lower() == "wiki"]

    random.seed(7)

    def pool_chunks(docs: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        items: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for d in docs:
            for ch in chunks_by_doc.get(d.get("doc_id"), []):
                items.append((d, ch))
        return items

    # Target counts
    target_sec = 10
    target_ir = 10
    target_news = 10
    target_pdh_min = 8  # minimum
    target_wiki_min = 2 # minimum

    sec_pool = pool_chunks(sec_docs)
    ir_pool = pool_chunks(ir_docs)
    news_pool = pool_chunks(newsroom_docs)
    pdh_pool = pool_chunks(prod_docs) + pool_chunks(dev_docs) + pool_chunks(help_docs)
    wiki_pool = pool_chunks(wiki_docs)

    # Build deterministic pools and fill quotas
    sec_pool = sec_pool
    ir_pool = ir_pool
    news_pool = news_pool
    pdh_pool = pdh_pool
    wiki_pool = wiki_pool

    random.shuffle(sec_pool); random.shuffle(ir_pool); random.shuffle(news_pool); random.shuffle(pdh_pool); random.shuffle(wiki_pool)
    picks: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    used_chunks = set()

    def add_from(pool, need):
        added = 0
        for d, ch in pool:
            if ch.get("chunk_id") in used_chunks:
                continue
            picks.append((d, ch))
            used_chunks.add(ch.get("chunk_id"))
            added += 1
            if added >= need:
                break
        return added

    # Fill minimum quotas
    add_from(sec_pool, target_sec)
    add_from(ir_pool, target_ir)
    add_from(news_pool, target_news)
    # product/dev/help at least 8
    cur_pdh = sum(1 for d,ch in picks if (d.get("doctype") or "").lower() in ("product","dev_docs","help_docs"))
    if cur_pdh < target_pdh_min:
        add_from(pdh_pool, target_pdh_min - cur_pdh)
    # wiki at least 2
    cur_wiki = sum(1 for d,ch in picks if (d.get("doctype") or "").lower() == "wiki")
    if cur_wiki < target_wiki_min:
        add_from(wiki_pool, target_wiki_min - cur_wiki)
    # Top up to 45 items with newsroom pool
    while len(picks) < 45:
        if add_from(news_pool, 1) == 0:
            break

    # Persona distribution: aim ~15 per persona for 45 items; adjust cyclically
    personas = ["vp_customer_experience", "cio", "vp_sales_ops"]
    total_needed = max(40, len(picks))
    # Difficulty rotation
    diffs = ["easy", "medium", "hard"]
    diff_counts = Counter()

    out_path = "data/interim/eval/salesforce_eval_seed.jsonl"
    ensure_dir(os.path.dirname(out_path))
    used_pairs = set()
    used_chunks = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (d, ch) in enumerate(picks):
            doc_id = d.get("doc_id")
            chunk_id = ch.get("chunk_id")
            text = ch.get("text") or ""
            title = (d.get("title") or ch.get("metadata_snapshot", {}).get("title") or "")
            url = ch.get("metadata_snapshot", {}).get("url") or d.get("final_url") or d.get("url") or ""
            pub = d.get("publish_date") or ""
            doctype = (d.get("doctype") or "").lower()
            # Choose persona
            # Force balanced persona rotation
            persona = personas[i % 3]
            # Keep persona shares balanced by rotating if needed
            # Build query
            base_kw = (title.split(" ")[0] if title else "Agentforce").lower()
            query = f"What does this document say about {base_kw}?"
            # Ensure uniqueness of (persona, query)
            attempt = 0
            while (persona, query) in used_pairs and attempt < 3:
                query = f"Summarize key points regarding {base_kw}"
                attempt += 1
            used_pairs.add((persona, query))
            # Keyphrases
            kps = pick_keyphrases(text, title)
            # Difficulty rotate
            difficulty = diffs[i % 3]

            eval_id = hashlib.sha1((chunk_id + persona + query).encode("utf-8")).hexdigest()[:12]
            rec = {
                "eval_id": eval_id,
                "persona": persona,
                "query_text": query,
                "expected_doc_id": doc_id,
                "expected_chunk_id": chunk_id,
                "expected_answer_keyphrases": kps,
                "source_type": d.get("doctype"),
                "created_from_url": url,
                "label_date": datetime.now(timezone.utc).date().isoformat(),
                "difficulty": difficulty,
                "notes": "auto-generated seed",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote seed set: {out_path} ({len(picks)} items)")


if __name__ == "__main__":
    main()
