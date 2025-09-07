#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple

from common import ensure_dir, now_iso


def normalize_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(words: List[str], k: int = 5) -> Set[str]:
    if len(words) < k:
        return set()
    return {" ".join(words[i : i + k]) for i in range(0, len(words) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def load_chunks() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Return (doc_map, chunks_by_doc) where doc_map[doc_id] has normalized metadata, and chunks_by_doc maps doc_id to list of chunk dicts."""
    # Load normalized docs
    doc_map: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob("data/interim/normalized/*.json"):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            doc_map[d.get("doc_id")] = d
        except Exception:
            continue
    # Load chunks
    chunks_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for cf in glob.glob("data/interim/chunks/*.chunks.jsonl"):
        with open(cf, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    ch = json.loads(line)
                    chunks_by_doc[ch.get("doc_id")].append(ch)
                except Exception:
                    continue
    # Sort by seq
    for k in chunks_by_doc:
        chunks_by_doc[k].sort(key=lambda x: int(x.get("seq_no") or 0))
    return doc_map, chunks_by_doc


def select_canonical(members: List[str], chunk_map: Dict[str, Dict[str, Any]], doc_map: Dict[str, Dict[str, Any]]) -> str:
    # Earliest publish_date, then longer word_count, then lex chunk_id
    def key(cid: str):
        ch = chunk_map[cid]
        doc = doc_map.get(ch.get("doc_id"), {})
        pd = doc.get("publish_date") or "9999-12-31"
        return (pd, -int(ch.get("word_count") or 0), cid)

    return sorted(members, key=key)[0]


def main():
    ap = argparse.ArgumentParser(description="Deduplicate near-duplicate chunks across corpus")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Max docs to process (0 = all)")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    ensure_dir("logs/dedupe")
    ensure_dir("data/interim/dedup")
    ensure_dir("data/interim/chunks/pre_dedupe")
    log_path = os.path.join("logs", "dedupe", datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")

    doc_map, chunks_by_doc = load_chunks()
    # Build flat list and maps
    chunk_map: Dict[str, Dict[str, Any]] = {}
    for doc_id, arr in chunks_by_doc.items():
        for ch in arr:
            chunk_map[ch.get("chunk_id")] = ch

    # Build shingles and inverted index
    shingle_size = 5
    inv: Dict[str, List[str]] = defaultdict(list)
    shingle_sets: Dict[str, Set[str]] = {}
    all_chunk_ids = list(chunk_map.keys())
    for cid in all_chunk_ids:
        words = normalize_words(chunk_map[cid].get("text") or "")
        shs = shingles(words, k=shingle_size)
        shingle_sets[cid] = shs
        for s in shs:
            inv[s].append(cid)

    # Candidate pairs via co-occurrence counts
    co_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for s, cids in inv.items():
        if len(cids) < 2:
            continue
        # To avoid O(n^2) blowup, cap list length
        if len(cids) > 2000:
            cids = cids[:2000]
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                co_counts[key] += 1

    threshold = 0.85
    dup_edges: Dict[str, Set[str]] = defaultdict(set)
    for (a, b), cnt in co_counts.items():
        sa = shingle_sets.get(a) or set()
        sb = shingle_sets.get(b) or set()
        if not sa or not sb:
            continue
        # quick upper bound; skip if cnt cannot reach threshold
        min_needed = threshold * min(len(sa), len(sb))
        if cnt < min_needed:
            continue
        jac = jaccard(sa, sb)
        if jac >= threshold:
            dup_edges[a].add(b)
            dup_edges[b].add(a)

    # Build clusters (connected components)
    visited: Set[str] = set()
    groups: List[List[str]] = []
    for cid in all_chunk_ids:
        if cid in visited:
            continue
        stack = [cid]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            for nb in dup_edges.get(cur, []):
                if nb not in visited:
                    stack.append(nb)
        if len(comp) >= 2:
            groups.append(sorted(comp))

    # Build dedup map
    dedup_groups = []
    to_remove: Set[str] = set()
    for members in groups:
        canonical = select_canonical(members, chunk_map, doc_map)
        dupes = [c for c in members if c != canonical]
        to_remove.update(dupes)
        # Pairwise jaccard stats
        js = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                js.append(jaccard(shingle_sets.get(members[i]) or set(), shingle_sets.get(members[j]) or set()))
        stats = {
            "pairwise_jaccard_min": round(min(js) if js else 0.0, 4),
            "pairwise_jaccard_max": round(max(js) if js else 0.0, 4),
            "members": len(members),
        }
        dedup_groups.append({
            "canonical_chunk_id": canonical,
            "duplicate_chunk_ids": dupes,
            "reason": "jaccard>=0.85",
            "stats": stats,
        })

    ensure_dir("data/interim/dedup")
    dedup_map = {
        "created_at": now_iso(),
        "near_duplicate_threshold": 0.85,
        "shingle_size": 5,
        "groups": dedup_groups,
    }
    with open("data/interim/dedup/dedup_map.json", "w", encoding="utf-8") as f:
        json.dump(dedup_map, f, ensure_ascii=False, indent=2)

    # Remove duplicates from per-doc files (backup originals)
    removed_per_doc: Dict[str, int] = defaultdict(int)
    before_counts: Dict[str, int] = {}
    after_counts: Dict[str, int] = {}
    for doc_id, arr in chunks_by_doc.items():
        src_path = os.path.join("data", "interim", "chunks", f"{doc_id}.chunks.jsonl")
        bak_path = os.path.join("data", "interim", "chunks", "pre_dedupe", f"{doc_id}.chunks.jsonl.bak")
        try:
            # Backup
            ensure_dir(os.path.dirname(bak_path))
            with open(src_path, "r", encoding="utf-8") as rf:
                original = rf.read()
            with open(bak_path, "w", encoding="utf-8") as wf:
                wf.write(original)
        except Exception:
            pass
        # Filter duplicates
        kept = [ch for ch in arr if ch.get("chunk_id") not in to_remove]
        before_counts[doc_id] = len(arr)
        after_counts[doc_id] = len(kept)
        removed_per_doc[doc_id] = before_counts[doc_id] - after_counts[doc_id]
        if not args.dry_run:
            tmp = src_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for ch in kept:
                    f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            os.replace(tmp, src_path)

    # Log per-doc coverage_ratio after dedupe
    for doc_id, arr in chunks_by_doc.items():
        kept = [ch for ch in arr if ch.get("chunk_id") not in to_remove]
        kept_wc = sum(int(ch.get("word_count") or 0) for ch in kept)
        doc_wc = int(doc_map.get(doc_id, {}).get("word_count") or 0)
        cov = kept_wc / max(1, doc_wc)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{doc_id},{doc_map.get(doc_id,{}).get('doctype')},chunks_before={before_counts.get(doc_id,0)},removed={removed_per_doc.get(doc_id,0)},chunks_after={after_counts.get(doc_id,0)},coverage_ratio={round(cov,4)}\n")

    print(f"Dedupe complete. Groups: {len(dedup_groups)}. Log: {log_path}")


if __name__ == "__main__":
    main()

