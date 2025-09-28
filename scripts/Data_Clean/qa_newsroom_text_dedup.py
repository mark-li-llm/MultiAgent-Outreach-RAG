#!/usr/bin/env python3
"""
Newsroom text-based dedup (read-only):

- Extract main text for each raw newsroom item (fallback to naive strip if BS4 rules unavailable)
- Compute features: word_count, char_len, normalized_text_sha256, SimHash-64 over tokens
- Build near-duplicate clusters using SimHash (candidate) + Jaccard on tokens (verify)
- Pick canonical per cluster by content-quality-first: max word_count, then newest fetched_at, then content_length
- Emit:
  - data/interim/dedup/newsroom_text_features.jsonl
  - data/interim/dedup/newsroom_dupe_clusters.jsonl
  - data/interim/dedup/newsroom_canonical.v2.jsonl
  - reports/qa/newsroom_dedup_plan.{json,md}

No raw files are moved. Optionally generates an actions shell (commented) for manual execution later.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def naive_html_to_text(html_bytes: bytes) -> Tuple[str, int, int]:
    raw = html_bytes.decode("utf-8", errors="replace")
    before = len(raw)
    # strip scripts/styles
    txt = re.sub(r"<script[\s\S]*?</script>", " ", raw, flags=re.I)
    txt = re.sub(r"<style[\s\S]*?</style>", " ", txt, flags=re.I)
    # remove tags
    txt = re.sub(r"<[^>]+>", " ", txt)
    # collapse whitespace
    txt = re.sub(r"\s+", " ", txt).strip()
    after = len(txt)
    return txt, before, after


def try_extract_main_text(raw_path: str) -> Tuple[str, int, int]:
    """Return (text, before_len, after_len). Prefer normalize_html, else naive."""
    b = open(raw_path, "rb").read()
    # Try project normalize_html
    try:
        from scripts.normalize_html import normalize_html_bytes, load_yaml as _ly  # type: ignore

        rules = _ly("configs/normalization.rules.yaml")
        text, before_len, after_len, _ = normalize_html_bytes(b, rules)
        # As a guard, if text looks empty, fallback to naive
        if not text or after_len <= 0:
            raise RuntimeError("empty text after normalize_html_bytes")
        return text, int(before_len or 0), int(after_len or 0)
    except Exception:
        # Fallback
        text, before_len, after_len = naive_html_to_text(b)
        return text, before_len, after_len


def normalize_for_compare(text: str) -> str:
    # similar to embedding_utils.normalize_text: lowercase, ASCII-like collapse, digits->0, whitespace collapse
    import unicodedata

    t = (text or "").lower()
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode("ascii", errors="ignore")
    t = t.replace("-", " ").replace("/", " ")
    t = re.sub(r"\d+", "0", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    # Reuse project tokenization idea: words + bigrams
    t = normalize_for_compare(text)
    toks = re.findall(r"[a-z0-9]{2,20}", t)
    bigrams: List[str] = []
    for i in range(len(toks) - 1):
        bigrams.append(f"bg:{toks[i]}_{toks[i+1]}")
    return toks + bigrams


def simhash64(tokens: List[str]) -> int:
    # Standard SimHash over 64 bits
    from hashlib import blake2b

    v = [0] * 64
    for tok in tokens:
        h = int.from_bytes(blake2b(tok.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(64):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    x = 0
    for i in range(64):
        if v[i] > 0:
            x |= (1 << i)
    return x


def hamming64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1 << 64) - 1)).count("1"))


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb)) / float(max(1, len(sa | sb)))


@dataclass
class Item:
    doc_id: str
    date: str
    slug: str
    fetched_at: str
    fetched_ts: float
    content_length: int
    meta_path: str
    raw_path: Optional[str]
    title: str
    http_status: int
    domain: str
    text: str
    word_count: int
    char_len: int
    text_sha256: str
    simhash: int
    tokens: List[str]


def sha256_hex_bytes(b: bytes) -> str:
    import hashlib

    return hashlib.sha256(b).hexdigest()


def load_items() -> List[Item]:
    items: List[Item] = []
    for mp in sorted(glob.glob("data/raw/newsroom/*.meta.json")):
        try:
            m = json.load(open(mp, "r", encoding="utf-8"))
        except Exception:
            continue
        fn = os.path.basename(mp)
        parts = fn.split("::")
        date = parts[2] if len(parts) > 2 else (m.get("visible_date") or "")
        slug = parts[3] if len(parts) > 3 else ""
        doc_id = m.get("doc_id")
        fetched_at = (m.get("fetched_at") or "").strip()
        ts = 0.0
        if fetched_at:
            try:
                ts = datetime.fromisoformat(fetched_at.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = 0.0
        raw_path = mp.replace(".meta.json", ".raw.html")
        if not os.path.exists(raw_path):
            raw_path = None
        text = ""
        wc = 0
        ch = 0
        if raw_path:
            t, before_len, after_len = try_extract_main_text(raw_path)
            text = t or ""
            wc = len(re.findall(r"\b\w+\b", text))
            ch = len(text)
        comp_norm = normalize_for_compare(text)
        toks = tokenize(comp_norm)
        sim = simhash64(toks) if toks else 0
        items.append(
            Item(
                doc_id=doc_id,
                date=date,
                slug=slug,
                fetched_at=fetched_at,
                fetched_ts=ts,
                content_length=int(m.get("content_length") or 0),
                meta_path=mp,
                raw_path=raw_path,
                title=m.get("visible_title") or "",
                http_status=int(m.get("http_status") or 0),
                domain=m.get("source_domain") or "",
                text=comp_norm,
                word_count=wc,
                char_len=ch,
                text_sha256=sha256_hex_bytes((comp_norm or "").encode("utf-8")),
                simhash=sim,
                tokens=toks,
            )
        )
    return items


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1


def build_clusters(items: List[Item], simhash_exact: int, simhash_near: int, jaccard_near: float) -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    n = len(items)
    dsu = DSU(n)
    edges: List[Dict[str, Any]] = []

    # Index by exact text to short-circuit
    by_sha: Dict[str, List[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_sha[it.text_sha256].append(i)
    for sha, idxs in by_sha.items():
        if len(idxs) > 1:
            base = idxs[0]
            for j in idxs[1:]:
                dsu.union(base, j)
                edges.append({"i": base, "j": j, "type": "EXACT", "sha": sha, "hamming": 0, "jacc": 1.0})

    # Pairwise (small corpus)
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if already exact-linked
            if dsu.find(i) == dsu.find(j):
                continue
            hi = items[i].simhash
            hj = items[j].simhash
            if hi == 0 or hj == 0:
                continue
            ham = hamming64(hi, hj)
            if ham <= simhash_exact:
                dsu.union(i, j)
                edges.append({"i": i, "j": j, "type": "EXACT_SIM", "hamming": ham})
                continue
            if ham <= simhash_near:
                # verify with token Jaccard
                jac = jaccard(items[i].tokens, items[j].tokens)
                if jac >= jaccard_near:
                    dsu.union(i, j)
                    edges.append({"i": i, "j": j, "type": "NEAR", "hamming": ham, "jacc": jac})

    groups: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        g = dsu.find(i)
        groups[g].append(i)
    clusters = [sorted(v) for v in groups.values()]
    clusters.sort(key=len, reverse=True)
    return clusters, edges


def pick_canonical(idx_list: List[int], items: List[Item]) -> int:
    # content_quality_first: max word_count, then newest fetched_ts, then max content_length
    def key(i: int):
        it = items[i]
        return (int(it.word_count or 0), float(it.fetched_ts or 0.0), int(it.content_length or 0))

    return sorted(idx_list, key=key, reverse=True)[0]


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Text-based newsroom dedup (read-only)")
    ap.add_argument("--simhash-exact", type=int, default=3, help="Hamming threshold for exact-sim")
    ap.add_argument("--simhash-near", type=int, default=10, help="Hamming threshold for near duplicates")
    ap.add_argument("--jaccard-near", type=float, default=0.95, help="Token Jaccard to confirm near-duplicates")
    args = ap.parse_args()

    items = load_items()
    out_dir = "data/interim/dedup"
    ensure_dir(out_dir)

    # Write features cache
    feat_path = os.path.join(out_dir, "newsroom_text_features.jsonl")
    with open(feat_path, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "doc_id": it.doc_id,
                "date": it.date,
                "slug": it.slug,
                "visible_title": it.title,
                "meta_path": it.meta_path,
                "raw_path": it.raw_path,
                "http_status": it.http_status,
                "fetched_at": it.fetched_at,
                "content_length": it.content_length,
                "word_count": it.word_count,
                "char_len": it.char_len,
                "text_sha256": it.text_sha256,
                "simhash64": format(it.simhash, "016x"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    clusters, edges = build_clusters(items, args.simhash_exact, args.simhash_near, args.jaccard_near)

    # Build cluster records
    cluster_recs: List[Dict[str, Any]] = []
    for cid, idxs in enumerate(clusters):
        if len(idxs) <= 1:
            continue
        can_idx = pick_canonical(idxs, items)
        can = items[can_idx]
        docs = [items[i] for i in idxs]
        keys = sorted({(d.date, d.slug) for d in docs})
        cluster_recs.append(
            {
                "cluster_id": cid,
                "size": len(idxs),
                "keys": [
                    {"date": k[0], "slug": k[1]} for k in keys
                ],
                "canonical": {
                    "doc_id": can.doc_id,
                    "fetched_at": can.fetched_at,
                    "word_count": can.word_count,
                    "content_length": can.content_length,
                    "text_sha256": can.text_sha256,
                },
                "duplicates": [
                    {
                        "doc_id": items[i].doc_id,
                        "fetched_at": items[i].fetched_at,
                        "word_count": items[i].word_count,
                        "content_length": items[i].content_length,
                        "text_sha256": items[i].text_sha256,
                    }
                    for i in idxs
                    if i != can_idx
                ],
            }
        )

    # Write clusters and canonical v2
    clusters_path = os.path.join(out_dir, "newsroom_dupe_clusters.jsonl")
    with open(clusters_path, "w", encoding="utf-8") as f:
        for r in cluster_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    canon_path = os.path.join(out_dir, "newsroom_canonical.v2.jsonl")
    # Build a canonical v2 manifest that includes singletons as well
    all_in_clusters = set()
    for r in cluster_recs:
        all_in_clusters.add(r["canonical"]["doc_id"])
        for d in r["duplicates"]:
            all_in_clusters.add(d["doc_id"])

    with open(canon_path, "w", encoding="utf-8") as out:
        # 1) clusters: write canonical + dup info
        for r in cluster_recs:
            can_id = r["canonical"]["doc_id"]
            out.write(
                json.dumps(
                    {
                        "cluster_id": r["cluster_id"],
                        "canonical_doc_id": can_id,
                        "duplicate_doc_ids": [d["doc_id"] for d in r["duplicates"]],
                        "picked_by": "content_quality_first",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        # 2) singletons
        for it in items:
            if it.doc_id not in all_in_clusters:
                out.write(
                    json.dumps(
                        {
                            "cluster_id": None,
                            "canonical_doc_id": it.doc_id,
                            "duplicate_doc_ids": [],
                            "picked_by": "singleton",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # QA plan report
    qa = {
        "summary": {
            "total_items": len(items),
            "clusters_gt1": sum(1 for r in cluster_recs if r["size"] > 1),
            "total_clustered_items": sum(r["size"] for r in cluster_recs),
            "singletons": len(items) - sum(r["size"] for r in cluster_recs),
        },
        "paths": {
            "features": feat_path,
            "clusters": clusters_path,
            "canonical_v2": canon_path,
        },
        "thresholds": {
            "simhash_exact_hamming": args.simhash_exact,
            "simhash_near_hamming": args.simhash_near,
            "jaccard_near": args.jaccard_near,
        },
        "notes": "No raw files were modified. Use canonical.v2 to filter downstream or to prepare a manual actions script.",
    }
    qa_dir = "reports/qa"
    ensure_dir(qa_dir)
    write_json(os.path.join(qa_dir, "newsroom_dedup_plan.json"), qa)

    # Markdown summary
    md = []
    md.append("# Newsroom Text Dedup Plan")
    md.append("")
    md.append(f"- Total items: {qa['summary']['total_items']}")
    md.append(f"- Clusters (>1): {qa['summary']['clusters_gt1']}")
    md.append(f"- Clustered items: {qa['summary']['total_clustered_items']}")
    md.append(f"- Singletons: {qa['summary']['singletons']}")
    md.append(f"- Thresholds: simhash_exact={args.simhash_exact}, simhash_near={args.simhash_near}, jaccard_near={args.jaccard_near}")
    md.append(f"- Features: {feat_path}")
    md.append(f"- Clusters: {clusters_path}")
    md.append(f"- Canonical v2: {canon_path}")
    md_path = os.path.join(qa_dir, "newsroom_dedup_plan.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("OK", feat_path)
    print("OK", clusters_path)
    print("OK", canon_path)
    print("OK", md_path)


if __name__ == "__main__":
    main()

