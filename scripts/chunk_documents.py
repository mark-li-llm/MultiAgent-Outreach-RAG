#!/usr/bin/env python3
import argparse
import glob
import io
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

from common import ensure_dir


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cl100k_token_count(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def phase_all() -> List[str]:
    return sorted(glob.glob("data/interim/normalized/*.json"))


def get_heading_positions(text: str) -> Tuple[List[Tuple[int, str]], List[int]]:
    positions: List[Tuple[int, str]] = []
    h_starts: List[int] = []
    offset = 0
    for line in text.splitlines(True):  # keepends
        if line.strip().lower().startswith("h2:") or line.strip().lower().startswith("h3:"):
            title = line.strip()[3:].strip()
            positions.append((offset, title))
            h_starts.append(offset)
        offset += len(line)
    return positions, h_starts


def get_first_h1(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.lower().startswith("h1:"):
            return s[3:].strip()
    return ""


def snap_to_boundary(pos: int, candidates: List[int], tol: int) -> int:
    best = pos
    best_delta = tol + 1
    for c in candidates:
        d = abs(c - pos)
        if d <= tol and d < best_delta:
            best = c
            best_delta = d
    return best


def plan_slices_with_boundaries(text: str, start: int, end: int, target_tokens: int, overlap_tokens: int, boundary_candidates: List[int], tol_chars: int) -> List[Tuple[int, int]]:
    seg = text[start : end + 1]
    seg_tokens = max(1, cl100k_token_count(seg))
    seg_len = len(seg)
    chars_per_token = seg_len / seg_tokens
    step_chars = int(max(1, round(chars_per_token * (target_tokens - overlap_tokens))))
    win_chars = int(max(1, round(chars_per_token * target_tokens)))
    # Candidate list filtered to segment
    cands = sorted([c for c in boundary_candidates if start <= c <= end])
    # Always include segment start
    if start not in cands:
        cands = [start] + cands
    slices: List[Tuple[int, int]] = []
    s = start
    while s <= end:
        # Predict next start
        pred_next = s + step_chars
        # Find candidate within tolerance closest to pred_next
        next_s = None
        if cands:
            closest = None
            best_delta = tol_chars + 1
            for c in cands:
                if c <= s:
                    continue
                d = abs(c - pred_next)
                if d <= tol_chars and d < best_delta:
                    best_delta = d
                    closest = c
            next_s = closest
        e = min(end, s + win_chars - 1)
        slices.append((s, e))
        if e == end:
            break
        s = next_s if next_s is not None else (s + step_chars)
    return slices


def chunk_doc(d: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    doc_id = d.get("doc_id")
    title = d.get("title") or ""
    text = d.get("text") or ""
    doctype = (d.get("doctype") or "").lower()
    pubdate = d.get("publish_date") or ""
    url = d.get("final_url") or d.get("url") or ""
    topic = d.get("topic") or ""
    persona = d.get("persona_tags") or []
    token_count = int(d.get("token_count") or 0)
    if token_count <= 0:
        token_count = cl100k_token_count(text)

    target = int(cfg.get("target_tokens", 800))
    overlap = int(cfg.get("overlap_tokens", 120))
    short_thresh = int(cfg.get("short_doc_threshold_tokens", 350))
    tol_chars = int(cfg.get("boundary_tolerance_chars", 50))

    chunks: List[Dict[str, Any]] = []

    def build_chunk(idx: int, s: int, e: int, local_heads: List[str]) -> Dict[str, Any]:
        # Title boost: title + first H1
        h1 = get_first_h1(text)
        boost_lines = []
        if title:
            boost_lines.append(title)
        if h1 and h1 != title:
            boost_lines.append(h1)
        boost = "\n".join(boost_lines).strip()

        body = text[s : e + 1]
        body = re.sub(r"\n{3,}", "\n\n", body)

        chunk_text = (boost + "\n\n" + body).strip() if boost else body
        wc = len(re.findall(r"\b\w+\b", chunk_text))
        tk = cl100k_token_count(chunk_text)
        return {
            "chunk_id": f"{doc_id}::chunk{idx:04d}",
            "doc_id": doc_id,
            "seq_no": idx,
            "text": chunk_text,
            "word_count": wc,
            "token_count": tk,
            "start_char": s,
            "end_char": e,
            "local_heads": local_heads[:2],
            "metadata_snapshot": {
                "company": "Salesforce",
                "doctype": d.get("doctype"),
                "date": pubdate,
                "url": url,
                "title": title,
                "topic": topic,
                "persona_tags": persona,
            },
        }

    if token_count < short_thresh or len(text.strip()) == 0:
        # Single chunk
        chunks.append(build_chunk(0, 0, max(0, len(text) - 1), []))
        return chunks

    # Build boundary candidates
    _, h2h3_positions = get_heading_positions(text)
    boundary_candidates = set(h2h3_positions)
    if doctype in ("10-k", "10-q", "8-k", "ars_pdf"):
        for span in d.get("sec_item_spans") or []:
            boundary_candidates.add(int(span.get("start_char") or 0))

    def local_heads_for_start(s: int) -> List[str]:
        heads, _ = get_heading_positions(text)
        prev = [t for (pos, t) in heads if pos <= s]
        prev_titles = prev[-2:][::-1] if prev else []
        return prev_titles

    # Split by spans for SEC, else whole doc
    segments: List[Tuple[int, int]] = []
    if doctype in ("10-k", "10-q", "8-k", "ars_pdf") and (d.get("sec_item_spans") or []):
        for span in d.get("sec_item_spans") or []:
            s = int(span.get("start_char") or 0)
            e = int(span.get("end_char") or (len(text) - 1))
            segments.append((s, min(e, len(text) - 1)))
    else:
        segments.append((0, max(0, len(text) - 1)))

    idx = 0
    for seg_s, seg_e in segments:
        # Build candidate list within segment
        cands_seg = sorted([c for c in boundary_candidates if seg_s <= c <= seg_e])
        if seg_s not in cands_seg:
            cands_seg = [seg_s] + cands_seg
        slices: List[Tuple[int, int]] = []
        # Sliding with boundary snapping (works for SEC and non-SEC; SEC still snaps to item/H2/H3 starts)
        slices = plan_slices_with_boundaries(text, seg_s, seg_e, target, overlap, list(boundary_candidates), tol_chars)
        # Merge tiny residuals if last slice < 120 tokens
        adjusted: List[Tuple[int, int]] = []
        for i, (s, e) in enumerate(slices):
            s2 = snap_to_boundary(s, list(boundary_candidates), tol_chars)
            adjusted.append((s2, e))
        # Merge residual
        merged: List[Tuple[int, int]] = []
        for i, (s, e) in enumerate(adjusted):
            if i == 0:
                merged.append((s, e))
                continue
            # Ensure overlap by moving start back if needed
            prev_s, prev_e = merged[-1]
            # compute tokens of previous
            prev_text = text[prev_s : prev_e + 1]
            prev_tokens = cl100k_token_count(prev_text)
            # compute tokens of current tentative
            cur_text = text[s : e + 1]
            cur_tokens = cl100k_token_count(cur_text)
            if cur_tokens < 120 and i == len(adjusted) - 1:
                # merge into previous if doesn't exceed target + 20%
                if prev_tokens + cur_tokens <= int(target * 1.2):
                    merged[-1] = (prev_s, e)
                    continue
            merged.append((s, e))

        for s, e in merged:
            lh = local_heads_for_start(s)
            chunks.append(build_chunk(idx, s, e, lh))
            idx += 1

    return chunks


def main():
    ap = argparse.ArgumentParser(description="Chunk normalized documents into retrieval-ready windows")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config("configs/chunking.config.json")
    ensure_dir("data/interim/chunks")
    ensure_dir("logs/chunk")
    log_path = os.path.join("logs", "chunk", datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")

    paths = phase_all()
    count = 0
    for p in paths:
        d = json.load(open(p, "r", encoding="utf-8"))
        doc_id = d.get("doc_id")
        if not d.get("text"):
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(f"{doc_id},{d.get('doctype')},chunks=0,expected=0,adj_jaccard_median=0.0,boundary_alignment=0.0\n")
            continue
        chunks = chunk_doc(d, cfg)
        outp = os.path.join("data", "interim", "chunks", f"{doc_id}.chunks.jsonl")
        if not args.dry_run:
            with open(outp, "w", encoding="utf-8") as f:
                for ch in chunks:
                    f.write(json.dumps(ch, ensure_ascii=False) + "\n")
        # expected count
        tk = int(d.get("token_count") or cl100k_token_count(d.get("text") or ""))
        expected = math.ceil(tk / max(1, (cfg["target_tokens"] - cfg["overlap_tokens"])) )
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"{doc_id},{d.get('doctype')},chunks={len(chunks)},expected={expected},adj_jaccard_median=NA,boundary_alignment=NA\n")
        count += 1
        if args.limit and count >= args.limit:
            break

    print(f"Chunked {count} docs. Log: {log_path}")


if __name__ == "__main__":
    main()
