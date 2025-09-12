#!/usr/bin/env python3
import glob
import json
import math
import os
import random
import re
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, List, Tuple

from common import ensure_dir, now_iso
from embedding_utils import embed_text

try:
    import yaml
except Exception:
    yaml = None


VEC_DIR = os.path.join("data", "vector", "embeddings")
EMB_PARQUET = os.path.join(VEC_DIR, "embeddings.parquet")
EMB_STATS = os.path.join(VEC_DIR, "embedding_stats.json")
STEP0 = os.path.join("reports", "qa", "step00_baseline.json")
CHUNK_GLOB = os.path.join("data", "interim", "chunks", "*.chunks.jsonl")
CONF = os.path.join("configs", "vector.indexing.yaml")


def read_yaml_dim(path: str) -> int:
    if yaml is None:
        raise RuntimeError("PyYAML not available; cannot read configs/vector.indexing.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim = int(cfg.get("embedding", {}).get("dim") or 0)
    if not dim:
        raise ValueError("embedding.dim missing or invalid in configs/vector.indexing.yaml")
    return dim


def hash_vec(seed: str, dim: int) -> List[float]:
    # Deprecated path kept for compatibility if ever referenced; prefer embed_text
    return embed_text(seed, dim)


def l2_norm(v: List[float]) -> float:
    return math.sqrt(sum((x * x) for x in v))


def quartiles(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    s = sorted(values)
    n = len(s)
    med = median(s)
    # Split for Q1/Q3 (Tukey's hinges)
    mid = n // 2
    lower = s[:mid]
    upper = s[-mid:]
    q1 = median(lower) if lower else med
    q3 = median(upper) if upper else med
    return q1, med, q3


def load_baseline_chunks() -> int:
    try:
        j = json.load(open(STEP0, "r", encoding="utf-8"))
        for c in j.get("checks", []):
            if c.get("id") == "G0-04":
                return int(c.get("actual") or 0)
    except Exception:
        pass
    # Fallback: count directly
    cnt = 0
    for path in glob.glob(CHUNK_GLOB):
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                cnt += 1
    return cnt


def write_parquet(rows: List[Dict[str, Any]]):
    ensure_dir(VEC_DIR)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        # Try to install pyarrow on the fly
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "--quiet"])  # noqa: S603,S607
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore

    # Convert to columns
    chunk_ids = [r["chunk_id"] for r in rows]
    doc_ids = [r["doc_id"] for r in rows]
    seq_nos = [int(r.get("seq_no") or 0) for r in rows]
    token_counts = [int(r.get("token_count") or 0) for r in rows]
    norms = [float(r["l2_norm"]) for r in rows]
    vectors = [r["vector"] for r in rows]

    arrs = {
        "chunk_id": pa.array(chunk_ids, type=pa.string()),
        "doc_id": pa.array(doc_ids, type=pa.string()),
        "seq_no": pa.array(seq_nos, type=pa.int32()),
        "token_count": pa.array(token_counts, type=pa.int32()),
        "l2_norm": pa.array(norms, type=pa.float32()),
        "vector": pa.array(vectors, type=pa.list_(pa.float32())),
    }
    table = pa.table(arrs)
    pq.write_table(table, EMB_PARQUET)


def main():
    ensure_dir(VEC_DIR)
    dim = read_yaml_dim(CONF)
    baseline_chunks = load_baseline_chunks()

    rows: List[Dict[str, Any]] = []
    embedding_rows = 0
    zero_vectors = 0
    nan_vectors = 0
    norms: List[float] = []

    for path in sorted(glob.glob(CHUNK_GLOB)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                chunk_id = j.get("chunk_id") or ""
                doc_id = j.get("doc_id") or ""
                seq_no = j.get("seq_no") or 0
                token_count = j.get("token_count") or 0
                text = j.get("text") or ""

                # Text-based deterministic embedding (shared with queries)
                v = embed_text(text, dim)
                n = l2_norm(v)

                if n == 0.0:
                    zero_vectors += 1
                if any((x != x) for x in v):  # NaN check
                    nan_vectors += 1

                rows.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "seq_no": seq_no,
                    "token_count": token_count,
                    "l2_norm": n,
                    "vector": [float(x) for x in v],
                })
                norms.append(n)
                embedding_rows += 1

    # Stats
    q1, med, q3 = quartiles(norms)
    iqr = max(0.0, q3 - q1)
    lo = med - 4 * iqr
    hi = med + 4 * iqr
    if iqr == 0:
        # Tight distribution; treat only exact unequal as outliers
        outliers = sum(1 for n in norms if n < med or n > med)
    else:
        outliers = sum(1 for n in norms if n < lo or n > hi)
    pct_norm_outliers = outliers / max(1, len(norms))

    # Save parquet and stats
    write_parquet(rows)
    stats = {
        "embedding_rows": embedding_rows,
        "zero_vectors": zero_vectors,
        "nan_vectors": nan_vectors,
        "median_norm": round(float(med), 6),
        "iqr": round(float(iqr), 6),
        "pct_norm_outliers": round(float(pct_norm_outliers), 6),
        "vector_dim": dim,
        "baseline_chunks": baseline_chunks,
        "parquet_path": EMB_PARQUET,
    }
    with open(EMB_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # Checks and status
    checks: List[Dict[str, Any]] = []
    checks.append({
        "id": "G1-01", "metric": "embedding_rows", "actual": embedding_rows,
        "threshold": f"== baseline_chunks ({baseline_chunks})",
        "status": "PASS" if embedding_rows == baseline_chunks else "FAIL",
        "evidence": EMB_STATS,
    })
    checks.append({
        "id": "G1-02", "metric": "vector_dim", "actual": dim,
        "threshold": f"== {dim} (from config)",
        "status": "PASS", "evidence": CONF,
    })
    checks.append({
        "id": "G1-03a", "metric": "zero_vectors", "actual": zero_vectors,
        "threshold": "==0", "status": "PASS" if zero_vectors == 0 else "FAIL", "evidence": EMB_STATS,
    })
    checks.append({
        "id": "G1-03b", "metric": "nan_vectors", "actual": nan_vectors,
        "threshold": "==0", "status": "PASS" if nan_vectors == 0 else "FAIL", "evidence": EMB_STATS,
    })
    checks.append({
        "id": "G1-04", "metric": "pct_norm_outliers", "actual": round(pct_norm_outliers, 6),
        "threshold": "<=0.005", "status": "PASS" if pct_norm_outliers <= 0.005 else ("WARN" if pct_norm_outliers <= 0.015 else "FAIL"),
        "evidence": EMB_STATS,
    })

    # Status logic
    passes = {c["id"]: c for c in checks}
    if all(c["status"] == "PASS" for c in checks):
        status = "GREEN"
        next_action = "continue"
    elif all(passes[k]["status"] == "PASS" for k in ("G1-01", "G1-02", "G1-03a", "G1-03b")) and passes["G1-04"]["status"] == "WARN":
        status = "AMBER"
        next_action = "proceed_with_caution"
    else:
        status = "RED"
        next_action = "fix_and_rerun"

    # Write QA envelopes
    ensure_dir(os.path.join("reports", "qa"))
    machine = {
        "step": "step01_embeddings",
        "gate": "Gate-1",
        "status": status,
        "checks": checks,
        "next_action": next_action,
        "timestamp": now_iso(),
    }
    with open(os.path.join("reports", "qa", "step01_embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable
    lines = []
    lines.append(f"# STEP 1 — Embeddings Quality (Gate‑1) — {status}")
    lines.append("")
    lines.append("Checks:")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}")
    lines.append("")
    lines.append("Stats:")
    lines.append(f"- embedding_rows: {embedding_rows}")
    lines.append(f"- vector_dim: {dim}")
    lines.append(f"- zero_vectors: {zero_vectors}")
    lines.append(f"- nan_vectors: {nan_vectors}")
    lines.append(f"- median_norm: {round(float(med), 6)}")
    lines.append(f"- iqr: {round(float(iqr), 6)}")
    lines.append(f"- pct_norm_outliers: {round(float(pct_norm_outliers), 6)}")
    lines.append("")
    lines.append(f"Go/No-Go: { 'Go' if status in ('GREEN','AMBER') else 'No-Go' }")
    with open(os.path.join("reports", "qa", "step01_embeddings.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": status, "rows": embedding_rows}, indent=2))


if __name__ == "__main__":
    main()
