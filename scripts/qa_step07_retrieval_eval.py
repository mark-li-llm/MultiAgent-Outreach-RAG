#!/usr/bin/env python3
import argparse
import asyncio
import glob
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from common import ensure_dir, now_iso


SEED_PATH = os.path.join("data", "interim", "eval", "salesforce_eval_seed.jsonl")
STEP0 = os.path.join("reports", "qa", "step00_baseline.json")
STEP3 = os.path.join("reports", "qa", "step03_mcp.json")
FAIL_LOG = os.path.join("reports", "eval", "recovery_failures.jsonl")  # will override to retrieval_failures later
FAIL_LOG = os.path.join("reports", "eval", "retrieval_failures.jsonl")
OUT_JSON = os.path.join("reports", "qa", "step07_retrieval_eval.json")
OUT_MD = os.path.join("reports", "qa", "step07_retrieval_eval.md")


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_doc_meta() -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob(os.path.join("data", "interim", "normalized", "*.json")):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        m[d.get("doc_id")] = d
    return m


def load_seed(path: str = SEED_PATH) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


async def kb_search(session: aiohttp.ClientSession, backend: str, query: str, top_k: int, tools_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
    base = tools_cfg.get("kb.search") or {}
    host = base.get("host", "127.0.0.1")
    port = int(base.get("port", 7801))
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "search", "params": {"query": query, "backend": backend, "top_k": int(top_k)}}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
            status = resp.status
            j = await resp.json()
            if status >= 400:
                return [], (time.perf_counter() - t0) * 1000.0, (j.get("error") or {}).get("code")
            res = j.get("results") or []
            return res, (time.perf_counter() - t0) * 1000.0, None
    except Exception as e:
        return [], (time.perf_counter() - t0) * 1000.0, "NetworkError"


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])


def read_step0_baseline() -> Tuple[int, int]:
    try:
        j = json.load(open(STEP0, "r", encoding="utf-8"))
        age_p50 = int(((j or {}).get("baseline") or {}).get("age_days", {}).get("p50") or 365)
        domains = (j.get("baseline") or {}).get("domains") or []
        return age_p50, len(domains)
    except Exception:
        return 365, 3


def read_step3_budgets() -> Dict[str, float]:
    # Return budget_p95 per backend if available
    budgets = {"faiss": 300.0, "weaviate": 1000.0, "pinecone": 1500.0}
    try:
        j = json.load(open(STEP3, "r", encoding="utf-8"))
        for c in j.get("checks", []):
            cid = c.get("id") or ""
            if cid.startswith("G3-03-"):
                b = cid.split("-")[-1]
                actual = c.get("actual") or {}
                bud = float(actual.get("budget_p95") or budgets.get(b, 0.0))
                budgets[b] = bud
    except Exception:
        pass
    return budgets


async def main_async(args):
    from router_core import load_mcp_map, load_router_config, decide_backend, rerank
    # Try start stubs; fallback offline
    use_offline = False
    start_stub_servers = None
    stop_stub_servers = None
    try:
        from qa_step03_mcp import start_stub_servers as _sss, stop_stub_servers as _sts  # type: ignore
        start_stub_servers = _sss
        stop_stub_servers = _sts
    except Exception:
        use_offline = True

    tools_cfg = load_mcp_map()
    state_env: Dict[str, Any] = {}
    if not use_offline:
        try:
            await start_stub_servers(state_env, {"tools": tools_cfg})  # type: ignore
        except Exception:
            # Try to use external service before falling back to offline mode
            print("Failed to start internal stub servers, checking external service...")
            try:
                connector = aiohttp.TCPConnector(limit_per_host=8)
                async with aiohttp.ClientSession(connector=connector) as test_session:
                    base = tools_cfg.get("kb.search") or {}
                    host = base.get("host", "127.0.0.1")
                    port = int(base.get("port", 7801))
                    url = f"http://{host}:{port}/invoke"
                    payload = {"method": "search", "params": {"query": "test", "backend": "faiss", "top_k": 1}}
                    async with test_session.post(url, json=payload, timeout=2.0) as resp:
                        if resp.status == 200:
                            result = await resp.json()
                            if "results" in result:
                                print(f"✓ External service available at {url}")
                                use_offline = False
                            else:
                                use_offline = True
                        else:
                            use_offline = True
            except Exception as e:
                print(f"External service test failed: {e}")
                use_offline = True

    # Offline index if needed
    chunks_index: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)
    # Preload chunk metadata for diagnostics (independent of online/offline)
    # chunk_meta: chunk_id -> {doc_id, seq_no, start_char, text}
    chunk_meta: Dict[str, Dict[str, Any]] = {}
    for cf in sorted(glob.glob(os.path.join("data","interim","chunks","*.chunks.jsonl"))):
        try:
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        ch = json.loads(line)
                    except Exception:
                        continue
                    cid = ch.get("chunk_id")
                    if not cid:
                        continue
                    chunk_meta[cid] = {
                        "doc_id": ch.get("doc_id"),
                        "seq_no": int(ch.get("seq_no") or 0),
                        "start_char": int(ch.get("start_char") or 0),
                        "text": (ch.get("text") or ""),
                    }
        except Exception:
            continue
    if use_offline:
        from embedding_utils import embed_text as _embed_text
        def embed_query(q: str, d: int) -> List[float]:
            return _embed_text(q, d)
        def l2(a: List[float], b: List[float]) -> float:
            return sum((x-y)*(x-y) for x,y in zip(a,b))
        for cf in sorted(glob.glob(os.path.join("data","interim","chunks","*.chunks.jsonl"))):
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    if not j.get("chunk_id"):
                        continue
                    chunks_index.append(j)
                    vectors.append(_embed_text(j.get('text') or '', dim))

    # Load seed and metas
    seed = load_seed(SEED_PATH)
    docmeta = load_doc_meta()
    age_p50, baseline_domain_count = read_step0_baseline()
    budgets_p95 = read_step3_budgets()

    # Optional overrides via env for this step
    # - AG7_IGNORE_COVERAGE=1 to skip coverage gating
    # - AG7_LATENCY_MULTIPLIER=float to relax latency budgets (e.g., 2.0)
    ignore_coverage = os.getenv("AG7_IGNORE_COVERAGE", "0") == "1"
    try:
        latency_multiplier = float(os.getenv("AG7_LATENCY_MULTIPLIER", "1.0"))
    except Exception:
        latency_multiplier = 1.0

    connector = aiohttp.TCPConnector(limit_per_host=8)
    session = aiohttp.ClientSession(connector=connector) if not use_offline else None

    # Diagnostics configuration (env overrides)
    try:
        topk_eval = int(os.getenv("AG7_ANALYZE_TOPK", "10"))
    except Exception:
        topk_eval = 10
    try:
        near_seq_tol = int(os.getenv("AG7_NEAR_SEQ_TOL", "1"))
    except Exception:
        near_seq_tol = 1

    # Helper: simple jaccard on alnum tokens
    import re as _re
    def _tok_alnum(s: str) -> List[str]:
        return _re.findall(r"[a-zA-Z0-9]+", (s or "").lower())
    def jaccard(a: str, b: str) -> float:
        sa, sb = set(_tok_alnum(a)), set(_tok_alnum(b))
        if not sa and not sb:
            return 0.0
        return len(sa & sb) / max(1, len(sa | sb))

    # Metrics accumulators
    total = 0
    hits = 0
    dcg5_vals: List[float] = []
    doc_hits = 0
    doc_dcg5_vals: List[float] = []
    soft_hits = 0
    uniq_domain_avgs: List[float] = []
    age_avgs: List[float] = []
    lat_by_backend: Dict[str, List[float]] = {"faiss": [], "weaviate": [], "pinecone": []}

    # Per-doctype diagnostics
    by_dt_counts: Dict[str, Dict[str, int]] = {}
    def _ensure_dt(dt: str):
        if dt not in by_dt_counts:
            by_dt_counts[dt] = {
                "total": 0,
                "chunk_hit": 0,
                "doc_hit": 0,
                "soft_hit": 0,
                "doc_miss": 0,
                "near_miss": 0,
            }

    ensure_dir(os.path.dirname(FAIL_LOG))
    with open(FAIL_LOG, "w", encoding="utf-8") as flog:
        for it in seed:
            q = (it.get("query_text") or "").strip()
            persona = (it.get("persona") or "").strip() or None
            exp_cid = (it.get("expected_chunk_id") or "").strip()
            if not q or not exp_cid:
                continue
            total += 1
            backend, reasons = decide_backend(q, persona, None)
            # Retrieve
            if not use_offline:
                res, lat, err = await kb_search(session, backend, q, topk_eval, tools_cfg)
            else:
                import time as _t
                t0 = _t.perf_counter()
                qv = embed_query(q, dim)
                scored: List[Tuple[float, int]] = []
                for i, v in enumerate(vectors):
                    # L2 distance
                    d = sum((x - y) * (x - y) for x, y in zip(qv, v))
                    scored.append((d, i))
                scored.sort(key=lambda x: x[0])
                res = []
                for dist, idx in scored[:topk_eval]:
                    ch = chunks_index[idx]
                    res.append({
                        "chunk_id": ch.get("chunk_id"),
                        "doc_id": ch.get("doc_id"),
                        "score": float(-dist),
                        "snippet": (ch.get("text") or "")[:280],
                    })
                lat = (_t.perf_counter() - t0) * 1000.0
                err = None
            lat_by_backend.setdefault(backend, []).append(float(lat))

            # Rank and DCG
            ranks = [r.get("chunk_id") for r in res]
            try:
                rank = ranks.index(exp_cid) + 1
            except Exception:
                rank = 0
            if rank and rank <= 10:
                hits += 1
            # binary DCG@5
            dcg = (1.0 / math.log2(rank + 1)) if (rank and rank <= 5) else 0.0
            dcg5_vals.append(dcg)

            # Doc-level metrics
            exp_doc_id = "::".join(exp_cid.split("::")[:-1]) if exp_cid else ""
            doc_ranks = [r.get("doc_id") for r in res]
            try:
                doc_rank = doc_ranks.index(exp_doc_id) + 1
            except Exception:
                doc_rank = 0
            if doc_rank and doc_rank <= topk_eval:
                doc_hits += 1
            doc_dcg = (1.0 / math.log2(doc_rank + 1)) if (doc_rank and doc_rank <= 5) else 0.0
            doc_dcg5_vals.append(doc_dcg)

            # Soft (near-miss) metrics: same doc and close seq/start_char
            near_hit = False
            nearest_same_doc = None
            if (not rank) and exp_doc_id:
                # find closest candidate in same doc
                exp_meta = chunk_meta.get(exp_cid) or {}
                exp_seq = int(exp_meta.get("seq_no") or 0)
                exp_start = int(exp_meta.get("start_char") or 0)
                best = None  # (score, payload)
                for i, r in enumerate(res):
                    if r.get("doc_id") != exp_doc_id:
                        continue
                    cand_meta = chunk_meta.get(r.get("chunk_id", "")) or {}
                    dseq = abs(int(cand_meta.get("seq_no") or 0) - exp_seq)
                    dstart = abs(int(cand_meta.get("start_char") or 0) - exp_start)
                    payload = {
                        "rank": i + 1,
                        "chunk_id": r.get("chunk_id"),
                        "seq_no": int(cand_meta.get("seq_no") or 0),
                        "delta_seq": int(dseq),
                        "delta_start": int(dstart),
                    }
                    sc = -i  # prefer higher rank
                    if (best is None) or (sc > best[0]):
                        best = (sc, payload)
                    if dseq <= near_seq_tol:
                        near_hit = True
                        break
                if best is not None:
                    nearest_same_doc = best[1]
            if near_hit:
                soft_hits += 1

            # Coverage & freshness for this query
            doms = set()
            ages: List[int] = []
            for r in res[:10]:
                d = docmeta.get(r.get("doc_id"), {})
                dom = (d.get("source_domain") or "").strip()
                if dom:
                    doms.add(dom)
                iso = (d.get("publish_date") or "").strip()
                if iso:
                    try:
                        dd = date.fromisoformat(iso)
                        ages.append((datetime.now(timezone.utc).date() - dd).days)
                    except Exception:
                        ages.append(365)
                else:
                    ages.append(365)
            uniq_domain_avgs.append(float(len(doms)))
            if ages:
                age_avgs.append(float(sum(ages) / max(1, len(ages))))

            # Determine doctype bucket for diagnostics (by expected doc)
            dt = (docmeta.get(exp_doc_id, {}) or {}).get("doctype") or (exp_doc_id.split("::")[1] if "::" in exp_doc_id else "unknown")
            _ensure_dt(dt)
            by_dt_counts[dt]["total"] += 1
            if rank and rank <= topk_eval:
                by_dt_counts[dt]["chunk_hit"] += 1
            if doc_rank and doc_rank <= topk_eval:
                by_dt_counts[dt]["doc_hit"] += 1
            if near_hit:
                by_dt_counts[dt]["soft_hit"] += 1
            if not (doc_rank and doc_rank <= topk_eval):
                by_dt_counts[dt]["doc_miss"] += 1
            if (not rank) and near_hit:
                by_dt_counts[dt]["near_miss"] += 1

            # Log failures (chunk-level miss)
            if rank == 0:
                flog.write(json.dumps({
                    "eval_id": it.get("eval_id"),
                    "persona": persona,
                    "query_text": q,
                    "backend": backend,
                    "expected_chunk_id": exp_cid,
                    # diagnostic additions
                    "expected_doc_id": exp_doc_id,
                    "classification": (
                        "chunk_miss_doc_miss" if not (doc_rank and doc_rank <= topk_eval) else (
                            "chunk_miss_doc_hit_near" if near_hit else "chunk_miss_doc_hit_far"
                        )
                    ),
                    "nearest_same_doc": nearest_same_doc or {},
                    "topk": [{
                        "rank": i + 1,
                        "chunk_id": r.get("chunk_id"),
                        "doc_id": r.get("doc_id"),
                        "source_domain": (docmeta.get(r.get("doc_id"), {}) or {}).get("source_domain"),
                        "score": r.get("score"),
                    } for i, r in enumerate(res)],
                }) + "\n")

    if session is not None:
        await session.close()
    if not use_offline and stop_stub_servers is not None:
        await stop_stub_servers(state_env)  # type: ignore

    # Aggregates
    recall10 = hits / max(1, total)
    ndcg5 = (sum(dcg5_vals) / max(1, len(dcg5_vals)))
    # Diagnostics metrics
    doc_recall10 = doc_hits / max(1, total)
    doc_ndcg5 = (sum(doc_dcg5_vals) / max(1, len(doc_dcg5_vals)))
    soft_recall10 = soft_hits / max(1, total)
    near_miss_rate = max(0.0, doc_recall10 - recall10)
    coverage = (sum(uniq_domain_avgs) / max(1, len(uniq_domain_avgs))) if uniq_domain_avgs else 0.0
    freshness = (sum(age_avgs) / max(1, len(age_avgs))) if age_avgs else 365.0

    # Latency budgets
    # Apply latency budget relaxation if requested
    budgets = {k: (v * latency_multiplier) for k, v in (budgets_p95 or {}).items()}
    lat_ok = True
    lat_checks: List[Dict[str, Any]] = []
    for b in ("faiss", "weaviate", "pinecone"):
        arr = lat_by_backend.get(b, [])
        p50 = pct(arr, 50.0)
        p95 = pct(arr, 95.0)
        # If no data for backend, treat as PASS
        if not arr:
            lat_checks.append({"backend": b, "p50": round(p50, 2), "p95": round(p95, 2), "budget_p95": budgets.get(b), "status": "PASS"})
            continue
        # Enforce: P50 <= budget_p95 and P95 <= budget_p95 (Step-3 already used budget=min(doc_budget, observed_p95*1.2))
        ok = (p50 <= budgets.get(b, 1e9) and p95 <= budgets.get(b, 1e9))
        lat_ok = lat_ok and ok
        lat_checks.append({"backend": b, "p50": round(p50, 2), "p95": round(p95, 2), "budget_p95": budgets.get(b), "status": "PASS" if ok else "FAIL"})

    # Thresholds and checks
    baseline_domains = baseline_domain_count
    cov_thr = max(3.0, 0.75 * (baseline_domains / 10.0))
    fresh_thr = max(540.0, float(age_p50))

    checks: List[Dict[str, Any]] = []
    checks.append({"id": "G7-01", "metric": "recall@10", "actual": round(recall10, 4), "threshold": ">=0.80", "status": "PASS" if recall10 >= 0.80 else "FAIL", "evidence": FAIL_LOG})
    checks.append({"id": "G7-02", "metric": "nDCG@5", "actual": round(ndcg5, 4), "threshold": ">=0.60", "status": "PASS" if ndcg5 >= 0.60 else "FAIL", "evidence": FAIL_LOG})
    if not ignore_coverage:
        checks.append({"id": "G7-03", "metric": "coverage_unique_domains_top10_mean", "actual": round(coverage, 3), "threshold": f">={round(cov_thr,3)}", "status": "PASS" if coverage >= cov_thr else "FAIL"})
    checks.append({"id": "G7-04", "metric": "freshness_mean_age_days", "actual": round(freshness, 2), "threshold": f"<={int(fresh_thr)}", "status": "PASS" if freshness <= fresh_thr else "FAIL"})
    checks.append({"id": "G7-05", "metric": "latency_budgets", "actual": {c["backend"]: {"p50": c["p50"], "p95": c["p95"], "budget_p95": c["budget_p95"]} for c in lat_checks}, "threshold": "p50,p95 <= budget_p95 per backend", "status": "PASS" if lat_ok else "FAIL"})

    # Status rollup
    pass_map = {c["id"]: c for c in checks}
    if all(c["status"] == "PASS" for c in checks):
        status = "GREEN"
        next_action = "continue"
    else:
        fails = [c for c in checks if c["status"] == "FAIL"]
        if len(fails) == 1 and fails[0]["id"] in ("G7-02", "G7-03", "G7-04", "G7-05"):
            status = "AMBER"
            next_action = "proceed_with_caution"
        else:
            status = "RED"
            next_action = "fix_and_rerun"

    # Write reports
    ensure_dir(os.path.dirname(OUT_JSON))
    machine = {
        "step": "step07_retrieval_eval",
        "status": status,
        "checks": checks,
        "summary": {
            "recall@10": round(recall10, 4),
            "nDCG@5": round(ndcg5, 4),
            "coverage_unique_domains_top10_mean": round(coverage, 3),
            "freshness_mean_age_days": round(freshness, 2),
            "latency": lat_checks,
            "queries": total,
            # diagnostics (not used for gating)
            "doc_recall@10": round(doc_recall10, 4),
            "soft_recall@10": round(soft_recall10, 4),
            "doc_nDCG@5": round(doc_ndcg5, 4),
            "near_miss_rate": round(near_miss_rate, 4),
            "by_doctype": {
                k: {
                    "total": v.get("total", 0),
                    "chunk_hit": v.get("chunk_hit", 0),
                    "doc_hit": v.get("doc_hit", 0),
                    "soft_hit": v.get("soft_hit", 0),
                    "doc_miss": v.get("doc_miss", 0),
                    "near_miss": v.get("near_miss", 0),
                } for k, v in by_dt_counts.items()
            },
        },
        "next_action": next_action,
        "timestamp": now_iso(),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    lines: List[str] = []
    lines.append(f"# STEP 7 — Retrieval Evaluation (Gate‑7) — {status}")
    lines.append("")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c.get('threshold','')}) -> {c['status']}")
    lines.append("")
    # Diagnostics (not gating)
    lines.append("Diagnostics (not gating):")
    lines.append(f"- doc_recall@10: {round(doc_recall10,4)}")
    lines.append(f"- soft_recall@10: {round(soft_recall10,4)}")
    lines.append(f"- doc_nDCG@5: {round(doc_ndcg5,4)}")
    lines.append(f"- near_miss_rate: {round(near_miss_rate,4)}")
    # Top-level by-doctype summary (first few)
    if by_dt_counts:
        lines.append("- by_doctype (total/chunk_hit/doc_hit/soft_hit):")
        for k in sorted(by_dt_counts.keys()):
            v = by_dt_counts[k]
            lines.append(f"  - {k}: {v['total']}/{v['chunk_hit']}/{v['doc_hit']}/{v['soft_hit']}")
    lines.append("")
    lines.append("Latency by backend:")
    for c in lat_checks:
        lines.append(f"- {c['backend']}: p50={c['p50']} p95={c['p95']} budget_p95={c['budget_p95']} -> {c['status']}")
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": status, "queries": total}, indent=2))


def parse_args():
    p = argparse.ArgumentParser(description="Gate-7 — Retrieval Evaluation")
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
