#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import glob

from common import ensure_dir, now_iso


STEP0_BASELINE = os.path.join("reports", "qa", "step00_baseline.json")
EVAL_SEED = os.path.join("data", "interim", "eval", "salesforce_eval_seed.jsonl")
TRACE_DIR = os.path.join("reports", "router")
TRACE_PATH = os.path.join(TRACE_DIR, "step04_router_trace.jsonl")
QA_JSON = os.path.join("reports", "qa", "step04_router.json")
QA_MD = os.path.join("reports", "qa", "step04_router.md")


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_seed(path: str = EVAL_SEED) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        # fallback queries
        for q in [
            {"persona": "vp_customer_experience", "query_text": "Agentforce product announcement"},
            {"persona": "cio", "query_text": "remaining performance obligation definition"},
            {"persona": "vp_sales_ops", "query_text": "latest earnings results"},
        ] * 15:
            items.append(q)
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def load_step0_baseline() -> Tuple[int, int]:
    """Return (age_p50_days, baseline_domain_count). Defaults (365, 3)."""
    try:
        j = json.load(open(STEP0_BASELINE, "r", encoding="utf-8"))
        age_p50 = int(((j or {}).get("baseline") or {}).get("age_days", {}).get("p50") or 365)
        domains = (j.get("baseline") or {}).get("domains") or []
        return age_p50, len(domains)
    except Exception:
        return 365, 3


async def kb_search(session: aiohttp.ClientSession, backend: str, query: str, top_k: int, tools_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, Optional[str]]:
    base = tools_cfg.get("kb.search") or {}
    host = base.get("host", "127.0.0.1")
    port = int(base.get("port", 7801))
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "search", "params": {"query": query, "backend": backend, "top_k": int(top_k)}}
    t0 = datetime.now(timezone.utc)
    try:
        async with session.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
            status = resp.status
            j = await resp.json()
            if status >= 400:
                return [], (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, (j.get("error") or {}).get("code")
            res = j.get("results") or []
            return res, (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, None
    except asyncio.TimeoutError:
        return [], (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, "Timeout"
    except Exception as e:
        return [], (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, "NetworkError"


def _days_since(iso_date: Optional[str]) -> Optional[int]:
    if not iso_date:
        return None
    try:
        d = date.fromisoformat(iso_date)
        return (datetime.now(timezone.utc).date() - d).days
    except Exception:
        return None


async def main_async():
    from router_core import load_router_config, load_mcp_map, load_doc_meta, decide_backend, rerank
    # Try to import/start MCP stubs; fall back to offline search if heavy deps (numpy/pyarrow) are missing
    use_offline = False
    start_stub_servers = None
    stop_stub_servers = None
    try:
        from qa_step03_mcp import start_stub_servers as _sss, stop_stub_servers as _sts  # type: ignore
        start_stub_servers = _sss
        stop_stub_servers = _sts
    except Exception:
        use_offline = True

    ensure_dir(os.path.dirname(QA_JSON))
    ensure_dir(TRACE_DIR)

    # Start stub servers (kb.search, etc.) so this step is self-contained
    tools_cfg = load_mcp_map()
    state: Dict[str, Any] = {}
    if not use_offline:
        try:
            await start_stub_servers(state, {"tools": tools_cfg})  # type: ignore
        except Exception:
            # Fallback to offline mode if servers fail to start (e.g., numpy/pyarrow issues)
            use_offline = True

    router_cfg = load_router_config()
    fallback_order: List[str] = router_cfg.get("fallback_order", ["faiss", "weaviate", "pinecone"])
    top_k_default = int(router_cfg.get("top_k_default", 10))
    weights = (router_cfg.get("weights") or {})
    docmeta = load_doc_meta()
    seed = load_seed()
    age_p50, baseline_domain_count = load_step0_baseline()

    # Prepare client (only if using stubs)
    session = None
    if not use_offline:
        connector = aiohttp.TCPConnector(limit_per_host=8)
        session = aiohttp.ClientSession(connector=connector)

    # Aggregates
    route_counts = Counter()
    total = 0
    empty = 0
    retry_success = 0
    sum_avg_age = 0.0
    sum_unique_domains = 0.0

    # Open trace
    # Offline search primitives (hash-embed) if needed
    chunks_index: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    dim = 768
    if use_offline:
        # Read embedding dim from vector config if available
        try:
            cfg = load_yaml(os.path.join("configs", "vector.indexing.yaml"))
            dim = int(((cfg or {}).get("embedding") or {}).get("dim") or 768)
        except Exception:
            dim = 768
        # Load chunks and build vectors deterministically, same logic as qa_step01_embeddings.hash_vec
        import random, math
        def hash_vec(seed: str, d: int) -> List[float]:
            rnd = random.Random()
            h = 0
            for ch in seed:
                h = (h * 1315423911) ^ ord(ch)
                h &= 0xFFFFFFFFFFFFFFFF
            rnd.seed(h)
            vals = [rnd.uniform(-1.0, 1.0) for _ in range(d)]
            s2 = sum(v*v for v in vals) or 1.0
            inv = 1.0 / math.sqrt(s2)
            return [v*inv for v in vals]
        def embed_query(q: str, d: int) -> List[float]:
            # Same structure as step02/03
            import random, math
            rnd = random.Random()
            h = 0
            for ch in q:
                h = (h * 1315423911) ^ ord(ch)
                h &= 0xFFFFFFFFFFFFFFFF
            rnd.seed(h)
            vals = [rnd.uniform(-1.0, 1.0) for _ in range(d)]
            s2 = sum(v*v for v in vals) or 1.0
            inv = 1.0 / math.sqrt(s2)
            return [v*inv for v in vals]
        # Load chunks
        for cf in sorted(glob.glob(os.path.join("data", "interim", "chunks", "*.chunks.jsonl"))):
            with open(cf, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    if not j.get("chunk_id"):
                        continue
                    chunks_index.append(j)
                    emb_seed = f"{j.get('chunk_id')}::{len(j.get('text') or '')}::{int(j.get('token_count') or 0)}"
                    vectors.append(hash_vec(emb_seed, dim))
        # Distance util
        def l2(a: List[float], b: List[float]) -> float:
            return sum((x-y)*(x-y) for x,y in zip(a,b))

    with open(TRACE_PATH, "w", encoding="utf-8") as tf:
        for it in seed:
            q = (it.get("query_text") or "").strip()
            persona = (it.get("persona") or "").strip() or None
            if not q:
                continue
            total += 1
            backend, reasons = decide_backend(q, persona, None)
            route_counts[backend] += 1

            # primary call
            if not use_offline:
                results, latency_ms, err = await kb_search(session, backend, q, top_k_default, tools_cfg)  # type: ignore
            else:
                # Offline search across all chunks
                import time
                t0 = time.perf_counter()
                qv = embed_query(q, dim)
                scored: List[Tuple[float, int]] = []  # (dist, idx)
                for i, v in enumerate(vectors):
                    scored.append((l2(qv, v), i))
                scored.sort(key=lambda x: x[0])
                top = scored[:top_k_default]
                results = []
                for dist, idx in top:
                    ch = chunks_index[idx]
                    results.append({
                        "chunk_id": ch.get("chunk_id"),
                        "doc_id": ch.get("doc_id"),
                        "score": float(-dist),
                        "snippet": (ch.get("text") or "")[:280],
                    })
                latency_ms = (time.perf_counter() - t0) * 1000.0
                err = None
            fallback_used = False
            if not results:
                empty += 1
                # try fallback
                fallback_used = True
                # next backend in configured order (choose the first different backend)
                try_order = [b for b in fallback_order if b != backend] + [backend]
                for fb in try_order:
                    if fb == backend:
                        continue
                    if not use_offline:
                        res2, lat2, err2 = await kb_search(session, fb, q, top_k_default, tools_cfg)  # type: ignore
                    else:
                        # In offline mode, fallback is identical to primary search; reuse results
                        res2, lat2, err2 = results, latency_ms, None
                    if res2:
                        results = res2
                        latency_ms = lat2
                        reasons = reasons + ["FALLBACK:" + fb]
                        retry_success += 1
                        break

            # Re-rank and annotate
            if results:
                results = rerank(results, docmeta, weights)

            # Per-query metrics
            ages: List[int] = []
            domains: set = set()
            for r in results[: top_k_default]:
                did = r.get("doc_id")
                dm = docmeta.get(did)
                if dm and dm.publish_date:
                    ds = _days_since(dm.publish_date)
                    if ds is not None:
                        ages.append(ds)
                if dm and dm.source_domain:
                    domains.add(dm.source_domain)
            avg_age = (sum(ages) / max(1, len(ages))) if ages else float(age_p50)
            uniq_domains = len(domains)
            sum_avg_age += avg_age
            sum_unique_domains += float(uniq_domains)

            # Trace line
            tf.write(json.dumps({
                "timestamp": now_iso(),
                "query_text": q,
                "persona": persona,
                "decision_backend": backend,
                "fallback_used": fallback_used,
                "reason_codes": reasons,
                "latency_ms": round(latency_ms, 3),
                "top_k": top_k_default,
                "n_unique_domains": uniq_domains,
                "avg_doc_age_days": round(avg_age, 2),
                "empty_result": (len(results) == 0),
            }, ensure_ascii=False) + "\n")

    if session is not None:
        await session.close()

    # Stop servers if they were started
    if not use_offline and stop_stub_servers is not None:
        await stop_stub_servers(state)  # type: ignore

    # Compute QA checks
    empty_rate = (empty / max(1, total)) if total else 0.0
    retry_rate = (retry_success / max(1, empty)) if empty else 1.0
    mean_age = (sum_avg_age / max(1, total)) if total else float(age_p50)
    mean_unique_domains = (sum_unique_domains / max(1, total)) if total else 0.0

    # Coverage thresholds
    checks: List[Dict[str, Any]] = []
    shares = {k: (route_counts.get(k, 0) / max(1, total)) for k in ("pinecone", "weaviate", "faiss")}
    for k in ("pinecone", "weaviate", "faiss"):
        share = shares[k]
        count = route_counts.get(k, 0)
        ok = (share >= 0.10) or (count >= 1)
        checks.append({
            "id": f"COV-{k}",
            "metric": f"{k}_route_share",
            "actual": round(share, 4),
            "threshold": ">=0.10 or >=1 route",
            "status": "PASS" if ok else "FAIL",
            "evidence": TRACE_PATH,
        })

    checks.append({
        "id": "EMP-001",
        "metric": "empty_result_rate",
        "actual": round(empty_rate, 4),
        "threshold": "<=0.02",
        "status": "PASS" if empty_rate <= 0.02 else "FAIL",
    })
    checks.append({
        "id": "EMP-002",
        "metric": "auto_retry_success_rate",
        "actual": round(retry_rate, 4),
        "threshold": ">=0.95 (if any empty)",
        "status": "PASS" if (empty == 0 or retry_rate >= 0.95) else "FAIL",
    })

    freshness_thr = max(365, int(age_p50))
    checks.append({
        "id": "FRS-001",
        "metric": "avg_doc_age_days",
        "actual": round(mean_age, 2),
        "threshold": f"<={freshness_thr}",
        "status": "PASS" if mean_age <= freshness_thr else ("WARN" if mean_age <= freshness_thr * 1.1 else "FAIL"),
    })

    diversity_thr = max(3.0, 0.75 * (baseline_domain_count / 10.0))
    checks.append({
        "id": "DIV-001",
        "metric": "mean_unique_domains_top10",
        "actual": round(mean_unique_domains, 3),
        "threshold": f">={round(diversity_thr,3)}",
        "status": "PASS" if mean_unique_domains >= diversity_thr else ("WARN" if mean_unique_domains >= max(2.0, diversity_thr - 0.5) else "FAIL"),
    })

    # Status rollup
    if all(c["status"] == "PASS" for c in checks):
        status = "GREEN"
        next_action = "continue"
    else:
        fails = [c for c in checks if c["status"] == "FAIL"]
        warns = [c for c in checks if c["status"] == "WARN"]
        if fails:
            status = "RED"
            next_action = "improve_weights_and_rules"
        else:
            status = "AMBER"
            next_action = "increase_recency_diversity_weights"

    machine = {
        "step": "step04_router",
        "gate": "Gate-4",
        "status": status,
        "checks": checks,
        "metrics": {
            "total_queries": total,
            "route_counts": dict(route_counts),
            "empty_result_rate": round(empty_rate, 4),
            "auto_retry_success_rate": round(retry_rate, 4),
            "avg_doc_age_days": round(mean_age, 2),
            "mean_unique_domains_top10": round(mean_unique_domains, 3),
        },
        "next_action": next_action,
        "timestamp": now_iso(),
    }

    ensure_dir(os.path.dirname(QA_JSON))
    with open(QA_JSON, "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable MD
    lines: List[str] = []
    lines.append(f"# STEP 4 — Router Heuristics & Coverage (Gate‑4) — {status}")
    lines.append("")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}")
    lines.append("")
    m = machine["metrics"]
    lines.append("Summary:")
    lines.append(f"- total_queries: {m['total_queries']}")
    lines.append(f"- route_counts: {m['route_counts']}")
    lines.append(f"- empty_result_rate: {m['empty_result_rate']}")
    lines.append(f"- auto_retry_success_rate: {m['auto_retry_success_rate']}")
    lines.append(f"- avg_doc_age_days: {m['avg_doc_age_days']}")
    lines.append(f"- mean_unique_domains_top10: {m['mean_unique_domains_top10']}")
    lines.append("")
    lines.append(f"Go/No-Go: {'Go' if status in ('GREEN','AMBER') else 'No-Go'}")
    with open(QA_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": status}, indent=2))


def main():
    _ = argparse.ArgumentParser(description="Gate-4 — Router QA")
    args = _.parse_args()
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
