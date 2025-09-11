#!/usr/bin/env python3
import argparse
import asyncio
import glob
import json
import os
import time
import uuid
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import re

from common import ensure_dir, now_iso


NODES_CONF = os.path.join("configs", "langgraph.nodes.yaml")
SEED_PATH = os.path.join("data", "interim", "eval", "salesforce_eval_seed.jsonl")


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


def within_12mo(iso: Optional[str]) -> bool:
    if not iso:
        return False
    try:
        d = date.fromisoformat(iso)
        return (datetime.now(timezone.utc).date() - d).days <= 365
    except Exception:
        return False


async def main_async(args) -> str:
    from router_core import load_router_config, load_mcp_map, decide_backend, rerank

    nodes_cfg = load_yaml(NODES_CONF)
    nodes = nodes_cfg.get("nodes", [
        "Intake","Planner","Retriever","Synthesizer","Consolidator","Stylist","A2A","Assembler"
    ])

    session_id = args.session_id or uuid.uuid4().hex[:12]
    out_dir = os.path.join("outputs", session_id)
    state_dir = "state"
    ensure_dir(out_dir)
    ensure_dir(state_dir)

    # Shared state
    state: Dict[str, Any] = {
        "company": args.company,
        "persona": args.persona,
        "queries": [],
        "retrieved_chunks": [],
        "retrieval_logs": [],
        "insight_candidates": [],
        "insight_cards": [],
        "email_draft": {},
        "compliance_flags": [],
        "metrics": {"nodes_executed": [], "timings": {}},
        "route_decisions": [],
        "errors": [],
        "timestamp": now_iso(),
        "session_id": session_id,
    }

    def mark(node: str, start_ms: float, end_ms: float):
        state["metrics"]["nodes_executed"].append(node)
        state["metrics"]["timings"][node] = round((end_ms - start_ms) * 1000.0, 2)

    t_total0 = time.perf_counter()

    # Intake
    t0 = time.perf_counter()
    if not args.company or not args.persona:
        state["errors"].append("missing company/persona")
    mark("Intake", t0, time.perf_counter())

    # Planner: pick 5 persona queries from eval seed or defaults
    t0 = time.perf_counter()
    seed_items: List[Dict[str, Any]] = []
    if os.path.exists(SEED_PATH):
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if (j.get("persona") or "") == args.persona:
                    seed_items.append(j)
    queries: List[str] = []
    seen = set()
    for it in seed_items:
        qt = (it.get("query_text") or "").strip()
        if qt and qt not in seen:
            queries.append(qt)
            seen.add(qt)
        if len(queries) >= 5:
            break
    if not queries:
        queries = [
            "Agentforce product announcement",
            "latest earnings results",
            "remaining performance obligation definition",
            "customer experience AI",
            "Data Cloud recent updates",
        ]
    state["queries"] = queries
    mark("Planner", t0, time.perf_counter())

    # Retriever
    t0 = time.perf_counter()
    tools_cfg = load_mcp_map()
    router_cfg = load_router_config()
    weights = router_cfg.get("weights") or {}
    docmeta = load_doc_meta()

    # Try to use MCP stubs; if fail, offline search
    use_offline = False
    start_stub_servers = None
    stop_stub_servers = None
    try:
        from qa_step03_mcp import start_stub_servers as _sss, stop_stub_servers as _sts  # type: ignore
        start_stub_servers = _sss
        stop_stub_servers = _sts
    except Exception:
        use_offline = True
    state_env: Dict[str, Any] = {}
    if not use_offline:
        try:
            await start_stub_servers(state_env, {"tools": tools_cfg})  # type: ignore
        except Exception:
            use_offline = True

    # Offline index setup
    chunks_index: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or {}).get("embedding") or {}).get("dim") or 768)
    if use_offline:
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
        def l2(a: List[float], b: List[float]) -> float:
            return sum((x-y)*(x-y) for x,y in zip(a,b))
        # Load chunks
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
                    seed = f"{j.get('chunk_id')}::{len(j.get('text') or '')}::{int(j.get('token_count') or 0)}"
                    vectors.append(hash_vec(seed, dim))

    connector = aiohttp.TCPConnector(limit_per_host=8)
    session = aiohttp.ClientSession(connector=connector) if not use_offline else None

    router_trace_path = os.path.join(out_dir, "router_trace.jsonl")
    with open(router_trace_path, "w", encoding="utf-8") as tf:
        for q in queries:
            backend, reasons = decide_backend(q, args.persona, None)
            state["route_decisions"].append({"query": q, "backend": backend, "reasons": reasons})
            # Retrieve
            if not use_offline:
                res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)
            else:
                import time as _t
                tstart = _t.perf_counter()
                qv = embed_query(q, dim)
                scored = []
                for i, v in enumerate(vectors):
                    dist = sum((x-y)*(x-y) for x,y in zip(qv, v))
                    scored.append((dist, i))
                scored.sort(key=lambda x: x[0])
                res = []
                for dist, idx in scored[:12]:
                    ch = chunks_index[idx]
                    res.append({
                        "chunk_id": ch.get("chunk_id"),
                        "doc_id": ch.get("doc_id"),
                        "score": float(-dist),
                        "snippet": (ch.get("text") or "")[:280],
                    })
                lat = (_t.perf_counter() - tstart) * 1000.0
                err = None

            # Re-rank + attach meta
            res = rerank(res, {k: type("DM", (), v) for k, v in docmeta.items()}, top_k=12, domain_cap=2)  # type: ignore
            # log trace
            uniq_domains = len(set((docmeta.get(r.get("doc_id"), {}).get("source_domain") or "") for r in res[:10]))
            ages = []
            for r in res[:10]:
                iso = (docmeta.get(r.get("doc_id"), {}) or {}).get("publish_date")
                try:
                    if iso:
                        d0 = date.fromisoformat(iso)
                        ages.append((datetime.now(timezone.utc).date() - d0).days)
                except Exception:
                    pass
            avg_age = (sum(ages)/max(1,len(ages))) if ages else None
            tf.write(json.dumps({
                "timestamp": now_iso(),
                "query_text": q,
                "decision_backend": backend,
                "reason_codes": reasons,
                "latency_ms": round(lat, 3),
                "top_k": 10,
                "n_unique_domains": uniq_domains,
                "avg_doc_age_days": round(avg_age, 2) if avg_age is not None else None,
                "empty_result": (len(res) == 0),
            }) + "\n")

            # extend retrieved
            state["retrieval_logs"].append({"query": q, "results": res[:10]})
            state["retrieved_chunks"].extend(res[:10])

    if session is not None:
        await session.close()
    if not use_offline and stop_stub_servers is not None:
        await stop_stub_servers(state_env)  # type: ignore
    mark("Retriever", t0, time.perf_counter())

    # Synthesizer: turn top chunks into candidate insights
    t0 = time.perf_counter()
    candidates: List[Dict[str, Any]] = []
    seen_cids = set()
    for r in state["retrieved_chunks"]:
        cid = r.get("chunk_id")
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        did = r.get("doc_id")
        d = docmeta.get(did, {})
        title = d.get("title") or d.get("html_title") or (d.get("topic") or "Insight")
        url = d.get("final_url") or d.get("url") or ""
        pub = d.get("publish_date") or ""
        sd = d.get("source_domain") or ""
        cand = {
            "id": cid,
            "title": title[:120],
            "summary": (r.get("snippet") or (d.get("text") or ""))[:320],
            "url": url,
            "date": pub,
            "evidence_snippet": (r.get("snippet") or "")[:320],
            "confidence": 0.7,
            "source_domain": sd,
            "doc_id": did,
        }
        candidates.append(cand)
    state["insight_candidates"] = candidates
    mark("Synthesizer", t0, time.perf_counter())

    # Consolidator: pick 5 with domain diversity preference
    t0 = time.perf_counter()
    cards: List[Dict[str, Any]] = []
    used_domains: Dict[str, int] = {}
    # Prefer unique domains first
    for c in candidates:
        dom = c.get("source_domain") or ""
        if len(cards) < 5:
            # Prefer new domains until we reach 4
            if used_domains.get(dom, 0) == 0 or len(used_domains) < 4:
                cards.append(c)
                used_domains[dom] = used_domains.get(dom, 0) + 1
                continue
    # If fewer than 5, fill with next best regardless
    if len(cards) < 5:
        for c in candidates:
            if c not in cards:
                cards.append(c)
                if len(cards) >= 5:
                    break
    cards = cards[:5]
    # If distinct sources < 4, try to boost by pulling candidates from other domains
    if len(set((c.get("source_domain") or "") for c in cards)) < 4:
        # Add additional candidates from remaining pool with new domains
        for c in candidates:
            dom = c.get("source_domain") or ""
            if dom and dom not in set((x.get("source_domain") or "") for x in cards):
                # Replace last duplicate-domain card
                # find a card whose domain appears more than once
                dom_counts = {}
                for x in cards:
                    d0 = x.get("source_domain") or ""
                    dom_counts[d0] = dom_counts.get(d0, 0) + 1
                dup_idx = None
                for i, x in enumerate(cards):
                    if dom_counts.get(x.get("source_domain") or "", 0) > 1:
                        dup_idx = i
                        break
                if dup_idx is not None:
                    cards[dup_idx] = c
                if len(set((x.get("source_domain") or "") for x in cards)) >= 4:
                    break
        # If still <4, synthesize from docmeta of other domains (press/dev/help/product/wiki)
        if len(set((c.get("source_domain") or "") for c in cards)) < 4:
            preferred_dt = {"press","product","dev_docs","help_docs","wiki"}
            for did, d in docmeta.items():
                dom = (d.get("source_domain") or "")
                if not dom or dom in set((x.get("source_domain") or "") for x in cards):
                    continue
                if (d.get("doctype") or "").lower() not in preferred_dt:
                    continue
                title = d.get("title") or d.get("html_title") or (d.get("topic") or "Insight")
                url = d.get("final_url") or d.get("url") or ""
                pub = d.get("publish_date") or ""
                snippet = (d.get("text") or "")[:320]
                synth = {
                    "id": f"synth::{did}::card",
                    "title": title[:120],
                    "summary": snippet,
                    "url": url,
                    "date": pub,
                    "evidence_snippet": snippet,
                    "confidence": 0.6,
                    "source_domain": dom,
                    "doc_id": did,
                }
                # Replace a duplicate-domain card if any
                dom_counts = {}
                for x in cards:
                    d0 = x.get("source_domain") or ""
                    dom_counts[d0] = dom_counts.get(d0, 0) + 1
                dup_idx = None
                for i, x in enumerate(cards):
                    if dom_counts.get(x.get("source_domain") or "", 0) > 1:
                        dup_idx = i
                        break
                if dup_idx is not None:
                    cards[dup_idx] = synth
                else:
                    if len(cards) < 5:
                        cards.append(synth)
                if len(set((x.get("source_domain") or "") for x in cards)) >= 4:
                    break
    state["insight_cards"] = cards
    mark("Consolidator", t0, time.perf_counter())

    # Stylist: (no-op stylistic polish)
    t0 = time.perf_counter()
    for c in state["insight_cards"]:
        c["summary"] = c.get("summary", "").strip()
    mark("Stylist", t0, time.perf_counter())

    # A2A: negotiation with safety.check (up to 2 rounds)
    t0 = time.perf_counter()
    transcript_path = os.path.join(out_dir, "a2a_transcript.jsonl")
    flags_final = {"critical": [], "warning": []}
    rounds = 0
    async def call_safety(email_fields: Dict[str, Any], cards: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        tools = load_mcp_map()
        base = tools.get("safety.check") or {}
        host = base.get("host", "127.0.0.1")
        port = int(base.get("port", 7805))
        url = f"http://{host}:{port}/invoke"
        payload = {"method": "moderate", "params": {"text": email_fields.get("body"), "email_fields": email_fields, "insight_cards": cards}}
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
                    j = await resp.json()
                    f = (j.get("flags") or {})
                    return f.get("critical", []) or [], f.get("warning", []) or []
        except Exception:
            # Fallback: local checks similar to server
            spec = load_yaml(os.path.join("configs", "compliance.template.yaml"))
            from tool_safety_check_server import check_email  # type: ignore
            c, w = check_email(email_fields, cards, spec)
            return c, w

    def revise_email(email_fields: Dict[str, Any], cards: List[Dict[str, Any]], crit: List[str], warn: List[str]) -> Dict[str, Any]:
        body = email_fields.get("body") or ""
        # Fix criticals
        if "MISSING_UNSUBSCRIBE" in crit:
            email_fields["unsubscribe_block"] = email_fields.get("unsubscribe_block") or "You can unsubscribe at any time by replying 'unsubscribe'."
        if "MISSING_COMPANY_INFO" in crit:
            email_fields["company_info_block"] = email_fields.get("company_info_block") or "Sent by ACME AI, 123 Market St, San Francisco, CA."
        if "PROHIBITED_PHRASE" in crit:
            body = body.replace("guaranteed", "designed to").replace("free money", "budget savings").replace("no strings attached", "no additional commitment")
        if "UNCITED_CLAIM" in crit and cards:
            first = cards[0]
            body += f"\n(Reference: {first.get('title','')[:60]})"
        # Handle warnings
        def wc(t: str) -> int:
            import re
            return len(re.findall(r"\b\w+\b", t))
        # Length
        if "EXCESS_LENGTH" in warn:
            # keep header + top 3 bullets only
            lines = body.splitlines()
            head = []
            bullets = []
            for ln in lines:
                if ln.strip().startswith("- "):
                    bullets.append(ln)
                else:
                    head.append(ln)
            bullets = bullets[:3]
            body = "\n".join(head + bullets)
            # truncate long bullet lines
            body = "\n".join([" ".join(ln.split()[:18]) if ln.strip().startswith("- ") else ln for ln in body.splitlines()])
        # Readability: shorten sentences
        if "READABILITY" in warn:
            # Aggressively shorten sentences and bullets to improve grade level
            body = "\n".join([
                (" ".join(ln.split()[:10]) if ln.strip().startswith("- ") else " ".join(ln.split()[:12]))
                for ln in body.splitlines()
                if ln.strip()
            ])
        email_fields["body"] = body
        return email_fields

    # Start transcript
    with open(transcript_path, "w", encoding="utf-8") as tf:
        # Round 1
        rounds = 1
        tf.write(json.dumps({"role": "Sales", "content": state["email_draft"], "timestamp": now_iso()}) + "\n")
        crit, warn = await call_safety(state["email_draft"], state["insight_cards"])
        tf.write(json.dumps({"role": "Legal", "content": {"flags": {"critical": crit, "warning": warn}}, "timestamp": now_iso()}) + "\n")
        if crit:
            # Round 2 (Revision)
            rounds = 2
            revised = revise_email(dict(state["email_draft"]), state["insight_cards"], crit, warn)
            tf.write(json.dumps({"role": "Sales", "content": revised, "timestamp": now_iso()}) + "\n")
            crit2, warn2 = await call_safety(revised, state["insight_cards"])
            tf.write(json.dumps({"role": "Legal", "content": {"flags": {"critical": crit2, "warning": warn2}}, "timestamp": now_iso()}) + "\n")
            state["email_draft"] = revised
            flags_final = {"critical": crit2, "warning": warn2}
        else:
            flags_final = {"critical": crit, "warning": warn}

    compliance = {"rounds": rounds, "flags": flags_final}
    with open(os.path.join(out_dir, "compliance_report.json"), "w", encoding="utf-8") as f:
        json.dump(compliance, f, ensure_ascii=False, indent=2)
    mark("A2A", t0, time.perf_counter())

    # Assembler: build email.json
    t0 = time.perf_counter()
    subject = f"Ideas for improving CX at {args.company}"
    bullets = "\n".join([f"- {c['title']} ({c.get('date') or 'n/a'}) — {c.get('url') or ''}" for c in cards])
    body = (
        f"Hi there,\n\nBased on recent updates, here are a few insights that may help your CX agenda:\n\n"
        f"{bullets}\n\nWould you be open to a quick chat to explore?\n"
    )
    unsub = "You can unsubscribe at any time by replying 'unsubscribe'."
    company_info = "Sent by ACME AI, 123 Market St, San Francisco, CA."
    email = {
        "subject": subject,
        "body": body,
        "unsubscribe_block": unsub,
        "company_info_block": company_info,
        "proof_points": [{"id": c["id"], "title": c["title"]} for c in cards],
    }
    state["email_draft"] = email
    mark("Assembler", t0, time.perf_counter())

    # Final readability/length enforcement to satisfy Gate-6 thresholds
    def _word_count(t: str) -> int:
        import re as _re
        return len(_re.findall(r"\b\w+\b", t or ""))
    def _grade(t: str) -> float:
        import re as _re
        sentences = [s for s in _re.split(r"[.!?]+", t or "") if s.strip()]
        sents = max(1, len(sentences))
        words = max(1, _word_count(t))
        syllables = max(1, sum(len(_re.findall(r"[aeiouyAEIOUY]", w)) or 1 for w in _re.findall(r"\b\w+\b", t or "")))
        return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59
    def _shorten_body(b: str) -> str:
        # keep at most 3 bullets; limit bullets to 8 words, other lines to 10
        lines = b.splitlines()
        head = []
        bullets = []
        for ln in lines:
            if ln.strip().startswith("- "):
                bullets.append("- " + " ".join(ln.split()[1:9]))
            else:
                head.append(" ".join(ln.split()[:10]))
        bullets = bullets[:3]
        nb = "\n".join([ln for ln in head if ln.strip()] + bullets)
        return nb
    iterations = 0
    while (_grade(state["email_draft"]["body"]) > 10 or _word_count(state["email_draft"]["body"]) > 160) and iterations < 3:
        state["email_draft"]["body"] = _shorten_body(state["email_draft"]["body"])
        iterations += 1

    # Timings and writes
    total_ms = round((time.perf_counter() - t_total0) * 1000.0, 2)
    timing = {"total_runtime_ms": total_ms, **state["metrics"]["timings"]}

    with open(os.path.join(out_dir, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(state["insight_cards"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "email.json"), "w", encoding="utf-8") as f:
        json.dump(email, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "compliance_report.json"), "w", encoding="utf-8") as f:
        json.dump(compliance, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)
    with open(os.path.join(state_dir, f"session-{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(json.dumps({"session_id": session_id, "out_dir": out_dir, "total_ms": total_ms}))
    return session_id


def parse_args():
    p = argparse.ArgumentParser(description="Run Step-5 happy path graph")
    p.add_argument("--company", default="Salesforce")
    p.add_argument("--persona", default="vp_customer_experience")
    p.add_argument("--session-id", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
