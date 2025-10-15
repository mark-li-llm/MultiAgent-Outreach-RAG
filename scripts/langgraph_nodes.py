#!/usr/bin/env python3
"""LangGraph node implementations for multi-agent RAG system."""
import glob
import json
import os
import time
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Tuple

import aiohttp
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from common import now_iso
from langgraph_state import AgentState
from router_core import load_router_config, load_mcp_map, decide_backend, rerank

load_dotenv()

# LLM Prompt Templates (copied from run_graph.py lines 30-90)
CONSOLIDATOR_SYSTEM_PROMPT = """You are a B2B research analyst consolidating RAG chunks into persona-aware insight cards.
- Preserve factual grounding strictly to the provided candidates.
- Do NOT invent IDs or sources.
- Write concise, executive-friendly copy.
- Tailor emphasis to the persona:
  * vp_customer_experience: NPS, CSAT, contact center, omnichannel, agent productivity, self-service, first contact resolution
  * cio: data integration, governance, security, TCO, platform, APIs, real-time
  * vp_sales_ops: pipeline, forecast accuracy, win rate, productivity, automation
"""

CONSOLIDATOR_USER_PROMPT = """Company: {company}
Persona: {persona}
Persona keywords to weave in naturally (only when relevant): {persona_keywords}

From these candidates (JSON), select exactly the same items (DO NOT add/remove), and for each:
- Improve 'title' (≤ 12 words) and 'summary' (1–2 sentences) with persona relevance.
- Keep 'id' exactly as given to preserve traceability.
- You may rephrase 'summary' but stay within the evidence.
- Add fields:
  persona_relevance: {{ "why_it_matters": str, "relevance_score": 1-5, "keywords_hit": [str] }}
  metric_impact: {{ "metric": str, "direction": "increase|decrease", "magnitude": "low|med|high" }}
  action_suggestion: str (1 actionable step for the recipient)

Return ONLY a JSON array of 5 objects with fields:
[id, title, summary, persona_relevance, metric_impact, action_suggestion]
(The original URL/date/doc_id/source_domain/evidence_snippet/confidence are preserved elsewhere via id.)

Candidates JSON:
{candidates_json}
"""

STYLIST_SYSTEM_PROMPT = """You are a senior B2B outbound email copywriter.
Write concise, evidence-based emails grounded ONLY in provided insight cards.
Compliance:
- No guarantees, no unsupported claims, no negative competitor statements.
- Keep an opt-out line and company info block as provided.
Style:
- 100–140 words, respectful, executive tone.
- CRITICAL OPENING REQUIREMENTS:
  * Start with a natural, contextual greeting (10-20 words)
  * Reference recent business developments or timely insights first
  * Implicitly acknowledge their priorities WITHOUT stating their job title
  * Create smooth transition to value proposition
  * Good opening examples:
    - "I noticed Salesforce's latest quarterly results and thought these insights might be relevant for your planning."
    - "Recent developments at Salesforce suggest opportunities that align with operational excellence priorities."
  * AVOID these patterns:
    - "Dear [Job Title]" or "Hi there"
    - "As [Job Title] focused on..."
    - "I recognize your mandate to..."
- 1–3 bullets that paraphrase the insights.
- Subject ≤ 12 words, concrete and benefit-oriented.
- Close by offering one or two optional follow-up paths (e.g., send a deeper summary, schedule time if helpful) without insisting on a meeting.
Persona voice:
- vp_customer_experience: formal yet empathetic; highlight NPS, CSAT, contact center efficiency, omnichannel and self-service gains; naturally reference customer experience priorities without stating title.
- cio: technically authoritative; emphasize integration, governance, security, platform scale, APIs, real-time data, and TCO discipline; keep language precise and risk-aware.
- vp_sales_ops: metrics-driven and operational; stress pipeline health, forecast accuracy, win rate, productivity, and automation; reference sales operations priorities without stating title.
"""

STYLIST_USER_PROMPT = """Company: {company}
Persona: {persona}
Persona keywords to weave in naturally (2–5 total, only if relevant): {persona_keywords}

Requirements:
- Start with a natural, contextual opening (NO "Dear [Title]" or "Hi there") that references recent developments or insights
- Implicitly acknowledge their priorities through context, NEVER explicitly state their job title
- Offer optional next steps (e.g., share a deeper dive, continue via email, or schedule a call) without demanding a meeting.

Use ONLY these insight cards (JSON) as evidence:
{insight_cards}

Write the final email fields as compact JSON with keys:
- subject: str (≤ 12 words)
- body: str (100–140 words, 1–3 bullets summarizing the insights, follow the requirements above, and maintain a professional tone)
- unsubscribe_block: str (use exactly: "You can unsubscribe at any time by replying 'unsubscribe'.")
- company_info_block: str (use exactly: "Sent by ACME AI, 123 Market St, San Francisco, CA.")

Return ONLY the JSON object.
"""

# Helper functions
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def log_llm_retry_event(session_id: str, attempt: int, error_id: str, input_ids: List[str], synth_count: int):
    """Log LLM retry event to JSONL for debugging (see docs/langgraph/001-llm-id-hallucination.md)."""
    from common import ensure_dir
    event = {
        "timestamp": now_iso(),
        "session_id": session_id,
        "node": "consolidator",
        "attempt": attempt,
        "max_attempts": 3,
        "error_type": "KeyError",
        "hallucinated_ids": [error_id],
        "expected_ids": input_ids,
        "synth_card_count": synth_count,
        "retry_reason": "LLM_ID_MISMATCH"
    }
    log_path = os.path.join("logs", "langgraph", "llm_retry_events.jsonl")
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_doc_meta() -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for p in glob.glob(os.path.join("data", "interim", "normalized", "*.json")):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        m[d.get("doc_id")] = d
    return m


async def kb_search(session: aiohttp.ClientSession, backend: str, query: str, top_k: int, tools_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, str]:
    """MCP kb.search client (copied from run_graph.py lines 114-131)."""
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


# ===== NODE IMPLEMENTATIONS =====

async def intake_node(state: AgentState) -> dict:
    """Validate company and persona inputs (run_graph.py lines 186-190)."""
    errors = []
    if not state.get("company") or not state.get("persona"):
        errors.append("missing company/persona")
    return {"errors": errors}


async def planner_node(state: AgentState) -> dict:
    """Generate 5 persona-specific queries from eval seed (run_graph.py lines 192-222)."""
    SEED_PATH = os.path.join("data", "interim", "eval", "salesforce_eval_seed.jsonl")
    seed_items: List[Dict[str, Any]] = []
    if os.path.exists(SEED_PATH):
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                if (j.get("persona") or "") == state["persona"]:
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

    # Load persona keywords
    eval_cfg = load_yaml(os.path.join("configs", "eval.prompts.yaml"))
    persona_keywords = (eval_cfg.get("personas", {}) or {}).get(state["persona"], [])

    return {"queries": queries, "persona_keywords": persona_keywords}


async def retriever_node(state: AgentState) -> dict:
    """Execute vector search via MCP kb.search (run_graph.py lines 224-441)."""
    tools_cfg = load_mcp_map()
    router_cfg = load_router_config()
    docmeta = load_doc_meta()

    retrieved_chunks = []
    retrieval_logs = []
    route_decisions = []

    connector = aiohttp.TCPConnector(limit_per_host=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        for q in state["queries"]:
            backend, reasons = decide_backend(q, state["persona"], None)
            route_decisions.append({"query": q, "backend": backend, "reasons": reasons})

            # Retrieve
            res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)

            # Re-rank + attach meta
            res = rerank(res, {k: type("DM", (), v) for k, v in docmeta.items()}, top_k=12, domain_cap=2)

            # Log and extend
            retrieval_logs.append({"query": q, "results": res[:10]})
            retrieved_chunks.extend(res[:10])

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_logs": retrieval_logs,
        "route_decisions": route_decisions,
    }


async def synthesizer_node(state: AgentState) -> dict:
    """Convert chunks to candidate insight objects (run_graph.py lines 443-471)."""
    docmeta = load_doc_meta()
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

    return {"insight_candidates": candidates}


async def consolidator_node(state: AgentState) -> dict:
    """LLM-enhance 5 selected insights with persona relevance (run_graph.py lines 473-587)."""
    candidates = state["insight_candidates"]

    # Select 5 with domain diversity (logic from lines 476-556)
    cards: List[Dict[str, Any]] = []
    used_domains: Dict[str, int] = {}
    for c in candidates:
        dom = c.get("source_domain") or ""
        if len(cards) < 5:
            if used_domains.get(dom, 0) == 0 or len(used_domains) < 4:
                cards.append(c)
                used_domains[dom] = used_domains.get(dom, 0) + 1
                continue

    # Fill to 5 if needed
    if len(cards) < 5:
        for c in candidates:
            if c not in cards:
                cards.append(c)
                if len(cards) >= 5:
                    break
    cards = cards[:5]

    # Domain diversity enforcement (run_graph.py lines 494-556)
    docmeta = load_doc_meta()
    if len(set((c.get("source_domain") or "") for c in cards)) < 4:
        # Add additional candidates from remaining pool with new domains
        for c in candidates:
            dom = c.get("source_domain") or ""
            if dom and dom not in set((x.get("source_domain") or "") for x in cards):
                # Replace last duplicate-domain card
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
                # Break only when we have BOTH ≥5 cards AND ≥4 domains
                if len(cards) >= 5 and len(set((x.get("source_domain") or "") for x in cards)) >= 4:
                    break

        # If still <4, synthesize from docmeta of other domains
        if len(set((c.get("source_domain") or "") for c in cards)) < 4:
            preferred_dt = {"press", "product", "dev_docs", "help_docs", "wiki"}
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
                # Break only when we have BOTH ≥5 cards AND ≥4 domains
                if len(cards) >= 5 and len(set((x.get("source_domain") or "") for x in cards)) >= 4:
                    break

    # Validation: Must have exactly 5 cards before LLM
    if len(cards) != 5:
        raise AssertionError(f"consolidator_node: Expected exactly 5 cards before LLM, got {len(cards)}")

    # Defensive retry mechanism for LLM ID hallucination (see docs/langgraph/001-llm-id-hallucination.md)
    import sys
    input_ids = [c["id"] for c in cards]
    synth_count = sum(1 for cid in input_ids if cid.startswith("synth::"))
    MAX_ATTEMPTS = 3
    cards_final = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # LLM enhancement
            llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
            consolidator_tmpl = ChatPromptTemplate.from_messages([
                ("system", CONSOLIDATOR_SYSTEM_PROMPT),
                ("user", CONSOLIDATOR_USER_PROMPT),
            ])

            consolidator_vars = {
                "company": state["company"],
                "persona": state["persona"],
                "persona_keywords": ", ".join(state.get("persona_keywords") or []),
                "candidates_json": json.dumps(cards, ensure_ascii=False),
            }

            resp = await llm.ainvoke(consolidator_tmpl.format_messages(**consolidator_vars))
            cards_llm = json.loads(resp.content)

            # Merge LLM fields back (KeyError occurs here if ID mismatch)
            by_id = {c["id"]: c for c in cards}
            cards_final = []
            for item in cards_llm:
                base = by_id[item["id"]]  # KeyError if LLM hallucinated ID
                base["title"] = item.get("title") or base["title"]
                base["summary"] = item.get("summary") or base["summary"]
                base["persona_relevance"] = item.get("persona_relevance")
                base["metric_impact"] = item.get("metric_impact")
                base["action_suggestion"] = item.get("action_suggestion")
                cards_final.append(base)

            # Success - break out of retry loop
            break

        except KeyError as e:
            # LLM hallucinated an ID not in input
            hallucinated_id = str(e).strip("'")

            if attempt < MAX_ATTEMPTS:
                # Log retry event
                log_llm_retry_event(
                    session_id=state.get("session_id", "unknown"),
                    attempt=attempt,
                    error_id=hallucinated_id,
                    input_ids=input_ids,
                    synth_count=synth_count
                )

                # Warn user
                print(f"⚠️  LLM retry {attempt}/{MAX_ATTEMPTS}: ID mismatch detected (hallucinated: {hallucinated_id[:80]}...), retrying...", file=sys.stderr)

                continue  # Retry
            else:
                # All attempts exhausted
                raise AssertionError(
                    f"consolidator_node: LLM ID hallucination after {MAX_ATTEMPTS} retries. "
                    f"Hallucinated ID: {hallucinated_id}. Expected one of: {input_ids[:3]}..."
                ) from e

    # Validation: Must have exactly 5 cards after LLM merge
    if len(cards_final) != 5:
        raise AssertionError(
            f"consolidator_node: Expected exactly 5 cards after LLM, got {len(cards_final)}. "
            f"LLM returned {len(cards_llm)} items from {len(cards)} input cards."
        )

    return {"insight_cards": cards_final}


async def stylist_node(state: AgentState) -> dict:
    """Generate email copy via LLM (run_graph.py lines 589-607)."""
    llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
    stylist_tmpl = ChatPromptTemplate.from_messages([
        ("system", STYLIST_SYSTEM_PROMPT),
        ("user", STYLIST_USER_PROMPT),
    ])

    stylist_vars = {
        "company": state["company"],
        "persona": state["persona"],
        "persona_keywords": ", ".join(state.get("persona_keywords") or []),
        "insight_cards": json.dumps(state["insight_cards"], ensure_ascii=False),
    }

    resp = await llm.ainvoke(stylist_tmpl.format_messages(**stylist_vars))
    email_fields = json.loads(resp.content)

    return {"email_draft": email_fields}


async def a2a_node(state: AgentState) -> dict:
    """Compliance negotiation with safety.check (run_graph.py lines 609-698).

    Phase 3: Full revision logic with conditional routing.
    """
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
            # Fallback: local checks
            spec = load_yaml(os.path.join("configs", "compliance.template.yaml"))
            from tool_safety_check_server import check_email
            c, w = check_email(email_fields, state["insight_cards"], spec)
            return c, w

    def revise_email(email_fields: Dict[str, Any], cards: List[Dict[str, Any]], crit: List[str], warn: List[str]) -> Dict[str, Any]:
        """Email revision logic (from run_graph.py lines 634-674)."""
        import re
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

    # Check current round state
    current_round = state.get("a2a_rounds", 0)

    # Call safety check on current email draft
    crit, warn = await call_safety(state["email_draft"], state["insight_cards"])

    # Record flags
    compliance_flags = [f"CRITICAL:{f}" for f in crit] + [f"WARN:{f}" for f in warn]

    # Increment round counter
    new_round = current_round + 1

    # If critical flags present and this is round 1, prepare for revision
    # The conditional edge will route back to Stylist
    email_draft = state["email_draft"]
    if crit and current_round < 1:
        # Apply revision logic - will be used when Stylist regenerates
        email_draft = revise_email(dict(state["email_draft"]), state["insight_cards"], crit, warn)

    return {
        "compliance_flags": compliance_flags,
        "a2a_rounds": new_round,
        "email_draft": email_draft,
    }


async def assembler_node(state: AgentState) -> dict:
    """Attach proof points and finalize output (run_graph.py lines 700-713)."""
    email = dict(state.get("email_draft") or {})

    # Safety defaults
    email.setdefault("unsubscribe_block", "You can unsubscribe at any time by replying 'unsubscribe'.")
    email.setdefault("company_info_block", "Sent by ACME AI, 123 Market St, San Francisco, CA.")

    # Proof points
    cards = state.get("insight_cards") or []
    email["proof_points"] = [{"id": c["id"], "title": c["title"]} for c in cards[:5]]

    return {"email_draft": email}
