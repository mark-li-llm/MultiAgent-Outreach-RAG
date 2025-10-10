#!/usr/bin/env python3
"""LangGraph-based agent orchestration (Phase 2 - full node logic)."""
import argparse
import asyncio
import json
import os
import time
import uuid
import re
from langgraph.graph import StateGraph, END
from langgraph_state import AgentState
from langgraph_nodes import (
    intake_node,
    planner_node,
    retriever_node,
    synthesizer_node,
    consolidator_node,
    stylist_node,
    a2a_node,
    assembler_node,
)
from common import ensure_dir, now_iso


def should_revise_email(state: AgentState) -> str:
    """Route to Stylist for revision or Assembler for final assembly."""
    critical_flags = [f for f in state.get("compliance_flags", []) if f.startswith("CRITICAL:")]
    rounds = state.get("a2a_rounds", 0)

    # If critical flags exist and we haven't exceeded 2 rounds, revise
    if critical_flags and rounds < 2:
        return "revise"
    return "assemble"


def build_graph() -> StateGraph:
    """Construct LangGraph StateGraph with 8 nodes."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("Intake", intake_node)
    workflow.add_node("Planner", planner_node)
    workflow.add_node("Retriever", retriever_node)
    workflow.add_node("Synthesizer", synthesizer_node)
    workflow.add_node("Consolidator", consolidator_node)
    workflow.add_node("Stylist", stylist_node)
    workflow.add_node("A2A", a2a_node)
    workflow.add_node("Assembler", assembler_node)

    # Add sequential edges
    workflow.set_entry_point("Intake")
    workflow.add_edge("Intake", "Planner")
    workflow.add_edge("Planner", "Retriever")
    workflow.add_edge("Retriever", "Synthesizer")
    workflow.add_edge("Synthesizer", "Consolidator")
    workflow.add_edge("Consolidator", "Stylist")
    workflow.add_edge("Stylist", "A2A")

    # Phase 3: Conditional edge for A2A revision loop
    workflow.add_conditional_edges(
        "A2A",
        should_revise_email,
        {
            "revise": "Stylist",  # Re-generate email (Round 2)
            "assemble": "Assembler",  # Proceed to final assembly
        }
    )
    workflow.add_edge("Assembler", END)

    return workflow


async def main_async(args):
    """Main entry point for LangGraph execution."""
    workflow = build_graph()
    app = workflow.compile()

    session_id = args.session_id or uuid.uuid4().hex[:12]
    out_dir = os.path.join("outputs", session_id)
    state_dir = "state"
    ensure_dir(out_dir)
    ensure_dir(state_dir)

    initial_state: AgentState = {
        "company": args.company,
        "persona": args.persona,
        "session_id": session_id,
        "timestamp": now_iso(),
        "queries": [],
        "persona_keywords": [],
        "retrieved_chunks": [],
        "retrieval_logs": [],
        "route_decisions": [],
        "insight_candidates": [],
        "insight_cards": [],
        "email_draft": {},
        "compliance_flags": [],
        "a2a_rounds": 0,
        "metrics": {"nodes_executed": [], "timings": {}},
        "errors": [],
    }

    t0 = time.perf_counter()

    # Invoke graph
    result = await app.ainvoke(initial_state)

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Final readability/length enforcement to satisfy Gate-6 thresholds (from run_graph.py lines 716-783)
    def _word_count(t: str) -> int:
        return len(re.findall(r"\b\w+\b", t or ""))
    def _grade(t: str) -> float:
        sentences = [s for s in re.split(r"[.!?]+", t or "") if s.strip()]
        sents = max(1, len(sentences))
        words = max(1, _word_count(t))
        syllables = max(1, sum(len(re.findall(r"[aeiouyAEIOUY]", w)) or 1 for w in re.findall(r"\b\w+\b", t or "")))
        return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59
    def _shorten_body(b: str) -> str:
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

    current_wc = _word_count(result["email_draft"]["body"])
    current_grade = _grade(result["email_draft"]["body"])

    # Priority 1: Word count hard limit
    if current_wc > 160:
        iterations = 0
        while current_wc > 160 and iterations < 3:
            result["email_draft"]["body"] = _shorten_body(result["email_draft"]["body"])
            current_wc = _word_count(result["email_draft"]["body"])
            iterations += 1

    # Priority 2: Readability grade with A2A trust
    elif current_grade > 15:
        # Check A2A result by extracting from compliance_flags
        critical_flags = [f for f in result.get("compliance_flags", []) if f.startswith("CRITICAL:")]
        if not critical_flags:
            # A2A passed - trust it, no truncation
            pass
        else:
            # A2A also flagged issues - try truncation with safeguard
            iterations = 0
            prev_grade = current_grade

            while current_grade > 10 and iterations < 3:
                new_body = _shorten_body(result["email_draft"]["body"])
                new_grade = _grade(new_body)

                # Safeguard: stop if grade gets worse
                if new_grade >= prev_grade:
                    break

                # Apply effective truncation
                result["email_draft"]["body"] = new_body
                prev_grade = new_grade
                current_grade = new_grade
                iterations += 1

    # Write outputs (match original format)
    with open(os.path.join(out_dir, "insights.json"), "w", encoding="utf-8") as f:
        json.dump(result["insight_cards"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "email.json"), "w", encoding="utf-8") as f:
        json.dump(result["email_draft"], f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
        json.dump({"total_runtime_ms": total_ms}, f, ensure_ascii=False, indent=2)

    # Write compliance report
    critical_flags = [f.replace("CRITICAL:", "") for f in result.get("compliance_flags", []) if f.startswith("CRITICAL:")]
    warning_flags = [f.replace("WARN:", "") for f in result.get("compliance_flags", []) if f.startswith("WARN:")]
    compliance = {
        "rounds": result.get("a2a_rounds", 1),
        "flags": {
            "critical": critical_flags,
            "warning": warning_flags,
        }
    }
    with open(os.path.join(out_dir, "compliance_report.json"), "w", encoding="utf-8") as f:
        json.dump(compliance, f, ensure_ascii=False, indent=2)

    # Write router trace
    with open(os.path.join(out_dir, "router_trace.jsonl"), "w", encoding="utf-8") as f:
        for rd in result.get("route_decisions", []):
            f.write(json.dumps({
                "timestamp": result["timestamp"],
                "query_text": rd.get("query"),
                "decision_backend": rd.get("backend"),
                "reason_codes": rd.get("reasons"),
            }) + "\n")

    # Write state
    with open(os.path.join(state_dir, f"session-{session_id}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(result), f, ensure_ascii=False, indent=2)

    print(json.dumps({"session_id": session_id, "out_dir": out_dir, "total_ms": total_ms}))
    return session_id


def parse_args():
    p = argparse.ArgumentParser(description="Run LangGraph-based agent workflow")
    p.add_argument("--company", default="Salesforce")
    p.add_argument("--persona", default="vp_customer_experience")
    p.add_argument("--session-id", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
