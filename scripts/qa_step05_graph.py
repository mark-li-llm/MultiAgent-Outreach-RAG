#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, date
from typing import Any, Dict, List

from common import ensure_dir, now_iso


def load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def distinct_sources(cards: List[Dict[str, Any]]) -> int:
    return len(set((c.get("source_domain") or "") for c in cards))


def count_recent(cards: List[Dict[str, Any]]) -> int:
    nowd = datetime.now(timezone.utc).date()
    c = 0
    for d in cards:
        iso = (d.get("date") or "").strip()
        if not iso:
            continue
        try:
            dd = date.fromisoformat(iso)
        except Exception:
            continue
        if (nowd - dd).days <= 365:
            c += 1
    return c


def main():
    ap = argparse.ArgumentParser(description="Gate-5 — Graph Happy Path QA")
    args = ap.parse_args()

    # Run the graph once
    proc = subprocess.run([sys.executable, os.path.join("scripts", "run_graph.py"), "--company", "Salesforce", "--persona", "vp_customer_experience"], capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr)
        raise SystemExit(1)
    try:
        info = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        info = {}
    sid = info.get("session_id")
    if not sid:
        raise SystemExit("Graph did not return session_id")
    out_dir = os.path.join("outputs", sid)
    state_path = os.path.join("state", f"session-{sid}.json")

    # Load artifacts
    state = load(state_path)
    insights = load(os.path.join(out_dir, "insights.json"))
    email = load(os.path.join(out_dir, "email.json"))
    timing = load(os.path.join(out_dir, "timing.json"))

    checks: List[Dict[str, Any]] = []

    # Node coverage
    expected_nodes = ["Intake","Planner","Retriever","Synthesizer","Consolidator","Stylist","A2A","Assembler"]
    nodes_executed = state.get("metrics", {}).get("nodes_executed", [])
    cov_ok = (nodes_executed == expected_nodes)
    checks.append({"id": "G5-01", "metric": "nodes_executed", "actual": nodes_executed, "threshold": "== all 8 in order", "status": "PASS" if cov_ok else "FAIL", "evidence": state_path})

    # Latency budget
    total_ms = float(timing.get("total_runtime_ms") or 0.0)
    thr = 30000.0
    lat_status = "PASS" if total_ms <= thr else ("WARN" if total_ms <= thr * 1.2 else "FAIL")
    checks.append({"id": "G5-02", "metric": "total_runtime_ms", "actual": int(total_ms), "threshold": f"<={int(thr)}", "status": lat_status, "evidence": os.path.join(out_dir, "timing.json")})

    # Insight count
    cnt = len(insights)
    checks.append({"id": "G5-03", "metric": "insight_cards", "actual": cnt, "threshold": "==5", "status": "PASS" if cnt == 5 else "FAIL", "evidence": os.path.join(out_dir, "insights.json")})

    # Distinct sources
    ds = distinct_sources(insights)
    checks.append({"id": "G5-04", "metric": "distinct_sources", "actual": ds, "threshold": ">=4", "status": "PASS" if ds >= 4 else "FAIL", "evidence": os.path.join(out_dir, "insights.json")})

    # Recency
    recent = count_recent(insights)
    checks.append({"id": "G5-05", "metric": "insights_within_12mo", "actual": recent, "threshold": ">=2", "status": "PASS" if recent >= 2 else "FAIL", "evidence": os.path.join(out_dir, "insights.json")})

    # Email schema
    required_fields = ["subject","body","unsubscribe_block","company_info_block","proof_points"]
    missing = [k for k in required_fields if not (email.get(k) or "")] 
    schema_ok = (len(missing) == 0 and isinstance(email.get("proof_points"), list))
    checks.append({"id": "G5-06", "metric": "email_schema_ok", "actual": schema_ok, "threshold": "==true", "status": "PASS" if schema_ok else "FAIL", "evidence": os.path.join(out_dir, "email.json")})

    # Proof points reference existing insights
    insight_ids = set(c.get("id") for c in insights)
    dangling = []
    for p in email.get("proof_points", []):
        if p.get("id") not in insight_ids:
            dangling.append(p.get("id"))
    pp_ok = (len(dangling) == 0)
    checks.append({"id": "G5-07", "metric": "proof_points_resolve", "actual": pp_ok, "threshold": "==true", "status": "PASS" if pp_ok else "FAIL", "evidence": os.path.join(out_dir, "email.json")})

    # Status rollup
    status = "GREEN" if all(c["status"] == "PASS" for c in checks) else ("AMBER" if any(c["status"] == "WARN" for c in checks) and not any(c["status"] == "FAIL" for c in checks) else "RED")

    machine = {
        "step": "step05_graph",
        "status": status,
        "checks": checks,
        "timestamp": now_iso(),
        "evidence": {"session_id": sid, "out_dir": out_dir, "state": state_path},
    }

    ensure_dir(os.path.join("reports","qa"))
    with open(os.path.join("reports","qa","step05_graph.json"), "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable
    lines = []
    lines.append(f"# STEP 5 — Graph Happy Path (Gate‑5) — {status}")
    lines.append("")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}")
    with open(os.path.join("reports","qa","step05_graph.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": status, "session_id": sid}, indent=2))


if __name__ == "__main__":
    main()

