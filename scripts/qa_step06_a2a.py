#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime, timezone, date
from typing import Any, Dict, List

from common import ensure_dir, now_iso


def load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def readability_grade(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, word_count(text))
    syllables = max(1, sum(len(re.findall(r"[aeiouyAEIOUY]", w)) or 1 for w in re.findall(r"\b\w+\b", text or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59


def main():
    ap = argparse.ArgumentParser(description="Gate-6 — A2A & Compliance QA")
    ap.add_argument("--session-id", required=True)
    args = ap.parse_args()

    sid = args.session_id
    out_dir = os.path.join("outputs", sid)

    insights = load(os.path.join(out_dir, "insights.json"))
    email = load(os.path.join(out_dir, "email.json"))
    comp = load(os.path.join(out_dir, "compliance_report.json"))

    checks: List[Dict[str, Any]] = []

    # Rounds <= 2
    rounds = int(comp.get("rounds") or 0)
    checks.append({"id": "G6-01", "metric": "negotiation_rounds", "actual": rounds, "threshold": "<=2", "status": "PASS" if rounds <= 2 else "FAIL", "evidence": os.path.join(out_dir, "a2a_transcript.jsonl")})

    # Critical flags == 0
    crit = comp.get("flags", {}).get("critical", []) or []
    checks.append({"id": "G6-02", "metric": "critical_flags_count", "actual": len(crit), "threshold": "==0", "status": "PASS" if len(crit) == 0 else "FAIL", "evidence": os.path.join(out_dir, "compliance_report.json")})

    # Length <= 160 words
    wc = word_count(email.get("body") or "")
    len_status = "PASS" if wc <= 160 else "FAIL"
    checks.append({"id": "G6-03", "metric": "email_body_words", "actual": wc, "threshold": "<=160", "status": len_status, "evidence": os.path.join(out_dir, "email.json")})

    # Readability <= Grade 10
    gr = readability_grade(email.get("body") or "")
    rd_status = "PASS" if gr <= 10 else "FAIL"
    checks.append({"id": "G6-04", "metric": "readability_grade", "actual": round(gr, 2), "threshold": "<=10", "status": rd_status, "evidence": os.path.join(out_dir, "email.json")})

    # Proof point references
    ids = set(c.get("id") for c in insights)
    dangling = []
    for p in email.get("proof_points", []):
        if p.get("id") not in ids:
            dangling.append(p.get("id"))
    ref_ok = (len(dangling) == 0)
    checks.append({"id": "G6-05", "metric": "proof_points_reference_ok", "actual": ref_ok, "threshold": "==true", "status": "PASS" if ref_ok else "FAIL", "evidence": os.path.join(out_dir, "email.json")})

    status = "GREEN" if all(c["status"] == "PASS" for c in checks) else ("AMBER" if any(c["status"] == "WARN" for c in checks) and not any(c["status"] == "FAIL" for c in checks) else "RED")

    machine = {"step": "step06_a2a", "status": status, "checks": checks, "timestamp": now_iso(), "evidence": {"session_id": sid, "out_dir": out_dir}}

    ensure_dir(os.path.join("reports","qa"))
    with open(os.path.join("reports","qa","step06_a2a.json"), "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append(f"# STEP 6 — A2A & Compliance Gate — {status}")
    lines.append("")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}")
    with open(os.path.join("reports","qa","step06_a2a.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": status}, indent=2))


if __name__ == "__main__":
    main()
