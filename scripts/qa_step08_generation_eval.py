#!/usr/bin/env python3
"""
Gate-8: Generation & Compliance Evaluation
Runs 10 end-to-end graph sessions across ≥3 personas.
Validates structural, compliance, and persona-alignment criteria.
"""
import argparse
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, now_iso


# Paths
PROMPTS_PATH = os.path.join("data", "interim", "eval", "generation_prompts.jsonl")
EVAL_CONFIG = os.path.join("configs", "eval.prompts.yaml")
MCP_CONFIG = os.path.join("configs", "mcp.tools.yaml")
COMPLIANCE_CONFIG = os.path.join("configs", "compliance.template.yaml")

# Output paths
GEN_METRICS_PATH = os.path.join("reports", "eval", "generation_metrics.json")
COMP_METRICS_PATH = os.path.join("reports", "eval", "compliance_metrics.json")
OUT_JSON = os.path.join("reports", "qa", "step08_generation_eval.json")
OUT_MD = os.path.join("reports", "qa", "step08_generation_eval.md")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config with fallback."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file (one JSON per line)."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def within_12mo(iso: Optional[str]) -> bool:
    """Check if date is within last 12 months (matching run_graph.py logic)."""
    if not iso:
        return False
    try:
        d = date.fromisoformat(iso)
        return (datetime.now(timezone.utc).date() - d).days <= 365
    except Exception:
        return False


def parse_run_graph_last_line(stdout: str) -> Dict[str, Any]:
    """
    Extract session info from run_graph.py stdout.
    Expects last line to be JSON with session_id, out_dir, total_ms.
    """
    lines = stdout.strip().split('\n')
    if not lines:
        raise ValueError("No output from run_graph.py")

    # Find last non-empty line
    for line in reversed(lines):
        line = line.strip()
        if line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    raise ValueError("No valid JSON found in run_graph.py output")


def validate_structure(insights: List[Dict[str, Any]], email: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate structural requirements:
    - Exactly 5 insights
    - ≥4 distinct source_domains
    - ≥2 insights with date within 12 months
    - Email has all required fields
    - All proof_points reference valid insight IDs
    """
    # Extract insight IDs
    ids = [c.get("id") for c in insights if c.get("id")]

    # Count distinct domains (case-insensitive)
    domains = set()
    for c in insights:
        domain = (c.get("source_domain") or "").strip().lower()
        if domain:
            domains.add(domain)

    # Count recent insights
    recent_count = sum(1 for c in insights if within_12mo(c.get("date")))

    # Check email schema
    required_fields = ["subject", "body", "unsubscribe_block", "company_info_block", "proof_points"]
    schema_ok = True
    for field in required_fields:
        if field == "proof_points":
            # Must be a list
            if not isinstance(email.get(field), list):
                schema_ok = False
        else:
            # Must be non-empty string
            if not (email.get(field) or "").strip():
                schema_ok = False

    # Check proof points resolution
    proof_points_resolve = True
    proof_points = email.get("proof_points", [])
    for pp in proof_points:
        if pp.get("id") not in ids:
            proof_points_resolve = False
            break

    return {
        "insights_count": len(insights),
        "distinct_sources": len(domains),
        "recent_count": recent_count,
        "email_schema_ok": schema_ok,
        "proof_points_resolve": proof_points_resolve
    }


def persona_keyword_hits(persona: str, email_body: str, eval_config_path: str = EVAL_CONFIG) -> int:
    """
    Count unique persona-specific keywords present in email body.
    Case-insensitive matching.
    """
    config = load_yaml(eval_config_path)
    personas = config.get("personas", {})

    # Get keywords for this persona
    keywords = personas.get(persona, [])
    if not keywords:
        return 0

    # Normalize body for matching
    body_lower = email_body.lower()

    # Count unique keyword hits
    hits = 0
    for keyword in keywords:
        if keyword.lower() in body_lower:
            hits += 1

    return hits


def word_count(text: str) -> int:
    """Count words in text (matching tool_safety_check_server.py)."""
    return len(re.findall(r"\b\w+\b", text or ""))


def readability_grade(text: str) -> float:
    """
    Flesch-Kincaid Grade approximation (matching tool_safety_check_server.py).
    """
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, word_count(text))
    syllables = max(1, sum(len(re.findall(r"[aeiouyAEIOUY]", w)) or 1
                           for w in re.findall(r"\b\w+\b", text or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59


async def maybe_start_stubs() -> Tuple[bool, str]:
    """
    Try to start MCP stub servers using qa_step03_mcp.
    Returns (started_successfully, service_mode).
    """
    try:
        # Try importing the MCP module
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qa_step03_mcp",
            os.path.join(os.path.dirname(__file__), "qa_step03_mcp.py")
        )
        if spec and spec.loader:
            qa_step03_mcp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(qa_step03_mcp)

            # Try to start servers
            cfg = load_yaml(MCP_CONFIG)
            state = {}
            servers = await qa_step03_mcp.start_stub_servers(state, cfg)
            if servers:
                return True, "mcp_stubs"
            else:
                return False, "offline"
    except Exception as e:
        # Failed to start stubs, will use offline mode
        print(f"Note: MCP stubs unavailable ({e}), using offline mode")
        return False, "offline"

    return False, "offline"


async def run_one_prompt(eval_id: str, company: str, persona: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Run a single generation via run_graph.py.
    Returns metrics dict with structural, compliance, performance data.
    """
    result = {
        "eval_id": eval_id,
        "persona": persona,
        "session_id": None,
        "out_dir": None,
        "structural": {},
        "compliance": {},
        "perf_ms": None,
        "error": None
    }

    # Run the graph
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            ["python3", "scripts/run_graph.py", "--company", company, "--persona", persona],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        perf_ms = int((time.perf_counter() - t0) * 1000)
        result["perf_ms"] = perf_ms

        if proc.returncode != 0:
            result["error"] = f"run_graph.py failed with code {proc.returncode}: {proc.stderr[:500]}"
            return result

        # Parse output
        try:
            last_line = parse_run_graph_last_line(proc.stdout)
            result["session_id"] = last_line.get("session_id")
            result["out_dir"] = last_line.get("out_dir")
        except Exception as e:
            result["error"] = f"Failed to parse run_graph.py output: {e}"
            return result

    except subprocess.TimeoutExpired:
        result["error"] = f"run_graph.py timed out after {timeout}s"
        return result
    except Exception as e:
        result["error"] = f"Failed to run run_graph.py: {e}"
        return result

    # Load outputs
    out_dir = result["out_dir"]
    if not out_dir or not os.path.exists(out_dir):
        result["error"] = f"Output directory not found: {out_dir}"
        return result

    try:
        # Load insights
        insights_path = os.path.join(out_dir, "insights.json")
        insights = load_json(insights_path) if os.path.exists(insights_path) else []

        # Load email
        email_path = os.path.join(out_dir, "email.json")
        email = load_json(email_path) if os.path.exists(email_path) else {}

        # Load compliance report
        compliance_path = os.path.join(out_dir, "compliance_report.json")
        compliance = load_json(compliance_path) if os.path.exists(compliance_path) else {}

        # Validate structure
        structural = validate_structure(insights, email)
        structural["persona_keyword_hits"] = persona_keyword_hits(persona, email.get("body", ""))
        result["structural"] = structural

        # Extract compliance data
        flags = compliance.get("flags", {})
        email_body = email.get("body", "")
        result["compliance"] = {
            "critical_flags": flags.get("critical", []),
            "warning_flags": flags.get("warning", []),
            "word_count": word_count(email_body),
            "readability_grade": readability_grade(email_body)
        }

    except Exception as e:
        result["error"] = f"Failed to process outputs: {e}"

    return result


def aggregate_metrics(runs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Aggregate metrics from all runs.
    Returns (generation_metrics, compliance_metrics).
    """
    # Separate successful runs from errors
    successful = [r for r in runs if not r.get("error")]
    failed = [r for r in runs if r.get("error")]

    # Generation metrics
    gen_runs = []
    for r in successful:
        gen_runs.append({
            "eval_id": r["eval_id"],
            "persona": r["persona"],
            "session_id": r["session_id"],
            "out_dir": r["out_dir"],
            "structural": r["structural"],
            "perf_ms": r["perf_ms"]
        })

    # Add failed runs
    for r in failed:
        gen_runs.append({
            "eval_id": r["eval_id"],
            "persona": r["persona"],
            "session_id": None,
            "out_dir": None,
            "structural": {
                "insights_count": 0,
                "distinct_sources": 0,
                "recent_count": 0,
                "email_schema_ok": False,
                "proof_points_resolve": False,
                "persona_keyword_hits": 0
            },
            "perf_ms": r.get("perf_ms"),
            "error": r.get("error")
        })

    # Calculate structural pass rate
    structural_passes = []
    for r in gen_runs:
        s = r["structural"]
        passes = (
            s["insights_count"] == 5 and
            s["distinct_sources"] >= 4 and
            s["recent_count"] >= 2 and
            s["email_schema_ok"] and
            s["proof_points_resolve"]
        )
        structural_passes.append(passes)

    structural_pass_rate = sum(structural_passes) / max(1, len(gen_runs))

    # Calculate persona keyword average
    keyword_hits = [r["structural"]["persona_keyword_hits"] for r in gen_runs]
    persona_keyword_hits_avg = sum(keyword_hits) / max(1, len(keyword_hits))

    generation_metrics = {
        "runs": gen_runs,
        "aggregates": {
            "structural_pass_rate": structural_pass_rate,
            "persona_keyword_hits_avg": persona_keyword_hits_avg
        },
        "timestamp": now_iso()
    }

    # Compliance metrics
    comp_runs = []
    for r in successful:
        comp_runs.append({
            "eval_id": r["eval_id"],
            "flags": {
                "critical": r["compliance"]["critical_flags"],
                "warning": r["compliance"]["warning_flags"]
            },
            "word_count": r["compliance"]["word_count"],
            "readability_grade": r["compliance"]["readability_grade"]
        })

    # Add failed runs as worst-case compliance
    for r in failed:
        comp_runs.append({
            "eval_id": r["eval_id"],
            "flags": {
                "critical": ["RUN_FAILED"],
                "warning": []
            },
            "word_count": 999,
            "readability_grade": 99.0
        })

    # Calculate compliance aggregates
    critical_flags_total = sum(len(r["flags"]["critical"]) for r in comp_runs)

    length_readability_passes = []
    for r in comp_runs:
        passes = r["word_count"] <= 160 and r["readability_grade"] <= 10.0
        length_readability_passes.append(passes)

    length_readability_pass_runs = sum(length_readability_passes)

    compliance_metrics = {
        "runs": comp_runs,
        "aggregates": {
            "critical_flags_total": critical_flags_total,
            "length_readability_pass_runs": length_readability_pass_runs
        },
        "timestamp": now_iso()
    }

    return generation_metrics, compliance_metrics


def compute_gate_status(gen_metrics: Dict[str, Any], comp_metrics: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Compute Gate-8 status based on thresholds.
    Returns (status, checks).
    """
    agg_gen = gen_metrics["aggregates"]
    agg_comp = comp_metrics["aggregates"]

    checks = [
        {
            "id": "G8-01",
            "metric": "structural_pass_rate",
            "actual": round(agg_gen["structural_pass_rate"], 4),
            "threshold": "==1.0",
            "status": "PASS" if agg_gen["structural_pass_rate"] == 1.0 else "FAIL",
            "evidence": "reports/eval/generation_metrics.json"
        },
        {
            "id": "G8-02",
            "metric": "critical_flags_total",
            "actual": agg_comp["critical_flags_total"],
            "threshold": "==0",
            "status": "PASS" if agg_comp["critical_flags_total"] == 0 else "FAIL",
            "evidence": "reports/eval/compliance_metrics.json"
        },
        {
            "id": "G8-03",
            "metric": "length_readability_pass_runs",
            "actual": agg_comp["length_readability_pass_runs"],
            "threshold": ">=9",
            "status": "PASS" if agg_comp["length_readability_pass_runs"] >= 9 else "FAIL",
            "evidence": "reports/eval/compliance_metrics.json"
        },
        {
            "id": "G8-04",
            "metric": "persona_keyword_hits_avg",
            "actual": round(agg_gen["persona_keyword_hits_avg"], 2),
            "threshold": ">=2.0",
            "status": "PASS" if agg_gen["persona_keyword_hits_avg"] >= 2.0 else "FAIL",
            "evidence": "reports/eval/generation_metrics.json"
        }
    ]

    # Determine overall status
    failures = sum(1 for c in checks if c["status"] == "FAIL")

    if failures == 0:
        status = "GREEN"
    elif failures == 1 and checks[2]["status"] == "FAIL":  # Only G8-03 failed
        status = "AMBER"
    elif failures == 1 and checks[3]["status"] == "FAIL":  # Only G8-04 failed
        status = "AMBER"
    else:
        status = "RED"

    return status, checks


def write_reports(
    gen_metrics: Dict[str, Any],
    comp_metrics: Dict[str, Any],
    status: str,
    checks: List[Dict[str, Any]],
    service_mode: str,
    total_runs: int
) -> None:
    """Write all report files."""
    # Ensure directories
    ensure_dir(os.path.dirname(GEN_METRICS_PATH))
    ensure_dir(os.path.dirname(COMP_METRICS_PATH))
    ensure_dir(os.path.dirname(OUT_JSON))

    # Write metrics files
    with open(GEN_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(gen_metrics, f, ensure_ascii=False, indent=2)

    with open(COMP_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(comp_metrics, f, ensure_ascii=False, indent=2)

    # Build QA envelope (machine)
    machine = {
        "step": "step08_generation_eval",
        "gate": "Gate-8",
        "status": status,
        "service_mode": service_mode,
        "checks": checks,
        "summary": {
            "runs": total_runs,
            "structural_pass_rate": gen_metrics["aggregates"]["structural_pass_rate"],
            "critical_flags_total": comp_metrics["aggregates"]["critical_flags_total"],
            "length_readability_pass_runs": comp_metrics["aggregates"]["length_readability_pass_runs"],
            "persona_keyword_hits_avg": gen_metrics["aggregates"]["persona_keyword_hits_avg"]
        },
        "timestamp": now_iso()
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Build MD report (human)
    lines = []
    lines.append(f"# STEP 8 — Generation & Compliance Evaluation (Gate‑8) — {status}")
    lines.append("")
    lines.append(f"**Service Mode**: {service_mode}")
    lines.append("")
    lines.append("**Checks:**")
    for c in checks:
        lines.append(f"- {c['id']}: {c['metric']} = {c['actual']} (threshold {c['threshold']}) -> {c['status']}")
    lines.append("")
    lines.append("**Summary:**")
    lines.append(f"- Total runs: {total_runs}")

    # Extract unique personas
    personas = list(set(r["persona"] for r in gen_metrics["runs"]))
    lines.append(f"- Personas tested: {', '.join(sorted(personas))}")

    # Add summary stats
    if gen_metrics["aggregates"]["structural_pass_rate"] == 1.0:
        lines.append("- All structural validations passed")
    else:
        fail_count = sum(1 for r in gen_metrics["runs"]
                        if not (r["structural"]["insights_count"] == 5 and
                               r["structural"]["distinct_sources"] >= 4 and
                               r["structural"]["recent_count"] >= 2 and
                               r["structural"]["email_schema_ok"] and
                               r["structural"]["proof_points_resolve"]))
        lines.append(f"- {fail_count} runs failed structural validation")

    if comp_metrics["aggregates"]["critical_flags_total"] == 0:
        lines.append("- No critical compliance flags detected")
    else:
        lines.append(f"- {comp_metrics['aggregates']['critical_flags_total']} critical compliance flags detected")

    lines.append("")

    # Add failure details if any
    errors = [r for r in gen_metrics["runs"] if r.get("error")]
    if errors:
        lines.append("**Failed Runs:**")
        for r in errors:
            lines.append(f"- {r['eval_id']} ({r['persona']}): {r.get('error', 'Unknown error')[:100]}")
        lines.append("")

    # Write MD
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


async def main_async(args) -> None:
    """Main async execution."""
    # Load prompts
    if not os.path.exists(args.prompts):
        print(json.dumps({
            "error": f"Prompts file not found: {args.prompts}",
            "hint": "Run: python3 scripts/build_eval_generation_prompts.py"
        }))
        raise SystemExit(1)

    prompts = load_jsonl(args.prompts)
    if not prompts:
        print(json.dumps({"error": "No prompts found in file"}))
        raise SystemExit(1)

    print(f"Loaded {len(prompts)} prompts")

    # Try to start MCP stubs
    started, service_mode = await maybe_start_stubs()
    if started:
        print(f"MCP stubs started successfully")
    else:
        print(f"Running in {service_mode} mode")

    # Run each prompt
    runs = []
    for i, prompt in enumerate(prompts, 1):
        print(f"Running prompt {i}/{len(prompts)}: {prompt['persona']} for {prompt['company']}...")
        result = await run_one_prompt(
            prompt["eval_id"],
            prompt["company"],
            prompt["persona"],
            timeout=args.timeout
        )
        runs.append(result)

        if result.get("error"):
            print(f"  ✗ Failed: {result['error'][:100]}")
        else:
            print(f"  ✓ Success: session_id={result['session_id']}")

    # Aggregate metrics
    gen_metrics, comp_metrics = aggregate_metrics(runs)

    # Compute gate status
    status, checks = compute_gate_status(gen_metrics, comp_metrics)

    # Write reports
    write_reports(gen_metrics, comp_metrics, status, checks, service_mode, len(runs))

    # Print final status
    print("")
    print(f"Gate-8 Status: {status}")
    print(f"Reports written to:")
    print(f"  - {OUT_JSON}")
    print(f"  - {OUT_MD}")
    print(f"  - {GEN_METRICS_PATH}")
    print(f"  - {COMP_METRICS_PATH}")

    # Exit with appropriate code
    if status == "RED":
        sys.exit(1)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Gate-8: Generation & Compliance Evaluation"
    )
    parser.add_argument(
        "--prompts",
        default=PROMPTS_PATH,
        help=f"Path to generation prompts JSONL (default: {PROMPTS_PATH})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout per run in seconds (default: 30)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run internal self-tests and exit"
    )

    args = parser.parse_args()

    # Self-test mode
    if args.self_test:
        print("Running self-tests...")

        # Test persona_keyword_hits
        test_body = "We can improve NPS and CSAT scores with our omnichannel solution"
        hits = persona_keyword_hits("vp_customer_experience", test_body)
        assert hits >= 2, f"Expected >=2 keyword hits, got {hits}"
        print("  ✓ persona_keyword_hits")

        # Test validate_structure
        mock_insights = [
            {"id": "i1", "source_domain": "salesforce.com", "date": "2025-01-01"},
            {"id": "i2", "source_domain": "techcrunch.com", "date": "2025-01-15"},
            {"id": "i3", "source_domain": "wikipedia.org", "date": "2024-06-01"},
            {"id": "i4", "source_domain": "sec.gov", "date": "2025-02-01"},
            {"id": "i5", "source_domain": "salesforce.com", "date": "2023-01-01"},
        ]
        mock_email = {
            "subject": "Test",
            "body": "Test body",
            "unsubscribe_block": "Unsubscribe",
            "company_info_block": "Company info",
            "proof_points": [{"id": "i1"}, {"id": "i3"}]
        }
        result = validate_structure(mock_insights, mock_email)
        assert result["insights_count"] == 5, f"Expected 5 insights, got {result['insights_count']}"
        assert result["distinct_sources"] >= 4, f"Expected >=4 sources, got {result['distinct_sources']}"
        assert result["recent_count"] >= 2, f"Expected >=2 recent, got {result['recent_count']}"
        assert result["email_schema_ok"], "Email schema validation failed"
        assert result["proof_points_resolve"], "Proof points resolution failed"
        print("  ✓ validate_structure")

        # Test word_count and readability
        test_text = "This is a test. It has two sentences."
        wc = word_count(test_text)
        assert 6 <= wc <= 10, f"Expected 6-10 words, got {wc}"
        grade = readability_grade(test_text)
        assert 0 <= grade <= 20, f"Expected grade 0-20, got {grade}"
        print("  ✓ word_count and readability_grade")

        print("All self-tests passed!")
        sys.exit(0)

    # Run main async logic
    asyncio.run(main_async(args))