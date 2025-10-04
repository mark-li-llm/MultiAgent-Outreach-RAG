#!/usr/bin/env python3
"""
Build deterministic generation evaluation prompts for Gate-8.
Generates 10 prompts across ≥3 personas for end-to-end testing.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from common import ensure_dir, now_iso


EVAL_PROMPTS_CONFIG = os.path.join("configs", "eval.prompts.yaml")
OUTPUT_PATH = os.path.join("data", "interim", "eval", "generation_prompts.jsonl")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config with fallback to empty dict."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_personas(path: str = EVAL_PROMPTS_CONFIG) -> Dict[str, List[str]]:
    """
    Load persona definitions from eval.prompts.yaml.
    Returns {persona_name: [keywords...]}
    """
    config = load_yaml(path)
    personas = config.get("personas", {})

    # Fallback to defaults if config missing
    if not personas:
        personas = {
            "vp_customer_experience": [
                "nps", "csat", "contact center", "omnichannel",
                "agent productivity", "self-service", "first contact resolution"
            ],
            "cio": [
                "data integration", "governance", "security", "tco",
                "platform", "apis", "real-time"
            ],
            "vp_sales_ops": [
                "pipeline", "forecast accuracy", "win rate",
                "productivity", "automation"
            ]
        }

    return personas


def deterministic_eval_id(company: str, persona: str, i: int) -> str:
    """
    Generate a deterministic 8-char hex ID from inputs.
    Ensures same inputs always produce same ID.
    """
    s = f"{company}:{persona}:{i}"
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def build_prompts(company: str, personas: List[str], total: int = 10) -> List[Dict[str, Any]]:
    """
    Build a list of generation prompt records.

    Args:
        company: Company name for all prompts
        personas: List of persona names to cycle through
        total: Total number of prompts to generate

    Returns:
        List of dicts with eval_id, company, persona, created_at
    """
    if not personas:
        raise ValueError("Need at least one persona")

    if len(personas) < 3:
        # Ensure we have at least 3 personas by repeating if needed
        available = list(personas)
        while len(personas) < 3:
            personas.append(available[len(personas) % len(available)])

    rows = []
    for i in range(total):
        # Round-robin through personas to ensure coverage
        persona = personas[i % len(personas)]
        eval_id = deterministic_eval_id(company, persona, i)

        rows.append({
            "eval_id": eval_id,
            "company": company,
            "persona": persona,
            "created_at": now_iso()
        })

    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    """Write rows as JSONL (one JSON per line)."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build generation evaluation prompts for Gate-8"
    )
    parser.add_argument(
        "--company",
        default="Salesforce",
        help="Company name for all prompts (default: Salesforce)"
    )
    parser.add_argument(
        "--total",
        type=int,
        default=10,
        help="Total number of prompts to generate (default: 10)"
    )
    parser.add_argument(
        "--personas",
        nargs="*",
        help="Specific personas to use (default: use all from config)"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Output JSONL path (default: {OUTPUT_PATH})"
    )

    args = parser.parse_args()

    # Validate total
    if args.total < 3:
        print(json.dumps({
            "error": "Need at least 3 prompts to cover multiple personas",
            "total_requested": args.total
        }))
        raise SystemExit(1)

    # Load or use specified personas
    all_personas = load_personas()

    if args.personas:
        # Use specified personas
        personas = args.personas
        # Validate they exist in config
        for p in personas:
            if p not in all_personas:
                print(f"Warning: Persona '{p}' not in config, using anyway")
    else:
        # Use first 3 personas from config (or all if fewer)
        personas = list(all_personas.keys())[:3]

    # Ensure we have at least 3 unique personas represented
    unique_personas = list(set(personas))
    if len(unique_personas) < 3:
        # Pad with additional personas from config if available
        for p in all_personas.keys():
            if p not in unique_personas:
                unique_personas.append(p)
            if len(unique_personas) >= 3:
                break

    # Build prompts
    try:
        prompts = build_prompts(args.company, unique_personas, args.total)
    except ValueError as e:
        print(json.dumps({"error": str(e)}))
        raise SystemExit(1)

    # Write output
    write_jsonl(prompts, args.output)

    # Count personas for summary
    persona_counts = {}
    for p in prompts:
        pname = p["persona"]
        persona_counts[pname] = persona_counts.get(pname, 0) + 1

    # Print summary
    summary = {
        "wrote": len(prompts),
        "path": args.output,
        "company": args.company,
        "personas": list(persona_counts.keys()),
        "persona_distribution": persona_counts,
        "timestamp": now_iso()
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()