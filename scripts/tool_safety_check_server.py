#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from aiohttp import web
import re


CONF = os.path.join("configs", "compliance.template.yaml")


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def readability_grade(text: str) -> float:
    # Flesch-Kincaid Grade approximation without external deps
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, word_count(text))
    syllables = max(1, sum(len(re.findall(r"[aeiouyAEIOUY]", w)) or 1 for w in re.findall(r"\b\w+\b", text or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59


def has_uncited_claim(text: str, insight_ids: List[str]) -> bool:
    # flag if we see strong quantifiers without any reference token like [#id] (not implemented) — heuristic only
    if re.search(r"\b(\d+%|double|guarantee|always|never)\b", text, re.I):
        return True
    return False


def prohibited_present(text: str, phrases: List[str]) -> bool:
    t = (text or "").lower()
    return any(p.lower() in t for p in phrases)


def check_email(email: Dict[str, Any], insights: List[Dict[str, Any]], spec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    critical: List[str] = []
    warning: List[str] = []
    body = email.get("body") or ""
    unsub = email.get("unsubscribe_block") or ""
    company = email.get("company_info_block") or ""
    insight_ids = [c.get("id") for c in insights]
    # Criticals
    if not unsub.strip():
        critical.append("MISSING_UNSUBSCRIBE")
    if not company.strip():
        critical.append("MISSING_COMPANY_INFO")
    if has_uncited_claim(body, insight_ids):
        critical.append("UNCITED_CLAIM")
    if prohibited_present(body, spec.get("prohibited_phrases", [])):
        critical.append("PROHIBITED_PHRASE")
    # Warnings
    max_words = int(next((r.get("params", {}).get("max_words") for r in spec.get("warning_rules", []) if r.get("id") == "EXCESS_LENGTH"), 160))
    if word_count(body) > max_words:
        warning.append("EXCESS_LENGTH")
    max_grade = int(next((r.get("params", {}).get("max_grade") for r in spec.get("warning_rules", []) if r.get("id") == "READABILITY"), 10))
    if readability_grade(body) > max_grade:
        warning.append("READABILITY")
    return critical, warning


async def handle_invoke(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"code": "InvalidJSON", "message": "Malformed JSON"}}, status=400)
    if (body.get("method") or "") != "moderate":
        return web.json_response({"error": {"code": "InvalidMethod", "message": "Unknown method"}}, status=400)
    params = body.get("params") or {}
    email = params.get("email_fields") or {"body": params.get("text") or ""}
    insights = params.get("insight_cards") or []
    spec = load_yaml(CONF)
    critical, warning = check_email(email, insights, spec)
    return web.json_response({"status": "ok", "flags": {"critical": critical, "warning": warning}})


async def handle_health(request: web.Request):
    return web.json_response({"status": "ok"})


def main():
    ap = argparse.ArgumentParser(description="Safety check server (method=moderate)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7805)
    args = ap.parse_args()
    app = web.Application()
    app.add_routes([web.get("/healthz", handle_health), web.post("/invoke", handle_invoke)])
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
