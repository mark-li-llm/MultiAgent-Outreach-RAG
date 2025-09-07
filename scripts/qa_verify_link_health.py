#!/usr/bin/env python3
import argparse
import glob
import json
import os
from urllib.parse import urlparse

from common import ensure_dir, now_iso


ALLOWLIST = [
    "sec.gov",
    "investor.salesforce.com",
    "salesforce.com",
    "developer.salesforce.com",
    "help.salesforce.com",
    "wikipedia.org",
]


def main():
    ap = argparse.ArgumentParser(description="QA for Gate G06 — Link Health & Canonicalization")
    args = ap.parse_args()

    # Load normalized docs
    docs = []
    for p in sorted(glob.glob("data/interim/normalized/*.json")):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
            docs.append(d)
        except Exception:
            continue

    docs_total = len(docs)
    ok = sum(1 for d in docs if d.get("link_ok") is True)
    link_ok_ratio = ok / max(1, docs_total)

    https = 0
    allow = 0
    non_https_urls = []
    by_final_url = {}
    for d in docs:
        fu = d.get("final_url") or d.get("url") or ""
        try:
            pr = urlparse(fu)
        except Exception:
            pr = None
        if pr and pr.scheme == "https":
            https += 1
        else:
            non_https_urls.append(fu)
        host = pr.netloc.lower() if pr else ""
        # Allow root domains and subdomains
        if any(host == root or host.endswith("." + root) for root in ALLOWLIST):
            allow += 1
        by_final_url.setdefault(fu, set()).add(d.get("doc_id"))

    https_ratio = https / max(1, docs_total)
    allow_ratio = allow / max(1, docs_total)
    collisions = sum(1 for k, v in by_final_url.items() if len(v) > 1)
    collision_sets = [list(v) for k, v in by_final_url.items() if len(v) > 1]

    checks = []
    checks.append({"id": "LNK-001", "metric": "link_ok_ratio", "actual": round(link_ok_ratio, 4), "threshold": "==1.0", "status": "PASS" if link_ok_ratio == 1.0 else "FAIL"})
    checks.append({"id": "LNK-002", "metric": "canonical_collisions", "actual": collisions, "threshold": "==0", "status": "PASS" if collisions == 0 else "FAIL"})
    checks.append({"id": "LNK-003", "metric": "allowlisted_domain_ratio", "actual": round(allow_ratio, 4), "threshold": "==1.0", "status": "PASS" if allow_ratio == 1.0 else "FAIL"})

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    machine = {
        "gate": "G06_LINK_HEALTH",
        "computed_at": now_iso(),
        "summary": {
            "docs_total": docs_total,
            "link_ok_ratio": round(link_ok_ratio, 4),
            "https_ratio": round(https_ratio, 4),
            "allowlisted_domain_ratio": round(allow_ratio, 4),
            "canonical_collisions": collisions,
        },
        "checks": checks,
        "status": status,
        "evidence": {
            "broken_docs": [d.get("doc_id") for d in docs if not d.get("link_ok")],
            "collision_sets": collision_sets,
            "non_https_urls": [u for u in non_https_urls if u],
            "log_path": "logs/link",
        },
    }

    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate06_link_health.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    # Human-readable
    s = machine["summary"]
    lines = []
    lines.append(f"# Gate G06 — Link Health & Canonicalization (Run {machine['computed_at']})")
    lines.append(f"Summary: {machine['status']}")
    lines.append("")
    chks = {c['id']: c for c in machine['checks']}
    lines.append(f"- link_ok_ratio: {s['link_ok_ratio']} (== 1.0) -> {chks['LNK-001']['status']}")
    lines.append(f"- canonical_collisions: {s['canonical_collisions']} (== 0) -> {chks['LNK-002']['status']}")
    lines.append(f"- allowlisted_domain_ratio: {s['allowlisted_domain_ratio']} (== 1.0) -> {chks['LNK-003']['status']}")
    lines.append(f"- https_ratio (informational): {s['https_ratio']}")
    lines.append("")
    lines.append("Failures & Actions:")
    if machine['status'] == 'PASS':
        lines.append("- None")
        lines.append("")
        lines.append("Proceed? (Y/N): Y")
    else:
        if chks['LNK-001']['status'] == 'FAIL':
            lines.append("- Fix broken links: re-fetch or replace with working alternates.")
        if chks['LNK-002']['status'] == 'FAIL':
            lines.append("- Resolve canonical collisions: merge/retire redundant docs and re-run.")
        if chks['LNK-003']['status'] == 'FAIL':
            lines.append("- Normalize URLs to allowed domains or remove offending docs.")
        lines.append("")
        lines.append("Proceed? (Y/N): N")
    with open("reports/qa/human_readable/gate06_link_health.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"status": machine["status"]}, indent=2))


if __name__ == "__main__":
    main()
