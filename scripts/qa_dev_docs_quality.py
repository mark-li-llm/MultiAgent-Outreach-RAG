#!/usr/bin/env python3
"""
QA script for dev_docs data quality verification.

Checks:
- File completeness (meta.json + raw.html pairs)
- Metadata field coverage
- Content size distribution
- URL validity and HTTP status
- Title and content extraction quality
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, load_json, now_iso


def analyze_dev_docs() -> Tuple[Dict[str, Any], List[str]]:
    """Analyze dev_docs quality and return (metrics, issues)."""
    data_dir = "data/raw/dev_docs"

    # Find all meta.json files
    meta_files = sorted(glob.glob(f"{data_dir}/*.meta.json"))

    if not meta_files:
        return {}, ["No dev_docs meta files found"]

    issues = []
    all_meta = []
    file_pairs = []
    content_sizes = []
    missing_html = []

    for meta_path in meta_files:
        # Load metadata
        try:
            meta = load_json(meta_path)
            all_meta.append(meta)
        except Exception as e:
            issues.append(f"Failed to load {meta_path}: {e}")
            continue

        # Check for corresponding HTML file
        html_path = meta_path.replace(".meta.json", ".raw.html")
        if os.path.exists(html_path):
            file_pairs.append((meta_path, html_path))
            # Get HTML file size
            try:
                size = os.path.getsize(html_path)
                content_sizes.append(size)
            except Exception as e:
                issues.append(f"Failed to get size of {html_path}: {e}")
        else:
            missing_html.append(meta_path)
            issues.append(f"Missing HTML file for {meta_path}")

    # Compute metrics
    total_meta = len(all_meta)
    total_pairs = len(file_pairs)
    pair_completeness = (total_pairs / total_meta) if total_meta > 0 else 0.0

    # HTTP status analysis
    http_200_count = sum(1 for m in all_meta if m.get("http_status") == 200)
    http_200_ratio = (http_200_count / total_meta) if total_meta > 0 else 0.0

    # Field coverage
    fields_to_check = ["doc_id", "source_domain", "source_bucket", "requested_url",
                       "final_url", "http_status", "content_type", "fetched_at",
                       "sha256_raw", "visible_title", "headline"]

    field_coverage = {}
    for field in fields_to_check:
        present = sum(1 for m in all_meta if m.get(field) is not None and m.get(field) != "")
        field_coverage[field] = round(present / total_meta, 4) if total_meta > 0 else 0.0

    # Title analysis
    missing_title = sum(1 for m in all_meta if not m.get("visible_title") and not m.get("headline"))
    title_coverage = 1.0 - (missing_title / total_meta) if total_meta > 0 else 0.0

    # Content size stats
    size_stats = {}
    if content_sizes:
        size_stats = {
            "min_bytes": min(content_sizes),
            "max_bytes": max(content_sizes),
            "mean_bytes": int(mean(content_sizes)),
            "median_bytes": int(median(content_sizes))
        }

    # Source domain consistency
    domains = Counter([m.get("source_domain") for m in all_meta])

    # SHA256 duplicates
    sha_hashes = [m.get("sha256_raw") for m in all_meta if m.get("sha256_raw")]
    sha_counts = Counter(sha_hashes)
    duplicates = {h: c for h, c in sha_counts.items() if c > 1}
    duplicate_rate = len(duplicates) / len(sha_hashes) if sha_hashes else 0.0

    # Compile metrics
    metrics = {
        "total_meta_files": total_meta,
        "total_html_files": total_pairs,
        "file_pair_completeness": round(pair_completeness, 4),
        "http_200_count": http_200_count,
        "http_200_ratio": round(http_200_ratio, 4),
        "title_coverage": round(title_coverage, 4),
        "missing_title_count": missing_title,
        "field_coverage": field_coverage,
        "content_size_stats": size_stats,
        "source_domains": dict(domains),
        "duplicate_sha256_rate": round(duplicate_rate, 4),
        "duplicate_count": len(duplicates),
        "missing_html_files": len(missing_html)
    }

    return metrics, issues


def build_checks(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build quality gate checks."""
    checks = []

    # Check 1: All files should have corresponding HTML
    checks.append({
        "id": "DEV-001",
        "metric": "file_pair_completeness",
        "actual": metrics.get("file_pair_completeness", 0.0),
        "threshold": "==1.0",
        "status": "PASS" if metrics.get("file_pair_completeness") == 1.0 else "FAIL",
        "description": "All meta.json files should have corresponding raw.html files"
    })

    # Check 2: All HTTP requests should succeed
    checks.append({
        "id": "DEV-002",
        "metric": "http_200_ratio",
        "actual": metrics.get("http_200_ratio", 0.0),
        "threshold": "==1.0",
        "status": "PASS" if metrics.get("http_200_ratio") == 1.0 else "FAIL",
        "description": "All fetched documents should have HTTP 200 status"
    })

    # Check 3: Title extraction should be complete
    checks.append({
        "id": "DEV-003",
        "metric": "title_coverage",
        "actual": metrics.get("title_coverage", 0.0),
        "threshold": "==1.0",
        "status": "PASS" if metrics.get("title_coverage") == 1.0 else "FAIL",
        "description": "All documents should have visible_title or headline"
    })

    # Check 4: Required field coverage
    field_cov = metrics.get("field_coverage", {})
    required_fields = ["doc_id", "source_domain", "source_bucket", "requested_url",
                       "final_url", "http_status", "fetched_at", "sha256_raw"]
    all_fields_complete = all(field_cov.get(f, 0.0) == 1.0 for f in required_fields)

    checks.append({
        "id": "DEV-004",
        "metric": "required_fields_complete",
        "actual": all_fields_complete,
        "threshold": "==True",
        "status": "PASS" if all_fields_complete else "FAIL",
        "description": f"Required metadata fields must be 100% present: {required_fields}"
    })

    # Check 5: No duplicate content
    checks.append({
        "id": "DEV-005",
        "metric": "duplicate_sha256_rate",
        "actual": metrics.get("duplicate_sha256_rate", 0.0),
        "threshold": "==0.0",
        "status": "PASS" if metrics.get("duplicate_sha256_rate") == 0.0 else "FAIL",
        "description": "No duplicate content (by SHA256 hash)"
    })

    # Check 6: Expected document count
    total_meta = metrics.get("total_meta_files", 0)
    checks.append({
        "id": "DEV-006",
        "metric": "total_documents",
        "actual": total_meta,
        "threshold": "==8",
        "status": "PASS" if total_meta == 8 else "FAIL",
        "description": "Should have exactly 8 dev_docs documents"
    })

    # Check 7: Minimum content size
    size_stats = metrics.get("content_size_stats", {})
    min_size = size_stats.get("min_bytes", 0)
    checks.append({
        "id": "DEV-007",
        "metric": "min_content_size",
        "actual": min_size,
        "threshold": ">=10000",
        "status": "PASS" if min_size >= 10000 else "FAIL",
        "description": "All documents should have substantial content (>=10KB)"
    })

    return checks


def generate_report(metrics: Dict[str, Any], checks: List[Dict[str, Any]],
                   issues: List[str]) -> Tuple[Dict[str, Any], str]:
    """Generate JSON and Markdown reports."""

    overall_status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"

    # Machine-readable JSON report
    json_report = {
        "gate": "QA_DEV_DOCS",
        "computed_at": now_iso(),
        "status": overall_status,
        "metrics": metrics,
        "checks": checks,
        "issues": issues[:100]  # Limit to first 100 issues
    }

    # Human-readable Markdown report
    md_lines = []
    md_lines.append("# Dev Docs Quality Report")
    md_lines.append(f"\n**Status**: {overall_status}")
    md_lines.append(f"**Generated**: {now_iso()}")

    md_lines.append("\n## Summary")
    md_lines.append(f"- Total documents: {metrics.get('total_meta_files', 0)}")
    md_lines.append(f"- Complete file pairs: {metrics.get('total_html_files', 0)}")
    md_lines.append(f"- HTTP 200 ratio: {metrics.get('http_200_ratio', 0.0):.2%}")
    md_lines.append(f"- Title coverage: {metrics.get('title_coverage', 0.0):.2%}")
    md_lines.append(f"- Duplicate rate: {metrics.get('duplicate_sha256_rate', 0.0):.2%}")

    size_stats = metrics.get("content_size_stats", {})
    if size_stats:
        md_lines.append("\n## Content Size Distribution")
        md_lines.append(f"- Min: {size_stats.get('min_bytes', 0):,} bytes")
        md_lines.append(f"- Max: {size_stats.get('max_bytes', 0):,} bytes")
        md_lines.append(f"- Mean: {size_stats.get('mean_bytes', 0):,} bytes")
        md_lines.append(f"- Median: {size_stats.get('median_bytes', 0):,} bytes")

    md_lines.append("\n## Quality Checks")
    md_lines.append("\n| ID | Metric | Status | Actual | Threshold |")
    md_lines.append("|-------|--------|--------|--------|-----------|")
    for check in checks:
        md_lines.append(f"| {check['id']} | {check['metric']} | {check['status']} | "
                       f"{check['actual']} | {check['threshold']} |")

    if issues:
        md_lines.append("\n## Issues Found")
        for i, issue in enumerate(issues[:20], 1):  # Show first 20 issues
            md_lines.append(f"{i}. {issue}")
        if len(issues) > 20:
            md_lines.append(f"\n... and {len(issues) - 20} more issues")

    md_lines.append("\n## Field Coverage")
    field_cov = metrics.get("field_coverage", {})
    for field, coverage in sorted(field_cov.items()):
        status = "✓" if coverage == 1.0 else "✗"
        md_lines.append(f"- {status} {field}: {coverage:.2%}")

    domains = metrics.get("source_domains", {})
    if domains:
        md_lines.append("\n## Source Domains")
        for domain, count in sorted(domains.items()):
            md_lines.append(f"- {domain}: {count} documents")

    md_report = "\n".join(md_lines)

    return json_report, md_report


def main():
    parser = argparse.ArgumentParser(description="QA dev_docs data quality")
    parser.add_argument("--output-dir", default="reports/qa",
                       help="Output directory for reports")
    args = parser.parse_args()

    print("Analyzing dev_docs quality...")
    metrics, issues = analyze_dev_docs()

    print("Building quality checks...")
    checks = build_checks(metrics)

    print("Generating reports...")
    json_report, md_report = generate_report(metrics, checks, issues)

    # Write reports
    ensure_dir(args.output_dir)

    json_path = os.path.join(args.output_dir, "qa_dev_docs_quality.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report: {json_path}")

    md_path = os.path.join(args.output_dir, "qa_dev_docs_quality.md")
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown report: {md_path}")

    # Print summary
    print(f"\n{md_report}")

    # Exit with appropriate code
    overall_status = json_report["status"]
    if overall_status == "FAIL":
        print("\n❌ Quality checks FAILED")
        exit(1)
    else:
        print("\n✅ Quality checks PASSED")
        exit(0)


if __name__ == "__main__":
    main()