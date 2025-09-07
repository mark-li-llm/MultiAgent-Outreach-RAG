#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, now_iso


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_normalized() -> List[str]:
    return sorted(glob.glob("data/interim/normalized/*.json"))


def list_raw_sidecars() -> List[str]:
    return sorted(glob.glob("data/raw/**/**/*.meta.json", recursive=True))


def eligible_raw(meta: Dict[str, Any]) -> bool:
    return int(meta.get("http_status") or 0) == 200


def compute_word_count(text: str) -> int:
    import re

    return len(re.findall(r"\b\w+\b", text))


def normalize_for_baseline(raw_path: str) -> Tuple[str, int, int]:
    # Import normalization function without writing outputs
    from normalize_html import normalize_html_bytes, extract_pdf_text

    if raw_path.endswith(".pdf"):
        text, _map, before_len, after_len = extract_pdf_text(raw_path)
        return text, before_len, after_len
    elif raw_path.endswith(".raw.html") or raw_path.endswith(".html"):
        rules = load_json_like_yaml("configs/normalization.rules.yaml")
        with open(raw_path, "rb") as f:
            text, before_len, after_len, _ = normalize_html_bytes(f.read(), rules)
        return text, before_len, after_len
    elif raw_path.endswith(".json"):
        text = json.dumps(load_json(raw_path))
        return text, len(text), len(text)
    else:
        return "", 0, 0


def load_json_like_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def build_baseline_if_missing() -> Dict[str, Any]:
    baseline_path = "reports/qa/gate00_baseline.json"
    # If baselines exist, return them
    if os.path.exists(baseline_path):
        return load_json(baseline_path)

    # Build adaptive baselines from first 3 docs per doctype
    # Prefer building baseline from existing normalized docs (if any)
    norm_paths = list_normalized()
    if norm_paths:
        per_dt_docs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for p in norm_paths:
            try:
                d = load_json(p)
            except Exception:
                continue
            dt = d.get("doctype")
            if len(per_dt_docs[dt]) < 3:
                per_dt_docs[dt].append(d)
        report: Dict[str, Any] = {"computed_at": now_iso(), "baselines": {}}
        for dt, docs in per_dt_docs.items():
            if not docs:
                continue
            wcs = [int(d.get("word_count") or 0) for d in docs]
            toks = [int(d.get("token_count") or 0) for d in docs]
            # Retention recompute for these docs
            rets: List[float] = []
            for d in docs:
                doc_id = d.get("doc_id")
                # find raw path
                raw_meta = None
                for mp in list_raw_sidecars():
                    m = load_json(mp)
                    if m.get("doc_id") == doc_id:
                        raw_meta = m
                        base_dir = os.path.dirname(mp)
                        break
                if not raw_meta:
                    continue
                raw_path = None
                for ext in (".raw.html", ".pdf", ".json"):
                    pp = os.path.join(base_dir, f"{doc_id}{ext}")
                    if os.path.exists(pp):
                        raw_path = pp
                        break
                if not raw_path:
                    continue
                text, before_len, after_len = normalize_for_baseline(raw_path)
                ret = (after_len / max(1, before_len)) if before_len else 1.0
                rets.append(ret)
            report["baselines"][dt] = {
                "median_word_count_doc": int(median(wcs)) if wcs else 0,
                "median_token_count_doc": int(median(toks)) if toks else 0,
                "median_chunks_per_doc": 1,
                "median_chunk_tokens": int(median(toks)) if toks else 0,
                "sec_item_coverage_ratio_median": 1.0 if dt in ("10-K", "10-Q", "8-K") else 0.0,
                "date_presence_rate": 0.0,
                "title_presence_rate": 1.0,
                "persona_tag_assignment_rate": 0.0,
                "raw_to_normalized_retention_ratio_median": float(median(rets)) if rets else 1.0,
            }
        ensure_dir("reports/qa/human_readable")
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open("reports/qa/human_readable/gate00_baseline.md", "w", encoding="utf-8") as f:
            f.write(f"# Gate G00 — Baseline (Run {now_iso()})\n")
            for dt, b in report.get("baselines", {}).items():
                f.write(f"- {dt}: median_word_count_doc {b['median_word_count_doc']}, retention {b['raw_to_normalized_retention_ratio_median']:.3f}\n")
        return report

    metas = []
    for mp in list_raw_sidecars():
        try:
            m = load_json(mp)
        except Exception:
            continue
        if eligible_raw(m):
            metas.append((m.get("doctype"), mp, m))
    metas.sort(key=lambda x: (x[0] or "", x[1]))

    by_type: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for dt, mp, m in metas:
        if len(by_type[dt]) < 3:
            by_type[dt].append((mp, m))

    report: Dict[str, Any] = {
        "computed_at": now_iso(),
        "baselines": {},
    }
    rules = load_json_like_yaml("configs/normalization.rules.yaml")
    for dt, items in by_type.items():
        word_counts = []
        token_counts = []
        ret_ratios = []
        title_presence = []
        date_presence = []
        for mp, m in items:
            base_dir = os.path.dirname(mp)
            doc_id = m.get("doc_id")
            # find raw
            raw_candidates = [
                os.path.join(base_dir, f"{doc_id}.raw.html"),
                os.path.join(base_dir, f"{doc_id}.pdf"),
                os.path.join(base_dir, f"{doc_id}.json"),
            ]
            raw_path = ""
            for c in raw_candidates:
                if os.path.exists(c):
                    raw_path = c
                    break
            if not raw_path:
                continue
            # Normalize this raw to compute baseline stats (in-memory)
            text, before_len, after_len = normalize_for_baseline(raw_path)
            wc = compute_word_count(text)
            try:
                from normalize_html import cl100k_token_count  # reuse function
            except Exception:
                def cl100k_token_count(t: str) -> int:
                    return len(t.split())
            token_counts.append(cl100k_token_count(text))
            word_counts.append(wc)
            ret = (after_len / max(1, before_len)) if before_len else 1.0
            ret_ratios.append(ret)
            title_presence.append(1 if (m.get("visible_title") or "") else 0)
            date_presence.append(1 if (m.get("visible_date") or m.get("rss_pubdate")) else 0)

        if not word_counts:
            continue
        report["baselines"][dt] = {
            "median_word_count_doc": int(median(word_counts)),
            "median_token_count_doc": int(median(token_counts)) if token_counts else 0,
            "median_chunks_per_doc": 1,  # placeholder (no chunking in Step 2)
            "median_chunk_tokens": int(median(token_counts)) if token_counts else 0,
            "sec_item_coverage_ratio_median": 1.0 if dt in ("10-K", "10-Q", "8-K") else 0.0,
            "date_presence_rate": (sum(date_presence) / max(1, len(date_presence))),
            "title_presence_rate": (sum(title_presence) / max(1, len(title_presence))),
            "persona_tag_assignment_rate": 0.0,
            "raw_to_normalized_retention_ratio_median": float(median(ret_ratios)) if ret_ratios else 1.0,
        }

    ensure_dir("reports/qa/human_readable")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("reports/qa/human_readable/gate00_baseline.md", "w", encoding="utf-8") as f:
        f.write(f"# Gate G00 — Baseline (Run {now_iso()})\n")
        for dt, b in report.get("baselines", {}).items():
            f.write(f"- {dt}: median_word_count_doc {b['median_word_count_doc']}, retention {b['raw_to_normalized_retention_ratio_median']:.3f}\n")
    return report


def compute_metrics(phase: str) -> Dict[str, Any]:
    baseline = build_baseline_if_missing()
    baselines = baseline.get("baselines", {})

    raw_metas = []
    for p in list_raw_sidecars():
        try:
            m = load_json(p)
        except Exception:
            continue
        if eligible_raw(m):
            raw_metas.append(m)
    raw_total = len(raw_metas)
    norm_paths = list_normalized()
    normalized_all = [load_json(p) for p in norm_paths]

    # Parse latest normalize log for drops
    latest_log = None
    try:
        logs = sorted(glob.glob("logs/normalize/*.log"))
        latest_log = logs[-1] if logs else None
        dropped_non_en = 0
        if latest_log:
            for line in open(latest_log, "r", encoding="utf-8", errors="ignore"):
                if "DROPPED_NON_EN" in line:
                    dropped_non_en += 1
    except Exception:
        dropped_non_en = 0

    # Build Phase A subset doc_ids deterministically (same selector as normalizer)
    import hashlib
    all_raw_ids = [m.get("doc_id") for m in raw_metas]
    if phase == "A":
        subset_ids = set([doc_id for doc_id in all_raw_ids if int(hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[-1], 16) % 2 == 0])
    else:
        subset_ids = set(all_raw_ids)

    normalized = [d for d in normalized_all if d.get("doc_id") in subset_ids]
    normalized_total = len(normalized)
    raw_total_phase = len(subset_ids)
    denom = max(1, raw_total_phase - dropped_non_en)
    normalized_coverage_ratio = normalized_total / denom

    lang_en_ratio = (sum(1 for d in normalized if (d.get("language") == "en")) / max(1, normalized_total))

    # Heading presence: look for lines starting with H1:/H2:/H3:
    heading_presence = sum(1 for d in normalized if any(s in (d.get("text") or "") for s in ["H1:", "H2:", "H3:"]))
    heading_presence_ratio = heading_presence / max(1, normalized_total)

    # PDF accounting
    # map doc_id -> raw ext
    raw_ext: Dict[str, str] = {}
    for mp in list_raw_sidecars():
        m = load_json(mp)
        if not eligible_raw(m):
            continue
        doc_id = m.get("doc_id")
        base_dir = os.path.dirname(mp)
        for ext in (".pdf", ".raw.html", ".json"):
            if os.path.exists(os.path.join(base_dir, f"{doc_id}{ext}")):
                raw_ext[doc_id] = ext
                break
    pdf_docs_seen = 0
    pdf_page_map_missing = 0
    for d in normalized:
        if raw_ext.get(d.get("doc_id")) == ".pdf":
            pdf_docs_seen += 1
            if "pdf_page_map" not in d:
                pdf_page_map_missing += 1

    # Per-doctype calculations
    per_dt: Dict[str, Dict[str, Any]] = {}
    # group normalized by doctype
    group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in normalized:
        group[d.get("doctype")].append(d)

    # documents below min word counts, collect ids
    docs_below_min_ids: List[str] = []
    for dt, docs in group.items():
        base = baselines.get(dt, {})
        base_median_wc = int(base.get("median_word_count_doc") or 400)
        doc_word_count_min = max(200, int(0.6 * base_median_wc))
        below = [d for d in docs if int(d.get("word_count") or 0) < doc_word_count_min]
        for d in below:
            docs_below_min_ids.append(d.get("doc_id"))

        # Retention recompute via raw
        ret_values: List[float] = []
        for d in docs:
            doc_id = d.get("doc_id")
            # find raw
            raw_meta_path = None
            for mp in list_raw_sidecars():
                m = load_json(mp)
                if m.get("doc_id") == doc_id:
                    raw_meta_path = mp
                    break
            if not raw_meta_path:
                continue
            base_dir = os.path.dirname(raw_meta_path)
            raw_path = None
            for ext in (".raw.html", ".pdf", ".json"):
                p = os.path.join(base_dir, f"{doc_id}{ext}")
                if os.path.exists(p):
                    raw_path = p
                    break
            if not raw_path:
                continue
            # in-memory normalization to compute retention
            from normalize_html import normalize_html_bytes, extract_pdf_text
            if raw_path.endswith(".pdf"):
                _text, _map, before_len, after_len = extract_pdf_text(raw_path)
            elif raw_path.endswith(".raw.html") or raw_path.endswith(".html"):
                rules = load_json_like_yaml("configs/normalization.rules.yaml")
                with open(raw_path, "rb") as f:
                    _text, before_len, after_len, _ = normalize_html_bytes(f.read(), rules)
            else:
                before_len = after_len = len(open(raw_path, "rb").read())
            ret = (after_len / max(1, before_len)) if before_len else 1.0
            ret_values.append(ret)

        retention_median = float(median(ret_values)) if ret_values else 1.0
        retention_threshold = float(base.get("raw_to_normalized_retention_ratio_median") or 1.0) - 0.05
        per_dt[dt] = {
            "baseline_median_word_count_doc": base_median_wc,
            "doc_word_count_min": doc_word_count_min,
            "docs_below_min_word_count": len(below),
            "retention_ratio_median": retention_median,
            "retention_threshold": retention_threshold,
        }

    # Checks
    checks: List[Dict[str, Any]] = []
    checks.append({
        "id": "STD-001", "metric": "normalized_coverage_ratio", "actual": round(normalized_coverage_ratio, 4), "threshold": ">=0.98",
        "status": "PASS" if normalized_coverage_ratio >= 0.98 else "FAIL",
    })
    checks.append({
        "id": "STD-002", "metric": "lang_en_ratio", "actual": round(lang_en_ratio, 4), "threshold": ">=0.95",
        "status": "PASS" if lang_en_ratio >= 0.95 else "FAIL",
    })
    total_below = sum(v.get("docs_below_min_word_count", 0) for v in per_dt.values())
    checks.append({
        "id": "STD-003", "metric": "min_word_count_violations", "actual": int(total_below), "threshold": "==0",
        "status": "PASS" if total_below == 0 else "FAIL",
    })
    press_ret = per_dt.get("press", {}).get("retention_ratio_median", 1.0)
    press_thr = per_dt.get("press", {}).get("retention_threshold", 0.0)
    checks.append({
        "id": "STD-004", "metric": "retention_ratio_median_press", "actual": round(press_ret, 4), "threshold": ">=baseline-0.05",
        "status": "PASS" if press_ret >= press_thr else "FAIL",
    })
    checks.append({
        "id": "STD-005", "metric": "heading_presence_ratio", "actual": round(heading_presence_ratio, 4), "threshold": ">=0.90",
        "status": "PASS" if heading_presence_ratio >= 0.90 else "FAIL",
    })
    checks.append({
        "id": "STD-006", "metric": "pdf_page_map_missing", "actual": int(pdf_page_map_missing), "threshold": "==0",
        "status": "PASS" if pdf_page_map_missing == 0 else "FAIL",
    })

    status = "PASS" if all(c.get("status") == "PASS" for c in checks) else "FAIL"

    # Evidence
    norm_ids = set(d.get("doc_id") for d in normalized)
    raw_ids = [rid for rid in all_raw_ids if rid in subset_ids]
    missing_norm = [r for r in raw_ids if r not in norm_ids]
    retention_samples = []
    # pick up to 5 samples from press
    for d in normalized[:5]:
        dt = d.get("doctype")
        retm = per_dt.get(dt, {}).get("retention_ratio_median", 1.0)
        retention_samples.append({"doc_id": d.get("doc_id"), "retention_ratio": retm})

    machine = {
        "gate": "G02_NORMALization",
        "phase": phase,
        "computed_at": now_iso(),
        "summary": {
            "raw_total": int(raw_total),
            "normalized_total": int(normalized_total),
            "dropped_non_en": int(dropped_non_en),
            "normalized_coverage_ratio": round(normalized_coverage_ratio, 4),
            "lang_en_ratio": round(lang_en_ratio, 4),
            "heading_presence_ratio": round(heading_presence_ratio, 4),
            "pdf_docs_seen": int(pdf_docs_seen),
            "pdf_page_map_missing": int(pdf_page_map_missing),
        },
        "per_doctype": per_dt,
        "checks": checks,
        "status": status,
        "evidence": {
            "missing_normalized_for_raw": missing_norm,
            "docs_below_min_word_count": docs_below_min_ids,
            "retention_samples": retention_samples,
            "log_path": latest_log or "",
        },
    }
    return machine


def write_reports(machine: Dict[str, Any]) -> None:
    ensure_dir("reports/qa/human_readable")
    with open("reports/qa/gate02_normalization.json", "w", encoding="utf-8") as f:
        json.dump(machine, f, ensure_ascii=False, indent=2)

    s = machine["summary"]
    checks = machine["checks"]
    phase = machine["phase"]
    ts = machine["computed_at"]

    def status_for(metric: str) -> str:
        for c in checks:
            if c.get("metric") == metric:
                return c.get("status")
        return "FAIL"

    lines = []
    lines.append(f"# Gate G02 — Normalization QA (Phase {phase}; Run {ts})")
    lines.append(f"Summary: {machine['status']}")
    lines.append("")
    lines.append("Coverage")
    lines.append(f"- Raw eligible docs: {s['raw_total']}")
    lines.append(f"- Normalized docs: {s['normalized_total']}")
    lines.append(f"- Dropped non-en: {s['dropped_non_en']}")
    lines.append(f"- normalized_coverage_ratio: {s['normalized_coverage_ratio']} (>= 0.98) -> {status_for('normalized_coverage_ratio')}")
    lines.append(f"- lang_en_ratio: {s['lang_en_ratio']} (>= 0.95) -> {status_for('lang_en_ratio')}")
    lines.append("")
    lines.append("Quality")
    total_below = next((c['actual'] for c in checks if c['id']=='STD-003'), 0)
    lines.append(f"- min_word_count_violations: {total_below} (== 0) -> {status_for('min_word_count_violations')}")
    lines.append("- retention_ratio_median (by doctype): ")
    for dt, v in machine.get("per_doctype", {}).items():
        ok = "PASS" if v.get("retention_ratio_median", 1.0) >= v.get("retention_threshold", 0.0) else "FAIL"
        lines.append(f"  - {dt}: {round(v.get('retention_ratio_median', 0.0),4)} (>= baseline-0.05) -> {ok}")
    lines.append(f"- heading_presence_ratio: {s['heading_presence_ratio']} (>= 0.90) -> {status_for('heading_presence_ratio')}")
    lines.append(f"- pdf_page_map_missing: {s['pdf_page_map_missing']} (== 0) -> {status_for('pdf_page_map_missing')}")
    lines.append("")
    lines.append("Actions:")
    any_fail = [c for c in checks if c.get("status") == "FAIL"]
    if any_fail:
        for c in any_fail:
            metric = c.get("metric")
            if metric == "normalized_coverage_ratio":
                lines.append("- Investigate missing normalized outputs; re-run normalization for affected docs.")
            elif metric == "lang_en_ratio":
                lines.append("- Confirm language detector; whitelist english domains.")
            elif metric == "min_word_count_violations":
                lines.append("- Relax boilerplate stripping; ensure content containers preserved.")
            elif metric == "retention_ratio_median_press":
                lines.append("- Improve retention by preserving main content containers; reduce aggressive removals.")
            elif metric == "heading_presence_ratio":
                lines.append("- Ensure heading nodes become text lines (H1/H2/H3).")
            elif metric == "pdf_page_map_missing":
                lines.append("- Include pdf_page_map key (list or null) for PDF docs.")
    else:
        lines.append("- None")
    lines.append("")
    lines.append(f"Proceed? (Y/N): {'Y' if machine['status']=='PASS' else 'N'}")

    with open("reports/qa/human_readable/gate02_normalization.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="QA for Gate G02 — Normalization")
    ap.add_argument("--phase", required=True, choices=["A", "B"])
    args = ap.parse_args()

    machine = compute_metrics(args.phase)
    write_reports(machine)
    print(json.dumps({"status": machine["status"]}, indent=2))


if __name__ == "__main__":
    main()
