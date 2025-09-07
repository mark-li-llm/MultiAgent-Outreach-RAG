# Gate G02 — Normalization QA (Phase A; Run 2025-09-07T19:46:54.340418+00:00)
Summary: FAIL

Coverage
- Raw eligible docs: 156
- Normalized docs: 48
- Dropped non-en: 0
- normalized_coverage_ratio: 1.0 (>= 0.98) -> PASS
- lang_en_ratio: 1.0 (>= 0.95) -> PASS

Quality
- min_word_count_violations: 9 (== 0) -> FAIL
- retention_ratio_median (by doctype): 
  - 10-K: 0.9925 (>= baseline-0.05) -> PASS
  - 8-K: 0.8007 (>= baseline-0.05) -> PASS
  - dev_docs: 0.9498 (>= baseline-0.05) -> PASS
  - help_docs: 0.3605 (>= baseline-0.05) -> PASS
  - press: 0.9114 (>= baseline-0.05) -> PASS
  - product: 0.6492 (>= baseline-0.05) -> PASS
  - wiki: 0.9328 (>= baseline-0.05) -> PASS
- heading_presence_ratio: 0.8958 (>= 0.90) -> FAIL
- pdf_page_map_missing: 0 (== 0) -> PASS

Actions:
- Relax boilerplate stripping; ensure content containers preserved.
- Ensure heading nodes become text lines (H1/H2/H3).

Proceed? (Y/N): N
