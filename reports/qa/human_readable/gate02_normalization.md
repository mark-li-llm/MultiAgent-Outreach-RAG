# Gate G02 — Normalization QA (Phase A; Run 2025-09-07T20:09:51.169055+00:00)
Summary: PASS

Coverage
- Raw eligible docs: 156
- Normalized docs: 44
- Dropped non-en: 0
- normalized_coverage_ratio: 1.0 (>= 0.98) -> PASS
- lang_en_ratio: 1.0 (>= 0.95) -> PASS

Quality
- min_word_count_violations: 0 (== 0) -> PASS
- retention_ratio_median (by doctype): 
  - 10-K: 0.9925 (>= baseline-0.05) -> PASS
  - 8-K: 0.988 (>= baseline-0.05) -> PASS
  - dev_docs: 0.9565 (>= baseline-0.05) -> PASS
  - press: 0.9117 (>= baseline-0.05) -> PASS
  - product: 0.6492 (>= baseline-0.05) -> PASS
  - wiki: 0.9328 (>= baseline-0.05) -> PASS
- heading_presence_ratio: 1.0 (>= 0.90) -> PASS
- pdf_page_map_missing: 0 (== 0) -> PASS

Actions:
- None

Proceed? (Y/N): Y
