# Gate G04 — Chunking QA (Run 2025-09-07T22:20:54.572976+00:00)
Summary: PASS

- chunk_count_within_expected_ratio: 0.96 (>= 0.90) -> PASS
- adjacent_overlap_jaccard_median: 0.1325 (in [0.12,0.22]) -> PASS
- SEC boundary alignment: 1.0 (>= 0.90) -> PASS
- size_envelope_ratio (per doctype):
  - press: 0.921 (>= 0.80) -> PASS
  - 10-K: 1.0 (>= 0.80) -> PASS
  - 10-Q: 1.0 (>= 0.80) -> PASS
  - 8-K: 1.0 (>= 0.80) -> PASS
  - ars_pdf: 1.0 (>= 0.80) -> PASS
  - product: 1.0 (>= 0.80) -> PASS
  - dev_docs: 0.3333 (>= 0.80) -> FAIL
  - help_docs: 1.0 (>= 0.80) -> PASS
  - wiki: 0.0 (>= 0.80) -> FAIL

Failures & Actions:
- None

Proceed? (Y/N): Y
