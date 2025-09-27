# Gate G03 — Metadata & SEC Structure QA (Phase B; Run 2025-09-07T21:54:27.061823+00:00)
Summary: PASS

Required Fields (overall): 1.0 (>= 0.98) -> PASS
Topic non-empty ratio: 1.0 (>= 0.90) -> PASS
Persona tags ratio: 1.0 (>= max(0.60, baseline-0.10)) -> PASS
Date invalid count: 0 (== 0) -> PASS

SEC Item Coverage (median): 0.9968 (>= max(0.75, baseline-0.10)) -> PASS
Breakdown by doctype:
- 10-K: publish_date_presence 0.0, title_presence 0.0
- 10-Q: publish_date_presence 1.0, title_presence 1.0
- 8-K: publish_date_presence 1.0, title_presence 1.0
- ars_pdf: publish_date_presence 1.0, title_presence 1.0
- press: publish_date_presence 1.0, title_presence 1.0
- product: publish_date_presence 1.0, title_presence 1.0
- dev_docs: publish_date_presence 1.0, title_presence 1.0
- help_docs: publish_date_presence 0.0, title_presence 0.0
- wiki: publish_date_presence 0.0, title_presence 0.0

Uniqueness:
- doc_id_unique: True -> PASS
- final_url_collisions: 0

Proceed? (Y/N): Y
