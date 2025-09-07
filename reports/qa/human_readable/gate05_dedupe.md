# Gate G05 — Deduplication QA (Run 2025-09-07T22:26:47.698859+00:00)
Summary: PASS

- global_duplicate_ratio: 0.1424 (<= 0.15) -> PASS
- non_adjacent_jaccard_p95: 0.2541 (<= 0.30) -> PASS
- coverage_ratio_median_overall: 1.1542 (>= 0.90) -> PASS
By Doctype Coverage Ratios (median):
- press: 1.1564318034906271
- 10-K: 0.008007961312994666
- 10-Q: 0.011845470453627676
- 8-K: 0.7171592775041051
- ars_pdf: 0.0006122889356422291
- product: 1.166400850611377
- dev_docs: 1.7178378378378378
- help_docs: 1.0
- wiki: 1.0754766600920447

Top duplicate clusters:
- crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0004 ← ['crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0000', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0001', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0002', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0003', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0005', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0006', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0007', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0008', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0009', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0010', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0011', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0012', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0013', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0014', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0015', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0016', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0017', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0018', 'crm::wiki::unknown::salesforce-wikipedia::6b727edd::chunk0019'] (size 20)
- crm::press::2025-06-23::news-details::1ee81dec::chunk0002 ← ['crm::press::2025-06-23::salesforce-launches-agentforce-3-to-solve-the-biggest-blockers-to-scaling-ai-age::9aa8f4e2::chunk0002', 'crm::press::2025-06-23::salesforce-launches-agentforce-3-to-solve-the-biggest-blockers-to-scaling-ai-age::b1bdb2c2::chunk0002'] (size 3)
- crm::press::2024-10-31::news-details::c5011c5a::chunk0012 ← ['crm::press::2025-09-03::news-details::2014af2d::chunk0012'] (size 2)
- crm::press::2024-10-31::news-details::c5011c5a::chunk0013 ← ['crm::press::2025-09-03::news-details::2014af2d::chunk0013'] (size 2)
- crm::press::2025-08-19::new-ways-to-pay-make-it-easier-than-ever-to-get-started-with-agentforce::d1363ec8::chunk0000 ← ['crm::press::2025-08-19::new-ways-to-pay-make-it-easier-than-ever-to-get-started-with-agentforce::d1363ec8::chunk0001'] (size 2)

Proceed? (Y/N): Y
