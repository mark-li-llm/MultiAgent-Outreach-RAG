# Gate G01 — Collection QA (Phase B; Run 2025-09-07T16:58:13.545888+00:00)
Summary: FAIL

Coverage by Source (target → actual)
- SEC: 6–7 → 6
- Investor PR: 20 (min 16) → 1
- Newsroom PR total: 30 (min 24) → 46
  - Corporate feed ≥10 → 10
  - Product feed ≥10 → 10
- Product+Dev+Help+Wiki total: 10±1 → 9

HTTP 200 ratio: 1.0 (threshold ≥0.99)
Raw exact duplicate rate: 0.0 (threshold ≤0.05)
PR recency: within 24mo 0.8043 (≥0.70), within 12mo 0.7826 (≥0.40)
Missing PR dates excluded from ratio: 1

Failures & Actions:
- COL-002 ir_docs_count: actual 1, threshold >=16. Increase --limit and verify date filtering.

Proceed? (Y/N): N
