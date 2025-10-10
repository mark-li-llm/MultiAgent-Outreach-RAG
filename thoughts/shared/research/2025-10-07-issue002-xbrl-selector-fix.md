---
date: 2025-10-07
status: resolved
tags: [xbrl, normalization, css-selectors, sec-filings, fix]
related_issues: [issue005, issue003, issue004]
related_plans: [2025-10-07-fix-xbrl-metadata-pollution]
---

# XBRL CSS Selector Fix: Validation and Implementation

## Problem

The XBRL fix plan proposed using CSS selectors like `ix\\:header` to remove XBRL metadata from SEC filings. However, testing revealed that **BeautifulSoup's CSS selector parser (soupsieve) rejects escaped colons** when using double-backslash syntax in YAML config.

### Root Cause

YAML single-quoted strings treat backslashes literally:
- YAML: `'ix\\:header'` → Python string: `ix\\:header` (double backslash)
- soupsieve interprets this as trying to escape `:header` as a pseudo-class
- Error: `SelectorSyntaxError: ':header' was detected as a pseudo-class`

## Solution

Use **CSS hex escape** for the colon character:
- Colon (`:`) → Hex code `\3A`
- Format: `ix\3A header` (space required after hex code)

This is the standard CSS escape syntax and is universally supported by soupsieve.

## Implementation

### Updated normalization.rules.yaml

```yaml
remove_selectors:
  # ... existing selectors ...
  # XBRL Inline format metadata removal (SEC filings)
  # Note: Colon escaped as \3A (hex code) for CSS selector compatibility
  - 'ix\3A header'       # XBRL header container (removes all nested children)
  - 'ix\3A hidden'       # Hidden metadata section
  - 'ix\3A resources'    # Context/unit definitions
  - 'ix\3A references'   # Schema references
  - 'xbrli\3A context'   # Individual context definitions
  - 'xbrli\3A unit'      # Unit definitions
```

### Validation Results

**Test File**: `scripts/test_xbrl_removal_e2e.py`

**Input**:
- File: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.raw.html`
- Size: 2,454,201 bytes
- XBRL tags: 2,124 `us-gaap:` occurrences, 748 `xbrli:context` definitions

**Output**:
- Size: 396,532 characters
- Reduction: **83.8%**
- XBRL artifacts: **0** (complete removal)
- Financial content: ✓ Preserved (revenue, income, fiscal year, consolidated, financial, statements, operations)

**First 500 chars of normalized text**:
```
crm-20250131
Table of Contents
UNITED STATES
SECURITIES AND EXCHANGE COMMISSION
Washington, D.C. 20549
FORM 10-K
☒ Annual report pursuant to Section 13 or 15(d) of the Securities Exchange Act of 1934
For the fiscal year ended January 31, 2025
Commission File Number: 001-32224
Salesforce, Inc.
(Exact name of Registrant as specified in its charter)
Delaware
...
```

## Alternative Approaches Tested

| Approach | Syntax | Result |
|----------|--------|--------|
| Single backslash | `ix\:header` | ✓ Works (but YAML quoting issue) |
| Double backslash | `ix\\:header` | ✗ Fails (pseudo-class error) |
| Hex escape | `ix\3Aheader` | ✓ Works |
| Hex with space | `ix\3A header` | ✓ Works (selected) |
| find_all() fallback | `find_all('ix:header')` | ✓ Works (requires code change) |

## Impact on XBRL Fix Plan

### What Changed
- **Line 176-195 of plan**: Replace `ix\\:header` syntax with `ix\3A header` hex escape syntax
- **Phase 1 success criteria**: Updated config validates and loads correctly ✓

### What Remains Unchanged
- Phase 2-6 execution steps (normalization, chunking, embeddings, indexing, Gate-7 eval)
- Expected metrics and success criteria
- Rollback strategy

## Next Steps

Execute the XBRL fix plan starting from **Phase 2** (Re-run Normalization):

```bash
# Phase 2: Clear and re-normalize SEC filings
cp -r data/interim/normalized data/backup/normalized_$(date +%Y%m%d_%H%M%S)
rm -f data/interim/normalized/crm::10-*.json data/interim/normalized/crm::8-*.json
conda run -n age python scripts/normalize_html.py --phase B

# Verify XBRL removal
grep -c 'us-gaap:' data/interim/normalized/crm::10-K*.json  # Should return 0
jq -r '.text' data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json | head -c 5000

# Phase 3-6: Continue with chunking, embeddings, indexing, evaluation
# (as documented in plan)
```

## References

- **Plan**: `thoughts/shared/plans/2025-10-07-fix-xbrl-metadata-pollution.md`
- **Test Scripts**:
  - `scripts/test_xbrl_selector_syntax.py` (initial validation)
  - `scripts/test_selector_escaping.py` (escape strategy comparison)
  - `scripts/test_xbrl_removal_e2e.py` (end-to-end validation)
- **CSS Hex Escape Spec**: https://www.w3.org/TR/CSS2/syndata.html#characters (Section 4.1.3)
- **soupsieve Documentation**: https://facelessuser.github.io/soupsieve/selectors/

## Conclusion

The CSS selector syntax issue is **resolved** using hex escape (`\3A`). The config-only fix approach from the original plan is **preserved** (no code changes required). The end-to-end test confirms complete XBRL removal with zero impact on financial content.

Status: ✅ **Ready for Phase 2 execution**
