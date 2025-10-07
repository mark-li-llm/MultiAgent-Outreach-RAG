# Ground Truth Correction Report
**Date**: 2025-10-07
**Scope**: 10-K/10-Q SEC filing queries (9 out of 46 total queries)

---

## Executive Summary

Verified all 10 SEC filing queries in the Gate-7 evaluation seed. Found **9 out of 10 queries had incorrect ground truth** annotations pointing to chunks that did not contain the answer. All 9 queries have been corrected with manually verified chunk IDs.

### Status
- ✅ **Corrected**: 9 queries
- ✅ **Already correct**: 1 query
- 📦 **Backup created**: `salesforce_eval_seed.jsonl.backup_20251007`
- 📝 **Original file renamed**: `salesforce_eval_seed.jsonl.old`

---

## Detailed Corrections

### 1. ✅ 10q_q1_revenue
**Query**: "What was Salesforce's total revenue for Q1 FY26?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0004 | **chunk0008** |
| Issue | XBRL metadata tags (no revenue data) | ✓ Contains "Total revenues $9,829" |

**Verification**: chunk0008 contains the Condensed Consolidated Statements of Operations with clear revenue breakdown:
- Subscription and support: $9,297M
- Professional services: $532M
- **Total revenues: $9,829M**

---

### 2. ✅ 10q_operating_cash
**Query**: "How much operating cash flow did Salesforce generate in Q1 FY26?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0031 | **chunk0009** |
| Issue | Litigation disclaimers (no cash flow) | ✓ Contains "Net cash provided by operating activities 6,476" |

**Verification**: chunk0009 contains the complete Consolidated Statements of Cash Flows showing:
- **Net cash provided by operating activities: $6,476M**

---

### 3. ✅ 10q_senior_notes
**Query**: "When do Salesforce's senior notes mature and what are the interest rates?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0019 | **chunk0049** |
| Issue | Asset impairment discussion (no debt info) | ✓ Contains complete senior notes table |

**Verification**: chunk0049 contains a full table with:

| Instrument | Maturity | Principal | Rate |
|------------|----------|-----------|------|
| 2028 Senior Notes | April 2028 | $1,500M | 3.70% |
| 2031 Senior Notes | July 2031 | $1,500M | 1.95% |
| 2051 Senior Notes | July 2051 | $2,000M | 2.90% |

---

### 4. ✅ 10q_share_repurchase
**Query**: "What is the status of Salesforce's share repurchase program?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0032 | **chunk0046** |
| Issue | Slack litigation proceedings (no repurchase) | ✓ Contains "$30.0 billion...Share Repurchase Program" |

**Verification**: chunk0046 explicitly states:
> "Our Board of Directors authorized a program to repurchase shares...for an aggregate total authorization of $30.0 billion"

---

### 5. ✅ 10q_ai_risks
**Query**: "What are the key risks Salesforce identifies related to AI and generative AI?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0050 | **chunk0070** |
| Issue | Counterparty credit risk (no AI content) | ✓ Contains AI risks discussion |

**Verification**: chunk0070 discusses AI risks:
> "We are increasingly building AI into many of our offerings, including generative and agentic AI...present emerging ethical issues...perceived or actual impact on human rights, privacy, employment..."

---

### 6. ✅ 10q_data_privacy
**Query**: "What regulatory compliance requirements does Salesforce face regarding data protection?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0051 | **chunk0073** |
| Issue | Strategic investment valuations (no GDPR) | ✓ Contains GDPR and data privacy framework |

**Verification**: chunk0073 discusses:
> "adopted the EU-U.S. Data Privacy Framework to foster EU-to-U.S. data transfers...CJEU decision...certain countries...have also passed...laws requiring...local data residency"

---

### 7. ✅ 10k_fy25_performance
**Query**: "What was Salesforce's full year FY25 financial performance?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0006 | **chunk0079** |
| Issue | XBRL metadata (no performance data) | ✓ Contains FY25 revenue table |

**Verification**: chunk0079 contains the full fiscal year revenue table:
- FY 2025 Total revenues: **$37,895M** (+9%)

---

### 8. ✅ 10k_agentforce_strategy
**Query**: "How does Salesforce describe its Agentforce AI agent strategy in the annual report?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0008 | **chunk0014** |
| Issue | Stock compensation XBRL tags (no Agentforce) | ✓ Contains Slack+Agentforce integration description |

**Verification**: chunk0014 describes:
> "Slack...is also deeply integrated with every Salesforce offering, including Agentforce, bringing a digital labor force in..."

---

### 9. ✅ 10k_sales_cloud_offerings
**Query**: "What capabilities does Salesforce Sales Cloud offer according to the 10-K?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0009 | **chunk0013** |
| Issue | Stock exchange listing info (no Sales Cloud) | ✓ Contains Sales Cloud product description |

**Verification**: chunk0013 describes:
> "Our Sales offering is an integrated platform that brings together the power of humans with AI agents...provides sales capabilities and tools..."

---

### 10. ✅ 8k_fy25_q4_announcement
**Query**: "What earnings announcement did Salesforce make in the February 2025 8-K filing?"

| Field | Before | After |
|-------|--------|-------|
| Expected chunk | chunk0000 | chunk0000 |
| Status | **Already correct** | ✓ No change needed |

---

## Impact Analysis

### Before Correction
- **Recall@10 would be artificially low**: Even perfect retrieval would be marked as failure
- **Evaluation metrics unusable**: nDCG, precision based on wrong ground truth
- **Misleading failure logs**: Actual successes recorded as failures

### After Correction
- **Valid evaluation baseline**: Ground truth now matches actual content
- **Reliable metrics**: Can trust Gate-7 recall/nDCG measurements
- **Actionable insights**: Failure logs now identify real retrieval issues

---

## Files Modified

| File | Status | Description |
|------|--------|-------------|
| `salesforce_eval_seed.jsonl` | ✅ Updated | Active eval seed with 9 corrections |
| `salesforce_eval_seed.jsonl.old` | 📦 Backup | Original file with wrong GT |
| `salesforce_eval_seed.jsonl.backup_20251007` | 📦 Backup | Timestamped backup |

---

## Verification Method

For each query:
1. Read the expected chunk content
2. Check if answer keywords appear in chunk
3. If <50% keywords match, search for correct chunk
4. Manually verify correct chunk contains full answer
5. Update expected_chunk_id and add correction note

---

## Recommendations

### Immediate
- ✅ **Done**: Use corrected seed file for all future Gate-7 runs
- ⚠️ **Important**: Re-run Gate-7 evaluation with corrected ground truth
- 📊 **Action**: Compare new recall@10 with previous results

### Short-term
- Verify remaining 36 queries (press releases, product docs, etc.)
- Add automated validation script to CI/CD

### Long-term
- Implement ground truth validation on file save
- Add chunk content preview in annotation tools
- Version-lock eval documents to prevent chunk ID drift

---

## Next Steps

1. **Re-run Gate-7** with corrected seed:
   ```bash
   conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py
   ```

2. **Compare results**:
   - Old recall@10 (with wrong GT)
   - New recall@10 (with correct GT)
   - Expected improvement: significant increase

3. **Verify remaining queries**: Run `scripts/fix_ground_truth.py` on queries 11-46

---

## Notes

- All corrections manually verified by reading actual chunk content
- Each corrected query has `[GT corrected: old -> new]` note appended
- Press release, product, dev docs, wiki queries not yet verified (assumed OK for now)
- Original backup preserved in multiple locations for safety
