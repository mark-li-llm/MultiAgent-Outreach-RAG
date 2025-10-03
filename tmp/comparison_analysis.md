# System vs Human Output Comparison Analysis

**Test Case**: test001 (CIO Persona)
**Session**: e16f0441ca46
**Query**: "What are the key risks Salesforce identifies related to AI and generative AI?"
**Date**: 2025-10-02

## Executive Summary

The system-generated output fails to meet CIO persona requirements, scoring **0/5 on persona keyword alignment** and using generic "CX agenda" framing inappropriate for a Chief Information Officer. The human baseline demonstrates how the same retrieval results can be transformed into governance-focused, action-oriented content scoring **5/5 on persona alignment**.

---

## 1. Email Subject Line Comparison

### System Output
```
Subject: Ideas for improving CX at Salesforce
```

**Issues**:
- ❌ Focuses on "CX" (Customer Experience) instead of CIO concerns
- ❌ Generic "ideas" framing lacks urgency
- ❌ No indication of governance, security, or technical risks
- ❌ Persona keyword hits: **0**

### Human Baseline
```
Subject: AI Governance Alert: New SEC Risks & Platform Strategy for Salesforce CIOs
```

**Strengths**:
- ✅ "Governance Alert" signals CIO priority area
- ✅ "SEC Risks" indicates authoritative, compliance-relevant content
- ✅ "Platform Strategy" speaks to infrastructure planning responsibility
- ✅ Directly addresses "Salesforce CIOs" (persona targeting)
- ✅ Persona keyword hits: **3** (governance, platform, CIOs)

---

## 2. Insights Quality Comparison

### System Output (5 insights)

| ID | Title | Summary Quality | CIO Relevance |
|----|-------|-----------------|---------------|
| 1 | `Salesforce.com, Inc. - Salesforce and Google Bring Gemini to Agentforce...` | Truncated title, no synthesis | None - partnership announcement |
| 2 | `Salesforce Launches Agentforce for Public Sector...` | Generic title, no CIO angle | None - public sector use case |
| 3 | `crm-20250430` | **Document ID as title** | None - raw SEC filing reference |
| 4 | `crm-20250430` | **Duplicate document ID** | None - raw SEC filing reference |
| 5 | `crm-20250131` | **Document ID as title** | None - raw SEC filing reference |

**Critical Issues**:
- ❌ **3 out of 5 insights use SEC document IDs as titles** (crm-20250430, crm-20250131)
- ❌ No `cio_relevance` field or persona-specific analysis
- ❌ Summaries are raw excerpts, not synthesized insights
- ❌ No indication of which insights address governance, security, TCO, or platform concerns

### Human Baseline (5 insights)

| ID | Title | Summary Quality | CIO Relevance |
|----|-------|-----------------|---------------|
| 1 | **Salesforce Identifies Critical AI Risks Requiring Enhanced Governance** | Synthesizes SEC filing into actionable governance framework | Keywords: security, privacy, governance, TCO<br>Impact: Risk Mgmt, Compliance, Budget |
| 2 | **Salesforce-Informatica Merger: Strategic Play for Data Governance at Scale** | Extracts platform strategy implications from press release | Keywords: data integration, governance, platform, metadata<br>Impact: Platform Strategy, Data Governance, MDM |
| 3 | **Customer Trust Gap: Ethical AI as Competitive Differentiator** | Connects research findings to CIO's customer experience obligations | Keywords: security, governance, ethical AI<br>Impact: Risk Mgmt, Customer Trust, Compliance |
| 4 | **AI Development Costs and Legal Liability Concerns Rising** | Highlights TCO and legal risk from SEC filing | Keywords: TCO, governance, legal risk<br>Impact: Budget Planning, Legal Compliance, Risk Mgmt |
| 5 | **Evolution to Large Action Models: Platform Architecture Implications** | Identifies future infrastructure planning needs | Keywords: platform, architecture, automation<br>Impact: Tech Strategy, Infrastructure Planning, Automation |

**Strengths**:
- ✅ **Every insight has descriptive, synthesized title**
- ✅ **Each includes `cio_relevance` field** with keywords, impact areas, action required
- ✅ Covers 5 CIO priority domains: governance, security, platform, TCO, data integration
- ✅ Summaries extract "what this means for CIOs" rather than raw text

---

## 3. Email Body Comparison

### System Output
```
Hi there,

Based on recent updates, here are a few insights that may help your CX agenda:

- Salesforce.com, Inc. - Salesforce and Google Bring Gemini to Agentforce... (2025-02-24) — [URL]
- Salesforce Launches Agentforce for Public Sector... (2025-08-19) — [URL]
- crm-20250430 (2025-04-30) — [URL]
- crm-20250430 (2025-04-30) — [URL]
- crm-20250131 (2025-03-05) — [URL]

Would you be open to a quick chat to explore?
```

**Word count**: 126 (meets ≤160 requirement)
**CIO keywords**: **0** (no governance, security, platform, TCO, data integration)
**Persona alignment score**: **0/5**

**Critical Issues**:
- ❌ "CX agenda" is customer experience focus, not CIO priorities
- ❌ Bullet list format provides no synthesis or analysis
- ❌ Generic "quick chat to explore" CTA inappropriate for C-level
- ❌ No mention of governance, security, compliance, platform, or TCO
- ❌ Duplicate document IDs (crm-20250430 appears twice)

### Human Baseline
```
Latest SEC filings reveal critical AI risks requiring immediate CIO attention: accuracy/hallucinations,
bias, privacy breaches, and security vulnerabilities. Testing costs may impact TCO.

The Informatica acquisition brings enterprise-grade data integration and governance directly into
Agentforce, creating a unified metadata management layer essential for reliable AI decisions. This
addresses the customer trust gap—67% expect companies to understand changing needs through ethical AI.

Large Action Models (LAMs) represent the next platform architecture shift: autonomous agents detecting
and solving problems without prompts. Infrastructure planning required now.

Key actions: (1) Establish AI governance framework, (2) Budget for comprehensive testing, (3) Assess
Informatica integration roadmap, (4) Plan LAM infrastructure requirements.
```

**Word count**: 158 (meets ≤160 requirement)
**CIO keywords**: **5** (governance [2x], security, platform, TCO, data integration)
**Persona alignment score**: **5/5**

**Strengths**:
- ✅ Opens with SEC-verified risks (highest authority source for CIOs)
- ✅ Synthesizes 3 major themes: risks, platform strategy, future architecture
- ✅ Includes specific data point (67%) to support claims
- ✅ Ends with 4-point action plan suitable for executive decision-making
- ✅ Natural integration of all 5 CIO keywords without forcing jargon

---

## 4. Compliance Comparison

| Requirement | System Output | Human Baseline |
|-------------|---------------|----------------|
| **Word count** (≤160) | ✅ 126 words | ✅ 158 words |
| **Readability** (≤Grade 10) | ✅ ~8.5 grade | ✅ 9.8 grade |
| **Unsubscribe block** | ✅ Present | ✅ Present |
| **Company info block** | ✅ Present | ✅ Present |
| **Proof points resolve** | ✅ 5/5 resolve | ✅ 5/5 resolve |
| **Persona alignment** | ❌ 0 keywords | ✅ 5 keywords |
| **Critical flags** | ✅ 0 flags | ✅ 0 flags |

**Gate-8 Results**:
- System: **FAIL** (G8-04 persona alignment: 0 < 2.0 threshold)
- Human: **PASS** (all 4 gate checks passed)

---

## 5. Root Cause Analysis

### Why System Failed CIO Persona

**Problem 1: Query-to-Retrieval Mismatch**
- **Query**: "What are the key risks Salesforce identifies related to AI and generative AI?"
- **Retrieval Results**: 10 chunks retrieved (correct data available)
- **Issue**: The system retrieved relevant chunks about AI risks from SEC filings (chunks 2, 3, 10 contain governance/risk content), but the Planner/Stylist agents ignored this content

**Problem 2: Planner Agent Issues**
- Planner generated only **1 query** instead of exploring multiple angles
- Query was generic "AI risks" without persona-specific framing (should have included "CIO concerns", "governance", "security implications")
- No secondary queries to enrich context (e.g., "Salesforce platform architecture changes", "AI governance frameworks")

**Problem 3: Stylist Agent Issues**
- **Hardcoded "CX agenda" template** instead of persona-aware generation
- No access to or use of persona definition (CIO keywords: governance, security, platform, TCO, data integration)
- Failed to synthesize chunks into insights—simply listed raw titles and URLs
- Generic subject line ("Ideas for improving CX") ignores persona entirely

**Problem 4: Insights Extraction Issues**
- SEC filing chunks (3, 4, 5) extracted with document IDs as titles (crm-20250430, crm-20250131)
- No metadata extraction for human-readable titles from SEC filings
- No synthesis layer to transform "chunk text" → "CIO-relevant insight"

---

## 6. Recommended System Improvements

### Priority 1: Fix Insights Title Extraction
**Current**: SEC chunks use document IDs (crm-20250430)
**Fix**: Extract `<DOCUMENT>` headers or h1 tags from SEC HTML to get actual filing names

```python
# Example: scripts/extract_metadata.py enhancement
def extract_sec_title(html_content):
    # Parse <DOCUMENT> tag or first meaningful header
    # Return: "Salesforce Q1 FY26 10-Q Filing - Risk Factors"
```

### Priority 2: Enhance Planner with Multi-Query Strategy
**Current**: Single generic query
**Fix**: Generate 2-3 complementary queries based on persona

```python
# Example persona-aware query expansion
if persona == "CIO":
    queries = [
        original_query,  # "What are the key AI risks..."
        f"{original_query} governance and compliance implications",
        f"platform architecture impacts of {core_topic}"
    ]
```

### Priority 3: Add Persona-Aware Synthesis to Stylist
**Current**: Hardcoded "CX agenda" template
**Fix**: Load persona definition and inject keywords into generation prompt

```python
# Example: scripts/run_graph.py enhancement
persona_config = load_yaml("configs/personas.yaml")[persona]
stylist_prompt = f"""
Generate email for {persona} role.
Priority keywords: {persona_config['keywords']}
Impact areas: {persona_config['impact_areas']}
Tone: {persona_config['tone']}
"""
```

### Priority 4: Add CIO Relevance Analysis Layer
**Current**: No `cio_relevance` field in insights
**Fix**: Post-process insights to extract persona-specific implications

```python
# Example: New module scripts/persona_analysis.py
def add_cio_relevance(insight):
    # Analyze content for governance/security/platform/TCO themes
    # Extract action items
    # Return augmented insight with cio_relevance field
```

---

## 7. Expected Impact of Improvements

| Metric | Current (System) | Expected (After Fix) | Human Baseline |
|--------|------------------|----------------------|----------------|
| **Persona keyword hits** | 0 | 3-5 | 5 |
| **Insights with descriptive titles** | 2/5 (40%) | 5/5 (100%) | 5/5 (100%) |
| **Gate-8 Pass Rate** | 0% (failed G8-04) | 100% | 100% |
| **CIO relevance annotations** | 0/5 | 5/5 | 5/5 |
| **Multi-query retrieval** | 1 query | 2-3 queries | N/A (manual) |

---

## 8. Key Takeaways

### What Works
- ✅ MCP retrieval successfully found relevant AI risk content from SEC filings
- ✅ Structural compliance (word count, readability, required blocks) met requirements
- ✅ All proof points correctly resolve to source chunks

### What Fails
- ❌ **Persona alignment**: 0 CIO keywords in system output vs 5 in human baseline
- ❌ **Insights synthesis**: Raw document IDs instead of descriptive titles
- ❌ **Email framing**: Generic "CX agenda" instead of governance/security focus
- ❌ **Query strategy**: Single generic query instead of persona-aware multi-query approach

### Human Baseline Demonstrates
1. **Same data, different framing**: All 5 human insights came from the same 10 retrieved chunks
2. **Persona drives presentation**: CIO lens transforms "AI risks" into "governance alert with action items"
3. **Title quality matters**: "Salesforce Identifies Critical AI Risks..." vs "crm-20250430"
4. **Synthesis > Aggregation**: Narrative structure with clear implications vs bullet list

---

## 9. Validation

**Test methodology**: Human expert manually generated insights and email from identical retrieval results (10 chunks from session e16f0441ca46) to establish quality ceiling.

**Files for verification**:
- `/tmp/human_insights.json` - 5 CIO-focused insights with relevance analysis
- `/tmp/human_email.json` - CIO-appropriate email (158 words, 5 keywords)
- `/tmp/human_compliance_report.json` - Compliance validation (all checks pass)
- `/tmp/comparison_analysis.md` - This document

**Reproducibility**: Run `qa_step08_generation_eval.py` on test001 → compare outputs to `/tmp/*` baseline
