---
date: 2025-10-07
status: draft
tags: [plan, sec-filings, xbrl, normalization, chunking, issue005]
related_issues: [issue005, issue003, issue004]
---

# Fix XBRL Inline Tags Metadata Pollution in SEC Filings

## Overview

SEC 10-K/10-Q filings use XBRL Inline format with `<ix:hidden>` sections containing thousands of metadata tags (CIK numbers, GAAP taxonomy URLs, date ranges, context definitions). The current `normalize_html.py` implementation uses BeautifulSoup's `.get_text()` method, which correctly removes XML tags but **preserves tag content**, causing XBRL metadata to pollute the first 2000+ characters of normalized text. This pushes actual financial content (revenue tables, MD&A sections) into later chunks, resulting in 0% chunk-level recall for SEC filing queries in Gate-7 evaluation.

**Problem Magnitude**:
- 10-K filing: First chunk (0-3773 chars) contains **only XBRL metadata noise**
- Actual financial statements appear in chunks 7+
- Gate-7 metrics: 10-K/10-Q filings achieve **0% chunk-level recall** (0/9 queries) vs 44.4% document-level recall (4/9)
- All 4 SEC filings in `data/raw/sec/` are affected

## Current State Analysis

### XBRL Inline Format Structure

SEC filings embed structured data using XBRL Inline (iXBRL) format:

```html
<body>
  <div style="display:none">
    <ix:header>
      <ix:hidden>
        <ix:nonNumeric name="dei:EntityCentralIndexKey">0001108524</ix:nonNumeric>
        <ix:nonNumeric name="dei:DocumentPeriodEndDate">2025-04-30</ix:nonNumeric>
        <ix:nonNumeric name="us-gaap:CostOfSalesMember">2025-02-01</ix:nonNumeric>
        <!-- Thousands of similar tags -->
      </ix:hidden>
      <ix:resources>
        <xbrli:context id="c-1">...</xbrli:context>
        <xbrli:context id="c-2">...</xbrli:context>
        <!-- Hundreds of context definitions -->
      </ix:resources>
    </ix:header>
  </div>

  <!-- Actual human-readable financial statements appear AFTER metadata -->
  <div class="financial-statements">
    <h1>Consolidated Statements of Operations</h1>
    ...
  </div>
</body>
```

**Key Issue**: The `<div style="display:none">` wrapper hides metadata visually but NOT during text extraction.

### Current Normalization Behavior

**File**: `scripts/normalize_html.py:83-137`

**Process**:
1. Line 85: `soup_all = BeautifulSoup(html, "html.parser")`
2. Lines 88-95: Optionally select preserved containers
3. Line 99: `before_text = kept_before.get_text("\n")`
4. Lines 104-107: Remove unwanted selectors (nav, footer, scripts, styles)
5. Line 129: `text = kept.get_text("\n")` ← **XBRL metadata extracted here**
6. Lines 130-133: Whitespace normalization

**Critical Gap**: No removal of `<ix:hidden>`, `<ix:header>`, or `<div style="display:none">` elements before text extraction.

**Result** (from `data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json`):
```json
{
  "text": "H1: crm-20250131\n\ncrm-20250131\n0001108524\nFALSE\nFY\n2025\nhttp://fasb.org/us-gaap/2024#PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization\n...\n(thousands of lines of XBRL identifiers)\n...\n(actual financial content appears much later)"
}
```

### Chunk-Level Impact

**File**: `data/interim/chunks/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.chunks.jsonl`

**Chunk 0** (first 3773 characters):
- 504 words extracted
- 1567 tokens counted
- **Content**: 100% XBRL metadata (CIK numbers, taxonomy URLs, date ranges, context IDs)
- **Useful financial information**: 0%

**Embedding Consequence**:
- OpenAI ada-002 embeds chunk 0 as "metadata soup"
- Queries like "What was Q1 FY26 revenue?" retrieve wrong chunks
- Expected financial statement chunks ranked too low (rank > 10)
- **Gate-7 Failure**: 0/9 SEC queries succeed at chunk-level (recall@10 = 0%)

## Desired End State

### Success Criteria

#### Automated Verification:
- [ ] Normalization stage removes all XBRL metadata: `grep -c 'us-gaap:' data/interim/normalized/crm::10-K*.json` returns 0
- [ ] First chunk contains actual financial content: inspect `head -1 data/interim/chunks/crm::10-K*.chunks.jsonl | jq -r '.text' | head -20` for readable statements
- [ ] All 4 SEC filings pass normalization: `python3 scripts/normalize_html.py --phase B` exits with status 0
- [ ] Gate-1 embeddings generation succeeds: `conda run -n age python scripts/qa_step01_embeddings.py` completes without errors
- [ ] Gate-2 FAISS index builds: `conda run -n ageFaiss python scripts/qa_step02_indexes.py` passes all checks
- [ ] Gate-7 chunk recall improves: `conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py` shows recall@10 > 0% for SEC queries

#### Manual Verification:
- [ ] Inspect normalized JSON for 10-K filing: verify no XBRL taxonomy URLs in first 5000 characters
- [ ] Review first 3 chunks of 10-K: confirm presence of readable financial statements (revenue tables, MD&A text)
- [ ] Compare before/after word counts: verify XBRL removal doesn't delete actual content (word count should decrease by ~10-15%)
- [ ] Check Gate-7 failure log: verify SEC query failures shift from `chunk_miss_doc_hit_far` to chunk hits

## What We're NOT Doing

- **Not** parsing XBRL semantically (extracting structured financial data as JSON)
- **Not** modifying FAISS index structure or embedding model
- **Not** changing chunking parameters (target_tokens, overlap_tokens remain unchanged)
- **Not** altering router heuristics or query routing logic
- **Not** creating separate normalization pipeline for SEC filings (solution applies universally via remove_selectors config)

## Implementation Approach

### Strategy

Add XBRL-specific removal selectors to `configs/normalization.rules.yaml` to strip hidden metadata sections **before** text extraction. This is the minimal, config-driven approach that:
1. Leverages existing removal infrastructure (`normalize_html.py:104-107`)
2. Requires zero code changes (pure configuration update)
3. Applies automatically to all future SEC filings
4. Maintains backward compatibility with non-SEC documents

### Alternative Approaches Considered

**Alternative 1**: Remove `<div style="display:none">` via CSS selector
- **Pros**: Simple, catches XBRL and other hidden content
- **Cons**: May remove legitimate hidden elements (accessibility content, print-only sections)
- **Decision**: Too broad; could delete useful content

**Alternative 2**: SEC-specific preprocessing in `normalize_html.py`
- **Pros**: Can implement sophisticated XBRL parsing
- **Cons**: Code complexity, violates existing config-driven design
- **Decision**: Over-engineered for this problem

**Alternative 3**: Post-normalization text filtering (regex-based cleanup)
- **Pros**: Doesn't touch normalization pipeline
- **Cons**: Hard to maintain regex patterns, fragile against XBRL schema changes
- **Decision**: Doesn't address root cause

**Selected Approach**: Configuration-based removal of XBRL namespaced tags
- **Pros**: Surgical precision, zero code changes, maintainable, extensible
- **Cons**: None identified

---

## Phase 1: Configuration Update

### Overview
Add XBRL tag removal selectors to normalization rules configuration.

### Changes Required

#### 1. Normalization Rules Configuration
**File**: `configs/normalization.rules.yaml`

**Current remove_selectors** (lines 1-11):
```yaml
remove_selectors:
  - script
  - style
  - nav
  - footer
  - "[role='navigation']"
  - "[role='banner']"
  - "[role='contentinfo']"
  - ".sidebar"
  - ".breadcrumb"
  - "[aria-label*='cookie']"
```

**Add XBRL-specific selectors**:
```yaml
remove_selectors:
  - script
  - style
  - nav
  - footer
  - "[role='navigation']"
  - "[role='banner']"
  - "[role='contentinfo']"
  - ".sidebar"
  - ".breadcrumb"
  - "[aria-label*='cookie']"
  # XBRL Inline format metadata removal
  - "ix\\:header"           # XBRL header container
  - "ix\\:hidden"           # Hidden metadata section
  - "ix\\:resources"        # Context/unit definitions
  - "ix\\:references"       # Schema references
  - "xbrli\\:context"       # Individual context definitions
  - "xbrli\\:unit"          # Unit definitions
```

**Rationale**:
- BeautifulSoup CSS selectors require escaping colons in namespaced tags: `ix:header` → `ix\\:header`
- Removes entire XBRL metadata trees before `.get_text()` extraction
- No impact on non-SEC documents (selectors match nothing if tags absent)

### Success Criteria

#### Automated Verification:
- [ ] Configuration file validates: `python3 -c "import yaml; yaml.safe_load(open('configs/normalization.rules.yaml'))"`
- [ ] Updated config loads without errors: `python3 scripts/normalize_html.py --dry-run --phase B --limit 1`

#### Manual Verification:
- [ ] YAML syntax correct (no parsing errors when loading config)
- [ ] Selector escaping correct (BeautifulSoup accepts `ix\\:header` format)

---

## Phase 2: Normalization Re-run

### Overview
Re-normalize all SEC filings with updated XBRL removal rules.

### Changes Required

#### 1. Clear Existing Normalized SEC Files
**Command**:
```bash
# Backup existing normalized files (optional safety measure)
cp -r data/interim/normalized data/backup/normalized_$(date +%Y%m%d_%H%M%S)

# Remove SEC normalized files to force regeneration
rm -f data/interim/normalized/crm::10-K*.json
rm -f data/interim/normalized/crm::10-Q*.json
rm -f data/interim/normalized/crm::8-K*.json
```

**Files Affected**:
- `data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json`
- `data/interim/normalized/crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866.json`
- `data/interim/normalized/crm::8-K::2025-05-28::q1-fy26-results-8-k::35792ff4.json`
- `data/interim/normalized/crm::8-K::2025-06-05::proxy-meeting-results-2025-06-05::47c09586.json`
- `data/interim/normalized/crm::8-K::2025-02-26::fy25-results-8-k::97457068.json`

#### 2. Re-run Normalization Pipeline
**Command**:
```bash
conda run -n age python scripts/normalize_html.py --phase B
```

**Expected Output**:
```
OK crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2 bytes_before=2345678 bytes_after=1987654 retention=0.847 lang=en wc=65234 tok=101234
OK crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866 bytes_before=1234567 bytes_after=1045678 retention=0.847 lang=en wc=42134 tok=65234
...
```

**Validation Checks**:
- Retention ratio ~0.85 (15% reduction from XBRL removal is expected)
- Word count decreases by 10-15% per document
- Token count decreases proportionally
- No `DROPPED_NON_EN` or `DROPPED_SHORT` for SEC filings

#### 3. Verify XBRL Metadata Removal
**Command**:
```bash
# Check for XBRL artifacts in normalized text
grep -c 'us-gaap:' data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json
grep -c 'dei:EntityCentralIndexKey' data/interim/normalized/crm::10-K*.json
grep -c 'xbrli:context' data/interim/normalized/crm::10-Q*.json

# All commands should return 0
```

**Inspect First 5000 Characters**:
```bash
jq -r '.text' data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json | head -c 5000
```

**Expected Content**: Should start with document title and Item 1 (Business), NOT XBRL metadata.

### Success Criteria

#### Automated Verification:
- [ ] Normalization completes successfully for all SEC filings: `echo $?` returns 0 after normalize_html.py
- [ ] Zero XBRL artifacts remain in normalized text: `grep -c 'us-gaap:' data/interim/normalized/crm::*.json` sums to 0
- [ ] Word counts decrease by 10-15%: compare old vs new `.word_count` in normalized JSON
- [ ] No documents dropped: count of normalized SEC files equals count of raw SEC files

#### Manual Verification:
- [ ] First 5000 characters of 10-K normalized text contain readable financial statements
- [ ] Document structure preserved: headings, paragraphs, and tables still present
- [ ] No over-removal: actual financial data (revenue numbers, dates) not deleted

---

## Phase 3: Re-chunk Documents

### Overview
Regenerate chunks from cleaned normalized text.

### Changes Required

#### 1. Clear Existing Chunk Files
**Command**:
```bash
# Backup existing chunks (optional)
cp -r data/interim/chunks data/backup/chunks_$(date +%Y%m%d_%H%M%S)

# Remove SEC chunk files to force regeneration
rm -f data/interim/chunks/crm::10-K*.chunks.jsonl
rm -f data/interim/chunks/crm::10-Q*.chunks.jsonl
rm -f data/interim/chunks/crm::8-K*.chunks.jsonl
```

#### 2. Re-run Chunking Pipeline
**Command**:
```bash
conda run -n age python scripts/chunk_documents.py
```

**Expected Behavior**:
- Chunk count per document may change slightly (XBRL removal shortens documents by ~15%)
- First chunk now starts with actual content instead of metadata noise
- SEC Item boundaries preserved (chunking still respects Item spans from `parse_sec_structures.py`)

#### 3. Verify First Chunk Content
**Command**:
```bash
# Inspect first chunk of 10-K filing
head -1 data/interim/chunks/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.chunks.jsonl | jq -r '.text' | head -50
```

**Expected Content**: Should contain title, Item 1 heading, and business description text.

**Should NOT Contain**:
- `0001108524` (CIK number)
- `http://fasb.org/us-gaap/2024#...` (taxonomy URLs)
- `us-gaap:CostOfSalesMember` (context IDs)
- `iso4217:USD` (unit definitions)

### Success Criteria

#### Automated Verification:
- [ ] Chunking completes successfully: `echo $?` returns 0 after chunk_documents.py
- [ ] First chunk has readable word count: `head -1 data/interim/chunks/crm::10-K*.chunks.jsonl | jq '.word_count'` returns > 100
- [ ] First chunk token count reasonable: `head -1 data/interim/chunks/crm::10-K*.chunks.jsonl | jq '.token_count'` returns 200-1000
- [ ] Total chunk count per doc decreases by ~10-15%: compare old vs new line counts in .chunks.jsonl files

#### Manual Verification:
- [ ] First 3 chunks of 10-K contain financial statements or business description
- [ ] No XBRL metadata visible in any chunk: spot-check 10 random chunks for taxonomy URLs
- [ ] SEC Item structure preserved: chunks still align with Item 1, 1A, 7, 7A, 8 boundaries

---

## Phase 4: Re-generate Embeddings

### Overview
Generate new OpenAI ada-002 embeddings for cleaned chunks.

### Changes Required

#### 1. Clear Existing Embeddings
**Command**:
```bash
# Backup existing embeddings Parquet file (optional)
cp data/vector/embeddings/embeddings.parquet data/backup/embeddings_$(date +%Y%m%d_%H%M%S).parquet

# Remove Parquet file to force full regeneration
rm -f data/vector/embeddings/embeddings.parquet
```

**Note**: The SHA256-based cache in `data/cache/embeddings/` will NOT match new chunk text (XBRL removal changes hash), so new API calls will be made automatically.

#### 2. Re-run Gate-1 Embedding Generation
**Command**:
```bash
conda run -n age python scripts/qa_step01_embeddings.py
```

**Expected Output**:
```json
{
  "gate": "G1",
  "status": "GREEN",
  "checks": {
    "G1-01": {"status": "PASS", "details": "Row count matches baseline chunks"},
    "G1-02": {"status": "PASS", "details": "Vector dimension == 1536"},
    "G1-03a": {"status": "PASS", "details": "Zero vectors: 0"},
    "G1-03b": {"status": "PASS", "details": "NaN vectors: 0"},
    "G1-04": {"status": "PASS", "details": "Outlier percentage: 0.1%"}
  }
}
```

**Cost Estimate**:
- 10-K filing: ~130 chunks × $0.0001/1K tokens × ~800 tokens/chunk ≈ $0.01
- 10-Q filing: ~90 chunks × similar rate ≈ $0.007
- Total for 5 SEC filings ≈ $0.05 (negligible)

### Success Criteria

#### Automated Verification:
- [ ] Gate-1 passes all checks: `jq '.status' reports/qa/step01_embeddings.json` returns `"GREEN"`
- [ ] Embedding dimension correct: `jq '.checks."G1-02".status' reports/qa/step01_embeddings.json` returns `"PASS"`
- [ ] Zero vectors absent: `jq '.checks."G1-03a".status' reports/qa/step01_embeddings.json` returns `"PASS"`
- [ ] Parquet row count matches chunk count: `python3 -c "import pyarrow.parquet as pq; print(len(pq.read_table('data/vector/embeddings/embeddings.parquet')))"`

#### Manual Verification:
- [ ] Embeddings Parquet file size reasonable (~1536 floats × chunk count × 4 bytes/float + overhead)
- [ ] No API errors in logs: check `logs/embeddings/*.log` for rate limits or failures
- [ ] Cache directory populated with new hashes: `ls data/cache/embeddings/ | wc -l` increases

---

## Phase 5: Rebuild FAISS Index

### Overview
Rebuild HNSW index with new embeddings.

### Changes Required

#### 1. Clear Existing FAISS Index
**Command**:
```bash
# Backup existing index files (optional)
cp -r data/vector/faiss data/backup/faiss_$(date +%Y%m%d_%H%M%S)

# Remove index files to force rebuild
rm -f data/vector/faiss/index.faiss
rm -f data/vector/faiss/idmap.parquet
rm -f data/vector/faiss/faiss_manifest.json
```

#### 2. Re-run Gate-2 Index Building
**Command**:
```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Expected Output**:
```json
{
  "gate": "G2",
  "status": "GREEN",
  "checks": {
    "G2-01": {"status": "PASS", "details": "FAISS index built successfully"},
    "G2-02": {"status": "PASS", "details": "ID map row count matches embeddings"},
    "G2-03": {"status": "PASS", "details": "HNSW parameters: M=32, efConstruction=200, efSearch=128"},
    "G2-04": {"status": "PASS", "details": "Index integrity check passed"},
    "G2-05": {"status": "PASS", "details": "Max reconstruction error: 0.0001 <= 0.001"},
    "G2-06": {"status": "PASS", "details": "Sanity search: >= 3 results per query"},
    "G2-07": {"status": "PASS", "details": "Keyword hit validation passed"}
  }
}
```

### Success Criteria

#### Automated Verification:
- [ ] Gate-2 passes all checks: `jq '.status' reports/qa/step02_indexes.json` returns `"GREEN"`
- [ ] FAISS index file exists: `test -f data/vector/faiss/index.faiss && echo PASS`
- [ ] ID map row count correct: `python3 -c "import pyarrow.parquet as pq; print(len(pq.read_table('data/vector/faiss/idmap.parquet')))"`
- [ ] Reconstruction error acceptable: `jq '.checks."G2-05".details' reports/qa/step02_indexes.json` shows <= 0.001

#### Manual Verification:
- [ ] Index file size reasonable (~vectors × M × 8 bytes + metadata)
- [ ] Sanity search returns relevant chunks for "earnings results" query
- [ ] No segfaults or OpenMP errors during index build (check console output)

---

## Phase 6: Run Gate-7 Retrieval Evaluation

### Overview
Evaluate chunk-level recall improvement for SEC filing queries.

### Changes Required

#### 1. Run Full Gate-7 Evaluation
**Command**:
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Expected Metrics** (before vs after):

| Metric | Before (with XBRL noise) | After (XBRL removed) | Target |
|--------|---------------------------|----------------------|--------|
| **Overall Chunk Recall@10** | 65.22% | **> 70%** | 80% |
| **Overall Chunk nDCG@5** | 34.41% | **> 45%** | 60% |
| **SEC-specific Chunk Recall** | 0% (0/9) | **> 50% (≥5/9)** | 80% |
| **Document-level Recall** | 44.4% (4/9) | **Maintained or improved** | - |

**Failure Classification Shift**:
- **Before**: `chunk_miss_doc_hit_far` (wrong chunks from correct document)
- **After**: Chunk hits (expected chunks appear in top-10)

#### 2. Inspect Failure Log
**Command**:
```bash
# Count failures by classification
jq -r '.classification' reports/eval/retrieval_failures.jsonl | sort | uniq -c

# Inspect SEC-specific failures
jq 'select(.expected_doc_id | contains("10-K") or contains("10-Q"))' reports/eval/retrieval_failures.jsonl
```

**Expected Improvement**: Fewer `chunk_miss_doc_hit_far` failures for SEC queries.

### Success Criteria

#### Automated Verification:
- [ ] Gate-7 evaluation runs to completion: `echo $?` returns 0
- [ ] SEC chunk recall improves to > 0%: manually compute from eval results
- [ ] Overall recall@10 improves: compare `jq '.metrics.recall10' reports/qa/step07_retrieval_eval.json` before vs after
- [ ] No new failures introduced: spot-check non-SEC query performance remains stable

#### Manual Verification:
- [ ] Inspect top-10 results for SEC queries: verify expected chunks now appear
- [ ] Review failure log: confirm `chunk_miss_doc_hit_far` cases decrease
- [ ] Compare trace logs: verify chunk ranks improve (expected chunk moves from rank 50+ to top-10)
- [ ] Validate no regressions: non-SEC queries (Wikipedia, press releases) maintain performance

---

## Testing Strategy

### Unit Tests
None required (pure configuration change, no code modifications).

### Integration Tests
**Covered by Quality Gates**:
- Gate-0: Baseline checks (document counts, metadata completeness)
- Gate-1: Embedding generation and validation
- Gate-2: FAISS index integrity and sanity search
- Gate-7: End-to-end retrieval evaluation with chunk/document recall metrics

### Manual Testing Steps

1. **Verify XBRL removal in normalized text**:
   ```bash
   jq -r '.text' data/interim/normalized/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.json | head -c 5000
   ```
   **Expected**: Readable financial statements, NO XBRL taxonomy URLs

2. **Inspect first chunk content**:
   ```bash
   head -1 data/interim/chunks/crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.chunks.jsonl | jq -r '.text' | head -50
   ```
   **Expected**: Item 1 (Business) or financial statements, NO metadata noise

3. **Query retrieval test** (manual via MCP stub):
   ```bash
   # Start MCP stub
   conda run -n age python scripts/qa_step03_mcp.py &

   # Query: "What was Salesforce's revenue for FY25?"
   # Expected: Chunks from Item 7 (MD&A) or Item 8 (Financials) in top-3 results
   ```

4. **Compare before/after Gate-7 reports**:
   ```bash
   # Render Markdown report
   cat reports/qa/step07_retrieval_eval.md

   # Check SEC-specific metrics (manually compute from JSONL)
   jq 'select(.expected_doc_id | contains("10-K") or contains("10-Q")) | .chunk_rank' reports/eval/retrieval_failures.jsonl
   ```

---

## Performance Considerations

### Computational Impact
- **Normalization**: Negligible (<1s additional per document to remove XBRL selectors)
- **Chunking**: ~10-15% fewer chunks per SEC document (shorter documents after XBRL removal)
- **Embeddings**: ~10-15% fewer API calls (proportional to chunk reduction), ~$0.05 total cost for re-embedding 5 SEC filings
- **FAISS indexing**: Slightly smaller index (~10% fewer vectors), faster build time (~1-2s reduction)
- **Query latency**: No change (index query time dominated by HNSW traversal, not index size)

### Storage Impact
- **Normalized JSON**: ~15% smaller (XBRL metadata removed)
- **Chunks JSONL**: ~10-15% fewer rows (proportional to chunk reduction)
- **Embeddings Parquet**: ~10% smaller (fewer vectors)
- **FAISS index**: ~10% smaller (fewer indexed vectors)

### Retrieval Quality Trade-offs
- **Benefit**: Chunks now contain actual financial content, improving semantic relevance
- **Risk**: None identified (XBRL metadata has zero semantic value for retrieval)

---

## Migration Notes

### Backward Compatibility
- **Configuration**: New selectors in `normalization.rules.yaml` are additive (no breaking changes)
- **Non-SEC documents**: XBRL selectors match nothing if tags absent (zero impact on Wikipedia, press releases, help docs)
- **Existing normalized data**: Can coexist with new normalized data (no format changes)

### Data Migration Strategy
**Option 1**: Incremental re-normalization (selected approach)
1. Update config (`configs/normalization.rules.yaml`)
2. Delete SEC normalized files only
3. Re-run normalization for SEC filings
4. Downstream pipeline auto-propagates changes (chunking, embeddings, indexing)

**Option 2**: Full pipeline reset
1. Delete all normalized, chunks, embeddings, indexes
2. Re-run all gates 0-7
3. **Drawback**: Unnecessary re-processing of non-SEC documents (~$2-3 OpenAI API cost)

**Recommended**: Option 1 (incremental)

### Rollback Plan
If Gate-7 metrics worsen unexpectedly:
1. Restore backed-up files:
   ```bash
   cp data/backup/normalized_TIMESTAMP/* data/interim/normalized/
   cp data/backup/chunks_TIMESTAMP/* data/interim/chunks/
   cp data/backup/embeddings_TIMESTAMP.parquet data/vector/embeddings/embeddings.parquet
   cp -r data/backup/faiss_TIMESTAMP/* data/vector/faiss/
   ```
2. Revert `configs/normalization.rules.yaml` (remove XBRL selectors)
3. Re-run Gate-7 evaluation

**Rollback Time**: <5 minutes (file copy operations only)

---

## References

### Original Issue
- `thoughts/shared/issues/issue005.md` - SEC filing retrieval pipeline investigation
- `thoughts/shared/issues/issue003.md` - XBRL metadata pollution discovery
- `thoughts/shared/issues/issue004.md` - Gate-7 RED status, per-doctype performance breakdown

### Related Research
- `thoughts/shared/research/2025-10-07-issue005-sec-filing-retrieval-pipeline.md` - Complete pipeline documentation
- `thoughts/shared/research/2025-10-07-issue003-retrieval-recall-investigation.md` - XBRL structure parsing failures
- `thoughts/shared/research/2025-10-07-issue004-retrieval-recall-investigation.md` - SEC filing 0% chunk recall analysis

### Implementation References
- `scripts/normalize_html.py:83-137` - HTML normalization function
- `scripts/normalize_html.py:194-327` - Document processing orchestration
- `configs/normalization.rules.yaml:1-11` - Current removal selectors
- `scripts/chunk_documents.py:105-228` - Chunking logic (SEC-aware segmentation)
- BeautifulSoup CSS selector documentation: https://www.crummy.com/software/BeautifulSoup/bs4/doc/#css-selectors

### XBRL Format References
- SEC XBRL documentation: https://www.sec.gov/structureddata/osd-inline-xbrl.html
- Inline XBRL specification: https://specifications.xbrl.org/work-product-index-inline-xbrl-inline-xbrl-1.1.html
- XBRL namespace prefixes: `ix:` (Inline XBRL), `xbrli:` (XBRL Instance), `dei:` (Document Entity Information), `us-gaap:` (US GAAP Taxonomy)

---

## Risk Assessment

### Low Risk
- **Configuration-only change**: No code modifications reduces regression risk
- **Isolated to SEC filings**: XBRL tags only exist in SEC documents, zero impact on other sources
- **Reversible**: Full rollback possible via backup restoration

### Medium Risk
- **Gate-7 improvement uncertainty**: XBRL removal is necessary but may not be sufficient to achieve 80% recall target (lexical reranking also broken per `qa_step03_mcp.py:122-144`)
- **Over-removal risk**: If `ix:` prefix matches legitimate inline elements in non-SEC docs (mitigated by namespace specificity)

### Mitigation Strategies
- **Before/after comparison**: Backup all intermediate data before regeneration
- **Incremental rollout**: Process SEC filings first, validate improvements before touching other document types
- **Comprehensive testing**: Run full Gate-0 through Gate-7 pipeline to detect unexpected side effects

---

## Next Steps After Implementation

### If Gate-7 Improves (Expected)
1. Document new baseline metrics in `reports/qa/step07_retrieval_eval.md`
2. Close Issue005 and related tickets
3. Update CLAUDE.md to document XBRL handling approach
4. Consider extending XBRL removal to other SEC filing types (EFFECT, S-1, DEF 14A) if added in future

### If Gate-7 Remains Low (Unlikely)
1. Investigate lexical reranking failure (`qa_step03_mcp.py:120-144` attempts to import missing `tokenize` function)
2. Consider SEC-specific reranking weights (boost chunks with financial keywords: "revenue", "income", "cash flow")
3. Evaluate table-aware chunking (preserve financial statement table structure during normalization)
4. Analyze ada-002 embedding quality for financial text (may require domain-specific fine-tuning)

---

## Appendix: XBRL Tag Inventory

**Common XBRL Namespaces in SEC Filings**:
- `ix:` - Inline XBRL wrapper tags (header, hidden, nonNumeric, nonFraction)
- `xbrli:` - XBRL Instance elements (context, unit, identifier, period, measure)
- `dei:` - Document & Entity Information (EntityCentralIndexKey, DocumentFiscalPeriodFocus)
- `us-gaap:` - US GAAP Taxonomy (CostOfSalesMember, FinanceLeaseLiabilityNoncurrent)
- `crm:` - Company-specific extensions (SubscriptionandSupportMember, segment)
- `srt:` - Standard Reference Taxonomy (ProductOrServiceAxis, StatementGeographicalAxis)

**Removal Strategy**:
- Target `ix:header`, `ix:hidden`, `ix:resources`, `ix:references` (top-level containers)
- Do NOT remove inline facts (`ix:nonNumeric`, `ix:nonFraction` embedded in narrative text) - these contain actual disclosure values
- Let BeautifulSoup's `.decompose()` recursively delete child elements (no need to list every sub-tag)
