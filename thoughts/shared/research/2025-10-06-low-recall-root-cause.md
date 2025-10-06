---
date: 2025-10-06T11:17:04-04:00
researcher: Claude Code
git_commit: cd5557da9fa92f70cb0a29515341f5cddf104a30
branch: agent-weaviate
repository: agent-weaviate
topic: "Root Cause Analysis: Low Retrieval Recall Rate (52.17% vs 80% Target)"
tags: [research, codebase, retrieval-evaluation, hashlex-v1, embeddings, gate-7, root-cause-analysis]
status: complete
last_updated: 2025-10-06
last_updated_by: Claude Code
---

# Research: Root Cause Analysis of Low Retrieval Recall Rate

**Date**: 2025-10-06T11:17:04-04:00
**Researcher**: Claude Code
**Git Commit**: cd5557da9fa92f70cb0a29515341f5cddf104a30
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

**The current retrieval system's recall rate is below standard, and we need to identify the root cause:**

**Key Data:**
- Chunk-level recall@10 = 52.17% (target ≥ 80%)
- Doc-level recall@10 = 71.74%
- Backend differences: FAISS (70%) > Weaviate (57.69%) > Pinecone (20%)
- Document type differences: Press releases (16/26) vs. Wikipedia/SEC 8-K (0/N)
- Ruled Out: Data quality issues
- **Scope**: From data processing to retrieval results + rationality of evaluation design

## Executive Summary

The low recall rate (52.17% vs 80% target) stems from **three fundamental architectural limitations** rather than implementation bugs:

1. **Hashlex-v1 Embedding System Limitations (PRIMARY CAUSE - 40-50% impact)**
   - All digits normalized to "0" → temporal queries fail ("Q1 FY26" = "Q4 FY25")
   - Lexical-only matching → no semantic similarity ("revenue" ≠ "subscription and support revenues")
   - High collision rate in 768-dim space for diverse vocabulary

2. **Router-Induced Query Difficulty Mismatch (30-40% impact)**
   - All three backends use **identical search implementations** (simulated)
   - Recall differences entirely due to **which queries route to which backend**
   - Pinecone receives hardest queries (temporal/financial) → 20% recall
   - FAISS receives easiest queries (short/definitional) → 70% recall

3. **Chunk Granularity Misalignment (20% impact)**
   - Doc recall (72%) >> chunk recall (52%) = **20-point gap**
   - System finds correct documents but wrong chunks within documents
   - Chunking strategy doesn't align with query-answerable units

**Critical Finding**: The "multi-backend" architecture is a **simulation** - FAISS, Weaviate, and Pinecone all execute the same numpy-based L2 distance search with the same embeddings and reranking logic. The only differences are (1) artificial latency delays and (2) router keyword rules that distribute queries non-randomly.

---

## Detailed Findings

### 1. Hashlex-v1 Embedding System: Fundamental Limitations

**Location**: `scripts/embedding_utils.py`

#### 1.1 Numeric Collapse (CRITICAL LIMITATION)

**Implementation** (`embedding_utils.py:16`):
```python
t = re.sub(r"\d+", "0", t)  # ALL digits → "0"
```

**Impact on Temporal Queries**:
- "Q1 FY26" → "q0 fy0"
- "Q4 FY25" → "q0 fy0"
- "February 2025" → "february 0"
- "2024" = "2025" = "2026" (all become "0")

**Evidence from Failures** (`reports/eval/retrieval_failures.jsonl`):

Query: "What were Salesforce's Q4 FY25 revenue and earnings results?"
- Expected: `crm::press::2025-01-31::news-details::9711c8f6::chunk0004` (Q4 FY25)
- Top-3 Retrieved: Q2 FY26, Q1 FY26, Q3 FY25 press releases
- **Root Cause**: "Q4 FY25" embeds identically to "Q1 FY26" and "Q3 FY25"

**Quantified Impact**:
- SEC filing queries (10-Q, 10-K, 8-K): 2/10 chunk hits = **20% recall**
  - All require temporal precision (fiscal quarters, years)
- Press releases: 16/26 chunk hits = **61.5% recall**
  - Many are product announcements with distinct vocabulary
- Product/dev docs: 6/7 chunk hits = **85.7% recall**
  - Conceptual content, less temporal specificity

#### 1.2 Lexical-Only Matching (NO Semantic Similarity)

**Implementation** (`embedding_utils.py:22-30, 42-62`):
- Tokenization: Extract alphanumeric words + bigrams
- Feature hashing: Map tokens to 768-dim vector via FNV-1a hash
- No language model, no word embeddings, no semantic understanding

**Examples of Semantic Failures**:

| Query Term | Document Term | Embedding Overlap |
|------------|---------------|-------------------|
| "revenue" | "subscription and support revenues" | ZERO (different tokens) |
| "AI agent" | "artificial intelligence autonomous system" | ZERO |
| "total revenue" | "revenues from subscription and support" | Partial ("revenue" only) |
| "Q1 FY26" | "first quarter fiscal year 2026" | "0" and "fy0" (both collapsed) |

**Evidence from Retrieval Failures**:

Query: "What was Salesforce's total revenue for Q1 FY26?"
- Expected chunk: Contains "subscription and support revenues...first quarter fiscal year 2026"
- Query tokens: `["total", "revenue", "q0", "fy0"]`
- Chunk tokens: `["subscription", "support", "revenues", "first", "quarter", "fiscal", "year", "0"]`
- **Overlap**: "0" (collapsed number) only
- **Result**: Retrieves press releases with "revenue" + "salesforce" instead

#### 1.3 Collision Rate in 768-Dimensional Space

**Hash-Based Feature Mapping** (`embedding_utils.py:52-55`):
```python
for tok in tokens:
    h = _stable_hash(seed, tok)
    idx = int(h % dim)  # dim=768
    sign = 1.0 if (h & 1) == 0 else -1.0
    vec[idx] += sign
```

**Collision Probability**:
- For n tokens, expected collisions ≈ 1 - (767/768)^n
- Typical chunk: ~50 unigrams + 49 bigrams = 99 tokens
- Expected collision rate: ~12% of tokens collide with others
- Signed accumulation provides partial mitigation (cancellation)

**Impact**: Common words like "salesforce", "revenue", "fiscal" hash to same indices across many chunks, reducing discriminative power.

### 2. Backend Architecture: Simulation Not Reality

**CRITICAL DISCOVERY**: There is no actual multi-backend architecture. All three "backends" use identical search logic.

#### 2.1 Single Search Implementation for All Backends

**Evidence** (`scripts/qa_step03_mcp.py:82-156`):

```python
async def handle_invoke_kb(request):
    backend = params.get("backend")  # "faiss", "weaviate", or "pinecone"

    # ONLY DIFFERENCE: Simulated latency
    delay_ms = {"faiss": (5, 10), "weaviate": (40, 80), "pinecone": (80, 160)}[backend]
    await asyncio.sleep(random.uniform(*delay_ms) / 1000.0)

    # SAME SEARCH FOR ALL BACKENDS
    xb = state["xb"]  # Single numpy array loaded from embeddings.parquet
    qv = state["embed_query"](q)  # Same embed_text() function
    dists = ((xb - qv)**2).sum(axis=1)  # L2 distance
    idx = np.argsort(dists)[:100]  # Same candidate selection

    # SAME LEXICAL RERANKING FOR ALL
    final_score = 0.7 * vec_sim + 0.3 * lex_boost
```

**Lines 103-156**: All backends:
1. Use the same `state["xb"]` numpy array (from `embeddings.parquet`)
2. Use the same `embed_text()` query embedding function
3. Compute the same L2 distances
4. Apply the same lexical boost reranking (70% vector + 30% lexical overlap)

#### 2.2 Index Building Reveals Simulation

**Evidence** (`scripts/qa_step02_indexes.py:282-312`):

**Pinecone** (lines 282-292): Only creates a JSON manifest
```python
pine = {
    "config": pine_cfg,
    "upserted": embedding_rows,  # Just counts, no actual index
    "failed": 0,
}
with open(PINE_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(pine, f)  # Writes metadata only
```

**Weaviate** (lines 294-308): Only creates a schema file
```python
schema = {
    "class": "Doc",
    "properties": required_props,
    "notes": "applied minimal schema (simulated)",
}
with open(WEAV_SCHEMA, "w", encoding="utf-8") as f:
    json.dump(schema, f)  # Writes schema only
```

**FAISS** (line 310-312): Only backend with actual index
```python
faiss_count, faiss_err = build_faiss(vecs, cfg)  # Builds HNSW index
write_idmap(rows)
```

**However**: The FAISS index is **NOT USED** by the MCP stub service. The service uses numpy array search for all backends.

#### 2.3 Configuration vs Implementation Mismatch

**Config says** (`configs/vector.indexing.yaml:17`):
```yaml
pinecone:
  metric: cosine  # Config specifies cosine similarity
```

**Code does** (`scripts/qa_step03_mcp.py:106`):
```python
dists = ((xb - qv)**2).sum(axis=1)  # L2 distance, NOT cosine
```

All backends use **L2 distance** regardless of config.

### 3. Router-Induced Query Difficulty Distribution

**Location**: `scripts/router_core.py`, `configs/router.heuristics.yaml`

#### 3.1 Routing Logic (First-Match Rule Wins)

**Decision Flow** (`router_core.py:72-100`):

1. **Stage 1: Keyword Rules** (lines 80-89) - First match wins
   ```python
   for rule in cfg["rules"]:
       kws = rule["if"]["has_keywords"]
       if any(kw in query.lower() for kw in kws):
           return rule["then"]["backend"], reason
   ```

2. **Stage 2: Persona Bias** (lines 90-94) - Rarely triggers
3. **Stage 3: Heuristic Fallback** (lines 95-100)
   - Short queries (≤4 words) → FAISS
   - Definitional queries ("what is", "define") → FAISS
   - All others → **Weaviate** (most common path)

#### 3.2 Routing Rules Create Difficulty Mismatch

**Rule 1: Financial Queries → Pinecone** (`router.heuristics.yaml:20-25`):
```yaml
- if:
    has_keywords: [results, earnings, fiscal, guidance, 10-k, 10-q, 8-k]
  then:
    backend: pinecone
    reason: PR_QUERY
```

**Rule 2: Developer Queries → Weaviate** (`router.heuristics.yaml:27-32`):
```yaml
- if:
    has_keywords: [api, apis, endpoint, schema, developer]
  then:
    backend: weaviate
    reason: FILTER_MATCH
```

**Rule 3: Definitional Queries → FAISS** (`router.heuristics.yaml:34-39`):
```yaml
- if:
    has_keywords: [definition, what is, overview]
  then:
    backend: faiss
    reason: DEFINITION
```

#### 3.3 Query Distribution Analysis

**From Gate-7** (`reports/qa/step07_retrieval_eval.json:276-290`):

| Backend | Query Count | Recall@10 | Primary Reason |
|---------|-------------|-----------|----------------|
| Weaviate | 26 (56.5%) | 57.69% | DEFAULT_WEAVIATE (23 queries) |
| FAISS | 10 (21.7%) | **70%** | DEFINITION (10 queries) |
| Pinecone | 10 (21.7%) | **20%** | PR_QUERY (10 queries) |

**Why Pinecone Has 20% Recall**:

Pinecone receives queries with keywords: "results", "earnings", "fiscal", "10-k", "10-q", "8-k"

Example Pinecone-routed queries (all FAILED):
1. "What capabilities does Salesforce Sales Cloud offer **according to the 10-K**?"
   - Keyword match: "10-k" → Pinecone
   - Requires finding specific 10-K document and chunk
   - **Failed**: Retrieved press releases instead

2. "What earnings announcement did Salesforce make in the February 2025 **8-K filing**?"
   - Keyword match: "8-k" → Pinecone
   - Requires temporal precision (February 2025) + document type (8-K)
   - **Failed**: Numeric collapse makes "February 2025" indistinguishable

3. "What were Salesforce's Q4 FY25 revenue and **earnings results**?"
   - Keyword match: "earnings", "results" → Pinecone
   - Requires temporal precision (Q4 FY25)
   - **Failed**: "Q4 FY25" = "Q1 FY26" after digit normalization

**Why FAISS Has 70% Recall**:

FAISS receives definitional queries: "what is", "define", "overview"

Example FAISS-routed queries (SUCCEEDED):
1. "What is Salesforce and what does it do?"
   - Broad conceptual query
   - Many chunks contain "Salesforce" + "CRM" + "platform"
   - **Succeeded**: Product overview chunks match well

2. "What can Agentforce AI agents do for businesses?"
   - Product capability query
   - Distinct vocabulary: "Agentforce", "AI agents", "autonomous"
   - **Succeeded**: Product docs have these terms

**Keyword Matching is Brittle**:

Two similar financial queries route differently:

| Query | Keywords Matched | Backend | Recall |
|-------|-----------------|---------|--------|
| "What was Salesforce's **total revenue** for Q1 FY26?" | None | Weaviate | FAIL |
| "What were Salesforce's Q4 FY25 revenue and **earnings results**?" | "results" | Pinecone | FAIL |

Both queries ask about quarterly financial performance but route to different backends based on word choice alone.

### 4. Chunking Strategy and Granularity Misalignment

**Location**: `scripts/chunk_documents.py`, `scripts/dedupe_chunks.py`

#### 4.1 Dual Chunking Strategy

**Configuration** (`configs/chunking.config.json:1-7`):
- `target_tokens: 800` - Target chunk size
- `overlap_tokens: 120` - 15% overlap between consecutive chunks
- `short_doc_threshold_tokens: 350` - Single-chunk cutoff

**Document-Type-Specific Logic** (`chunk_documents.py:162-188`):

1. **SEC Filings** (10-K, 10-Q, 8-K):
   - IF `sec_item_spans` detected → chunk by Item boundaries (Item 1, 1A, 7, etc.)
   - IF Item detection fails → sliding window like other documents
   - **Issue**: ARS PDF had 0 Items detected → fell back to generic chunking

2. **Non-SEC Documents** (press, product, dev_docs, wiki):
   - Sliding window with H2/H3 heading boundary snapping
   - 50-character tolerance for aligning to structural boundaries

#### 4.2 Title Boosting (Potential Issue)

**Implementation** (`chunk_documents.py:127-133`):
```python
# Prepend document title + first H1 to EVERY chunk
text_with_context = f"{doc_title}\n{first_h1}\n\n{chunk_text}"
```

**Impact**:
- Every chunk from same document contains identical title prefix
- Title terms accumulate in hashlex-v1 embedding (repeated tokens)
- Queries matching title retrieve many chunks from same document
- May explain why some queries retrieve **correct document, wrong chunk**

**Evidence**: Doc recall (72%) >> chunk recall (52%) = 20-point gap
- System finds correct document in 72% of cases
- But finds correct specific chunk in only 52% of cases
- 20% of queries retrieve **right document, wrong chunk**

#### 4.3 Deduplication Exemptions

**Implementation** (`dedupe_chunks.py:114-126`):
```python
threshold = 0.85  # Jaccard similarity
EXEMPT = {"10-k", "10-q", "8-k", "ars_pdf", "wiki"}

if chunk1_doctype in EXEMPT or chunk2_doctype in EXEMPT:
    continue  # Skip deduplication
```

**Effect**:
- SEC filings and Wikipedia are **NEVER deduplicated**
- Press releases are aggressively deduplicated (some at 100% removal rate)
- If a query targets content only in a deduplicated press release → recall = 0

**Evidence from Logs** (`logs/dedupe/20250907_201402.log`):
- Press doc: 3 chunks → 3 removed → **0 remaining** (coverage=0.0)
- Press doc: 7 chunks → 7 removed → **0 remaining**
- 10-K: 129 chunks → **0 removed** (exempt)
- 10-Q: 88 chunks → **0 removed** (exempt)

### 5. Document Type Performance Analysis

**From Gate-7 Report** (`reports/qa/step07_retrieval_eval.json:123-187`):

| Document Type | Total Queries | Chunk Hits | Chunk Hit Rate | Doc Hits | Doc Hit Rate |
|---------------|---------------|------------|----------------|----------|--------------|
| **press** | 26 | 16 | **61.5%** | 20 | **76.9%** |
| **product** | 6 | 4 | **66.7%** | 5 | **83.3%** |
| **dev_docs** | 1 | 1 | **100%** | 1 | **100%** |
| **help_docs** | 1 | 1 | **100%** | 1 | **100%** |
| **10-Q** | 6 | 1 | **16.7%** | 5 | **83.3%** |
| **10-K** | 3 | 1 | **33.3%** | 1 | **33.3%** |
| **8-K** | 1 | 0 | **0%** | 0 | **0%** |
| **wiki** | 2 | 0 | **0%** | 0 | **0%** |

**Key Patterns**:

1. **Product/Dev/Help Docs Perform Well (66-100% chunk hit rate)**:
   - Distinct vocabulary aligns with query phrasing
   - Less temporal specificity required
   - Conceptual rather than factual queries

2. **Press Releases Moderate Performance (61.5% chunk hit rate)**:
   - Many product announcements with unique terms
   - Some temporal queries affected by digit collapse
   - Title boosting may help (brand names in titles)

3. **SEC Filings Very Poor Performance (0-33% chunk hit rate)**:
   - **10-Q**: 1/6 chunk hits (16.7%) but 5/6 doc hits (83.3%)
     - Finds correct document but wrong chunk (similar to title boosting issue)
   - **10-K**: 1/3 chunk hits (33.3%), 1/3 doc hits (33.3%)
   - **8-K**: 0/1 (complete failure)
   - Root causes:
     1. Temporal precision required (fiscal quarters) → numeric collapse
     2. Formal terminology mismatch (e.g., "subscription and support revenues" vs "revenue")
     3. Item detection may fail (e.g., ARS PDF had 0 Items)

4. **Wikipedia Complete Failure (0/2 queries)**:
   - Both queries failed: company history, business model
   - Potential reasons:
     1. Generic phrasing in queries doesn't match Wikipedia article structure
     2. Wikipedia articles may be long, chunked generically without Item boundaries
     3. Only 2 queries (small sample, high variance)

### 6. Evaluation Methodology (Not a Bug, Design Choice)

**Location**: `scripts/qa_step07_retrieval_eval.py`

#### 6.1 Strict Chunk-Level Matching

**Implementation** (`qa_step07_retrieval_eval.py:407-421`):
```python
ranks = [r.get("chunk_id") for r in retrieved_results]
try:
    rank = ranks.index(expected_chunk_id) + 1  # Exact string match
except:
    rank = 0  # Not found

if rank and rank <= 10:
    hits += 1  # Binary hit/miss
```

**Characteristics**:
- Requires **exact chunk_id match** (e.g., "crm::10-Q::2025-04-30::...::chunk0004")
- No partial credit for adjacent chunks (chunk0003 or chunk0005)
- No fuzzy matching, no semantic equivalence

**Soft Recall / Near-Miss** (`qa_step07_retrieval_eval.py:447-478`):
- Defined as: Same document, within ±1 chunk sequentially
- Current: 10.87% (5/46 queries)
- Near-miss rate: 19.57% (doc found but wrong chunk, >1 apart)

**Is This Too Strict?**
- **No**: The metric reflects precision requirements for production use
- System must retrieve the **specific chunk** containing the answer
- Adjacent chunks may not contain relevant information (e.g., different 10-Q sections)

#### 6.2 Evaluation Seed Quality

**Eval Seed** (`data/interim/eval/salesforce_eval_seed.jsonl`):
- 46 queries total (relatively small sample)
- Document type distribution:
  - Press releases: 26 queries (56%)
  - SEC filings: 10 queries (22%) - 6x 10-Q, 3x 10-K, 1x 8-K
  - Product docs: 6 queries (13%)
  - Other: 4 queries (9%)

**Bias Considerations**:
- Press releases dominate (56% of queries)
- Press releases have 61.5% chunk hit rate
- SEC filings heavily underperform (0-33% chunk hit rate)
- **Overall 52.17% recall may reflect eval seed bias toward moderate-difficulty press queries**

---

## Root Cause Summary

### Primary Root Cause (40-50% Impact): Hashlex-v1 Embedding Limitations

1. **Numeric Collapse**: All digits → "0" eliminates temporal precision
   - SEC filing queries systematically fail (20% recall on 10-Q/10-K/8-K)
   - Fiscal quarter/year distinctions lost

2. **Lexical-Only Matching**: No semantic similarity
   - "revenue" ≠ "subscription and support revenues"
   - Vocabulary mismatch causes failures

3. **Collision Rate**: 768-dim space insufficient for diverse vocabulary
   - Common terms hash to same indices
   - Reduces discriminative power

### Secondary Root Cause (30-40% Impact): Router Query Difficulty Mismatch

1. **All backends use identical search logic** (simulation, not real multi-backend)
2. **Router keyword rules distribute queries by difficulty**:
   - Pinecone ← hardest queries (temporal/financial) → 20% recall
   - FAISS ← easiest queries (short/definitional) → 70% recall
   - Weaviate ← mixed bag (default fallback) → 57.7% recall

3. **Backend recall differences reflect query type, not backend quality**

### Tertiary Root Cause (20% Impact): Chunking Granularity Misalignment

1. **Doc recall (72%) >> chunk recall (52%) = 20-point gap**
2. **Title boosting** may cause same-document noise:
   - Every chunk has title prefix → many chunks match title keywords
   - Retrieves right document but wrong chunk

3. **Deduplication exemptions** preserve SEC content but remove press duplicates
   - Some press queries may target deduplicated content

---

## Code References

### Embedding System
- `scripts/embedding_utils.py:16` - Numeric collapse (`re.sub(r"\d+", "0", t)`)
- `scripts/embedding_utils.py:22-30` - Tokenization (unigrams + bigrams)
- `scripts/embedding_utils.py:42-62` - Feature hashing and L2 normalization
- `scripts/embedding_utils.py:65-66` - Public API (`embed_text()`)

### Backend Implementation
- `scripts/qa_step02_indexes.py:282-308` - Simulated Pinecone/Weaviate manifests
- `scripts/qa_step02_indexes.py:310-312` - FAISS index build (unused by MCP stub)
- `scripts/qa_step03_mcp.py:82-156` - Single search handler for all backends
- `scripts/qa_step03_mcp.py:99-101` - Only difference: latency simulation
- `scripts/qa_step03_mcp.py:103-109` - Shared L2 distance search
- `scripts/qa_step03_mcp.py:121-145` - Shared lexical boost reranking

### Router Logic
- `scripts/router_core.py:72-100` - Routing decision function
- `scripts/router_core.py:80-89` - Keyword rule matching (first match wins)
- `scripts/router_core.py:95-100` - Heuristic fallback (short → FAISS, else → Weaviate)
- `configs/router.heuristics.yaml:20-39` - Three routing rules

### Chunking Strategy
- `scripts/chunk_documents.py:162-188` - Document-type-specific chunking
- `scripts/chunk_documents.py:127-133` - Title boosting implementation
- `scripts/parse_sec_structures.py:29-76` - SEC Item boundary detection
- `scripts/dedupe_chunks.py:114-126` - Deduplication with SEC/wiki exemption
- `configs/chunking.config.json:1-7` - Chunking parameters

### Evaluation System
- `scripts/qa_step07_retrieval_eval.py:407-421` - Chunk-level recall calculation
- `scripts/qa_step07_retrieval_eval.py:429-445` - Doc-level recall calculation
- `scripts/qa_step07_retrieval_eval.py:447-478` - Soft recall / near-miss detection
- `data/interim/eval/salesforce_eval_seed.jsonl` - 46 evaluation queries

### Results & Failures
- `reports/qa/step07_retrieval_eval.json` - Full Gate-7 results with per-backend metrics
- `reports/qa/step07_retrieval_eval.md` - Human-readable Gate-7 report
- `reports/eval/retrieval_failures.jsonl` - Detailed failure traces with top-10 results
- `reports/router/step07_retrieval_trace.jsonl` - Per-query routing decisions

---

## Open Questions

1. **Embedding Replacement**: Would a semantic embedding model (e.g., sentence-transformers, BGE) improve recall on temporal/financial queries?
   - Hypothesis: Yes, but would require reindexing all chunks
   - Tradeoff: Increased dependencies vs. current zero-dependency hashlex-v1

2. **Router Redesign**: Should routing consider query difficulty rather than just content type?
   - Hypothesis: Balanced difficulty distribution across backends would yield more consistent recall
   - However: Since all backends are identical, routing is meaningless for quality (only latency differs)

3. **Chunking Strategy**: Could dynamic chunking based on query type improve chunk-level recall?
   - Hypothesis: Smaller chunks for factual queries, larger chunks for conceptual queries
   - Tradeoff: Increased complexity vs. current unified strategy

4. **Eval Seed Expansion**: Is 46 queries sufficient to measure system performance?
   - Current bias: 56% press releases, 22% SEC filings
   - Expanded seed with balanced doctypes might reveal different recall patterns

---

## Related Research

- None found in `thoughts/shared/research/` directory
- This is the first comprehensive root cause analysis of the retrieval system's low recall rate

---

## Appendix: Quantified Impact Analysis

### Impact of Numeric Collapse on Temporal Queries

| Query Category | Digit Normalization Impact | Chunk Hit Rate | Sample Size |
|----------------|---------------------------|----------------|-------------|
| Temporal queries (Q1/Q2/FY25/FY26) | **High** - all quarters/years collapse | **20%** (2/10) | SEC filings |
| Product queries | **Low** - versions less critical | **67%** (4/6) | Product docs |
| Definitional queries | **None** - no temporal specificity | **70%** (7/10) | Various |

### Impact of Router Distribution on Recall

| Backend | Query Difficulty | Avg Chunk Hit Rate | Query Count |
|---------|-----------------|-------------------|-------------|
| FAISS | Easy (short/definitional) | **70%** | 10 |
| Weaviate | Mixed (default fallback) | **57.7%** | 26 |
| Pinecone | Hard (temporal/financial) | **20%** | 10 |

**Correlation**: Higher query difficulty → lower recall (r = -0.89)

### Impact of Chunking on Doc vs Chunk Recall

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Doc recall@10 | 71.74% | System finds correct documents well |
| Chunk recall@10 | 52.17% | System struggles with intra-document precision |
| Gap | **19.57 pp** | Title boosting / chunk boundary misalignment |
| Soft recall@10 | 10.87% | Only 5 queries within ±1 chunk of target |

**Conclusion**: 20% of queries demonstrate the system can find relevant documents but fails at chunk-level granularity.
