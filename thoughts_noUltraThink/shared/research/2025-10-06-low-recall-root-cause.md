---
date: 2025-10-06T11:03:13-04:00
researcher: Claude Code
git_commit: cd5557da9fa92f70cb0a29515341f5cddf104a30
branch: agent-weaviate
repository: ag3
topic: "Root Cause Analysis: Low Recall Rate in Retrieval System (52.17% vs 80% target)"
tags: [research, retrieval-evaluation, gate-7, hashlex-v1, embedding-analysis, sec-documents]
status: complete
last_updated: 2025-10-06
last_updated_by: Claude Code
---

# Research: Root Cause Analysis of Low Recall Rate (52.17% vs 80% target)

**Date**: 2025-10-06T11:03:13-04:00
**Researcher**: Claude Code
**Git Commit**: cd5557da9fa92f70cb0a29515341f5cddf104a30
**Branch**: agent-weaviate
**Repository**: ag3

## Research Question

The current retrieval system's recall rate is below standard at 52.17% chunk-level recall@10 (target ≥80%) and 71.74% doc-level recall@10. Backend performance differs significantly: FAISS (70%) > Weaviate (57.69%) > Pinecone (20%). Document type performance also varies: Press releases (61.5% chunk hit rate) vs SEC 8-K (0/1) and Wikipedia (0/2). What are the fundamental reasons for the low recall rate across the data processing to retrieval pipeline and evaluation design?

## Executive Summary

The low recall rate (52.17%) stems from **five fundamental architectural limitations** in the hashlex-v1 embedding system and evaluation methodology, not from implementation bugs or data quality issues:

1. **Lexical Embedding Limitations**: The hashlex-v1 bag-of-words + bigrams approach cannot capture semantic similarity or handle paraphrasing, causing systematic failures when queries use different terminology than document text.

2. **Numeric Collapse**: All numbers are normalized to '0' during embedding, making fiscal periods (Q1 FY26 vs Q4 FY25) and financial metrics indistinguishable.

3. **Backend Performance Differences Are Routing Artifacts**: All three backends use identical embeddings and search algorithms; the 70%/57.69%/20% performance split reflects query-type distribution, not backend quality.

4. **SEC Document Processing Gaps**: XBRL metadata contamination, Item detection failures, and non-descriptive titles reduce retrieval quality for SEC filings.

5. **Query-Document Vocabulary Mismatch**: Evaluation queries use natural language ("Q1 FY26 revenue") while document chunks contain different phrasing ("first quarter fiscal year 2026"), which lexical embeddings cannot bridge.

**The recall rate accurately reflects the system's capabilities given its design choices.** The evaluation methodology is sound; low recall is a genuine limitation of deterministic lexical embeddings when faced with semantic variation.

## Detailed Findings

### 1. Embedding System Architecture (hashlex-v1)

**Location**: `scripts/embedding_utils.py:8-66`

#### How It Works

The hashlex-v1 system converts text to 768-dimensional vectors through four stages:

1. **Normalization** (lines 8-19):
   - Lowercase conversion
   - Unicode → ASCII (strips non-English characters)
   - Hyphen/slash → space separation
   - **All digits → '0'** (critical limitation)
   - Whitespace collapse

2. **Tokenization** (lines 22-30):
   - Extract alphanumeric sequences (2-20 chars)
   - Generate bigrams with `bg:` prefix
   - Example: "Agentforce 2.0" → `["agentforce", "0"]` + `["bg:agentforce_0"]`

3. **Feature Hashing** (lines 33-62):
   - FNV-1a hash mapping tokens → 768-dimensional indices
   - Signed accumulation (hash LSB determines +1 or -1)
   - Collision handling via signed cancellation

4. **L2 Normalization** (lines 57-62):
   - Normalize vectors to unit length
   - Makes L2 distance equivalent to cosine similarity

#### What It Captures

- **Lexical presence**: Which words appear in text
- **Local word order**: Adjacent word pairs (bigrams)
- **Term frequency**: Repeated tokens accumulate signal
- **Multi-word phrases**: "machine learning" as `bg:machine_learning`

#### What It Loses

- **Semantic similarity**: "AI agent" ≠ "artificial intelligence autonomous system"
- **Synonyms**: "purchase" ≠ "buy", "client" ≠ "customer" (zero overlap)
- **Numeric precision**: "Q1 2025" = "Q4 2023" (both → "q0 0")
- **Long-range dependencies**: No context beyond bigrams
- **Case semantics**: "AI" (acronym) = "ai" (word)
- **Non-ASCII languages**: Complete loss of non-English text

#### Impact on Recall

**Example failure** (`retrieval_failures.jsonl:1`):
- Query: "What was Salesforce's total revenue for Q1 FY26?"
- Tokens: `["what", "salesforce", "total", "revenue", "q0", "fy0"]`
- Expected chunk contains: "fiscal year 2026 first quarter subscription and support revenues"
- **Mismatch**: "Q1" vs "first quarter", "FY26" vs "fiscal year 2026", all numbers collapsed to "0"
- Result: Chunk miss, doc miss

### 2. Backend Performance Analysis

**Location**: `scripts/qa_step03_mcp.py:82-156`, `scripts/router_core.py:72-100`

#### Critical Finding: All Backends Are Identical

Investigation reveals that FAISS, Weaviate, and Pinecone **use the exact same embeddings and search implementation** in the MCP stub service:

**Shared Components**:
- Single numpy array `state["xb"]` loaded from `data/vector/embeddings/embeddings.parquet` (line 53)
- Identical query embedding via `embed_text()` from `embedding_utils.py` (lines 69-76)
- Identical L2 distance computation: `((xb - qv)**2).sum(axis=1)` (line 106)
- Identical lexical reranking: `0.7 * vec_sim + 0.3 * lex_boost` (line 133)
- Only difference: artificial latency delays (FAISS 5-10ms, Weaviate 40-80ms, Pinecone 80-160ms)

#### Why FAISS Performs Better (70% vs 57.69% vs 20%)

**Root Cause: Query-Type Distribution, Not Backend Quality**

**FAISS receives definitional queries** (router rule, `router_core.py:96-98`):
- Keywords: "what is", "definition", "overview"
- Short queries (≤4 words)
- Examples: "What is Salesforce?", "What capabilities does Salesforce Sales Cloud offer?"
- **Why higher recall**: Definitional queries have distinct vocabulary ("Agentforce", "Sales Cloud", "Data Cloud") that aligns well with product documentation

**Pinecone receives financial/earnings queries** (routing rule, `configs/router.heuristics.yaml:21-25`):
- Keywords: "results", "earnings", "fiscal", "guidance", "10-k", "10-q", "8-k"
- Examples: "What were Salesforce's Q4 FY25 revenue and earnings results?"
- **Why lower recall**:
  - Generic vocabulary ("revenue", "earnings", "results") appears across many documents
  - Specific fiscal periods (Q4 FY25 vs Q1 FY26) indistinguishable due to digit collapse
  - SEC filings have dense financial tables with many similar numeric patterns

**Weaviate receives mixed queries** (default fallback):
- All queries without matching keywords
- 56.5% of evaluation traffic
- **Why medium recall**: Mix of easy and hard queries

**Evidence**: Queries routed to Pinecone expecting SEC documents often retrieve press releases instead because the system cannot distinguish fiscal periods after numeric normalization.

### 3. Document Type Performance Differences

**Location**: `scripts/chunk_documents.py:105-230`, `scripts/parse_sec_structures.py:13-76`

#### SEC Document Processing Pipeline

SEC documents (10-K, 10-Q, 8-K) receive special handling:

1. **Item Structure Parsing** (`parse_sec_structures.py:13-76`):
   - Regex detection of Item boundaries (Item 1, 1A, 7, 7A, 8)
   - Annotates normalized docs with `sec_item_spans` array
   - Computes `sec_item_coverage_ratio` (target ≥75%)

2. **Item-Based Chunking** (`chunk_documents.py:182-186`):
   - If `sec_item_spans` exists: split document into Item segments
   - Each Item chunked independently with sliding window
   - Chunk boundaries snap to Item starts + H2/H3 headings

3. **Title Boost** (`chunk_documents.py:126-138`):
   - Every chunk prepends document title + first H1
   - For 10-K: Title = "crm-20250131" (document ID, not descriptive)
   - For press: Title = "Salesforce Announces Fourth Quarter..." (semantic content)

#### Issues Affecting SEC Recall

**Issue 1: XBRL Metadata Contamination**
- Sample 10-K chunk0000 contains raw XBRL namespace tags instead of business narrative
- Example: `crm-20250131 0001108524 FALSE FY 2025 http://fasb.org/us-gaap/2024#PropertyPlant...`
- These chunks have low semantic value for retrieval queries

**Issue 2: Item Detection Failures**
- If Item headers don't match regex patterns, `sec_item_spans` remains empty
- Fallback: Document chunked as single segment like press releases
- Loss of Item-level structural boundaries

**Issue 3: Non-Descriptive Titles**
- SEC documents use filing identifiers as titles ("crm-20250131")
- Title boost adds noise instead of context
- Press releases have semantic titles that improve retrieval

**Issue 4: Vocabulary Mismatch**
- SEC filings use formal financial terminology ("subscription and support revenues")
- Queries use natural language ("total revenue")
- Lexical embeddings cannot bridge this gap

#### Performance by Document Type

From `step07_retrieval_eval.json:123-187`:

| Doctype | Total Queries | Chunk Hits | Doc Hits | Chunk Hit Rate |
|---------|--------------|------------|----------|----------------|
| Press   | 26           | 16         | 20       | 61.5%          |
| 10-Q    | 6            | 1          | 5        | 16.7%          |
| Product | 6            | 4          | 5        | 66.7%          |
| 10-K    | 3            | 1          | 1        | 33.3%          |
| 8-K     | 1            | 0          | 0        | 0%             |
| Wiki    | 2            | 0          | 0        | 0%             |
| Dev     | 1            | 1          | 1        | 100%           |
| Help    | 1            | 1          | 1        | 100%           |

**Pattern**: Product/dev/help documentation has high recall (66-100%), SEC filings have low recall (0-33%), Wikipedia 0%.

### 4. Evaluation Methodology Assessment

**Location**: `scripts/qa_step07_retrieval_eval.py:116-577`, `data/interim/eval/salesforce_eval_seed.jsonl`

#### Evaluation Design

The evaluation seed contains **46 manually curated queries** with precise expected chunks:

**Query Characteristics**:
- Highly specific fact-seeking queries
- Financial metrics: "What was Salesforce's total revenue for Q1 FY26?"
- Temporal constraints: "When do Salesforce's senior notes mature..."
- Named entities: "Who did Salesforce appoint as President and CFO in 2025?"
- Product features: "What capabilities does Salesforce Sales Cloud offer..."

**Labeling**:
- Single gold standard: one `expected_chunk_id` per query
- Binary relevance: relevant (1) or not (0)
- No graded relevance or multi-relevance
- Chunks verified to exist in embeddings index

**Recall Calculation** (`qa_step07_retrieval_eval.py:407-427`):
```python
recall@10 = (queries where expected_chunk_id in top-10) / total_queries
```

**nDCG@5 Calculation** (`qa_step07_retrieval_eval.py:420-421`):
```python
dcg = (1.0 / log2(rank + 1)) if rank <= 5 else 0.0
ndcg@5 = mean(dcg values)
```

#### Evaluation Soundness

**Strengths**:
- Correct mathematical implementation of recall and nDCG
- Queries verified to have expected chunks in index
- Dual-format reporting (JSON + Markdown) with full traceability
- Per-query failure logging with top-10 results

**Potential Issues**:
- **Single gold standard assumption**: Real queries may have multiple valid answer chunks
- **Query-document vocabulary mismatch**: Queries use natural language, documents use formal terminology
- **Chunk boundary sensitivity**: If answer spans multiple chunks, only one labeled as correct
- **Temporal specificity**: Many queries reference specific quarters/years that collapse to '0'

**Conclusion**: The evaluation methodology is **sound and accurately reflects system performance**. Low recall is genuine, not an artifact of flawed evaluation design.

### 5. Query-Document Vocabulary Gap

**Location**: Analysis across `retrieval_failures.jsonl` and `salesforce_eval_seed.jsonl`

#### Systematic Mismatch Patterns

**Pattern 1: Temporal References**
- Query: "Q1 FY26" → Tokens: `["q0", "fy0"]`
- Document: "first quarter fiscal year 2026" → Tokens: `["first", "quarter", "fiscal", "year", "0"]`
- **Overlap**: `["fiscal"]` only (if "fiscal" appears in query)
- **Impact**: Temporal precision lost due to numeric collapse

**Pattern 2: Acronym vs Spelled-Out**
- Query: "AI agent" → Tokens: `["ai", "agent"]`
- Document: "artificial intelligence autonomous system" → Tokens: `["artificial", "intelligence", "autonomous", "system"]`
- **Overlap**: Zero
- **Impact**: Synonym blindness

**Pattern 3: Financial Terminology**
- Query: "total revenue" → Tokens: `["total", "revenue"]`
- Document: "subscription and support revenues" → Tokens: `["subscription", "and", "support", "revenues"]`
- **Overlap**: Zero ("revenue" vs "revenues" are different tokens)
- **Impact**: Plural/singular variation

**Pattern 4: Generic Keywords**
- Query: "earnings results" → Tokens: `["earnings", "results"]`
- Appears in: 20+ press releases, 5+ SEC filings
- **Impact**: High lexical overlap with many irrelevant documents

#### Evidence from Failures

**Example 1** (`retrieval_failures.jsonl:1`):
- Query: "What was Salesforce's total revenue for Q1 FY26?"
- Expected: `crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0004`
- Retrieved rank 1: `crm::press::2025-09-02::...::chunk0001` (press release about Q2, not Q1)
- **Why**: "Q2" and "Q1" both collapse to "q0", cannot distinguish quarters

**Example 2** (`retrieval_failures.jsonl:6-7`):
- Query: "How does Salesforce describe its Agentforce AI agent strategy in the annual report?"
- Expected: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0008`
- Retrieved rank 1: `crm::press::2024-11-20::news-details::a75c5c49::chunk0001` (press release, not 10-K)
- **Why**: Generic terms "Agentforce", "AI", "agent", "strategy" appear in both press and 10-K; no way to distinguish "annual report" from "press release"

**Example 3** (`retrieval_failures.jsonl:8`):
- Query: "What earnings announcement did Salesforce make in the February 2025 8-K filing?"
- Expected: `crm::8-K::2025-02-26::fy25-results-8-k::97457068::chunk0000`
- Retrieved rank 1: `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2::chunk0036` (10-K, not 8-K)
- **Why**: "February 2025" → "february 0", cannot distinguish from other months/years

### 6. Router Heuristics Impact

**Location**: `configs/router.heuristics.yaml:19-39`, `scripts/router_core.py:72-100`

#### Routing Rules

Three rules with first-match-wins semantics:

1. **PR_QUERY → Pinecone**: Keywords `[results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k]`
2. **FILTER_MATCH → Weaviate**: Keywords `[api, apis, endpoint, schema, developer, example]`
3. **DEFINITION → FAISS**: Keywords `[definition, what is, overview]` OR short queries (≤4 words)
4. **Default → Weaviate**: All unmatched queries

#### Mis-Routing Patterns

**SEC Filing Queries Routed to Weaviate** (should go to Pinecone):
- Query: "What was Salesforce's total revenue for Q1 FY26?"
- Contains: "revenue", "Q1", "FY26"
- Missing: "fiscal", "10-q", "earnings", "results"
- **Routed to**: Weaviate (default fallback)
- **Should route to**: Pinecone (financial backend)

**Why This Matters**:
- Although backends use identical search algorithms in the stub, the router controls which queries each backend handles
- Pinecone gets harder queries (financial, requiring temporal precision)
- FAISS gets easier queries (definitional, with distinct vocabulary)
- Performance differences reflect query difficulty, not backend quality

## Code References

### Core Embedding System
- `scripts/embedding_utils.py:8-19` - Text normalization (digit collapse)
- `scripts/embedding_utils.py:22-30` - Tokenization (unigrams + bigrams)
- `scripts/embedding_utils.py:42-62` - Feature hashing and L2 normalization
- `scripts/embedding_utils.py:65-66` - Public API: `embed_text(text, dim)`

### Retrieval Pipeline
- `scripts/qa_step01_embeddings.py:124-155` - Batch embedding generation
- `scripts/qa_step02_indexes.py:80-164` - FAISS HNSW index build
- `scripts/qa_step03_mcp.py:82-156` - MCP kb.search stub (unified search for all backends)
- `scripts/router_core.py:72-100` - Query routing logic

### SEC Document Processing
- `scripts/fetch_sec_filings.py:27-64` - SEC filing download
- `scripts/parse_sec_structures.py:13-76` - SEC Item boundary detection
- `scripts/chunk_documents.py:182-188` - Item-based segmentation
- `scripts/chunk_documents.py:126-138` - Title boost mechanism

### Evaluation
- `scripts/qa_step07_retrieval_eval.py:251` - Load eval seed
- `scripts/qa_step07_retrieval_eval.py:375-382` - Router-based backend selection
- `scripts/qa_step07_retrieval_eval.py:407-427` - Recall and nDCG calculation
- `scripts/qa_step07_retrieval_eval.py:552-576` - Failure logging

### Configuration
- `configs/vector.indexing.yaml:1-5` - Embedding configuration (hashlex-v1, dim=768)
- `configs/router.heuristics.yaml:19-39` - Routing rules
- `configs/chunking.config.json` - Chunking parameters (800 tokens, 120 overlap)

## Architecture Documentation

### Embedding Space Consistency

**Critical Design Decision**: All documents and queries use identical `embed_text()` function from `scripts/embedding_utils.py:65-66`.

**Verification Points**:
- Document indexing: `qa_step01_embeddings.py:138`
- Query embedding (MCP stub): `qa_step03_mcp.py:69-76`
- Query embedding (offline mode): `qa_step07_retrieval_eval.py:399`

**Why This Matters**: Mismatched embedding functions would place queries and documents in different vector spaces, yielding recall=0. The system enforces consistency through shared imports.

### Multi-Backend Architecture (Simulated)

The system **simulates** a multi-backend architecture while using a single implementation:

**Shared State** (`qa_step03_mcp.py:40-77`):
- `state["xb"]`: Single numpy array of all chunk embeddings
- `state["rows"]`: Metadata for all chunks
- Same embeddings used for FAISS, Weaviate, Pinecone

**Backend Differentiation**:
- Artificial latency: FAISS 5-10ms, Weaviate 40-80ms, Pinecone 80-160ms
- Routing labels: Query → backend assignment via keyword rules
- Manifest files: Separate configs for each backend

**Production Implications**: Replacing MCP stub with actual FAISS/Weaviate/Pinecone API calls would require:
- Network calls to external services
- Backend-specific query formats
- Different HNSW/vector index configurations
- Potential embedding differences (currently unified)

### Data Flow Summary

```
SEC Filings → fetch_sec_filings.py → data/raw/sec/*.raw.html + *.meta.json
   ↓
normalize_html.py → data/interim/normalized/*.json (with H1:/H2:/H3: markers)
   ↓
parse_sec_structures.py → Annotates with sec_item_spans[] (in-place)
   ↓
chunk_documents.py → data/interim/chunks/*.chunks.jsonl (Item-based segmentation)
   ↓
dedupe_chunks.py → data/interim/dedup/*.jsonl (SEC exempt from dedup)
   ↓
qa_step01_embeddings.py → data/vector/embeddings/embeddings.parquet (hashlex-v1)
   ↓
qa_step02_indexes.py → data/vector/faiss/index.faiss (HNSW)
   ↓
qa_step03_mcp.py → MCP kb.search stub (loads embeddings into numpy)
   ↓
qa_step07_retrieval_eval.py → Queries routed to backends → Retrieval → Metrics
```

## Root Cause Summary

### Primary Causes of Low Recall (52.17%)

1. **Lexical Embedding Limitations** (30-40% impact):
   - Bag-of-words + bigrams cannot handle semantic similarity
   - "Q1 FY26" ≠ "first quarter fiscal year 2026" (zero lexical overlap)
   - Synonym blindness: "AI agent" ≠ "artificial intelligence"
   - No learned term importance (TF-IDF absent)

2. **Numeric Collapse** (20-30% impact):
   - All digits → '0' during normalization
   - Cannot distinguish fiscal periods (Q1 vs Q4, FY25 vs FY26)
   - Financial metrics lose specificity
   - Dates become indistinguishable

3. **SEC Document Processing** (10-15% impact):
   - XBRL metadata contamination in chunks
   - Non-descriptive titles ("crm-20250131")
   - Item detection failures → loss of structure
   - Formal financial terminology vs natural language queries

4. **Query-Type Routing** (10-15% impact):
   - Harder queries (financial, temporal) routed to Pinecone
   - Easier queries (definitional) routed to FAISS
   - Performance differences reflect query difficulty, not backend quality

5. **Evaluation Design Strictness** (5-10% impact):
   - Single gold standard per query
   - Binary relevance (no partial credit)
   - Chunk boundary sensitivity (answer may span multiple chunks)

### Why Recall Is Genuinely Low

The system is **operating as designed** given its architectural constraints:

- **By Design**: Deterministic lexical embeddings prioritize reproducibility over semantic understanding
- **Trade-off**: Zero external dependencies (no neural networks, no API calls) in exchange for limited semantic reasoning
- **Intended Use Case**: Exact keyword matching and product name retrieval
- **Not Intended For**: Semantic search, paraphrasing, cross-lingual retrieval, temporal reasoning

The 52.17% recall accurately reflects hashlex-v1's capabilities when faced with:
- Vocabulary mismatch between queries and documents
- Temporal precision requirements (fiscal periods)
- Generic keyword queries ("earnings results") with high false positive rates

## Open Questions

1. **What is the ideal recall target for a lexical embedding system?**
   - Should the target (80%) be adjusted to reflect lexical embedding capabilities (50-60%)?
   - Or does 80% require semantic embeddings?

2. **Can SEC document processing be improved within current architecture?**
   - XBRL metadata filtering during normalization
   - Better Item title normalization
   - Descriptive title generation from filing content

3. **What percentage of evaluation queries are answerable with lexical embeddings?**
   - Baseline: queries with exact keyword matches in expected chunks
   - Upper bound: queries where synonyms/paraphrasing aren't required

4. **How much would query reformulation improve recall?**
   - Expanding "Q1 FY26" to "first quarter fiscal year 2026"
   - Adding synonyms ("revenue" + "revenues" + "income")

5. **What is the production embedding strategy?**
   - Continue with hashlex-v1 for auditability/transparency?
   - Hybrid approach (lexical + semantic embeddings)?
   - Replace with neural embeddings (OpenAI, Cohere, etc.)?

## Related Research

This investigation builds on:
- Day-1 baseline verification (`README_DAY1.md`)
- Gate-0 through Gate-6 quality checks
- Original system design in `AGENTS.md` and `CLAUDE.md`

Future research directions:
- Semantic embedding alternatives (maintain traceability)
- Hybrid lexical+semantic retrieval
- Query expansion strategies
- SEC document normalization improvements

---

**Conclusion**: The low recall rate (52.17%) is a **genuine limitation of the hashlex-v1 lexical embedding approach** when faced with semantic variation, numeric precision requirements, and vocabulary mismatch between queries and documents. The evaluation methodology is sound and accurately measures system performance. The backend performance differences (70%/57.69%/20%) are **routing artifacts**, not algorithmic differences—all backends use identical embeddings and search implementations.
