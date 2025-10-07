---
date: 2025-10-07T14:38:32-04:00
researcher: Unknown
git_commit: e6597f5ac21d7a1d210428bddbc5dd75374fde90
branch: agent-weaviate
repository: agent-weaviate
topic: "Query Generation Logic Investigation (Issue #6)"
tags: [research, codebase, issue006, eval, query-generation, gate-7]
status: complete
last_updated: 2025-10-07
last_updated_by: Unknown
---

# Research: Query Generation Logic Investigation (Issue #6)

**Date**: 2025-10-07 14:38:32 EDT
**Researcher**: Unknown
**Git Commit**: e6597f5ac21d7a1d210428bddbc5dd75374fde90
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

Check the query generation logic of the eval function and see if there are any problems with the current query.

## Summary

Investigation of the query generation logic in the evaluation system reveals a **fundamental disconnect** between the automated query generation implementation (`build_eval_seed.py`) and the actual evaluation queries being used. The `build_eval_seed.py` script generates overly generic template-based queries like "What does this document say about Salesforce?", but the actual evaluation seed file (`salesforce_eval_seed.jsonl`) contains manually-curated, highly specific queries like "What was Salesforce's total revenue for Q1 FY26?".

This indicates that the automated query generation is **not being used in practice** because it produces poor quality queries that don't represent realistic information needs. The evaluation system currently relies entirely on manual query authoring, which doesn't scale and creates a bottleneck for expanding the evaluation dataset.

## Detailed Findings

### 1. Query Generation Implementation (build_eval_seed.py)

**Location**: `scripts/build_eval_seed.py:183-189`

The automated query generation uses a simple template-based approach:

```python
base_kw = (title.split(" ")[0] if title else "Agentforce").lower()
query = f"What does this document say about {base_kw}?"
# Fallback if duplicate:
query = f"Summarize key points regarding {base_kw}"
```

**Problems with this approach**:

1. **Overly Generic**: Extracts only the first word of the document title (e.g., "Salesforce") and creates vague questions
2. **No Context**: Ignores specific financial metrics, product names, dates, or technical details present in the document
3. **Poor Information Need Modeling**: Doesn't reflect how real users query for information (specific metrics vs. general summaries)
4. **Limited Variation**: Only two query templates, leading to repetitive and similar queries
5. **No Semantic Alignment**: Query doesn't capture the specific information present in the expected chunk

**Examples of what it would generate**:
- "What does this document say about Salesforce?"
- "What does this document say about Q1?"
- "Summarize key points regarding Salesforce"

### 2. Actual Evaluation Queries (salesforce_eval_seed.jsonl)

**Location**: `data/interim/eval/salesforce_eval_seed.jsonl`

The actual eval seed contains manually-curated, realistic queries that differ dramatically from the template output:

**Sample queries from eval seed**:
- Line 1: `"What was Salesforce's total revenue for Q1 FY26?"` (specific financial metric)
- Line 2: `"How much operating cash flow did Salesforce generate in Q1 FY26?"` (specific metric with temporal context)
- Line 3: `"When do Salesforce's senior notes mature and what are the interest rates?"` (specific debt structure question)
- Line 5: `"What are the key risks Salesforce identifies related to AI and generative AI?"` (specific risk category)
- Line 18: `"What is Salesforce AgentExchange and when was it launched?"` (product + temporal detail)

**Quality characteristics of actual queries**:
- Specific financial metrics (revenue, cash flow, earnings)
- Temporal precision (Q1 FY26, FY25, specific dates)
- Product names (AgentExchange, Agentforce 2.0)
- Technical details (senior notes, interest rates, maturity dates)
- Regulatory/compliance focus (GDPR, data protection)
- Role-appropriate personas (CFO asks financial questions, CIO asks technical questions)

### 3. Query Processing Pipeline (qa_step07_retrieval_eval.py)

**Location**: `scripts/qa_step07_retrieval_eval.py:368-403`

The evaluation system loads queries from the eval seed and processes them without any transformation:

```python
# Line 369: Direct extraction
q = (it.get("query_text") or "").strip()

# Line 375: Router decision based on keywords
backend, reasons = decide_backend(q, persona, None)

# Line 386: Embed query using OpenAI ada-002
qv = embed_query(q, dim)  # Calls embed_text() from embedding_utils.py

# Lines 387-394: Retrieve top-10 via L2 distance
for idx, v in enumerate(vectors):
    dist = sum((x-y)*(x-y) for x,y in zip(qv, v))
    ranked.append((cids[idx], -dist))
```

**Key observations**:
- No preprocessing or normalization of query text before embedding
- Queries sent to OpenAI ada-002 API as-is (line 386)
- Relies entirely on OpenAI's internal preprocessing
- Router uses keyword matching on lowercased query (line 375)

### 4. Embedding Consistency (embedding_utils.py)

**Location**: `scripts/embedding_utils.py:86-133`

Both documents and queries use the same `embed_text()` function:

```python
def embed_text(text: str, dim: int) -> List[float]:
    # 1. Validate dimension (must be 1536 for ada-002)
    if dim != ADA002_DIM:
        raise ValueError(f"OpenAI ada-002 requires dim={ADA002_DIM}")

    # 2. Handle empty text
    if not text.strip():
        return [0.001] * 1536

    # 3. Check cache (SHA256 hash of text)
    cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]

    # 4. Call OpenAI API (with retry logic)
    embedding = _call_openai_api(text)  # model="text-embedding-ada-002"

    # 5. Return raw vector (no normalization)
    return embedding
```

**Validation**: All code paths confirmed to use this function:
- Gate-1 documents: `qa_step01_embeddings.py:173` → `embed_batch()` → `embed_text()`
- Gate-7 queries: `qa_step07_retrieval_eval.py:235` → `embed_text()`
- Gate-3 MCP stub: `qa_step03_mcp.py:73` → `embed_text()`
- Gate-2 FAISS test: `qa_step02_indexes.py:218` → `embed_text()`

**Result**: No embedding mismatch issues. Documents and queries exist in the same vector space.

### 5. Router Logic (router_core.py)

**Location**: `scripts/router_core.py:72-100`

The router selects backends using keyword matching:

```python
# Line 78: Lowercase query for matching
ql = (query or "").lower()

# Lines 84-89: First-match-wins keyword rules
for rule in rules:
    keywords = rule.get("keywords", [])
    if any(kw in ql for kw in keywords):
        return rule["backend"], rule["reason"]

# Lines 91-94: Persona bias
if persona in persona_preferences:
    return persona_preferences[persona], "PERSONA_MATCH"

# Lines 96-100: Default heuristics
if len(ql.split()) <= 4:
    return "faiss", "SHORT_QUERY"
```

**Keyword routing examples**:
- `["results", "earnings", "fiscal", "guidance"]` → pinecone (PR_QUERY)
- `["api", "apis", "endpoint", "schema"]` → weaviate (FILTER_MATCH)
- `["definition", "what is", "overview"]` → faiss (DEFINITION)

**Persona mappings**:
- `vp_sales_ops` → pinecone
- `cio` → weaviate
- `vp_customer_experience` → faiss

### 6. Discrepancy Analysis

**Template-Generated Query**:
```
Query: "What does this document say about Salesforce?"
Quality: Generic, vague
Specificity: None
User intent: Unclear (exploratory?)
Expected behavior: Broad document retrieval
```

**Manually-Curated Query**:
```
Query: "What was Salesforce's total revenue for Q1 FY26?"
Quality: High specificity
Specificity: Financial metric + temporal + entity
User intent: Clear (specific data point)
Expected behavior: Retrieve exact section with revenue figure
```

**Impact on Retrieval Evaluation**:
- Generic queries may retrieve multiple relevant sections (ambiguous ground truth)
- Specific queries have clear success criteria (exact chunk or fail)
- Current low recall (65.22% chunk recall@10) may partially stem from query quality mismatch
- SEC filing failures (0% recall on 10-Q/10-K) indicate semantic gap between queries and dense formal text

### 7. Retrieval Failure Patterns (retrieval_failures.jsonl)

**Location**: `reports/eval/retrieval_failures.jsonl`

Analysis of failures shows consistent pattern:

**Query**: "What was Salesforce's total revenue for Q1 FY26?"
**Expected chunk**: `crm::10-Q::2025-04-30::...::chunk0004` (SEC 10-Q filing)
**Retrieved**: Press releases from `investor.salesforce.com` (ranks 1-10)
**Classification**: `chunk_miss_doc_miss` (neither chunk nor document found)

**Query**: "What was Salesforce's full year FY25 financial performance?"
**Expected chunk**: `crm::10-K::2025-03-05::...::chunk0006` (SEC 10-K filing)
**Retrieved**: Press releases about FY25 results
**Classification**: `chunk_miss_doc_miss`

**Pattern**: Financial queries retrieve press releases instead of SEC filings, despite both containing similar revenue/earnings information. This suggests:
- Press releases use more natural language similar to queries
- SEC filings contain tables, legal language, or formatting that doesn't embed well with OpenAI ada-002
- The embedding model may favor conversational text over formal/structured documents

### 8. Historical Context (from thoughts/ directory)

**Related research documents found**:

1. **`thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md`** - Root cause analysis revealing four interrelated factors causing low recall (52.17% chunk recall@10)

2. **`thoughts/shared/research/2025-10-07-issue003-retrieval-recall-investigation.md`** - Investigation of five interconnected root causes for Gate-7 retrieval failures (65.22% chunk recall, 34.41% nDCG@5)

3. **`thoughts/shared/research/2025-10-07-issue004-retrieval-recall-investigation.md`** - Critical investigation confirming system uses OpenAI ada-002 (1536-dim) NOT hashlex-v1 (768-dim), with Gate-7 RED status analysis

4. **`thoughts/shared/issues/issue002.md`** - Root cause investigation for low recall rate
5. **`thoughts/shared/issues/issue003.md`** - Gate-7 RED status investigation (0% recall on SEC filings)
6. **`thoughts/shared/issues/issue005.md`** - SEC 10-K/10-Q filings achieving 0% chunk-level recall

**Cross-cutting themes**:
- Evaluation methodology issues (ground truth generation)
- SEC filing recall failures (structured text embedding problems)
- Router reranking effectiveness (17.39% near-miss rate)
- Query generation quality (this investigation)

## Code References

### Query Generation
- `scripts/build_eval_seed.py:183-189` - Template-based query generation logic
- `scripts/build_eval_seed.py:42-66` - Keyphrase extraction
- `scripts/build_eval_seed.py:159-180` - Persona assignment
- `data/interim/eval/salesforce_eval_seed.jsonl:1-20` - Manually-curated queries

### Query Processing
- `scripts/qa_step07_retrieval_eval.py:368-403` - Query evaluation loop
- `scripts/qa_step07_retrieval_eval.py:233-235` - Query embedding wrapper
- `scripts/embedding_utils.py:86-133` - OpenAI ada-002 embedding implementation
- `scripts/router_core.py:72-100` - Backend routing logic

### Configuration
- `configs/vector.indexing.yaml:2-4` - Embedding model (openai-ada-002, dim=1536)
- `configs/router.heuristics.yaml:19-40` - Keyword routing rules

### Evaluation Reports
- `reports/qa/step07_retrieval_eval.md` - Retrieval evaluation summary (65.22% recall@10)
- `reports/eval/retrieval_failures.jsonl` - Failed query traces with diagnostics

## Problems Identified

### Problem 1: Automated Query Generation Not Used
**Impact**: High
**Description**: The `build_eval_seed.py` script generates overly generic queries, but the actual eval seed contains manually-curated specific queries. This creates a maintenance bottleneck and limits eval dataset scalability.

**Evidence**:
- Template generates: "What does this document say about Salesforce?"
- Actual queries: "What was Salesforce's total revenue for Q1 FY26?"
- 46 queries in eval seed, all manually authored

### Problem 2: Template Quality Insufficient
**Impact**: High
**Description**: The template-based approach produces queries that don't reflect realistic user information needs or capture specific content from documents.

**Evidence**:
- Only uses first word of title as base keyword
- Two query templates total
- No extraction of financial metrics, dates, product names, or technical details

### Problem 3: No Query Variation Strategy
**Impact**: Medium
**Description**: The system doesn't generate multiple query variations per document chunk, limiting coverage of different ways users might seek the same information.

**Evidence**:
- One query per chunk
- No query expansion or reformulation
- No paraphrasing or synonym substitution

### Problem 4: Semantic Gap Between Queries and Documents
**Impact**: High
**Description**: Manually-curated queries use natural language ("What was revenue?") while source chunks state facts ("Revenue was $X million"). This gap must be bridged by embeddings, and ada-002 struggles with formal/structured text.

**Evidence**:
- SEC filings: 0-10% chunk recall
- Press releases: 77% chunk recall
- retrieval_failures.jsonl shows consistent pattern of retrieving press over SEC filings

### Problem 5: No Query-Specific Preprocessing
**Impact**: Medium
**Description**: Queries are embedded as-is with no preprocessing, normalization, or enhancement. The system relies entirely on OpenAI's internal processing.

**Evidence**:
- Line 369: Direct extraction `q = it.get("query_text")`
- Line 386: Direct embedding `qv = embed_query(q, dim)`
- No expansion, no stopword removal, no entity extraction

### Problem 6: Persona-Query Mismatch Possible
**Impact**: Low
**Description**: The automated generator assigns personas via round-robin rotation (i % 3), which doesn't align persona expertise with document content type.

**Evidence**:
- Line 180: `persona = personas[i % 3]`
- A CFO persona might be assigned to a technical developer doc
- Manually-curated queries properly align personas (CFO asks financial questions, CIO asks technical)

### Problem 7: Scalability Bottleneck
**Impact**: High
**Description**: Manual query authoring doesn't scale. Expanding evaluation coverage to 100+ documents requires proportional human effort.

**Evidence**:
- Current: 46 manually-curated queries
- Target scale: 100+ documents × 3-5 queries each = 300-500 queries
- No automation path available

## Architecture Documentation

### Current Query Generation Flow

```
Document Chunks (*.chunks.jsonl)
    ↓
[AUTOMATED BUT UNUSED]
Select 45+ chunks across doc types (build_eval_seed.py:106-152)
    ↓
Template-based generation (build_eval_seed.py:183-189)
    - Extract first word of title
    - Create "What does this document say about {word}?"
    ↓
Output: Generic, low-quality queries
========================================
[MANUAL PROCESS IN PRACTICE]
Human reviews document chunks
    ↓
Manually authors realistic, specific queries
    ↓
Output: salesforce_eval_seed.jsonl (high-quality queries)
```

### Retrieval Evaluation Flow

```
Load eval seed (qa_step07_retrieval_eval.py:251)
    ↓
For each query:
    Extract query text (line 369)
        ↓
    Route to backend (line 375: router_core.decide_backend)
        - Keyword matching on lowercased query
        - Persona bias lookup
        - Default heuristics
        ↓
    Embed query (line 386: embedding_utils.embed_text)
        - Check cache (SHA256 hash)
        - Call OpenAI ada-002 API
        - Return 1536-dim vector
        ↓
    Retrieve top-10 (line 387-394: L2 distance)
        ↓
    Check recall (line 409-414: expected_chunk_id in results)
        ↓
    Log failures (line 553-576: retrieval_failures.jsonl)
```

### Embedding Consistency

```
Documents (Gate-1)
    ↓
embedding_utils.embed_batch() → embed_text()
    ↓
OpenAI ada-002 API (1536-dim)
    ↓
data/vector/embeddings/embeddings.parquet

Queries (Gate-7)
    ↓
embedding_utils.embed_text()
    ↓
OpenAI ada-002 API (1536-dim)
    ↓
In-memory vector for retrieval

[SAME FUNCTION, SAME MODEL, SAME VECTOR SPACE]
```

## Related Research

**Retrieval recall investigations**:
- `thoughts/shared/research/2025-10-07-issue002-low-recall-investigation.md` - 52.17% recall@10 analysis
- `thoughts/shared/research/2025-10-07-issue003-retrieval-recall-investigation.md` - Gate-7 RED status (65.22% recall, 34.41% nDCG)
- `thoughts/shared/research/2025-10-07-issue004-retrieval-recall-investigation.md` - ada-002 vs hashlex-v1 confirmation

**Issue tickets**:
- `thoughts/shared/issues/issue002.md` - Low recall investigation request
- `thoughts/shared/issues/issue003.md` - Gate-7 RED status (0% SEC recall)
- `thoughts/shared/issues/issue005.md` - SEC filing recall failure (0/9 queries)
- `thoughts/shared/issues/issue006.md` - Query generation investigation request (this document)

## Open Questions

1. **Can LLMs generate realistic queries automatically?**
   - Could GPT-4 analyze document chunks and generate specific, realistic queries similar to the manually-curated ones?

2. **Should query generation target specific information types?**
   - Financial metrics (revenue, cash flow, earnings)
   - Product features (capabilities, launch dates)
   - Risk factors (compliance, security, data protection)
   - Technical specifications (APIs, endpoints, schemas)

3. **How many query variations per chunk are optimal?**
   - Current: 1 query per chunk
   - Alternative: 3-5 variations per chunk to test different query formulations

4. **Should queries be chunk-specific or document-specific?**
   - Current: Chunk-level (expected_chunk_id)
   - Alternative: Document-level with multiple valid chunks

5. **Does query quality explain SEC filing recall failures?**
   - Generic queries may retrieve press releases (natural language) over SEC filings (formal/structured)
   - More specific queries might improve SEC recall by reducing semantic ambiguity

6. **Should there be query-specific preprocessing?**
   - Entity extraction (dates, metrics, product names)
   - Query expansion (synonyms, related terms)
   - Hybrid search (keyword + semantic)

7. **Is the persona-to-query alignment important for evaluation?**
   - Do CFO-authored queries have different retrieval patterns than CIO-authored queries?
   - Should personas guide query generation (not just assignment)?

## Recommendations

While this document focuses on documenting existing issues, the following areas warrant further investigation:

1. **Query generation automation**: Explore LLM-based query synthesis that maintains the specificity of manual queries
2. **Query variation coverage**: Investigate generating multiple query formulations per document chunk
3. **Persona-content alignment**: Study whether aligning persona expertise with document type improves evaluation realism
4. **Query preprocessing**: Research whether entity extraction or query expansion improves retrieval
5. **Evaluation ground truth**: Re-examine whether chunk-level ground truth is appropriate vs document-level
6. **SEC filing handling**: Investigate specialized preprocessing for structured/tabular documents

## Conclusion

The query generation logic in `build_eval_seed.py` produces overly generic template-based queries that are **not used in practice**. The evaluation system relies entirely on manually-curated, specific queries that better represent realistic user information needs. This creates a scalability bottleneck as the evaluation dataset grows. The template approach fundamentally lacks the specificity required for effective retrieval evaluation - it cannot extract financial metrics, product names, temporal details, or technical specifications that characterize real queries.

The embedding consistency between documents and queries is verified and correct - both use `embed_text()` with OpenAI ada-002 (1536-dim). The retrieval recall issues (65.22% chunk recall@10, 0% on SEC filings) stem from other factors: semantic gap between natural language queries and formal document text, document type bias in embeddings, and router effectiveness.

Addressing the query generation quality gap requires either:
1. Automating high-quality query synthesis (e.g., using LLMs to generate specific queries)
2. Accepting manual curation as necessary for evaluation quality
3. Redesigning evaluation methodology to use document-level rather than chunk-level ground truth
