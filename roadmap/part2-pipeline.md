---
date: 2025-10-20T15:54:47-04:00
researcher: Claude Code
git_commit: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
branch: agent-weaviate
repository: agent-weaviate
topic: "Data Pipeline & Storage - Complete 13-Stage Pipeline Documentation"
tags: [research, codebase, pipeline, data-processing, quality-gates, architecture]
status: complete
last_updated: 2025-10-20
last_updated_by: Claude Code
---

# Research: Data Pipeline & Storage - Complete 13-Stage Pipeline Documentation

**Date**: 2025-10-20T15:54:47-04:00
**Researcher**: Claude Code
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

Document the **complete data pipeline** from raw document collection through processing to final artifacts, covering all 13 stages with implementation details, data formats, storage locations, and quality gates.

## Summary

The multi-agent RAG system implements a 13-stage gated data pipeline that transforms raw HTML/PDF documents into vector-indexed, retrieval-ready text chunks. The pipeline is split into two major phases:

**Stages 1-5 (Data Collection & Processing)**:
- Collection → Normalization → Metadata → Chunking → Deduplication
- Transforms 7 source types (SEC filings, press releases, dev docs, etc.) into deduplicated chunks
- Produces dual-format outputs at each stage (raw content + JSON metadata)

**Stages 6-13 (Vector Infrastructure & Evaluation)**:
- Embedding (Gate-1) → Indexing (Gate-2) → MCP Tools (Gate-3)
- Routing (Gate-4) → Graph (Gate-5) → A2A (Gate-6) → Retrieval Eval (Gate-7) → Generation Eval (Gate-8)
- Validates quality at each gate with automated checks

The system processes 80+ documents into 536+ chunks with 1536-dimensional embeddings, indexed across 3 vector stores (FAISS, Weaviate, Pinecone) for different query types. All stages emit dual-format reports (JSON + Markdown) for audit-ready traceability.

## 1. Overview

### Pipeline Architecture

The data pipeline consists of **13 sequential stages** organized into two major phases:

#### Phase 1: Data Collection & Processing (Stages 1-5)
1. **Collection** - 7 fetch scripts retrieve documents from web sources
2. **Normalization** - HTML/PDF → clean structured text
3. **Metadata Extraction** - Enrich with dates, topics, personas
4. **Chunking** - Split into retrieval-ready segments (800 tokens)
5. **Deduplication** - Remove near-duplicates (Jaccard similarity)

#### Phase 2: Vector Infrastructure & Evaluation (Stages 6-13)
6. **Embedding (Gate-1)** - OpenAI ada-002 (1536-dim)
7. **Indexing (Gate-2)** - FAISS/Weaviate/Pinecone indexes
8. **MCP Tools (Gate-3)** - Local service stubs (kb.search, web.fetch, etc.)
9. **Routing (Gate-4)** - Query routing heuristics
10. **Graph Execution (Gate-5)** - 8-node LangGraph pipeline
11. **Agent-to-Agent (Gate-6)** - Compliance negotiation
12. **Retrieval Evaluation (Gate-7)** - recall@10, nDCG@5
13. **Generation Evaluation (Gate-8)** - 10-run structural validation

### Purpose of Each Stage

**Collection**: Gathers raw documents from 7 sources (SEC, investor news, newsroom, dev docs, help docs, product pages, Wikipedia) with metadata tracking.

**Normalization**: Parses HTML/PDF, extracts text, removes navigation/boilerplate, preserves headings, detects language, applies domain-specific rules.

**Metadata**: Enriches documents with publish dates, topics (11 categories), personas (3 types), URLs, SEC item boundaries.

**Chunking**: Splits documents into 800-token chunks with 120-token overlap, respects heading boundaries, adds title boosts.

**Deduplication**: Identifies near-duplicates using 5-gram Jaccard similarity (≥0.85 threshold), selects canonical chunks.

**Embedding**: Generates vector embeddings via OpenAI ada-002, caches results (SHA-256 keys), validates L2 norms.

**Indexing**: Builds 3 vector indexes—FAISS (general), Weaviate (dev docs), Pinecone (press/financial)—with round-trip integrity tests.

**MCP Tools**: Validates 5 local service stubs for health, contract conformance, and latency budgets.

**Routing**: Routes queries to appropriate backends using keyword rules, persona bias, fallback mechanisms.

**Graph Execution**: Runs full 8-node pipeline (Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler).

**Agent-to-Agent**: Negotiates compliance constraints, enforces length/readability limits, validates references.

**Retrieval Evaluation**: Tests retrieval quality on 40+ labeled queries (recall@10 ≥ 0.80, nDCG@5 ≥ 0.60).

**Generation Evaluation**: Runs 10 end-to-end sessions, validates structure, compliance, persona alignment.

## 2. Architecture & Design

### Pipeline Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: DATA COLLECTION                          │
│  7 fetch_*.py scripts → data/raw/<bucket>/{doc_id}.{raw.html,meta.json}│
│  Sources: SEC, investor_news, newsroom, dev_docs, help_docs,         │
│           product, wikipedia                                          │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: NORMALIZATION                             │
│  normalize_html.py → data/interim/normalized/{doc_id}.json           │
│  Transformations: HTML→text, heading extraction, language detection  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 STAGE 3: METADATA EXTRACTION                         │
│  extract_metadata.py → Updates data/interim/normalized/*.json        │
│  Enrichments: dates, topics (11), personas (3), URLs                 │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: CHUNKING                               │
│  chunk_documents.py → data/interim/chunks/{doc_id}.chunks.jsonl      │
│  Algorithm: 800-token sliding window, 120-token overlap             │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    STAGE 5: DEDUPLICATION                            │
│  dedupe_chunks.py → Rewrites data/interim/chunks/*.chunks.jsonl      │
│  Method: 5-gram Jaccard similarity (≥0.85), canonical selection     │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 6: EMBEDDING (Gate-1)                        │
│  qa_step01_embeddings.py → data/vector/embeddings/embeddings.parquet │
│  Model: OpenAI ada-002 (1536-dim), cached in data/cache/embeddings/ │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 7: INDEXING (Gate-2)                         │
│  qa_step02_indexes.py → data/vector/{faiss,weaviate,pinecone}/      │
│  Indexes: FAISS HNSW, Weaviate schema, Pinecone manifest            │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  STAGE 8: MCP TOOLS (Gate-3)                         │
│  qa_step03_mcp.py → Validates local stubs on ports 7801-7805        │
│  Tools: kb.search, web.fetch, link.resolve, crm.lookup, safety.check│
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   STAGE 9: ROUTING (Gate-4)                          │
│  qa_step04_router.py → reports/router/step04_router_trace.jsonl     │
│  Strategy: Keyword rules → Persona bias → Fallback                  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                STAGE 10: GRAPH EXECUTION (Gate-5)                    │
│  qa_step05_graph.py → outputs/{session_id}/{insights,email,timing}  │
│  Nodes: 8 (Intake → Planner → Retriever → ... → Assembler)         │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│              STAGE 11: AGENT-TO-AGENT (Gate-6)                       │
│  qa_step06_a2a.py → Validates compliance from outputs/{session_id}/ │
│  Checks: Critical flags==0, length≤160 words, readability≤Grade 10  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│            STAGE 12: RETRIEVAL EVALUATION (Gate-7)                   │
│  qa_step07_retrieval_eval.py → reports/eval/retrieval_failures.jsonl│
│  Metrics: recall@10 ≥0.80, nDCG@5 ≥0.60, coverage ≥3 domains       │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│           STAGE 13: GENERATION EVALUATION (Gate-8)                   │
│  qa_step08_generation_eval.py → reports/eval/{generation,compliance}│
│  Runs: 10 sessions, validates structure, compliance, persona        │
└──────────────────────────────────────────────────────────────────────┘
```

### Stage Dependencies

**Sequential Dependencies**:
- Stage 2 (Normalization) requires Stage 1 (Collection) output
- Stage 3 (Metadata) requires Stage 2 output + Stage 1 sidecars
- Stage 4 (Chunking) requires Stage 3 output
- Stage 5 (Deduplication) requires Stage 4 output
- Stage 6 (Embedding) requires Stage 5 output
- Stage 7 (Indexing) requires Stage 6 output
- Stages 8-13 can run independently after Stage 7 completes

**Parallel Execution**:
- All 7 fetch scripts can run concurrently
- Normalization can process Phase A and Phase B in parallel (hash-based split)
- Metadata extraction can process Phase A and Phase B in parallel
- Gates 4-6 can run independently (evaluation only)
- Gate 7 and Gate 8 can run independently

### Data Transformations

**Collection → Normalization**:
- Input: HTML bytes + metadata JSON
- Output: Structured JSON with clean text
- Key transformations:
  - Remove navigation, footers, scripts, styles, XBRL tags
  - Convert headings to H1:/H2:/H3: markers
  - Preserve content via CSS selectors (article, main, .content)
  - Normalize whitespace (max 2 consecutive newlines)
  - Detect language via langdetect

**Normalization → Metadata**:
- Input: Normalized JSON with text field
- Output: Same JSON enriched with metadata fields
- Key transformations:
  - Extract publish_date from HTML meta, sidecar, or text regex
  - Assign topics via 11-category keyword matching
  - Assign persona_tags via keyword matching (3 personas)
  - Resolve title from 4-level precedence waterfall
  - Validate dates (1999-01-01 to today)

**Metadata → Chunking**:
- Input: Normalized JSON with metadata
- Output: JSONL file (one chunk per line)
- Key transformations:
  - Split by SEC item boundaries (for 10-K/10-Q/8-K)
  - Apply 800-token sliding window with 120-token overlap
  - Snap chunk boundaries to H2/H3 headings (50-char tolerance)
  - Prepend title boost to each chunk
  - Add local_heads context (last 2 headings before chunk)

**Chunking → Deduplication**:
- Input: JSONL chunk files
- Output: Rewritten JSONL with duplicates removed
- Key transformations:
  - Normalize text to lowercase alphanumeric tokens
  - Generate 5-gram shingles
  - Compute Jaccard similarity for candidate pairs
  - Select canonical chunk (earliest date, longest word count)
  - Backup originals to pre_dedupe/ before rewriting

**Deduplication → Embedding**:
- Input: Deduplicated chunks (JSONL)
- Output: Parquet table with chunk_id, vector columns
- Key transformations:
  - Extract text from each chunk
  - Call OpenAI ada-002 API with caching (SHA-256 keys)
  - Validate L2 norms (median ~1.0, outliers <0.5%)
  - Store as 1536-dimensional float32 vectors

**Embedding → Indexing**:
- Input: Parquet embeddings + chunk metadata
- Output: 3 vector indexes
- Key transformations:
  - FAISS: Build HNSW index (M=32, efConstruction=200)
  - Weaviate: Apply schema, simulate batch insert
  - Pinecone: Create manifest with upsert tracking
  - Perform round-trip integrity tests (max error <0.001)

## 3. File Inventory

### Pipeline Scripts (41 total)

#### Collection Scripts (7 fetch_*.py)
- `scripts/fetch_dev_docs.py` - Developer documentation (developer.salesforce.com)
- `scripts/fetch_help_docs.py` - Help articles (help.salesforce.com)
- `scripts/fetch_investor_news.py` - Investor relations press releases (RSS)
- `scripts/fetch_newsroom_rss.py` - Corporate/product press releases (2 RSS feeds)
- `scripts/fetch_product_docs.py` - Product overview pages
- `scripts/fetch_sec_filings.py` - SEC filings (10-K, 10-Q, 8-K, annual reports)
- `scripts/fetch_wikipedia.py` - Wikipedia articles

#### Processing Scripts (4)
- `scripts/normalize_html.py` - HTML→normalized JSON transformation
- `scripts/extract_metadata.py` - Metadata enrichment (in-place updates)
- `scripts/chunk_documents.py` - Document→chunks transformation
- `scripts/dedupe_chunks.py` - Chunk deduplication

#### Quality Gate Scripts (18)
**Data Pipeline Verification (5 qa_verify_*.py)**:
- `scripts/qa_verify_collection.py` - Gate G01: Collection coverage
- `scripts/qa_verify_normalization.py` - Gate G02: Normalization quality
- `scripts/qa_verify_metadata.py` - Gate G03: Metadata completeness
- `scripts/qa_verify_chunking.py` - Gate G04: Chunking boundaries
- `scripts/qa_verify_dedupe.py` - Gate G05: Deduplication effectiveness

**Infrastructure & Evaluation (10 qa_step*.py)**:
- `scripts/qa_step00_baseline.py` - Gate 0: Baseline snapshot
- `scripts/qa_step01_embeddings.py` - Gate 1: Embedding generation
- `scripts/qa_step02_indexes.py` - Gate 2: Index building
- `scripts/qa_step03_mcp.py` - Gate 3: MCP tool health
- `scripts/qa_step04_router.py` - Gate 4: Query routing
- `scripts/qa_step05_graph.py` - Gate 5: Graph execution
- `scripts/qa_step06_a2a.py` - Gate 6: Agent-to-agent negotiation
- `scripts/qa_step07_retrieval_eval.py` - Gate 7: Retrieval quality
- `scripts/qa_step08_generation_eval.py` - Gate 8: Generation quality
- `scripts/qa_step08_debug.py` - Gate 8 debug: Deep pipeline inspection

#### Supporting Scripts (12)
- `scripts/common.py` - Shared utilities (fetch, logging, rate limiting)
- `scripts/embedding_utils.py` - Embedding cache and OpenAI API calls
- `scripts/ingest_manual_html.py` - Manual HTML ingestion
- `scripts/ingest_manual_ir_html.py` - Manual IR HTML ingestion
- `scripts/build_eval_seed.py` - Evaluation set creation
- `scripts/build_eval_generation_prompts.py` - Generation prompt builder
- `scripts/build_inventory_csv.py` - Document inventory CSV
- `scripts/parse_sec_structures.py` - SEC filing structure parser
- `scripts/link_health_check.py` - URL health verification
- `scripts/test_title_extraction.py` - Title extraction testing
- `scripts/fix_investor_news_metadata.py` - Metadata correction
- `scripts/apply_ground_truth_fixes.py` - Ground truth updates

### Data Directories

```
data/
├── raw/                      # Original fetched documents (7 buckets)
│   ├── dev_docs/             # Developer documentation
│   ├── help_docs/            # Help articles
│   ├── investor_news/        # Investor relations press
│   ├── newsroom/             # Corporate/product press
│   ├── product/              # Product pages
│   ├── sec/                  # SEC filings
│   └── wikipedia/            # Wikipedia articles
│
├── interim/                  # Processing artifacts
│   ├── normalized/           # Stage 2: Clean JSON documents
│   ├── chunks/               # Stage 4: Chunked JSONL files
│   │   └── pre_dedupe/       # Backup before deduplication
│   ├── dedup/                # Stage 5: Dedup maps and clusters
│   └── eval/                 # Evaluation test sets
│       ├── salesforce_eval_seed.jsonl
│       ├── salesforce_eval_queries_v2.jsonl
│       └── generation_prompts.jsonl
│
├── vector/                   # Embeddings and indexes
│   ├── embeddings/           # Stage 6: Parquet embeddings + stats
│   ├── faiss/                # Stage 7: FAISS HNSW index + ID map
│   ├── weaviate/             # Stage 7: Weaviate schema + manifest
│   └── pinecone/             # Stage 7: Pinecone manifest
│
├── cache/                    # Embedding cache (SHA-256 keys)
│   └── embeddings/           # {16-char-hex}.json files
│
├── final/                    # Production artifacts
│   ├── inventory/            # Master document catalog (CSV)
│   ├── dictionaries/         # Metadata extraction rules
│   ├── rules/                # Normalization rules
│   └── reports/              # Health and verification reports
│
└── backup/                   # Timestamped snapshots
    ├── chunks_YYYYMMDD_HHMMSS/
    ├── normalized_YYYYMMDD_HHMMSS/
    └── embeddings_YYYYMMDD_HHMMSS.parquet
```

### Sample Data Files

**Raw HTML + Metadata**:
```
data/raw/dev_docs/crm::dev_docs::unknown::agent-api-developer-guide::6f4b900c.raw.html
data/raw/dev_docs/crm::dev_docs::unknown::agent-api-developer-guide::6f4b900c.meta.json
```

**Normalized Document**:
```
data/interim/normalized/crm::dev_docs::unknown::agent-api-developer-guide::6f4b900c.json
```

**Chunked Document**:
```
data/interim/chunks/crm::dev_docs::unknown::agent-api-developer-guide::6f4b900c.chunks.jsonl
```

**Embeddings**:
```
data/vector/embeddings/embeddings.parquet  (536 rows × 3 columns: chunk_id, doc_id, vector)
data/vector/embeddings/embedding_stats.json
```

**FAISS Index**:
```
data/vector/faiss/index.faiss  (binary HNSW index)
data/vector/faiss/idmap.parquet  (chunk_id → faiss_id mapping)
data/vector/faiss/faiss_manifest.json  (configuration)
```

## 4. Core Components Deep Dive

### Collection Stage: 7 Fetch Scripts

All collection scripts share a common async architecture using `aiohttp`, `RateLimiter`, and `fetch_with_retries()` from `common.py`. They output dual files per document: `.raw.html` (or `.pdf`) + `.meta.json`.

#### fetch_sec_filings.py

**Location**: `scripts/fetch_sec_filings.py`

**Data Source**: SEC Edgar archives for Salesforce (ticker CRM, CIK 1108524)

**Configuration**: Hardcoded list of 6 filings at lines 27-64:
- 10-K annual reports (FY25, FY24)
- 10-Q quarterly reports (FY26-Q1, FY25-Q4)
- 8-K current reports (selected events)
- Annual report PDFs

**Process**:
1. Iterates through `SEC_ITEMS` list (line 160)
2. Skips items with existing `.meta.json` showing HTTP 200 (lines 146-162)
3. Fetches via `fetch_with_retries()` with SSL disabled for sec.gov (line 74)
4. Extracts title from HTML `<title>` tag (line 90)
5. Parses date from meta tags or sidecar (line 92)
6. Builds doc_id: `crm::{doctype}::{date}::{slug}::{hash8}` (line 96)
7. Writes raw HTML/PDF only on HTTP 200 (line 129)
8. Always writes `.meta.json` regardless of status (line 131)

**Output Directory**: `data/raw/sec/`

**Example Output**:
```
crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.raw.html
crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2.meta.json
```

#### fetch_investor_news.py

**Location**: `scripts/fetch_investor_news.py`

**Data Source**: Salesforce Investor Relations RSS feed (https://investor.salesforce.com/rss/pressrelease.aspx)

**Parameters**:
- `--since`: Lower bound date (default: "2024-01-01")
- `--until`: Upper bound date (default: None)
- `--limit`: Max articles (default: 20)

**Process**:
1. Fetches RSS feed XML (line 152)
2. Parses XML with `ET.fromstring()` (line 154)
3. Extracts `<item>` elements with link, title, pubDate (lines 70-86)
4. Filters by date window via `within_window()` (line 155)
5. Limits selection (line 156)
6. Fetches each article HTML concurrently (lines 158-163)
7. Prefers RSS pubdate over HTML meta for doc_id (line 101)
8. Records `rss_pubdate` and `notes: "feed=ir_rss"` in metadata (lines 125-126)

**Output Directory**: `data/raw/investor_news/`

**Date Handling**: Converts RFC 822 dates from RSS to ISO format via `to_iso_date_from_rfc822()` (lines 36-45)

#### fetch_newsroom_rss.py

**Location**: `scripts/fetch_newsroom_rss.py`

**Data Source**: Two Salesforce Newsroom RSS feeds:
1. Corporate press releases (https://www.salesforce.com/news/content-types/press-releases-corporate/feed/)
2. Product press releases (https://www.salesforce.com/news/content-types/press-releases-product/feed/)

**Parameters**:
- `--since`, `--until`: Date window filtering
- `--limit`: Total items across both feeds (default: 30)
- `--use-index`: Enable index page crawling for additional items

**Process**:
1. Splits limit equally across 2 feeds: `per_feed = max(1, total_limit // 2)` (line 164)
2. Fetches each feed XML (lines 168-185)
3. Parses items, filters by date, limits selection
4. Fetches article HTML with domain filtering (lines 100-103)
5. Optional: If `--use-index` and total_saved < limit, crawls https://www.salesforce.com/news/all-news-press-salesforce/ for additional items (lines 187-197)
6. Extracts links via regex from index HTML (lines 136-151)

**Output Directory**: `data/raw/newsroom/`

**Metadata Notes**: Records feed source as `"feed=corporate"` or `"feed=product"` (line 121)

#### fetch_dev_docs.py, fetch_help_docs.py, fetch_product_docs.py

**Locations**:
- `scripts/fetch_dev_docs.py`
- `scripts/fetch_help_docs.py`
- `scripts/fetch_product_docs.py`

**Data Sources**:
- Dev docs: developer.salesforce.com (Agent API guides)
- Help docs: help.salesforce.com (Copilot introduction)
- Product: www.salesforce.com (Agentforce, Data Cloud overview)

**Configuration**: Hardcoded URL lists at lines 26-31 (dev), line 24 (help), lines 26-30 (product)

**Process** (identical across all three):
1. Loads URL list
2. Checks for existing `.meta.json` (deduplication)
3. Fetches via `fetch_with_retries()` (line 36)
4. Extracts title from HTML (line 40)
5. Sets doctype to "dev_docs", "help_docs", or "product" (line 42)
6. Writes raw HTML + metadata (lines 76-79)

**Output Directories**:
- `data/raw/dev_docs/`
- `data/raw/help_docs/`
- `data/raw/product/`

**Date Handling**: All three set `visible_date: None` since documentation pages typically lack publication dates

#### fetch_wikipedia.py

**Location**: `scripts/fetch_wikipedia.py`

**Data Source**: Single Wikipedia page (https://en.wikipedia.org/wiki/Salesforce)

**Process**:
1. Makes HEAD request to capture Last-Modified header (lines 46-50)
2. Fetches full HTML (line 52)
3. Records `last_modified_http` in metadata (line 71)
4. Sets doctype to "wiki" (line 36)

**Output Directory**: `data/raw/wikipedia/`

**Unique Feature**: Only fetch script that captures HTTP Last-Modified header for publish date

### Normalization Stage: normalize_html.py

**Location**: `scripts/normalize_html.py`

**Purpose**: Transforms raw HTML/PDF into clean, structured JSON documents with extracted text and metadata.

**Input**:
- Raw files: `data/raw/**/*.{raw.html,pdf}`
- Metadata: `data/raw/**/*.meta.json`
- Configuration: `configs/normalization.rules.yaml`

**Output**: `data/interim/normalized/{doc_id}.json`

**Key Functions**:

**`normalize_html_bytes()` (lines 83-137)**:
Core HTML processing pipeline:
1. Parse HTML with BeautifulSoup (line 85)
2. Preserve content matching `preserve_selectors` (article, main, .content) (lines 88-97)
3. Remove unwanted elements matching `remove_selectors` (nav, footer, script, style, XBRL tags) (lines 104-107)
4. Convert structural elements:
   - `<br>` → newlines (lines 112-113)
   - `<h1>`, `<h2>`, `<h3>` → "H1:", "H2:", "H3:" markers (lines 116-119)
   - Block elements → trailing newlines (lines 122-125)
5. Strip tracking params from links (line 127)
6. Extract text with `get_text("\n")` (line 129)
7. Normalize whitespace:
   - Collapse horizontal whitespace to single space (line 130)
   - Collapse newlines (line 131)
   - Limit consecutive newlines to 2 (line 132)

**Domain-Specific Rules** (lines 216-224):
- **Investor Relations**: Relaxes removal selectors to keep nearly all content (only removes script, style, cookies)
- **Help/Dev Documentation**: Preserves sidebar and breadcrumb for context

**Content Fallback** (lines 226-236):
- If help_docs or dev_docs has <200 words, re-extracts without selector filtering
- Ensures minimal content even for sparse pages

**Language Detection** (lines 248-261):
- Uses `langdetect` library on first 8000 characters (lines 39-47)
- Whitelists known English domains (sec.gov, salesforce.com, developer.salesforce.com, etc.) (lines 250-258)
- Drops non-English documents (line 259-261)

**Publish Date Resolution** (lines 263-281):
Priority waterfall:
1. HTML meta tags (`article:published_time`, `pubdate`, `date`) (lines 267-270)
2. `visible_date` from `.meta.json` sidecar (lines 272-279)
3. `rss_pubdate` from `.meta.json` (lines 272-279)

**Quality Filtering** (lines 287-291):
- SEC 8-K filings with <200 words are dropped as "DROPPED_SHORT"

**Output Schema** (lines 293-318):
```json
{
  "doc_id": "crm::press::2024-12-17::...",
  "company": "Salesforce",
  "doctype": "press",
  "title": "Introducing Agentforce 2.0...",
  "publish_date": "2024-12-17",
  "url": "https://...",
  "final_url": "https://...",
  "source_domain": "www.salesforce.com",
  "section": "body",
  "topic": "",
  "persona_tags": [],
  "language": "en",
  "text": "H1: Introducing Agentforce 2.0...\n...",
  "word_count": 3372,
  "token_count": 4417,
  "ingestion_ts": "2025-10-07T20:09:31.846961+00:00",
  "hash_sha256": "2a85d23a45a41b425f35b413df6dbff9...",
  "html_title": "Introducing Agentforce 2.0...",
  "meta_published_time": null,
  "last_modified_http": null,
  "byline": null,
  "press_location": null,
  "ticker_mentions": [],
  "pdf_page_map": null
}
```

### Metadata Stage: extract_metadata.py

**Location**: `scripts/extract_metadata.py`

**Purpose**: Enriches normalized documents with structured metadata (titles, dates, topics, personas).

**Input**:
- Normalized docs: `data/interim/normalized/*.json`
- Sidecars: `data/raw/**/*.meta.json`
- Configs: `configs/metadata.dictionary.yaml`, `configs/eval.prompts.yaml`

**Output**: Updates `data/interim/normalized/*.json` in place

**Phase Selection** (lines 14-18):
- Phase A: SHA-1 hash of doc_id ends in even hex digit
- Phase B: SHA-1 hash of doc_id ends in odd hex digit
- Enables parallel processing without coordination

**Title Extraction** (lines 58-77):
4-level precedence waterfall:
1. `html_title` from normalized doc (line 60-62)
2. `title` from existing field (line 63-64)
3. First H1 from document text (line 65-67)
4. `visible_title` from sidecar (line 68-70)
5. Slug from doc_id: `"agentforce-3-announcement"` → `"Agentforce 3 Announcement"` (lines 71-76)

**Publish Date Extraction** (lines 80-130):
Doctype-specific logic:
- **SEC** (10-K, 10-Q, 8-K, ars_pdf): Prefers `visible_date`, falls back to fetch timestamp
- **Press**: Prefers `meta_published_time`, then `visible_date`, then `rss_pubdate`
- **Product/Dev/Help**: Regex extracts "Updated: March 5, 2024" from text, falls back to `last_modified_http`
- **Wikipedia**: Uses only `last_modified_http`

All dates coerced to ISO format via `coerce_date_iso()` (lines 201-211):
- Validated: between 1999-01-01 and today
- Set to `None` if out of range

**Topic Assignment** (lines 133-166):
Keyword matching on title + text:
- 11 topics: Agentforce, Agent API, Data Cloud, Earnings, Partnership, Executive, Security, Compliance, GenAI, Industry Solutions, Platform, AI
- Returns pipe-delimited string: `"Agentforce|Partnership|Platform"`

**Persona Assignment** (lines 168-178):
Keyword matching using `configs/eval.prompts.yaml`:
- 3 personas: vp_customer_experience, cio, vp_sales_ops
- Returns sorted list: `["cio", "vp_sales_ops"]`

**Press Recency Flags** (lines 219-229):
For doctype=="press" with valid publish_date:
- `is_within_12mo`: boolean (days ≤ 365)
- `is_within_24mo`: boolean (days ≤ 730)

**In-Place Mutation** (lines 231-256):
Uses atomic write pattern:
1. Updates fields only if value changed (lines 232-238)
2. Writes to `{path}.tmp` (line 252)
3. Atomic rename with `os.replace()` (line 255)
4. Skips write if no changes detected

### Chunking Stage: chunk_documents.py

**Location**: `scripts/chunk_documents.py`

**Purpose**: Splits documents into retrieval-ready chunks with overlap and boundary awareness.

**Input**:
- Normalized docs: `data/interim/normalized/*.json`
- Configuration: `configs/chunking.config.json`

**Output**: `data/interim/chunks/{doc_id}.chunks.jsonl` (one chunk per line)

**Configuration** (lines 118-121):
- `tokenizer`: "cl100k_base" (GPT-4 tokenizer via tiktoken)
- `target_tokens`: 800
- `overlap_tokens`: 120
- `short_doc_threshold_tokens`: 350 (no chunking below this)
- `boundary_tolerance_chars`: 50 (snap tolerance for headings/SEC items)

**Algorithm**:

**1. Short Document Handling** (lines 162-165):
- If token_count < 350, creates single chunk with entire document
- Adds title boost (prepended title + first H1)

**2. Boundary Detection** (lines 167-173):
- **H2/H3 Headings**: Scans for lines starting with "h2:" or "h3:" (lines 35-45)
- **SEC Item Boundaries**: Extracts `start_char` from `sec_item_spans` for 10-K/10-Q/8-K (lines 170-173)
- Combined boundary set used for snapping

**3. Segmentation** (lines 180-188):
- **SEC Documents**: Split by `sec_item_spans` boundaries, chunk each item independently
- **Other Documents**: Treat entire document as single segment

**4. Sliding Window** (lines 67-102):
- Calculate `chars_per_token` ratio for segment (lines 69-73)
- `step_chars = chars_per_token * (target - overlap)` (line 78)
- `win_chars = chars_per_token * target` (line 79)
- Slide window with boundary snapping (lines 82-101)

**5. Boundary Snapping** (lines 56-64):
- For each candidate boundary within 50 chars of predicted position
- Uses closest boundary if found, else uses predicted position

**6. Residual Merging** (lines 205-223):
- If final chunk < 120 tokens
- Merges into previous chunk if combined ≤ 960 tokens (1.2 × target)
- Prevents tiny trailing chunks

**7. Title Boosting** (lines 125-138):
- Prepends document title + first H1 to each chunk
- Format: `"{title}\n\nH1: {first_h1}\n\n{chunk_body}"`

**8. Local Heading Context** (lines 174-178, 226-227):
- Finds last 2 H2/H3 headings before chunk start
- Stores in `local_heads` field for retrieval context

**Output Schema** (lines 141-160):
```json
{
  "chunk_id": "crm::press::2024-11-25::..::chunk0000",
  "doc_id": "crm::press::2024-11-25::..::88c1752e",
  "seq_no": 0,
  "text": "<title boost>\n\n<chunk body>",
  "word_count": 662,
  "token_count": 789,
  "start_char": 0,
  "end_char": 4201,
  "local_heads": ["Most Recent H2/H3", "Previous H2/H3"],
  "metadata_snapshot": {
    "company": "Salesforce",
    "doctype": "press",
    "date": "2024-11-25",
    "url": "https://...",
    "title": "How new digital workers will lead to an unlimited age",
    "topic": "",
    "persona_tags": []
  }
}
```

### Deduplication Stage: dedupe_chunks.py

**Location**: `scripts/dedupe_chunks.py`

**Purpose**: Identifies and removes near-duplicate chunks using Jaccard similarity on 5-gram shingles.

**Input**:
- Normalized docs: `data/interim/normalized/*.json` (for metadata)
- Chunks: `data/interim/chunks/*.chunks.jsonl`

**Output**:
- Dedup map: `data/interim/dedup/dedup_map.json`
- Backups: `data/interim/chunks/pre_dedupe/*.chunks.jsonl.bak`
- Updated chunks: `data/interim/chunks/*.chunks.jsonl` (rewritten in place)

**Algorithm**:

**1. Text Normalization** (lines 14-15):
- Regex extracts lowercase alphanumeric tokens: `r"[a-z0-9]+"`
- Example: `"Hello, World! 123"` → `["hello", "world", "123"]`

**2. K-Shingling** (lines 18-21):
- Creates overlapping 5-word sequences
- Example: `["the", "quick", "brown", "fox", "jumps", "over"]` →
  ```
  {"the quick brown fox jumps", "quick brown fox jumps over"}
  ```

**3. Inverted Index** (lines 86-96):
- Maps each shingle → list of chunk IDs containing it
- Enables efficient candidate generation

**4. Candidate Pairs** (lines 98-112):
- For each shingle's posting list:
  - Cap at 2000 chunks (prevents O(n²) explosion) (line 104-105)
  - Generate all pairs from posting list (lines 106-107)
  - Count co-occurrences in `co_counts` dictionary (line 112)

**5. Jaccard Filtering** (lines 114-134):
- Threshold: 0.85 (line 114)
- Exempt doctypes: 10-K, 10-Q, 8-K, ars_pdf, wiki (line 116, 123-126)
- Quick upper-bound check before computing Jaccard (lines 128-130)
- Jaccard formula: `|A ∩ B| / |A ∪ B|` (lines 24-27)
- Creates bidirectional edges in graph (lines 133-134)

**6. Connected Components** (lines 136-154):
- Iterative DFS to find clusters (lines 139-152)
- Records groups with 2+ members (lines 153-154)

**7. Canonical Selection** (lines 56-64):
Priority order:
1. Earliest `publish_date` (line 61)
2. Longest `word_count` (negated at line 62)
3. Lexicographic `chunk_id` (line 62)

**8. Backup and Rewrite** (lines 191-216):
- Creates `pre_dedupe/` directory (line 76)
- Copies original files to backup (lines 199-203)
- Filters out duplicates (line 207)
- Writes to temp file, atomic replace (lines 212-216)

**Dedup Map Schema** (lines 181-188):
```json
{
  "created_at": "2025-10-20T...",
  "near_duplicate_threshold": 0.85,
  "shingle_size": 5,
  "groups": [
    {
      "canonical_chunk_id": "doc123_chunk05",
      "duplicate_chunk_ids": ["doc456_chunk12", "doc789_chunk03"],
      "reason": "jaccard>=0.85",
      "stats": {
        "pairwise_jaccard_min": 0.87,
        "pairwise_jaccard_max": 0.92,
        "members": 3
      }
    }
  ]
}
```

## 5. Configuration & Settings

### configs/normalization.rules.yaml

**Purpose**: Defines HTML cleaning rules for text extraction.

**Structure** (26 lines total):

**`remove_selectors`** (lines 1-19):
CSS selectors for elements to remove:
- Navigation: `nav`, `footer`, `.breadcrumb`, `.sidebar`
- Tracking: `[aria-label*="cookie"]`, `.share`, `.social`, `.newsletter`
- Technical: `script`, `style`, `noscript`
- XBRL (SEC): `ix\3A header`, `ix\3A hidden`, `ix\3A resources`, `xbrli\3A context`, `xbrli\3A unit`

**`preserve_selectors`** (lines 20-25):
Content areas to preserve (first match wins):
- `article`
- `main`
- `.content`
- `.entry-content`
- `#content`

**`newline_blocks`** (lines 26-30):
Block-level elements that should have trailing newlines:
- `p`, `div`, `section`, `li`

**`heading_levels`** (lines 31-34):
Heading tags converted to markers:
- `h1` → "H1: {text}\n"
- `h2` → "H2: {text}\n"
- `h3` → "H3: {text}\n"

**Usage**: Loaded by `normalize_html.py` at line 341

### configs/metadata.dictionary.yaml

**Purpose**: Schema documentation for metadata fields (not enforced, reference only).

**Structure** (26 fields defined):

**Required Fields**:
- `doc_id` (string): Unique identifier
- `company` (string): "Salesforce"
- `doctype` (string): Document type (press, 10-K, dev_docs, etc.)
- `title` (string): Document title
- `publish_date` (string, nullable): YYYY-MM-DD format
- `url` (string): Source URL
- `final_url` (string): After redirects
- `source_domain` (string): Netloc of URL
- `section` (string): "body" (reserved for future use)
- `topic` (string): Pipe-delimited topics
- `persona_tags` (array): List of persona keys
- `language` (string): ISO code (e.g., "en")
- `text` (string): Normalized text content
- `word_count` (integer): Word count
- `token_count` (integer): Token count (cl100k_base)
- `ingestion_ts` (string): ISO timestamp
- `hash_sha256` (string): Content hash

**Optional Fields**:
- `html_title` (string, nullable): From HTML `<title>` tag
- `meta_published_time` (string, nullable): From meta tags
- `last_modified_http` (string, nullable): From HTTP header
- `byline` (string, nullable): Author (reserved)
- `press_location` (string, nullable): Location (reserved)
- `ticker_mentions` (array): Stock tickers (reserved)
- `pdf_page_map` (any, nullable): PDF page boundaries

**Usage**: Referenced by `extract_metadata.py` but not validated programmatically

### configs/chunking.config.json

**Purpose**: Defines chunking algorithm parameters.

**Structure**:
```json
{
  "tokenizer": "cl100k_base",
  "target_tokens": 800,
  "overlap_tokens": 120,
  "short_doc_threshold_tokens": 350,
  "boundary_tolerance_chars": 50
}
```

**Parameters**:
- `tokenizer`: tiktoken encoding name (GPT-4 tokenizer)
- `target_tokens`: Target chunk size in tokens
- `overlap_tokens`: Token overlap between consecutive chunks
- `short_doc_threshold_tokens`: Documents below this are kept as single chunk
- `boundary_tolerance_chars`: Maximum distance to snap chunk boundaries to headings/SEC items

**Usage**: Loaded by `chunk_documents.py` at line 240

### configs/vector.indexing.yaml

**Purpose**: Embedding and index configuration.

**Structure** (example):
```yaml
embedding:
  model: text-embedding-ada-002
  provider: openai
  dimension: 1536
  batch_size: 100

faiss:
  index_type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128

weaviate:
  host: localhost
  port: 8080
  class_name: SalesforceChunk

pinecone:
  environment: us-west1-gcp
  index_name: salesforce-rag
  metric: cosine
```

**Usage**:
- `qa_step01_embeddings.py` loads embedding config at line 28
- `qa_step02_indexes.py` loads index config at line 21

### configs/router.heuristics.yaml

**Purpose**: Query routing rules and heuristics.

**Structure** (example):
```yaml
keyword_rules:
  - keywords: ["sec", "10-k", "10-q", "earnings", "revenue"]
    backend: faiss
    boost: 1.5

  - keywords: ["agent api", "developer", "sdk", "integration"]
    backend: weaviate
    boost: 1.5

  - keywords: ["press", "announcement", "news"]
    backend: pinecone
    boost: 1.5

persona_bias:
  cfo: faiss
  cio: weaviate
  vp_sales_ops: pinecone

fallback:
  primary: faiss
  secondary: weaviate
```

**Usage**: Loaded by `qa_step04_router.py` at lines 188-190

### configs/mcp.tools.yaml

**Purpose**: MCP service endpoint definitions.

**Structure** (example):
```yaml
services:
  kb_search:
    port: 7801
    methods: [search]
    timeout_ms: 5000

  web_fetch:
    port: 7802
    methods: [fetch]
    timeout_ms: 3000

  link_resolve:
    port: 7803
    methods: [resolve]
    timeout_ms: 1000

  crm_lookup:
    port: 7804
    methods: [lookup]
    timeout_ms: 2000

  safety_check:
    port: 7805
    methods: [check]
    timeout_ms: 500
```

**Usage**: Loaded by `qa_step03_mcp.py` at line 24

### configs/langgraph.nodes.yaml

**Purpose**: Agent graph topology and node configuration.

**Structure** (example):
```yaml
nodes:
  - name: Intake
    timeout_s: 5

  - name: Planner
    timeout_s: 15

  - name: Retriever
    timeout_s: 30

  - name: Synthesizer
    timeout_s: 45

  - name: Consolidator
    timeout_s: 30

  - name: Stylist
    timeout_s: 20

  - name: A2A
    timeout_s: 30

  - name: Assembler
    timeout_s: 10

edges:
  - [Intake, Planner]
  - [Planner, Retriever]
  - [Retriever, Synthesizer]
  - [Synthesizer, Consolidator]
  - [Consolidator, Stylist]
  - [Stylist, A2A]
  - [A2A, Assembler]
```

**Usage**:
- Graph execution scripts read node order
- Debug script reads timeouts at `qa_step08_debug.py:905-909`

### configs/eval.prompts.yaml

**Purpose**: Persona definitions with keyword lists for matching.

**Structure** (lines 1-24):
```yaml
personas:
  vp_customer_experience:
    keywords:
      - nps
      - csat
      - contact center
      - omnichannel
      - agent productivity
      - self-service
      - first contact resolution

  cio:
    keywords:
      - data integration
      - governance
      - security
      - tco
      - platform
      - apis
      - real-time

  vp_sales_ops:
    keywords:
      - pipeline
      - forecast accuracy
      - win rate
      - productivity
      - automation
```

**Usage**:
- `extract_metadata.py` loads at line 26 for persona assignment
- `qa_step08_generation_eval.py` loads at line 26 for keyword hit calculation

### configs/compliance.template.yaml

**Purpose**: Compliance rules for email generation.

**Structure** (example):
```yaml
rules:
  critical:
    - id: COMP-001
      description: No unsubstantiated claims
      check: proof_points_required

    - id: COMP-002
      description: No forward-looking statements without disclaimer
      check: forward_looking_disclaimer

  warnings:
    - id: COMP-W01
      description: Prefer active voice
      check: passive_voice_ratio < 0.3

    - id: COMP-W02
      description: Readability target
      check: flesch_kincaid_grade <= 10

length_limits:
  email_body_words: 160
  subject_line_chars: 60

style_guidelines:
  tone: professional
  persona_alignment_required: true
```

**Usage**: Loaded by `qa_step08_generation_eval.py` at line 28

## 6. Data Structures & Schemas

### Raw HTML Schema (.raw.html + .meta.json)

**Files**:
- `{doc_id}.raw.html` - Raw HTML bytes as fetched
- `{doc_id}.meta.json` - Fetch metadata

**Metadata Schema**:
```json
{
  "doc_id": "crm::dev_docs::unknown::agent-api-developer-guide::6f4b900c",
  "source_domain": "developer.salesforce.com",
  "source_bucket": "dev_docs",
  "doctype": "dev_docs",
  "requested_url": "https://developer.salesforce.com/docs/einstein/genai/guide/agent-api.html",
  "final_url": "https://developer.salesforce.com/docs/einstein/genai/guide/agent-api.html",
  "redirect_chain": [],
  "http_status": 200,
  "content_type": "text/html",
  "content_length": 48292,
  "fetched_at": "2025-09-07T20:32:23.176246+00:00",
  "sha256_raw": "973046c2a1f8...",
  "visible_title": "Agent API Developer Guide",
  "visible_date": null,
  "rss_pubdate": null,
  "headline": "Agent API Developer Guide",
  "notes": "manual_save=1",
  "latency_ms": 123
}
```

### Normalized JSON Schema

**File**: `{doc_id}.json` in `data/interim/normalized/`

**Schema**:
```json
{
  "doc_id": "crm::press::2024-12-17::introducing-agentforce-2-0::64bc03ee",
  "company": "Salesforce",
  "doctype": "press",
  "title": "Introducing Agentforce 2.0: The Digital Labor Platform",
  "publish_date": "2024-12-17",
  "url": "https://www.salesforce.com/news/press-releases/2024/12/17/agentforce-2-0/",
  "final_url": "https://www.salesforce.com/news/press-releases/2024/12/17/agentforce-2-0/",
  "source_domain": "www.salesforce.com",
  "section": "body",
  "topic": "Agentforce|Partnership|Platform",
  "persona_tags": ["cio"],
  "language": "en",
  "text": "H1: Introducing Agentforce 2.0...\n\n...",
  "word_count": 3372,
  "token_count": 4417,
  "ingestion_ts": "2025-10-07T20:09:31.846961+00:00",
  "hash_sha256": "2a85d23a45a41b425f35b413df6dbff9f249d36e0606441eadc1accc1ee29927",
  "html_title": "Introducing Agentforce 2.0",
  "meta_published_time": "2024-12-17T08:00:00+00:00",
  "last_modified_http": null,
  "byline": null,
  "press_location": null,
  "ticker_mentions": [],
  "pdf_page_map": null,
  "is_within_12mo": true,
  "is_within_24mo": true
}
```

### Chunk Schema

**File**: `{doc_id}.chunks.jsonl` in `data/interim/chunks/`

**Format**: JSONL (one JSON object per line)

**Schema**:
```json
{
  "chunk_id": "crm::press::2024-11-25::how-new-digital-workers::88c1752e::chunk0000",
  "doc_id": "crm::press::2024-11-25::how-new-digital-workers::88c1752e",
  "seq_no": 0,
  "text": "How new digital workers will lead to an unlimited age\n\nH1: How new digital workers will lead to an unlimited age\n\nThe rise of AI agents marks a fundamental shift in how work gets done...",
  "word_count": 662,
  "token_count": 789,
  "start_char": 0,
  "end_char": 4201,
  "local_heads": ["Recent Developments", "Industry Impact"],
  "metadata_snapshot": {
    "company": "Salesforce",
    "doctype": "press",
    "date": "2024-11-25",
    "url": "https://time.com/7178872/agents-unlimited-age/",
    "title": "How new digital workers will lead to an unlimited age",
    "topic": "Agentforce|AI|Platform",
    "persona_tags": ["cio", "vp_sales_ops"]
  }
}
```

### Embedding Schema

**File**: `embeddings.parquet` in `data/vector/embeddings/`

**Schema**:
- `chunk_id` (string): Primary key
- `doc_id` (string): Parent document reference
- `vector` (list[float32]): 1536-dimensional embedding

**Parquet Structure**:
```
ParquetFile(536 rows, 3 columns)
  chunk_id: string
  doc_id: string
  vector: list<element: float>[1536]
```

**Stats File** (`embedding_stats.json`):
```json
{
  "embedding_rows": 536,
  "zero_vectors": 0,
  "nan_vectors": 0,
  "median_norm": 1.0,
  "iqr": 0.0,
  "pct_norm_outliers": 0.0,
  "vector_dim": 1536,
  "baseline_chunks": 565,
  "parquet_path": "data/vector/embeddings/embeddings.parquet"
}
```

### FAISS Manifest Schema

**File**: `faiss_manifest.json` in `data/vector/faiss/`

**Schema**:
```json
{
  "index_type": "HNSW",
  "metric": "L2",
  "dim": 1536,
  "count": 536,
  "roundtrip_error_max": 0.0,
  "paths": {
    "index": "data/vector/faiss/index.faiss",
    "idmap": "data/vector/faiss/idmap.parquet"
  },
  "params": {
    "type": "HNSW",
    "metric": "L2",
    "M": 32,
    "efConstruction": 200,
    "efSearch": 128
  }
}
```

### Metadata Fields

**Core Identifiers**:
- `doc_id`: Format `crm::{doctype}::{date}::{slug}::{hash8}`
- `chunk_id`: Format `{doc_id}::chunk{seq_no:04d}`

**Temporal Fields**:
- `publish_date`: YYYY-MM-DD (validated: 1999-01-01 to today)
- `fetched_at`: ISO 8601 timestamp with timezone
- `ingestion_ts`: ISO 8601 timestamp with timezone
- `is_within_12mo`: Boolean (for press docs)
- `is_within_24mo`: Boolean (for press docs)

**Content Fields**:
- `text`: Cleaned text with H1:/H2:/H3: markers
- `word_count`: Regex word boundary count `\b\w+\b`
- `token_count`: cl100k_base tokenizer count
- `hash_sha256`: SHA-256 of text bytes

**Classification Fields**:
- `doctype`: press, 10-K, 10-Q, 8-K, ars_pdf, dev_docs, help_docs, product, wiki
- `topic`: Pipe-delimited (11 categories)
- `persona_tags`: Array (3 personas)
- `language`: ISO code (e.g., "en")

**Structural Fields** (chunks only):
- `seq_no`: Sequential index within document (0-based)
- `start_char`: Character offset in original text
- `end_char`: Character offset in original text
- `local_heads`: Last 2 H2/H3 headings before chunk

## 7. External Dependencies

### Web Scraping Libraries

**aiohttp** (async HTTP client):
- Used by: All 7 fetch scripts
- Purpose: Concurrent HTTP requests with connection pooling
- Features: Custom headers, SSL control, redirect tracking

**BeautifulSoup** (HTML parser):
- Used by: `normalize_html.py`, `common.py` (title extraction)
- Purpose: HTML parsing and element selection
- Parser: `html.parser` (built-in)

**pdfminer.six** (PDF text extraction):
- Used by: `normalize_html.py`
- Purpose: Extract text from SEC PDF filings
- Function: `pdfminer.high_level.extract_text()`

### HTML Parsers

**BeautifulSoup with html.parser**:
- CSS selector support for content preservation/removal
- Tag conversion (br → newline, h1 → "H1:" marker)
- Link tracking parameter removal

**lxml** (optional):
- Not used directly, but BeautifulSoup can use lxml parser if available
- Faster than html.parser for large documents

### SEC Edgar API

**Access Pattern**:
- Direct URL fetching (no official API used)
- URLs: `https://www.sec.gov/Archives/edgar/data/{CIK}/{filing-id}.htm`
- Rate limiting: 6 requests per second (via RateLimiter)
- SSL verification: Disabled for sec.gov (line in `common.py:251-252`)

**Filing Types Fetched**:
- 10-K: Annual reports
- 10-Q: Quarterly reports
- 8-K: Current reports (material events)
- Annual report PDFs

**Example URLs** (from `fetch_sec_filings.py:27-64`):
```
https://www.sec.gov/Archives/edgar/data/1108524/000110852425000012/crm-20250131.htm
https://www.sec.gov/cgi-bin/viewer?action=view&cik=1108524&accession_number=0001108524-24-000009
```

### Language Detection

**langdetect**:
- Used by: `normalize_html.py:39-47`
- Purpose: Detect document language
- Sample size: First 8000 characters
- Fallback: Returns "en" on error

**Whitelisting**:
- Known English domains bypassed (sec.gov, salesforce.com, etc.)
- Reduces false negatives for technical documentation

### Token Counting

**tiktoken**:
- Used by: `normalize_html.py`, `chunk_documents.py`, `qa_step07_retrieval_eval.py`
- Purpose: Accurate token counting for OpenAI models
- Encoding: `cl100k_base` (GPT-4 tokenizer)
- Fallback: Word count (`len(text.split())`) if tiktoken unavailable

### Embedding API

**OpenAI API (ada-002)**:
- Used by: `qa_step01_embeddings.py`, `embedding_utils.py`
- Model: `text-embedding-ada-002`
- Dimension: 1536
- Rate limiting: Built into API client
- Caching: SHA-256 keyed cache in `data/cache/embeddings/`

**Cache Structure** (`embedding_utils.py:20-67`):
- Key: 16-character SHA-256 prefix of input text
- Value: JSON with `text_hash` (MD5) and `embedding` (1536 floats)
- Hit rate: Reduces API costs for repeated processing

### Vector Stores

**FAISS** (Facebook AI Similarity Search):
- Used by: `qa_step02_indexes.py`
- Index type: HNSW (Hierarchical Navigable Small World)
- Metric: L2 distance
- Parameters: M=32, efConstruction=200, efSearch=128
- Environment: `ageFaiss` conda env (Python 3.12) to avoid OpenMP conflicts

**Weaviate**:
- Used by: `qa_step02_indexes.py`
- Schema: Minimal with required properties (chunk_id, text, metadata)
- Backend: Local or cloud instance
- Purpose: Dev docs indexing

**Pinecone**:
- Used by: `qa_step02_indexes.py`
- Metric: Cosine similarity
- Purpose: Press/financial content indexing
- Manifest only (no actual upserts in current implementation)

## 8. Execution & Usage

### Running Collection Scripts

**Basic Usage**:
```bash
# Fetch SEC filings
python3 scripts/fetch_sec_filings.py --limit 6

# Fetch investor news (with date filter)
python3 scripts/fetch_investor_news.py --since 2024-01-01 --limit 20

# Fetch newsroom (with index crawling)
python3 scripts/fetch_newsroom_rss.py --limit 30 --use-index

# Fetch dev/help/product docs
python3 scripts/fetch_dev_docs.py
python3 scripts/fetch_help_docs.py
python3 scripts/fetch_product_docs.py

# Fetch Wikipedia
python3 scripts/fetch_wikipedia.py
```

**Common Arguments**:
- `--dry-run`: Preview without writing files
- `--limit N`: Max items to fetch (0 = unlimited)
- `--concurrency N`: Concurrent requests (default: 4)
- `--since YYYY-MM-DD`: Lower bound date (RSS feeds only)
- `--until YYYY-MM-DD`: Upper bound date (RSS feeds only)

**Parallel Execution**:
```bash
# Run all fetch scripts concurrently in background
for script in fetch_*.py; do
  python3 scripts/$script &
done
wait
```

### Running Processing Scripts

**Normalization** (Phase-based):
```bash
# Phase A (even hash)
python3 scripts/normalize_html.py --phase A

# Phase B (odd hash)
python3 scripts/normalize_html.py --phase B

# Both phases in parallel
python3 scripts/normalize_html.py --phase A &
python3 scripts/normalize_html.py --phase B &
wait
```

**Metadata Extraction** (Phase-based):
```bash
# Phase A
python3 scripts/extract_metadata.py --phase A

# Phase B
python3 scripts/extract_metadata.py --phase B

# Parallel
python3 scripts/extract_metadata.py --phase A &
python3 scripts/extract_metadata.py --phase B &
wait
```

**Chunking**:
```bash
# All documents
python3 scripts/chunk_documents.py

# Preview first 10
python3 scripts/chunk_documents.py --dry-run --limit 10
```

**Deduplication**:
```bash
# Full deduplication
python3 scripts/dedupe_chunks.py

# Dry run
python3 scripts/dedupe_chunks.py --dry-run
```

### Running Quality Gates

**Sequential Execution** (recommended):
```bash
# Gate 0: Baseline
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step00_baseline.py

# Gate 1: Embeddings (requires OPENAI_API_KEY in .env)
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py

# Gate 2: Indexes (MUST use ageFaiss environment!)
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate 3: MCP tools
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py

# Gate 4: Router
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step04_router.py

# Gate 5: Graph
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step05_graph.py

# Gate 6: A2A
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step06_a2a.py --session-id <session_id>

# Gate 7: Retrieval evaluation (with relaxed budgets)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py

# Gate 8: Generation evaluation
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step08_generation_eval.py
```

**Environment Variables**:

**Gate 1**:
- `AG1_AUTO_CONFIRM=1`: Skip cost confirmation prompt
- `OPENAI_API_KEY`: Required for embedding API

**Gate 7**:
- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating (for dev/test)
- `AG7_LATENCY_MULTIPLIER=3.0`: Relax latency budgets (3× baseline)
- `AG7_DEBUG=1`: Enable debug logging
- `AG7_TRACE=1`: Enable trace logging
- `AG7_ANALYZE_TOPK=10`: Top-k for analysis (default: 10)
- `AG7_NEAR_SEQ_TOL=2`: Sequence tolerance for near-miss (default: 2)

### Full Pipeline Execution

**End-to-End** (all stages):
```bash
#!/bin/bash
set -e

# 1. Collection (parallel)
for script in fetch_*.py; do
  python3 scripts/$script &
done
wait

# 2. Normalization (parallel phases)
python3 scripts/normalize_html.py --phase A &
python3 scripts/normalize_html.py --phase B &
wait

# 3. Metadata (parallel phases)
python3 scripts/extract_metadata.py --phase A &
python3 scripts/extract_metadata.py --phase B &
wait

# 4. Chunking
python3 scripts/chunk_documents.py

# 5. Deduplication
python3 scripts/dedupe_chunks.py

# 6. Embeddings (Gate 1)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG1_AUTO_CONFIRM=1 \
  python scripts/qa_step01_embeddings.py

# 7. Indexes (Gate 2) - MUST use ageFaiss!
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss \
  python scripts/qa_step02_indexes.py

# 8. MCP (Gate 3)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/qa_step03_mcp.py

# 9. Router (Gate 4)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/qa_step04_router.py

# 10. Retrieval Evaluation (Gate 7)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py

echo "Pipeline complete!"
```

### Graph Workflow Execution

**LangGraph Implementation** (recommended):
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Original Implementation** (for comparison):
```bash
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Output Locations**:
- State: `state/session-{session_id}.json`
- Insights: `outputs/{session_id}/insights.json`
- Email: `outputs/{session_id}/email.json`
- Timing: `outputs/{session_id}/timing.json`
- Compliance: `outputs/{session_id}/compliance_report.json`

## 9. Code Patterns & Conventions

### Pattern 1: Fetch Script Naming
**Convention**: `fetch_<source_type>.py`

**Examples**:
- `fetch_dev_docs.py`
- `fetch_sec_filings.py`
- `fetch_investor_news.py`

**Shared Structure**:
- Async main with `aiohttp.ClientSession`
- Rate limiting via `RateLimiter` (6 RPS)
- Dual output: `.raw.html` + `.meta.json`
- Deduplication via existing `.meta.json` check

### Pattern 2: Phase-Based Processing
**Implementation**: Hash-based deterministic split

```python
def phase_select(doc_id: str, phase: str) -> bool:
    h = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return (int(h[-1], 16) % 2 == 0) if phase == "A" else (int(h[-1], 16) % 2 == 1)
```

**Usage**:
- `normalize_html.py --phase A|B`
- `extract_metadata.py --phase A|B`

**Benefit**: Enables parallel processing without coordination

### Pattern 3: Document ID Format
**Convention**: `crm::{doctype}::{date}::{slug}::{hash8}`

**Generation** (`common.py:322-326`):
```python
def build_doc_id(doctype: str, date_str: Optional[str], slug_base: str, url_for_hash: str) -> str:
    date_part = date_str or "unknown"
    slug = slugify(slug_base or "document")
    tail = sha1_8(strip_tracking_params(url_for_hash))
    return f"crm::{doctype}::{date_part}::{slug}::{tail}"
```

**Examples**:
- `crm::10-K::2025-03-05::fy25-form-10-k::4c26cea2`
- `crm::press::2024-12-17::introducing-agentforce-2-0::64bc03ee`
- `crm::dev_docs::unknown::agent-api-developer-guide::6f4b900c`

### Pattern 4: Dual-Format Reports
**Convention**: JSON (machine) + Markdown (human)

**Structure**:
```python
# JSON report
machine = {
    "gate": "G01_COLLECTION",
    "computed_at": now_iso(),
    "summary": {...},
    "checks": [{...}],
    "status": "PASS"
}
write_json("reports/qa/gate01_collection.json", machine)

# Markdown report
lines = ["# Gate G01 — Collection QA", "Summary: PASS", ...]
write("reports/qa/human_readable/gate01_collection.md", "\n".join(lines))
```

**Used by**: All 18 QA scripts

### Pattern 5: Atomic File Writes
**Implementation**: Temp file + `os.replace()`

```python
if not args.dry_run:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
```

**Used by**:
- `extract_metadata.py:252-255`
- `dedupe_chunks.py:212-216`

**Benefit**: Prevents partial writes on crash

### Pattern 6: Rate-Limited Async Fetching
**Implementation** (`common.py:187-198, 228-294`):

```python
class RateLimiter:
    def __init__(self, rps: float = 6.0):
        self.rps = rps
        self.last_call = 0.0
        self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            sleep_time = max(0, 1.0 / self.rps - (now - self.last_call))
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.last_call = time.monotonic()

async def fetch_with_retries(session, limiter, url, ...):
    await limiter.wait()
    # ... fetch logic with exponential backoff
```

**Used by**: All 7 fetch scripts

### Pattern 7: Logging with Timestamps
**Implementation** (`common.py:201-214`):

```python
def build_logger() -> Tuple[logging.Logger, str]:
    ensure_dir("logs/fetch")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/fetch/{ts}.log"
    logger = logging.getLogger(f"fetch_{ts}")
    # ... add file and stdout handlers
    return logger, log_path
```

**Used by**: Collection and processing scripts

### Pattern 8: Configuration Loading
**YAML**:
```python
def load_yaml(path: str) -> Dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
```

**JSON**:
```python
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

**Used by**: Processing and QA scripts

### Pattern 9: Error Handling
**Convention**: Log and continue

```python
for path in paths:
    try:
        # ... processing
    except Exception as e:
        logger.error(f"ERROR {doc_id} {e}")
        continue
```

**Used by**: All processing loops

### Pattern 10: Status Conventions
**Binary**: `"PASS"` / `"FAIL"`

**Traffic Light**: `"GREEN"` / `"AMBER"` / `"RED"`

**Usage**:
- Individual checks: `PASS` / `FAIL` / `WARN`
- Overall gates: `GREEN` / `AMBER` / `RED`
- `AMBER`: Marginal pass with warnings

## 10. Testing & Verification

### Gate G01: Collection QA

**Script**: `scripts/qa_verify_collection.py`

**Purpose**: Validates raw document collection coverage, quality, and recency.

**Checks**:
1. **COL-001**: SEC docs count ≥ 6
2. **COL-002**: IR docs count ≥ 16
3. **COL-003**: Newsroom total ≥ 24
4. **COL-004**: Newsroom per feed ≥ 10
5. **COL-005**: Product/dev/help/wiki total in [9, 11]
6. **COL-006**: HTTP 200 ratio ≥ 0.99
7. **COL-007**: Exact duplicate rate ≤ 0.05
8. **COL-008**: Press within 24mo ratio ≥ 0.70
9. **COL-009**: Press within 12mo ratio ≥ 0.40

**Reports**:
- `reports/qa/gate01_collection.json`
- `reports/qa/human_readable/gate01_collection.md`

### Gate G02: Normalization QA

**Script**: `scripts/qa_verify_normalization.py`

**Purpose**: Validates text normalization quality, coverage, and content retention.

**Checks**:
1. **STD-001**: Normalized coverage ≥ 0.98
2. **STD-002**: English detection ≥ 0.95
3. **STD-003**: Min word count violations == 0
4. **STD-004**: Retention ratio median (press) ≥ baseline - 0.05
5. **STD-005**: Heading presence ≥ 0.90
6. **STD-006**: PDF page map missing == 0

**Reports**:
- `reports/qa/gate02_normalization.json`
- `reports/qa/human_readable/gate02_normalization.md`

### Gate G03: Metadata & SEC Structure QA

**Script**: `scripts/qa_verify_metadata.py`

**Purpose**: Validates metadata completeness, SEC item coverage, and data integrity.

**Checks**:
1. **META-001**: Required fields presence ≥ 0.98
2. **META-002**: SEC item coverage median ≥ 0.75
3. **META-003**: Topic nonempty ratio ≥ 0.90
4. **META-004**: Persona tags ratio ≥ 0.60
5. **META-005**: Doc ID unique == true
6. **META-006**: Invalid date count == 0

**Reports**:
- `reports/qa/gate03_metadata.json`
- `reports/qa/human_readable/gate03_metadata.md`

### Gate G04: Chunking QA

**Script**: `scripts/qa_verify_chunking.py`

**Purpose**: Validates chunk size distribution, overlap consistency, and SEC boundary alignment.

**Checks**:
1. **CHK-001**: Chunk count within expected ≥ 0.90
2. **CHK-002**: Size envelope ratio (press) ≥ 0.80
3. **CHK-003**: Adjacent overlap Jaccard median in [0.12, 0.22]
4. **CHK-004**: SEC boundary alignment ≥ 0.90

**Reports**:
- `reports/qa/gate04_chunking.json`
- `reports/qa/human_readable/gate04_chunking.md`

### Gate G05: Deduplication QA

**Script**: `scripts/qa_verify_dedupe.py`

**Purpose**: Validates duplicate removal effectiveness while preserving content coverage.

**Checks**:
1. **DED-001**: Global duplicate ratio ≤ 0.15
2. **DED-002**: Non-adjacent Jaccard P95 ≤ 0.30
3. **DED-003**: Coverage ratio median ≥ 0.90

**Reports**:
- `reports/qa/gate05_dedupe.json`
- `reports/qa/human_readable/gate05_dedupe.md`

### Gate 0: Baseline

**Script**: `scripts/qa_step00_baseline.py`

**Purpose**: Establishes baseline metrics for corpus.

**Checks**:
1. **G0-01**: Baseline docs ≥ 80
2. **G0-02**: Publish date percent ≥ 0.98
3. **G0-03**: Seed eval size ≥ 40
4. **G0-04**: Baseline chunks ≥ baseline docs
5. **G0-05**: Domain count ≥ 3

**Metrics**:
- Document/chunk counts
- Age distribution (p50, p90, buckets)
- Token count distribution (p50, p90)

**Reports**:
- `reports/qa/step00_baseline.json`
- `reports/qa/step00_baseline.md`

### Gate 1: Embeddings

**Script**: `scripts/qa_step01_embeddings.py`

**Purpose**: Generates and validates text embeddings.

**Checks**:
1. **G1-01**: Embedding rows == baseline chunks
2. **G1-02**: Vector dimension == config dimension (1536)
3. **G1-03a**: Zero vectors == 0
4. **G1-03b**: NaN vectors == 0
5. **G1-04**: Norm outliers ≤ 0.005

**Metrics**:
- L2 norm statistics (median, IQR, outliers)
- Embedding row count
- Vector dimension

**Reports**:
- `reports/qa/step01_embeddings.json`
- `reports/qa/step01_embeddings.md`

### Gate 2: Indexes

**Script**: `scripts/qa_step02_indexes.py`

**Purpose**: Builds and validates vector indexes.

**Checks**:
1. **G2-01**: Pinecone upsert rate ≥ 0.98
2. **G2-02**: Weaviate upsert rate ≥ 0.98
3. **G2-03**: FAISS count ratio ≥ 0.98
4. **G2-04**: Missing required metadata ≤ 0.02
5. **G2-05**: FAISS round-trip error ≤ 0.001
6. **G2-06**: Sanity search min top-k ≥ 3
7. **G2-07**: Keyword hit min top-10 ≥ 1

**Metrics**:
- Index counts per backend
- Round-trip integrity test results
- Sanity search results (3 test queries)

**Reports**:
- `reports/qa/step02_indexes.json`
- `reports/qa/step02_indexes.md`
- `data/final/reports/index_health.json`

### Gate 3: MCP Tools

**Script**: `scripts/qa_step03_mcp.py`

**Purpose**: Validates MCP tool endpoints.

**Checks**:
1. **G3-01**: Health endpoints OK == 5
2. **G3-02**: Contract OK rate == 1.0
3. **G3-03**: Latency budgets (P50/P95 per backend)
4. **G3-04**: Timeout rate == 0.0

**Tests**:
- Health checks (5 tools)
- Contract validation (valid/invalid requests)
- Latency sampling (15 queries × 3 backends)

**Reports**:
- `reports/qa/step03_mcp.json`
- `reports/qa/step03_mcp.md`
- `logs/mcp/step03_probes.jsonl`

### Gate 4: Router

**Script**: `scripts/qa_step04_router.py`

**Purpose**: Validates query routing logic and diversity.

**Checks**:
1. **COV-{backend}**: Route share ≥ 0.10 OR count ≥ 1 (per backend)
2. **EMP-001**: Empty result rate ≤ 0.02
3. **EMP-002**: Auto retry success rate ≥ 0.95
4. **FRS-001**: Avg doc age ≤ max(365, age_p50)
5. **DIV-001**: Mean unique domains (top 10) ≥ 2.4

**Metrics**:
- Route counts per backend
- Empty result rate and retry success
- Average document age
- Domain diversity

**Reports**:
- `reports/qa/step04_router.json`
- `reports/qa/step04_router.md`
- `reports/router/step04_router_trace.jsonl`

### Gate 7: Retrieval Evaluation

**Script**: `scripts/qa_step07_retrieval_eval.py`

**Purpose**: Evaluates retrieval quality on labeled test set.

**Checks**:
1. **G7-01**: Recall@10 ≥ 0.80
2. **G7-02**: nDCG@5 ≥ 0.60
3. **G7-03**: Coverage ≥ max(3.0, 0.75 × baseline_domains / 10)
4. **G7-04**: Freshness ≤ max(540, baseline_age_p50)
5. **G7-05**: Latency budgets (all backends within P95)

**Metrics**:
- Recall@k (k=1,3,5,10,20)
- nDCG@5
- Soft recall@10 (near-miss recovery)
- Near-miss rate
- Rank statistics (P50/P75/P90/max)
- Per-backend quality breakdown
- Per-doctype breakdown

**Reports**:
- `reports/qa/step07_retrieval_eval.json`
- `reports/qa/step07_retrieval_eval.md`
- `reports/eval/retrieval_failures.jsonl`
- `reports/router/step07_retrieval_trace.jsonl`

### Gate 8: Generation Evaluation

**Script**: `scripts/qa_step08_generation_eval.py`

**Purpose**: Runs 10 end-to-end sessions and validates outputs.

**Checks**:
1. **G8-01**: Structural pass rate == 1.0
2. **G8-02**: Critical flags total == 0
3. **G8-03**: Length/readability pass runs ≥ 9
4. **G8-04**: Persona keyword hits avg ≥ 2.0

**Validations per run**:
- Insight count == 5
- Distinct sources ≥ 4
- Recent count ≥ 2 (within 12mo)
- Email schema (all required fields)
- Proof points resolve

**Metrics**:
- Structural pass rate
- Critical/warning flag counts
- Length/readability pass runs
- Persona keyword hits (avg, p50, p95)

**Reports**:
- `reports/qa/step08_generation_eval.json`
- `reports/qa/step08_generation_eval.md`
- `reports/eval/generation_metrics.json`
- `reports/eval/compliance_metrics.json`

## 11. Known Issues & Limitations

### Rate Limiting on SEC Edgar

**Issue**: SEC limits requests to 10 per second, violations may result in IP blocks.

**Mitigation**:
- `RateLimiter` set to 6 RPS (conservative) in `common.py:20`
- Configurable via `AR_GLOBAL_RPS` environment variable
- Exponential backoff for 429 responses in `fetch_with_retries()`

**References**:
- `common.py:187-198` (RateLimiter class)
- `common.py:238-244` (backoff logic)

### HTML Parsing Edge Cases

**Issue 1: XBRL Tags in SEC Filings**
- SEC HTML filings contain XBRL inline tags (ix:header, ix:hidden, xbrli:context)
- These are removed via CSS selectors in `normalization.rules.yaml:13-19`
- Escaped colon syntax: `ix\3A header` → `ix:header`

**Issue 2: Dynamic Content**
- JavaScript-rendered content not captured (fetch scripts only retrieve initial HTML)
- Affects: Some Salesforce help docs with client-side rendering
- Mitigation: Manual ingestion via `ingest_manual_html.py` if needed

**Issue 3: PDF Table Extraction**
- `pdfminer.six` extracts text but loses table structure
- SEC tables become unstructured text blocks
- Mitigation: Not addressed (acceptable for MVP)

**References**:
- `normalize_html.py:104-107` (removal logic)
- `normalize_html.py:140-150` (PDF extraction)

### OpenMP Conflicts with FAISS

**Issue**: Installing pip `faiss-cpu` in `age` environment causes `OMP Error #15` due to multiple OpenMP runtimes.

**Symptoms**:
```
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
Segmentation fault
```

**Root Cause**: Conda's numpy and pip's faiss-cpu both link OpenMP, causing runtime conflicts.

**Mitigation**:
- **Use separate environments**: `age` (Python 3.13) for most tasks, `ageFaiss` (Python 3.12) for Gate-2
- **Never install pip faiss-cpu in age environment**
- `ageFaiss` uses conda-forge faiss-cpu which resolves OpenMP correctly

**References**:
- `CLAUDE.md:80-88` (documented gotcha)
- `docs/envs.md` (environment details)
- `docs/troubleshooting.md` (debugging steps)

### Embedding Consistency Requirements

**Issue**: Recall drops to 0% if documents and queries use different embedding functions.

**Root Cause**: Vector similarity requires identical embedding models and preprocessing.

**Requirements**:
1. Both documents AND queries MUST use `embed_text()` from `embedding_utils.py`
2. Same preprocessing (no extra whitespace normalization)
3. Same OpenAI model (`text-embedding-ada-002`)
4. Dimension must be 1536 for both

**Mitigation**:
- Centralized embedding via `embedding_utils.py:76-80`
- Cache ensures consistency across runs
- Gate-1 validation checks dimension and norms

**References**:
- `embedding_utils.py:76-80` (embed_text function)
- `CLAUDE.md:92-99` (documented gotcha)

### Date Parsing Ambiguity

**Issue**: Multiple date formats across sources require fuzzy parsing.

**Examples**:
- RSS: `"Mon, 17 Dec 2024 08:00:00 GMT"` (RFC 822)
- HTML meta: `"2024-12-17T08:00:00+00:00"` (ISO 8601)
- SEC: `"2025-03-05"` (YYYY-MM-DD)
- Text: `"Updated: March 5, 2024"` (human-readable)

**Mitigation**:
- `coerce_date()` tries 8 different formats in `common.py:148-184`
- Validation rejects dates before 1999 or in future
- Fallback to `None` if all parsers fail

**References**:
- `common.py:148-184` (coerce_date function)
- `extract_metadata.py:80-130` (date resolution logic)

### Short Document Filtering

**Issue**: 8-K filings can be very short (< 200 words) but still valid.

**Current Behavior**: Normalized 8-K docs with < 200 words are dropped as "DROPPED_SHORT".

**Justification**: Short 8-Ks typically lack substantive content (e.g., "No material changes").

**Trade-off**: May lose some valid short filings.

**References**:
- `normalize_html.py:287-291` (filtering logic)

### Language Detection False Negatives

**Issue**: Technical documentation with code snippets may be misclassified as non-English.

**Mitigation**:
- Whitelists known English domains (sec.gov, salesforce.com, etc.)
- Samples only first 8000 characters (before code sections)
- Returns "en" on detection errors

**References**:
- `normalize_html.py:39-47` (detect_lang function)
- `normalize_html.py:250-258` (whitelisting)

## 12. References

### Related Documentation

- **docs/architecture.md** — Detailed system design, LangGraph orchestration, embedding system, multi-index routing
- **docs/commands.md** — Complete command reference for all 41 scripts
- **docs/configuration.md** — Configuration file deep dive with tuning guidelines
- **docs/troubleshooting.md** — Debug playbook for common issues
- **docs/evaluation.md** — Quality gates and metrics definitions
- **docs/envs.md** — Environment setup and package details

### Part 3: Embeddings (Gate-1)

**Covered in**: Stage 6 of this document (lines 457-470)

**Key Topics**:
- OpenAI ada-002 (1536-dim)
- Cache mechanism (SHA-256 keys)
- L2 norm validation
- Embedding stats (median, IQR, outliers)

**See also**: `docs/architecture.md` for embedding system design

### Part 7: Quality Gates (Gates 0-8)

**Covered in**: Section 10 of this document (lines 1152-1335)

**Key Topics**:
- 13 quality gates (G01-G05, G0-G8)
- Validation checks and thresholds
- Dual-format reports (JSON + Markdown)
- Status conventions (PASS/FAIL, GREEN/AMBER/RED)

**See also**: `docs/evaluation.md` for metric definitions and thresholds

### External Resources

**SEC Edgar**:
- Main site: https://www.sec.gov
- EDGAR search: https://www.sec.gov/edgar/searchedgar/companysearch.html
- CIK lookup: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001108524
- Rate limiting: https://www.sec.gov/os/webmaster-faq#developers

**OpenAI Embeddings**:
- API docs: https://platform.openai.com/docs/guides/embeddings
- Model: text-embedding-ada-002 (1536 dimensions)
- Pricing: https://openai.com/pricing

**FAISS**:
- GitHub: https://github.com/facebookresearch/faiss
- Wiki: https://github.com/facebookresearch/faiss/wiki
- HNSW paper: https://arxiv.org/abs/1603.09320

**Weaviate**:
- Docs: https://weaviate.io/developers/weaviate
- Schema: https://weaviate.io/developers/weaviate/manage-data/collections

**Pinecone**:
- Docs: https://docs.pinecone.io
- Quickstart: https://docs.pinecone.io/guides/get-started/quickstart

### Code References by Stage

**Stage 1 (Collection)**:
- `scripts/fetch_sec_filings.py:67-133` — SEC fetch logic
- `scripts/fetch_investor_news.py:89-140` — RSS parsing and article fetch
- `scripts/fetch_newsroom_rss.py:154-199` — Dual RSS feeds + index crawling
- `scripts/common.py:228-294` — Shared fetch_with_retries function

**Stage 2 (Normalization)**:
- `scripts/normalize_html.py:83-137` — Core HTML processing
- `scripts/normalize_html.py:216-236` — Domain-specific rules and fallbacks
- `configs/normalization.rules.yaml:1-34` — Cleaning rules

**Stage 3 (Metadata)**:
- `scripts/extract_metadata.py:58-77` — Title extraction (4-level waterfall)
- `scripts/extract_metadata.py:80-130` — Publish date resolution (doctype-specific)
- `scripts/extract_metadata.py:133-178` — Topic and persona assignment
- `configs/metadata.dictionary.yaml:1-26` — Schema documentation

**Stage 4 (Chunking)**:
- `scripts/chunk_documents.py:67-102` — Sliding window with boundary snapping
- `scripts/chunk_documents.py:125-160` — Chunk building with title boost
- `configs/chunking.config.json` — Chunking parameters

**Stage 5 (Deduplication)**:
- `scripts/dedupe_chunks.py:14-27` — Text normalization and Jaccard similarity
- `scripts/dedupe_chunks.py:86-154` — Inverted index and clustering
- `scripts/dedupe_chunks.py:56-64` — Canonical selection

**Stage 6 (Embedding, Gate-1)**:
- `scripts/qa_step01_embeddings.py:140-224` — Embedding generation and validation
- `scripts/embedding_utils.py:76-80` — OpenAI API calls
- `scripts/embedding_utils.py:20-67` — Cache loading and saving

**Stage 7 (Indexing, Gate-2)**:
- `scripts/qa_step02_indexes.py:80-164` — FAISS index building
- `scripts/qa_step02_indexes.py:212-271` — Sanity search tests
- `configs/vector.indexing.yaml` — Index configuration

**Stage 8-13 (Gates 3-8)**:
- `scripts/qa_step03_mcp.py:40-204` — MCP stub server setup
- `scripts/qa_step04_router.py:262-396` — Query routing logic
- `scripts/qa_step07_retrieval_eval.py:367-576` — Retrieval evaluation loop
- `scripts/qa_step08_generation_eval.py:664-680` — Generation execution

## Open Questions

None. This research comprehensively documents the 13-stage data pipeline as implemented.

---

## Metadata

- **Total Scripts**: 41 (7 fetch, 4 processing, 18 QA, 12 supporting)
- **Total Gates**: 13 (5 data pipeline verification + 8 infrastructure/evaluation gates)
- **Data Directories**: 6 main (raw, interim, vector, cache, final, backup)
- **Configuration Files**: 10 YAML/JSON files in `configs/`
- **Document Count**: 80+ documents → 536+ chunks
- **Embedding Dimension**: 1536 (OpenAI ada-002)
- **Vector Stores**: 3 (FAISS, Weaviate, Pinecone)
- **MCP Tools**: 5 (kb.search, web.fetch, link.resolve, crm.lookup, safety.check)
- **LangGraph Nodes**: 8 (Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler)
