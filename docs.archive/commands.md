# Command Reference

Complete reference for all scripts and commands in the ag3 system.

## Conda Path Configuration

**Conda executable**: `/Users/liyunxiao/anaconda3/bin/conda`

All `conda` commands in this document use the full path above. For brevity, examples may show `conda run -n age ...` but the full path `/Users/liyunxiao/anaconda3/bin/conda run -n age ...` should be used.

## Quality Gates (Main Pipeline)

Run gates in sequence to validate the pipeline. Each gate produces dual-format reports (JSON + Markdown) in `reports/qa/`.

### Gate-0: Baseline Checks

```bash
conda run -n age python scripts/qa_step00_baseline.py
```

**Purpose**: Validate directory structure, file permissions, and basic system requirements

**Output**:
- `reports/qa/step00_baseline.json`
- `reports/qa/step00_baseline.md`

### Gate-1: Generate Embeddings

```bash
conda run -n age python scripts/qa_step01_embeddings.py
```

**Purpose**: Generate OpenAI ada-002 text embeddings for all chunks

**Requirements**:
- `OPENAI_API_KEY` in `.env` file
- Deduplicated chunks in `data/interim/dedup/`

**Output**:
- `data/vector/embeddings/embeddings.parquet`
- `reports/qa/step01_embeddings.json`
- `reports/qa/step01_embeddings.md`

**Environment Variables**:
- `AG1_AUTO_CONFIRM=1`: Skip cost confirmation prompt (auto-proceed)
- `OPENAI_API_KEY`: OpenAI API key (required)

### Gate-2: Build and Validate Indexes

```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Purpose**: Build FAISS index and validate index health

**Requirements**:
- Embeddings from Gate-1
- **Must use `ageFaiss` environment** (NOT `age`)

**Output**:
- `data/vector/faiss/index.faiss`
- `data/final/reports/index_health.json`
- `reports/qa/step02_indexes.json`
- `reports/qa/step02_indexes.md`

**Critical**: Use `ageFaiss` environment to avoid OpenMP conflicts

### Gate-3: Validate MCP Tools

```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**Purpose**: Start MCP stub services and validate tool health and contracts

**Output**:
- `reports/qa/step03_mcp.json`
- `reports/qa/step03_mcp.md`
- `logs/mcp/step03_probes.jsonl`

**Ports Used**: 7801-7805 (localhost)

### Gate-4: Test Router Heuristics

```bash
conda run -n age python scripts/qa_step04_router.py
```

**Purpose**: Test query routing heuristics across FAISS/Weaviate/Pinecone

**Output**:
- `reports/qa/step04_router.json`
- `reports/qa/step04_router.md`

### Gate-5: Validate LangGraph Orchestration

```bash
conda run -n age python scripts/qa_step05_graph.py
```

**Purpose**: Validate LangGraph workflow execution (node transitions, state management)

**Output**:
- `reports/qa/step05_graph.json`
- `reports/qa/step05_graph.md`

### Gate-6: Agent-to-Agent Compliance Checks

```bash
conda run -n age python scripts/qa_step06_a2a.py --session-id <SESSION>
```

**Purpose**: Validate A2A handoff protocols and compliance negotiation

**Arguments**:
- `--session-id`: Session identifier for tracking

**Output**:
- `reports/qa/step06_a2a.json`
- `reports/qa/step06_a2a.md`
- `outputs/<session-id>/compliance_report.json`

### Gate-7: Retrieval Evaluation

```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

**Purpose**: End-to-end retrieval quality assessment (recall@10, nDCG@5, latency)

**Output**:
- `reports/qa/step07_retrieval_eval.json`
- `reports/qa/step07_retrieval_eval.md`
- `reports/eval/retrieval_failures.jsonl`
- `reports/router/step07_retrieval_trace.jsonl`

**Environment Variables**:
- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating (recommended for initial runs)
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets (e.g., 3.0 for 3x tolerance)
- `AG7_ANALYZE_TOPK=<int>`: Retrieval cut-off for evaluation (default: 10)
- `AG7_TOPK_SLICES="1,3,5,10"`: Additional @k slices for recall curves
- `AG7_NEAR_SEQ_TOL=<int>`: Near-miss tolerance in chunks within same doc (default: 1)
- `AG7_TRACE=1`: Enable per-query trace JSONL (default: enabled)
- `AG7_TRACE_TOPK=<int>`: Number of top-K items to capture in trace (default: 10)
- `AG7_TRACE_SUCCESSES=1`: Include successes in trace (default: enabled; set 0 for misses only)
- `AG7_DEBUG=1`: Umbrella debug switch enabling tracing (default: enabled)

### Gate-8: Generation & Compliance Evaluation

```bash
conda run -n age python scripts/qa_step08_generation_eval.py
```

**Purpose**: End-to-end email generation quality and compliance validation (10 runs across ≥3 personas)

**Output**:
- `reports/qa/step08_generation_eval.json`
- `reports/qa/step08_generation_eval.md`
- `reports/eval/generation_metrics.json`
- `reports/eval/compliance_metrics.json`

## Data Collection

Scripts to fetch documents from various sources.

### SEC Filings

```bash
python3 scripts/fetch_sec_filings.py
```

**Purpose**: Fetch SEC 10-K/10-Q filings
**Output**: `data/raw/sec/`

### Product Documentation

```bash
python3 scripts/fetch_product_docs.py
```

**Purpose**: Fetch product documentation
**Output**: `data/raw/product/`

### Developer Documentation

```bash
python3 scripts/fetch_dev_docs.py
```

**Purpose**: Fetch developer documentation
**Output**: `data/raw/dev_docs/`

### Help Documentation

```bash
python3 scripts/fetch_help_docs.py
```

**Purpose**: Fetch help articles
**Output**: `data/raw/help_docs/`

### Wikipedia

```bash
python3 scripts/fetch_wikipedia.py
```

**Purpose**: Fetch Wikipedia articles
**Output**: `data/raw/wikipedia/`

### Newsroom RSS

```bash
python3 scripts/fetch_newsroom_rss.py
```

**Purpose**: Fetch press releases from newsroom RSS feed
**Output**: `data/raw/newsroom/`

### Investor News

```bash
python3 scripts/fetch_investor_news.py
```

**Purpose**: Fetch investor news articles
**Output**: `data/raw/investor_news/`

### Manual Ingestion (HTML)

```bash
python3 scripts/ingest_manual_html.py
```

**Purpose**: Ingest HTML files from `data/manual_inbox/`
**Output**: `data/raw/manual/`

**Note**: Place HTML files in `data/manual_inbox/` before running

### Manual Ingestion (IR HTML)

```bash
python3 scripts/ingest_manual_ir_html.py
```

**Purpose**: Ingest investor relations HTML from `data/manual_inbox/`
**Output**: `data/raw/investor_relations/`

## Data Processing

Scripts to process and transform raw documents.

### Parse SEC Structures

```bash
python3 scripts/parse_sec_structures.py
```

**Purpose**: Parse SEC filing structure (sections, tables)
**Output**: `data/interim/sec_parsed/`

### Extract Metadata

```bash
python3 scripts/extract_metadata.py
```

**Purpose**: Extract structured metadata from documents
**Config**: `configs/metadata.dictionary.yaml`
**Output**: `data/interim/metadata/`

### Chunk Documents

```bash
python3 scripts/chunk_documents.py
```

**Purpose**: Split documents into retrievable chunks
**Config**: `configs/chunking.config.json`
**Output**: `data/interim/chunks/`

### Deduplicate Chunks

```bash
python3 scripts/dedupe_chunks.py
```

**Purpose**: Remove duplicate chunks
**Output**: `data/interim/dedup/`

### Build Evaluation Seed

```bash
python3 scripts/build_eval_seed.py
```

**Purpose**: Build evaluation seed dataset for quality gates
**Output**: `data/interim/eval/eval_seed.jsonl`

## Verification & Utilities

Scripts to verify individual pipeline stages.

### Verify Collection

```bash
python3 scripts/qa_verify_collection.py
```

**Purpose**: Verify raw document collection completeness

### Verify Normalization

```bash
python3 scripts/qa_verify_normalization.py
```

**Purpose**: Verify text normalization rules applied correctly
**Config**: `configs/normalization.rules.yaml`

### Verify Metadata

```bash
python3 scripts/qa_verify_metadata.py
```

**Purpose**: Verify metadata extraction completeness

### Verify Chunking

```bash
python3 scripts/qa_verify_chunking.py
```

**Purpose**: Verify document chunking quality

### Verify Deduplication

```bash
python3 scripts/qa_verify_dedupe.py
```

**Purpose**: Verify deduplication effectiveness

### Verify Link Health

```bash
python3 scripts/qa_verify_link_health.py
```

**Purpose**: Check health of external links in documents

### Verify Evaluation Seed

```bash
python3 scripts/qa_verify_eval_seed.py
```

**Purpose**: Verify evaluation seed dataset quality

### Verify Day-1 Signoff

```bash
python3 scripts/qa_verify_day1_signoff.py
```

**Purpose**: Verify Day-1 milestone completion

### Build Inventory CSV

```bash
python3 scripts/build_inventory_csv.py
```

**Purpose**: Build document inventory CSV
**Output**: `data/final/inventory/inventory.csv`

### Link Health Check

```bash
python3 scripts/link_health_check.py
```

**Purpose**: Check health of links in indexed documents
**Output**: `data/final/reports/link_health.json`

### Verify Day-1 Milestones

```bash
python3 scripts/verify_day1_milestones.py
```

**Purpose**: Comprehensive Day-1 milestone verification
**Output**: `reports/qa/day1_milestones.{json,md}`

## MCP & Agent Services

Scripts to run agent services and workflows.

### Start MCP Stub Services

```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**Purpose**: Start local MCP stub services
**Ports**: 7801-7805 (localhost)
**Config**: `configs/mcp.tools.yaml`

### Tool Safety Check Server

```bash
python3 scripts/tool_safety_check_server.py
```

**Purpose**: Run tool safety check server (MCP safety.check)
**Port**: 7805 (localhost)

### Execute Original Graph Workflow

```bash
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Purpose**: Execute original (non-LangGraph) workflow

**Arguments**:
- `--company`: Company name for research
- `--persona`: Persona role (must exist in `icl/persona/`)
- `--session-id`: Session identifier for output tracking

**Output**:
- `outputs/<session-id>/insights.json`
- `outputs/<session-id>/email.json`
- `outputs/<session-id>/compliance_report.json`
- `outputs/<session-id>/timing.json`
- `outputs/<session-id>/router_trace.jsonl`
- `state/session-<session-id>.json`

### Execute LangGraph Workflow

```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Purpose**: Execute LangGraph StateGraph workflow

**Arguments**: Same as `run_graph.py` above

**Output**: Identical format to `run_graph.py`

**Note**: Both implementations produce identical outputs and pass the same quality gates

### Visualize LangGraph

```bash
conda run -n age python scripts/visualize_graph.py
```

**Purpose**: Generate visual representation of LangGraph workflow

**Output**:
- `reports/graphs/agent_workflow.mmd` (Mermaid diagram)
- `reports/graphs/agent_workflow.png` (PNG image)

## Environment Variables Reference

### Gate-1 (Embedding Generation)

- `AG1_AUTO_CONFIRM=1`: Skip cost confirmation prompt
- `OPENAI_API_KEY`: OpenAI API key (required)

### Gate-7 (Retrieval Evaluation)

- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets
- `AG7_ANALYZE_TOPK=<int>`: Retrieval cut-off (default: 10)
- `AG7_TOPK_SLICES="1,3,5,10"`: @k slices for recall curves
- `AG7_NEAR_SEQ_TOL=<int>`: Near-miss tolerance (default: 1)
- `AG7_TRACE=1`: Enable trace JSONL (default: enabled)
- `AG7_TRACE_TOPK=<int>`: Trace top-K items (default: 10)
- `AG7_TRACE_SUCCESSES=1`: Include successes in trace
- `AG7_DEBUG=1`: Enable debug tracing

### General

- `AR_USER_AGENT`: Custom user agent for web requests
- `AR_GLOBAL_RPS`: Rate limiting for HTTP requests (requests per second)

## Common Workflows

### Initial Setup

```bash
# 1. Create environments
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml

# 2. Set up API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Collect data (example)
python3 scripts/fetch_sec_filings.py
python3 scripts/fetch_product_docs.py

# 4. Process data
python3 scripts/extract_metadata.py
python3 scripts/chunk_documents.py
python3 scripts/dedupe_chunks.py

# 5. Run quality gates
conda run -n age python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
conda run -n age python scripts/qa_step03_mcp.py
```

### Full Quality Gate Run

```bash
# Run all gates in sequence
conda run -n age python scripts/qa_step00_baseline.py
conda run -n age python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
conda run -n age python scripts/qa_step03_mcp.py
conda run -n age python scripts/qa_step04_router.py
conda run -n age python scripts/qa_step05_graph.py
conda run -n age python scripts/qa_step06_a2a.py --session-id test-session
conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py
conda run -n age python scripts/qa_step08_generation_eval.py
```

### Debug Retrieval Issues

```bash
# Run Gate-7 with full tracing
conda run -n age \
  AG7_DEBUG=1 \
  AG7_TRACE=1 \
  AG7_TRACE_SUCCESSES=1 \
  AG7_LATENCY_MULTIPLIER=5.0 \
  python scripts/qa_step07_retrieval_eval.py

# Inspect failures
cat reports/eval/retrieval_failures.jsonl | jq .
cat reports/router/step07_retrieval_trace.jsonl | jq .
```

### Compare Graph Implementations

```bash
# Run both implementations with same inputs
SESSION_ID="compare-test"

# Original
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id "${SESSION_ID}-original"

# LangGraph
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id "${SESSION_ID}-langgraph"

# Compare outputs
diff outputs/${SESSION_ID}-original/insights.json outputs/${SESSION_ID}-langgraph/insights.json
diff outputs/${SESSION_ID}-original/email.json outputs/${SESSION_ID}-langgraph/email.json
```

## Related Documentation

- **[architecture.md](architecture.md)** - Detailed system design and architecture
- **[configuration.md](configuration.md)** - Configuration file documentation
- **[evaluation.md](evaluation.md)** - Quality gates and evaluation metrics
- **[troubleshooting.md](troubleshooting.md)** - Debug guide and common issues
- **[envs.md](envs.md)** - Environment setup and package details
