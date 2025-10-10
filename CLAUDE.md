# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent RAG (Retrieval-Augmented Generation) system for Sales/IR/PR outreach that automates trusted-source research and audit-ready email generation with step-level traceability. The system implements a gated data pipeline with quality checks at each stage, emitting both machine-readable JSON and human-readable Markdown reports.

## Architecture

### Data Pipeline Stages

The system follows a multi-stage pipeline with quality gates:

1. **Collection** (`fetch_*.py`, `ingest_*.py`): Gather documents from various sources
2. **Normalization** (`qa_verify_normalization.py`): Apply text cleaning rules
3. **Metadata Extraction** (`extract_metadata.py`): Extract structured metadata
4. **Chunking** (`chunk_documents.py`): Split documents into retrievable units
5. **Deduplication** (`dedupe_chunks.py`): Remove duplicate content
6. **Embedding** (Gate-1): Generate text vectors using OpenAI text-embedding-ada-002
7. **Indexing** (Gate-2): Build FAISS/Weaviate/Pinecone indexes
8. **MCP Tools** (Gate-3): Validate tool health and contracts
9. **Routing** (Gate-4): Test query routing heuristics
10. **Graph Orchestration** (Gate-5): Validate LangGraph workflows
11. **A2A Compliance** (Gate-6): Agent-to-agent handoffs and compliance checks
12. **Retrieval Evaluation** (Gate-7): End-to-end retrieval quality assessment
13. **Generation Evaluation** (Gate-8): End-to-end email generation quality and compliance validation

### Multi-Agent Architecture (A2A)

The system uses LangGraph to orchestrate agent-to-agent interactions:

- **Planner**: Routing and policy selection using heuristics from `configs/router.heuristics.yaml`
- **Retriever**: Executes MCP `kb.search` tool across multiple vector backends
- **Consolidator**: LLM-enhanced persona-aware insight card generation (uses ChatOpenAI with model="gpt-5-nano")
- **Stylist**: LLM-based email generation with compliance checking (uses ChatOpenAI with model="gpt-5-nano")

Agent nodes and timeouts are defined in `configs/langgraph.nodes.yaml`.

**LLM Configuration**: The system uses `gpt-5-nano` as the LLM model for both Consolidator and Stylist agents. This model name is intentionally set to `gpt-5-nano` in `scripts/run_graph.py` (line 176).

### LangGraph Orchestration

The system provides **two implementations** for agent orchestration:

1. **Original**: `scripts/run_graph.py` - Custom sequential orchestration
2. **LangGraph**: `scripts/run_graph_langgraph.py` - Full LangGraph StateGraph implementation

Both implementations maintain 100% backward compatibility with identical output formats and quality gate thresholds.

#### LangGraph Architecture

**Implementation Files**:
- `scripts/run_graph_langgraph.py` - Main graph builder and execution
- `scripts/langgraph_nodes.py` - 8 agent node implementations
- `scripts/langgraph_state.py` - Typed state schema (AgentState TypedDict)
- `scripts/visualize_graph.py` - Graph visualization generator

**StateGraph Structure**: Type-safe state management with field-level accumulators

```python
class AgentState(TypedDict):
    # Input fields
    company: str
    persona: str
    session_id: str

    # Accumulated fields (using Annotated[..., add])
    retrieved_chunks: Annotated[List[Dict], add]
    compliance_flags: Annotated[List[str], add]
    errors: Annotated[List[str], add]

    # Replaced fields
    queries: List[str]
    insight_cards: List[Dict]
    email_draft: Dict
    a2a_rounds: int
```

**Graph Topology**: 8 nodes with conditional A2A routing

```
Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A
                                                                        ↓
                                                            (no critical flags)
                                                                        ↓
                                                                   Assembler → END
                                                                        ↑
                                                              (critical flags & rounds<2)
                                                                        ↓
                                                      ← ← ← ← ← ← Stylist (Round 2)
```

**Conditional Edges**:
- **A2A → Stylist** (revise): If critical compliance flags exist AND rounds < 2
- **A2A → Assembler** (assemble): Otherwise

**Node Functions** (`scripts/langgraph_nodes.py`):
1. **intake_node**: Input validation (company, persona)
2. **planner_node**: Generate 5 persona-specific queries from eval seed
3. **retriever_node**: Execute MCP kb.search across FAISS/Weaviate/Pinecone backends
4. **synthesizer_node**: Convert chunks to candidate insight objects
5. **consolidator_node**: LLM-enhanced persona-aware insight refinement (ChatOpenAI, gpt-5-nano)
6. **stylist_node**: LLM-based email generation (ChatOpenAI, gpt-5-nano)
7. **a2a_node**: Compliance negotiation with MCP safety.check (up to 2 rounds with revision logic)
8. **assembler_node**: Attach proof points and finalize output

**Graph Visualization**:
```bash
conda run -n age python scripts/visualize_graph.py
# Generates: reports/graphs/agent_workflow.{mmd,png}
```

**Execution**:
```bash
# LangGraph implementation
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session

# Original implementation
conda run -n age python scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Output Artifacts** (identical format for both implementations):
- `outputs/<session-id>/insights.json` - 5 enhanced insight cards
- `outputs/<session-id>/email.json` - Generated email with proof points
- `outputs/<session-id>/compliance_report.json` - A2A negotiation results
- `outputs/<session-id>/timing.json` - Per-node execution times
- `outputs/<session-id>/router_trace.jsonl` - Query routing decisions
- `state/session-<session-id>.json` - Full state snapshot

**Quality Gates**: Both implementations pass Gates 5, 6, and 8 with identical thresholds.

### Text Embedding System (OpenAI ada-002)

**Location**: `scripts/embedding_utils.py`

**Process**:
1. Uses OpenAI `text-embedding-ada-002` API
2. Implements caching (SHA-256 keys) in `data/cache/embeddings/` to minimize API calls
3. Retry logic with exponential backoff (3 attempts) for API failures
4. Returns 1536-dimensional vectors (normalized by OpenAI)

**Critical**:
- Both documents and queries MUST use the same `embed_text(text, dim)` function to ensure they exist in the same vector space. Mismatched embeddings will result in recall=0.
- Requires `OPENAI_API_KEY` in `.env` file (create manually: `echo "OPENAI_API_KEY=your-key" > .env`)
- Batch processing available via `embed_batch()` to reduce API costs

**Configuration**: `configs/vector.indexing.yaml` specifies:
- `embedding.model: openai-ada-002`
- `embedding.dim: 1536`
- `embedding.batch_size: 20` (reduced to avoid 8192 token limit)

### Multi-Index Routing

**Router logic**: `scripts/router_core.py`
**Heuristics config**: `configs/router.heuristics.yaml`

The router selects between FAISS, Weaviate, and Pinecone backends based on:
- Keyword matching rules (first match wins)
- Persona bias (optional per-persona preferences)
- Weighted scoring (similarity 0.5, recency 0.3, diversity 0.2)
- Fallback order when no rule matches

### MCP (Model Context Protocol) Tools

**Stub service**: `scripts/qa_step03_mcp.py`
**Configuration**: `configs/mcp.tools.yaml`

Local stub services run on localhost ports 7801-7805:
- `kb.search` (7801): Knowledge base search across vector backends
- `web.fetch` (7802): Web content fetching
- `link.resolve` (7803): URL resolution
- `crm.lookup` (7804): CRM data lookup
- `safety.check` (7805): Compliance and safety validation

These are designed to run offline and can be swapped for production services by updating the config.

## Environment Setup

### Conda Path Configuration

**Conda executable**: `/Users/liyunxiao/anaconda3/bin/conda`

All `conda` commands in this document should use the full path above.

### Two-Environment Architecture

This project uses **two separate conda environments** to avoid OpenMP runtime conflicts:

#### `age` (Python 3.13) — Primary Environment
- **Use for**: Most tasks including Gate-1 (embeddings), Gate-7 (retrieval eval), routing, MCP stubs
- **Critical**: DO NOT install pip `faiss-cpu` in this environment (causes OMP Error #15)
- **Key packages**: aiohttp, pyyaml, pyarrow>=21, numpy>=2.3, openblas, llvm-openmp

#### `ageFaiss` (Python 3.12) — FAISS-Only Environment
- **Use for**: Gate-2 (FAISS index builds) and FAISS health checks only
- **Key packages**: faiss-cpu=1.9.*, numpy=1.26.*, scipy, pyarrow=21.*

### Environment Creation

```bash
conda env create -f envs/age.yaml
conda env create -f envs/ageFaiss.yaml
```

See `docs/envs.md` for detailed environment documentation.

## Key Commands

### Quality Gates (Main Pipeline)

Run gates in sequence to validate the pipeline:

```bash
# Gate-0: Baseline checks
conda run -n age python scripts/qa_step00_baseline.py

# Gate-1: Generate embeddings (text vectors)
conda run -n age python scripts/qa_step01_embeddings.py

# Gate-2: Build and validate FAISS index
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate-3: Validate MCP tool health and contracts
conda run -n age python scripts/qa_step03_mcp.py

# Gate-4: Test router heuristics
conda run -n age python scripts/qa_step04_router.py

# Gate-5: Validate LangGraph orchestration
conda run -n age python scripts/qa_step05_graph.py

# Gate-6: Agent-to-agent compliance checks
conda run -n age python scripts/qa_step06_a2a.py --session-id <SESSION>

# Gate-7: Retrieval evaluation (recall@10, nDCG@5, latency)
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py

# Gate-8: Generation & compliance evaluation (10 runs across ≥3 personas)
conda run -n age python scripts/qa_step08_generation_eval.py
```

### Data Collection

```bash
# Fetch from various sources
python3 scripts/fetch_sec_filings.py
python3 scripts/fetch_product_docs.py
python3 scripts/fetch_dev_docs.py
python3 scripts/fetch_help_docs.py
python3 scripts/fetch_wikipedia.py
python3 scripts/fetch_newsroom_rss.py
python3 scripts/fetch_investor_news.py

# Manual ingestion (HTML files in data/manual_inbox/)
python3 scripts/ingest_manual_html.py
python3 scripts/ingest_manual_ir_html.py
```

### Data Processing

```bash
# Parse SEC filing structures
python3 scripts/parse_sec_structures.py

# Extract metadata
python3 scripts/extract_metadata.py

# Chunk documents
python3 scripts/chunk_documents.py

# Deduplicate chunks
python3 scripts/dedupe_chunks.py

# Build evaluation seed
python3 scripts/build_eval_seed.py
```

### Verification & Utilities

```bash
# Verify individual pipeline stages
python3 scripts/qa_verify_collection.py
python3 scripts/qa_verify_normalization.py
python3 scripts/qa_verify_metadata.py
python3 scripts/qa_verify_chunking.py
python3 scripts/qa_verify_dedupe.py
python3 scripts/qa_verify_link_health.py
python3 scripts/qa_verify_eval_seed.py
python3 scripts/qa_verify_day1_signoff.py

# Build inventory CSV
python3 scripts/build_inventory_csv.py

# Check link health
python3 scripts/link_health_check.py

# Day-1 milestone verification
python3 scripts/verify_day1_milestones.py
```

### MCP & Agent Services

```bash
# Start local MCP stub services
conda run -n age python scripts/qa_step03_mcp.py

# Run tool safety check server
python3 scripts/tool_safety_check_server.py

# Execute LangGraph workflow
python3 scripts/run_graph.py
```

## Directory Structure

```
ag3/
├── configs/                      # YAML configuration files
│   ├── vector.indexing.yaml      # Embedding and index settings
│   ├── router.heuristics.yaml    # Query routing logic
│   ├── mcp.tools.yaml            # MCP service endpoints
│   ├── langgraph.nodes.yaml      # Agent graph orchestration
│   ├── metadata.dictionary.yaml  # Metadata extraction rules
│   ├── normalization.rules.yaml  # Text normalization rules
│   ├── eval.prompts.yaml         # Evaluation prompt templates
│   ├── agents.schema.yaml        # Agent schema definitions
│   ├── compliance.template.yaml  # Compliance check templates
│   └── chunking.config.json      # Document chunking parameters
│
├── scripts/                      # Processing and QA scripts (41 total)
│   ├── embedding_utils.py        # OpenAI ada-002 embedding with caching and retry logic
│   ├── router_core.py            # Query routing and reranking logic
│   ├── qa_step*.py               # Quality gate scripts (Gates 0-7)
│   ├── fetch_*.py                # Data collection scripts
│   ├── ingest_*.py               # Manual data ingestion
│   ├── qa_verify_*.py            # Individual stage verification
│   └── [other utilities]
│
├── data/                         # Data artifacts (organized by stage)
│   ├── raw/                      # Original fetched documents
│   │   ├── sec/                  # SEC filings
│   │   ├── product/              # Product documentation
│   │   ├── dev_docs/             # Developer documentation
│   │   ├── help_docs/            # Help articles
│   │   ├── newsroom/             # Press releases
│   │   ├── investor_news/        # Investor news
│   │   └── wikipedia/            # Wikipedia articles
│   ├── manual_inbox/             # Manual HTML ingestion staging
│   ├── interim/                  # Intermediate processing artifacts
│   │   ├── normalized/           # Normalized documents
│   │   ├── chunks/               # Chunked documents
│   │   ├── dedup/                # Deduplicated chunks
│   │   └── eval/                 # Evaluation datasets
│   ├── vector/                   # Vector embeddings and indexes
│   │   ├── embeddings/           # Generated embeddings (Parquet)
│   │   ├── faiss/                # FAISS indexes
│   │   ├── weaviate/             # Weaviate manifests
│   │   └── pinecone/             # Pinecone manifests
│   ├── cache/                    # Caching layer
│   │   └── embeddings/           # OpenAI API response cache (SHA-256 keys)
│   ├── final/                    # Production-ready artifacts
│   │   ├── reports/              # Index health reports
│   │   ├── inventory/            # Document inventory
│   │   ├── dictionaries/         # Metadata dictionaries
│   │   └── rules/                # Normalization rules
│   └── backup/                   # Backups and historical data
│
├── reports/                      # Quality assurance reports
│   ├── qa/                       # Gate reports (JSON + Markdown)
│   │   ├── step0*.{json,md}      # Gate outputs (dual format)
│   │   └── [other gates]
│   ├── eval/                     # Evaluation artifacts
│   │   └── retrieval_failures.jsonl  # Failed retrieval traces
│   └── router/                   # Router trace logs
│       └── step07_retrieval_trace.jsonl
│
├── logs/                         # Runtime logs (organized by component)
│
├── outputs/                      # Generated outputs (emails, etc.)
│
├── state/                        # Persistent state
│
├── envs/                         # Conda environment definitions
│   ├── age.yaml                  # Primary environment (Python 3.13)
│   └── ageFaiss.yaml             # FAISS environment (Python 3.12)
│
├── docs/                         # Documentation
│   ├── envs.md                   # Environment setup details
│   └── README.md
│
├── icl/                          # In-context learning templates
│   ├── persona/                  # Persona definitions
│   └── templates/                # Prompt templates
│
├── worktrees/                    # Git worktrees (IGNORED by .gitignore)
│
├── README.md                     # Main project documentation
├── README_DAY1.md                # Day-1 milestone documentation
├── AGENTS.md                     # Agent/automation guidelines
└── CLAUDE.md                     # This file (Claude Code guidance)
```

## Configuration Files

### `configs/vector.indexing.yaml`
Defines embedding and index settings:
- Embedding model (`openai-ada-002`), dimensions (1536), batch size (20)
- FAISS HNSW parameters (M=32, efConstruction=200, efSearch=128)
- Pinecone and Weaviate manifests (simulated, no network)

### `configs/router.heuristics.yaml`
Query routing heuristics:
- Weighting: similarity (0.5), recency (0.3), diversity (0.2)
- Keyword-based routing rules (press/financial → Pinecone, developer → Weaviate, definitions → FAISS)
- Persona bias (optional per-role preferences)
- Fallback order: [faiss, weaviate, pinecone]

### `configs/mcp.tools.yaml`
MCP service endpoints:
- Localhost ports 7801-7805
- Timeout budgets (2000ms default)

### `configs/langgraph.nodes.yaml`
Agent graph configuration:
- Node list: Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler
- Per-node timeout budgets (2s to 10s)

### `configs/metadata.dictionary.yaml`
Metadata extraction rules for structured fields

### `configs/normalization.rules.yaml`
Text cleaning and normalization patterns

### `configs/eval.prompts.yaml`
Evaluation prompt templates for quality assessment

### `configs/agents.schema.yaml`
Agent schema definitions and validation rules

### `configs/compliance.template.yaml`
Compliance check templates for generated content

## Important Environment Variables

### Gate-7 (Retrieval Evaluation)
- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating (recommended for initial runs)
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets (e.g., 3.0 for 3x tolerance)
- `AG7_ANALYZE_TOPK=<int>`: Retrieval cut-off for evaluation (default: 10)
- `AG7_TOPK_SLICES="1,3,5,10"`: Additional @k slices for recall curves
- `AG7_NEAR_SEQ_TOL=<int>`: Near-miss tolerance in chunks within same doc (default: 1)
- `AG7_TRACE=1`: Enable per-query trace JSONL (default: enabled)
- `AG7_TRACE_TOPK=<int>`: Number of top-K items to capture in trace (default: 10)
- `AG7_TRACE_SUCCESSES=1`: Include successes in trace (default: enabled; set 0 for misses only)
- `AG7_DEBUG=1`: Umbrella debug switch enabling tracing (default: enabled)

### Gate-1 (Embedding Generation)
- `AG1_AUTO_CONFIRM=1`: Skip cost confirmation prompt (auto-proceed with embedding generation)
- `OPENAI_API_KEY`: OpenAI API key (required, set in `.env` file)

### General
- `AR_USER_AGENT`: Custom user agent for web requests
- `AR_GLOBAL_RPS`: Rate limiting for HTTP requests (requests per second)

## Troubleshooting

### OpenMP Runtime Conflicts
**Symptom**: `OMP Error #15` or segfault during FAISS operations

**Cause**: Mixing pip `faiss-cpu` (which bundles libomp) with conda OpenBLAS+OpenMP in the same environment

**Fix**:
1. Always run Gate-2 (FAISS index builds) in the `ageFaiss` environment
2. NEVER install pip `faiss-cpu` in the `age` environment
3. If already installed, recreate the environment from scratch: `conda env remove -n age && conda env create -f envs/age.yaml`

### Recall=0 in Gate-7
**Symptom**: Retrieval evaluation shows 0% recall despite having indexed documents

**Cause**: Mismatched embeddings between documents and queries (e.g., using different embedding functions or random vectors)

**Fix**: Ensure both document indexing (Gate-1) and query processing use `embed_text()` from `scripts/embedding_utils.py` with the same dimensionality (1536)

### OpenAI API Errors
**Symptom**: Embedding generation fails with API errors or rate limits

**Cause**: Missing or invalid `OPENAI_API_KEY`, network issues, or rate limiting

**Fix**:
1. Ensure `.env` file exists with valid `OPENAI_API_KEY` (create with: `echo "OPENAI_API_KEY=your-key" > .env`)
2. Check network connection to OpenAI API
3. For rate limits, the retry logic will handle transient errors (3 attempts with exponential backoff)
4. Use cached embeddings when possible (cache stored in `data/cache/embeddings/`)

### Port Conflicts
**Symptom**: "Port busy" errors when starting MCP services on ports 7801-7805

**Fix**:
1. Check for existing stub services: `lsof -i :7801-7805`
2. Stop existing services or kill processes using these ports
3. Alternatively, update `configs/mcp.tools.yaml` to use different ports

### PDF Glyph Noise
**Symptom**: Chunks contain CID-like tokens or rendering artifacts from PDF extraction

**Impact**: May affect retrieval quality and embedding generation

**Mitigation**: Consider implementing a PDF-specific preprocessing step or enhancing the reranker

### JSONL Parse Errors
**Symptom**: Errors reading intermediate JSONL files

**Fix**: Ensure UTF-8 encoding and valid JSON on each line; check logs for corrupted entries

### Missing Dependencies
**Symptom**: Import errors for packages like `yaml`, `pyarrow`, `aiohttp`

**Fix**: Recreate the conda environment from the appropriate YAML file

## Quality Gates & Outputs

Each quality gate produces dual-format reports (JSON for machines, Markdown for humans):

| Gate | Script | JSON Report | Markdown Report | Additional Artifacts |
|------|--------|-------------|-----------------|---------------------|
| Gate-0 | `qa_step00_baseline.py` | `reports/qa/step00_baseline.json` | `reports/qa/step00_baseline.md` | N/A |
| Gate-1 | `qa_step01_embeddings.py` | `reports/qa/step01_embeddings.json` | `reports/qa/step01_embeddings.md` | `data/vector/embeddings/embeddings.parquet` |
| Gate-2 | `qa_step02_indexes.py` | `reports/qa/step02_indexes.json` | `reports/qa/step02_indexes.md` | `data/final/reports/index_health.json`, FAISS indexes |
| Gate-3 | `qa_step03_mcp.py` | `reports/qa/step03_mcp.json` | `reports/qa/step03_mcp.md` | N/A |
| Gate-4 | `qa_step04_router.py` | `reports/qa/step04_router.json` | `reports/qa/step04_router.md` | N/A |
| Gate-5 | `qa_step05_graph.py` | `reports/qa/step05_graph.json` | `reports/qa/step05_graph.md` | N/A |
| Gate-6 | `qa_step06_a2a.py` | `reports/qa/step06_a2a.json` | `reports/qa/step06_a2a.md` | N/A |
| Gate-7 | `qa_step07_retrieval_eval.py` | `reports/qa/step07_retrieval_eval.json` | `reports/qa/step07_retrieval_eval.md` | `reports/eval/retrieval_failures.jsonl`, `reports/router/step07_retrieval_trace.jsonl` |
| Gate-8 | `qa_step08_generation_eval.py` | `reports/qa/step08_generation_eval.json` | `reports/qa/step08_generation_eval.md` | `reports/eval/generation_metrics.json`, `reports/eval/compliance_metrics.json` |

**Report Locations**:
- Machine-readable: `reports/qa/step*.json`
- Human-readable: `reports/qa/step*.md`
- Evaluation traces: `reports/eval/` and `reports/router/`

## Evaluation Metrics (Gate-7)

The retrieval evaluation (Gate-7) computes:

- **recall@10**: Proportion of relevant documents retrieved in top 10 results
- **nDCG@5**: Normalized Discounted Cumulative Gain at rank 5 (ranking quality)
- **coverage**: Optional check that all indexed documents are reachable
- **freshness**: Time-based relevance of retrieved documents
- **latency**: Response time budgets (median, p95, p99) per backend

Failures are logged to `reports/eval/retrieval_failures.jsonl` with full query context and retrieved results for debugging.

## Evaluation Metrics (Gate-8)

The generation evaluation (Gate-8) validates end-to-end email generation quality:

- **structural_pass_rate**: All runs must produce exactly 5 insights, ≥4 distinct sources, ≥2 recent items (within 12 months), valid email schema, and resolvable proof points
- **critical_flags_total**: Must be 0 (no critical compliance violations)
- **length_readability_pass_runs**: ≥9 out of 10 runs must pass (≤160 words, grade ≤10.0)
- **persona_keyword_hits_avg**: Average ≥2.0 persona-specific keywords per email

**Thresholds**:
- G8-01: structural_pass_rate == 1.0
- G8-02: critical_flags_total == 0
- G8-03: length_readability_pass_runs >= 9
- G8-04: persona_keyword_hits_avg >= 2.0

**Status**: GREEN (all pass), AMBER (only G8-03 or G8-04 fails), RED (G8-01 or G8-02 fails, or multiple failures)

Generated outputs are saved per session in `outputs/<session_id>/` with full traceability to source insights.

## Scale & Performance

- **Current scale**: Designed and verified on 100+ documents (~1.6k chunks)
- **FAISS latency**: Median sub-second local retrieval
- **Weaviate/Pinecone**: Simulated manifests (no network required for development)
- **Horizontal scaling**: Stateless stages and services; indexes can be sharded externally

## Code Conventions

When working with this codebase:

1. **Embedding consistency**: Always use `embed_text()` from `scripts/embedding_utils.py` for both documents and queries (requires `OPENAI_API_KEY` in `.env`)
2. **Environment discipline**: Use `age` for most tasks, `ageFaiss` only for Gate-2 FAISS builds
3. **Report preservation**: Maintain dual JSON+Markdown report format; don't change schemas without updating consumers
4. **Config-driven behavior**: Prefer adding environment variables or config options over hardcoding
5. **Minimal dependencies**: Avoid adding heavyweight packages; keep the system lightweight and portable
6. **No auto-install**: Don't add automatic package installation to scripts (breaks environment reproducibility)
7. **Traceability**: Preserve evidence links and provenance chains in reports
8. **Stateless design**: Keep stages independent and replayable

## Git Worktrees

The `worktrees/` directory contains git worktrees for parallel development branches (e.g., `agent-faiss`, `agent-pinecone`, `agent-weaviate`, `agent-test`). This directory is excluded via `.gitignore` to prevent conflicts.

## Quick Start

1. **Create environments**:
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
   /Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
   ```

2. **Set up environment variables**:
   ```bash
   # Create .env file with your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

3. **Build embeddings and indexes**:
   ```bash
   # Gate-1: Generate embeddings (requires OPENAI_API_KEY in .env)
   /Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py

   # Gate-2: Build FAISS index
   /Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```

4. **Validate MCP tools**:
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py
   ```

5. **Run retrieval evaluation**:
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py
   ```

6. **Inspect results**:
   ```bash
   cat reports/qa/step07_retrieval_eval.md
   ```

## Additional Resources

- **Main documentation**: `README.md` (architecture overview, system design)
- **Agent guidelines**: `AGENTS.md` (automation-friendly runbook)
- **Day-1 milestones**: `README_DAY1.md` (initial milestone documentation)
- **Environment details**: `docs/envs.md` (conda environment deep dive)

## Notes for Claude Code

- **Conda path**: Always use `/Users/liyunxiao/anaconda3/bin/conda` as the conda executable
- This is a research/evaluation system designed for traceability and reproducibility
- All stages emit dual-format reports (JSON + Markdown) for both machines and humans
- The OpenAI ada-002 embedding model requires an API key (set in `.env`) and implements caching to minimize costs
- Dependencies: `openai`, `python-dotenv`, `tenacity` (for retry logic)
- MCP stubs run locally on localhost (no network required for development)
- For production use, swap MCP stub endpoints in `configs/mcp.tools.yaml` to point to real services
- The two-environment architecture is critical: mixing FAISS pip packages into the main environment causes OpenMP crashes
