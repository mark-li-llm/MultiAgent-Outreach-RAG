# System Architecture

This document provides detailed technical specifications for the multi-agent RAG system architecture.

## Overview

The system implements a multi-stage gated data pipeline for Sales/IR/PR outreach that automates trusted-source research and audit-ready email generation with step-level traceability. All stages emit dual-format reports (JSON + Markdown) for both machine and human consumption.

## Data Pipeline Stages

The system follows a multi-stage pipeline with quality gates at each checkpoint:

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

## Multi-Agent Architecture (A2A)

The system uses LangGraph to orchestrate agent-to-agent interactions across four primary agents:

### Agent Roles

- **Planner**: Routing and policy selection using heuristics from `configs/router.heuristics.yaml`
- **Retriever**: Executes MCP `kb.search` tool across multiple vector backends (FAISS, Weaviate, Pinecone)
- **Consolidator**: LLM-enhanced persona-aware insight card generation
  - Uses ChatOpenAI with model="gpt-5-nano"
  - Generates 5 insight cards per session
- **Stylist**: LLM-based email generation with compliance checking
  - Uses ChatOpenAI with model="gpt-5-nano"
  - Applies persona-specific tone and formatting

Agent nodes and timeouts are defined in `configs/langgraph.nodes.yaml`.

**LLM Configuration**: The system uses `gpt-5-nano` as the LLM model for both Consolidator and Stylist agents. This model name is intentionally set to `gpt-5-nano` in `scripts/run_graph.py` (line 176).

## LangGraph Orchestration

The system provides **two implementations** for agent orchestration:

1. **Original**: `scripts/run_graph.py` - Custom sequential orchestration
2. **LangGraph**: `scripts/run_graph_langgraph.py` - Full LangGraph StateGraph implementation

Both implementations maintain 100% backward compatibility with identical output formats and quality gate thresholds.

### LangGraph Implementation Details

**Implementation Files**:
- `scripts/run_graph_langgraph.py` - Main graph builder and execution
- `scripts/langgraph_nodes.py` - 8 agent node implementations
- `scripts/langgraph_state.py` - Typed state schema (AgentState TypedDict)
- `scripts/visualize_graph.py` - Graph visualization generator

### StateGraph Structure

Type-safe state management with field-level accumulators:

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

**Key Design Choices**:
- Accumulated fields use `Annotated[..., add]` to append values across node executions
- Replaced fields overwrite previous values
- TypedDict provides static type checking for state schema

### Graph Topology

8 nodes with conditional A2A routing:

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

This allows up to 2 rounds of compliance negotiation before final assembly.

### Node Functions

Detailed node implementations in `scripts/langgraph_nodes.py`:

1. **intake_node**: Input validation (company, persona)
   - Validates required fields exist
   - Initializes session state

2. **planner_node**: Generate 5 persona-specific queries from eval seed
   - Loads persona definition from `icl/persona/`
   - Generates queries tailored to persona role

3. **retriever_node**: Execute MCP kb.search across FAISS/Weaviate/Pinecone backends
   - Uses router heuristics to select backend
   - Aggregates results from multiple indexes

4. **synthesizer_node**: Convert chunks to candidate insight objects
   - Extracts metadata and provenance
   - Creates structured insight candidates

5. **consolidator_node**: LLM-enhanced persona-aware insight refinement
   - Uses ChatOpenAI (gpt-5-nano)
   - Refines insights for persona relevance
   - Selects top 5 insights

6. **stylist_node**: LLM-based email generation
   - Uses ChatOpenAI (gpt-5-nano)
   - Applies persona tone and formatting
   - Generates email draft with compliance hooks

7. **a2a_node**: Compliance negotiation with MCP safety.check
   - Validates email against compliance rules
   - Flags critical violations
   - Up to 2 rounds with revision logic

8. **assembler_node**: Attach proof points and finalize output
   - Links insights to source documents
   - Generates final artifacts (JSON + Markdown)

### Graph Visualization

Generate visual representation of the graph:

```bash
conda run -n age python scripts/visualize_graph.py
# Generates: reports/graphs/agent_workflow.{mmd,png}
```

The visualization shows node connections, conditional edges, and data flow.

### Execution

Both implementations support identical command-line interfaces:

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

### Output Artifacts

Identical format for both implementations:

- `outputs/<session-id>/insights.json` - 5 enhanced insight cards
- `outputs/<session-id>/email.json` - Generated email with proof points
- `outputs/<session-id>/compliance_report.json` - A2A negotiation results
- `outputs/<session-id>/timing.json` - Per-node execution times
- `outputs/<session-id>/router_trace.jsonl` - Query routing decisions
- `state/session-<session-id>.json` - Full state snapshot

### Quality Gates

Both implementations pass Gates 5, 6, and 8 with identical thresholds:

- **Gate-5**: Graph orchestration validation (node execution, state transitions)
- **Gate-6**: A2A compliance checks (handoff protocols, negotiation rounds)
- **Gate-8**: Generation quality (structural integrity, compliance, readability)

## Text Embedding System (OpenAI ada-002)

**Location**: `scripts/embedding_utils.py`

The embedding system is critical to retrieval quality and must be used consistently across all stages.

### Process

1. Uses OpenAI `text-embedding-ada-002` API
2. Implements caching (SHA-256 keys) in `data/cache/embeddings/` to minimize API calls
3. Retry logic with exponential backoff (3 attempts) for API failures
4. Returns 1536-dimensional vectors (normalized by OpenAI)

### Critical Requirements

- **Consistency**: Both documents and queries MUST use the same `embed_text(text, dim)` function to ensure they exist in the same vector space. Mismatched embeddings will result in recall=0.
- **API Key**: Requires `OPENAI_API_KEY` in `.env` file (create manually: `echo "OPENAI_API_KEY=your-key" > .env`)
- **Batch Processing**: Use `embed_batch()` to reduce API costs when embedding multiple texts
- **Caching**: Embeddings are cached using SHA-256 hash keys to avoid redundant API calls

### Configuration

Settings defined in `configs/vector.indexing.yaml`:

```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20  # Reduced to avoid 8192 token limit
```

### Key Functions

```python
def embed_text(text: str, dim: int = 1536) -> np.ndarray:
    """Generate embedding for a single text string."""
    # Returns cached embedding if available
    # Otherwise calls OpenAI API with retry logic

def embed_batch(texts: List[str], dim: int = 1536) -> List[np.ndarray]:
    """Generate embeddings for multiple texts in a single API call."""
    # More efficient than multiple embed_text() calls
```

## Multi-Index Routing

**Router logic**: `scripts/router_core.py`
**Heuristics config**: `configs/router.heuristics.yaml`

The router dynamically selects between FAISS, Weaviate, and Pinecone backends based on query characteristics and persona context.

### Selection Criteria

The router uses a rule-based system with the following priority:

1. **Keyword matching rules** (first match wins)
   - Press releases, financial → Pinecone
   - Developer documentation → Weaviate
   - Definitions, general knowledge → FAISS

2. **Persona bias** (optional per-persona preferences)
   - Different personas may prefer different backends
   - Configurable in `configs/router.heuristics.yaml`

3. **Weighted scoring** when no rule matches
   - Similarity: 0.5
   - Recency: 0.3
   - Diversity: 0.2

4. **Fallback order**: [faiss, weaviate, pinecone]

### Routing Trace

All routing decisions are logged to JSONL for audit and debugging:
- `outputs/<session-id>/router_trace.jsonl` - Per-query routing decisions
- `reports/router/step07_retrieval_trace.jsonl` - Evaluation-time routing trace

## MCP (Model Context Protocol) Tools

**Stub service**: `scripts/qa_step03_mcp.py`
**Configuration**: `configs/mcp.tools.yaml`

MCP provides a standardized interface for tool invocation across the agent system.

### Local Stub Services

Development uses local stub services on localhost ports 7801-7805:

| Tool | Port | Purpose |
|------|------|---------|
| `kb.search` | 7801 | Knowledge base search across vector backends |
| `web.fetch` | 7802 | Web content fetching |
| `link.resolve` | 7803 | URL resolution |
| `crm.lookup` | 7804 | CRM data lookup |
| `safety.check` | 7805 | Compliance and safety validation |

### Design Principles

- **Offline-first**: Stubs run locally without network dependencies
- **Contract-first**: Tools define JSON schemas for inputs/outputs
- **Swappable**: Update `configs/mcp.tools.yaml` to point to production services
- **Timeout-aware**: Each tool has configurable timeout budgets (default: 2000ms)

### Health Checks

Gate-3 validates:
- All MCP services are reachable
- Services respond within timeout budgets
- Response schemas match contracts
- Error handling works correctly

## Scale & Performance

Current system performance characteristics:

- **Current scale**: Designed and verified on 100+ documents (~1.6k chunks)
- **FAISS latency**: Median sub-second local retrieval
- **Weaviate/Pinecone**: Simulated manifests (no network required for development)
- **Horizontal scaling**: Stateless stages and services; indexes can be sharded externally
- **Cache hit rate**: Embedding cache typically achieves >80% hit rate after initial run

### Performance Bottlenecks

1. **OpenAI API**: Rate limits can slow embedding generation (use caching and batching)
2. **FAISS index build**: Scales with O(n log n) for n chunks
3. **LLM generation**: Stylist and Consolidator nodes are slowest (5-10s per call)

## Code Conventions

When working with this architecture:

1. **Embedding consistency**: Always use `embed_text()` from `scripts/embedding_utils.py` for both documents and queries
2. **Environment discipline**: Use `age` for most tasks, `ageFaiss` only for Gate-2 FAISS builds
3. **Report preservation**: Maintain dual JSON+Markdown report format; don't change schemas without updating consumers
4. **Config-driven behavior**: Prefer adding environment variables or config options over hardcoding
5. **Minimal dependencies**: Avoid adding heavyweight packages; keep the system lightweight and portable
6. **No auto-install**: Don't add automatic package installation to scripts (breaks environment reproducibility)
7. **Traceability**: Preserve evidence links and provenance chains in reports
8. **Stateless design**: Keep stages independent and replayable

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
│   ├── qa_step*.py               # Quality gate scripts (Gates 0-8)
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
│   ├── architecture.md           # This file (detailed system design)
│   ├── commands.md               # Complete command reference
│   ├── configuration.md          # Config file deep dive
│   ├── troubleshooting.md        # Debug playbook
│   ├── evaluation.md             # Quality gates & metrics
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
└── CLAUDE.md                     # Claude Code guidance (core reference)
```

## Related Documentation

- **[commands.md](commands.md)** - Complete command reference for all scripts
- **[configuration.md](configuration.md)** - Detailed configuration file documentation
- **[evaluation.md](evaluation.md)** - Quality gates and evaluation metrics
- **[troubleshooting.md](troubleshooting.md)** - Debug guide and common issues
- **[envs.md](envs.md)** - Environment setup and package details