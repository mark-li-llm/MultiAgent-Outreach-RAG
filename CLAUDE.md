# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Multi-agent RAG system for Sales/IR/PR outreach with audit-ready email generation and step-level traceability. Implements a gated data pipeline with quality checks at each stage, emitting dual-format reports (JSON + Markdown).

**For detailed architecture**: See [docs/architecture.md](docs/architecture.md)

## Quick Reference

### Pipeline Stages (13 Gates)

1-5: Collection → Normalization → Metadata → Chunking → Deduplication
6-8: Embedding (Gate-1) → Indexing (Gate-2) → MCP Tools (Gate-3)
9-13: Routing (Gate-4) → Graph (Gate-5) → A2A (Gate-6) → Retrieval Eval (Gate-7) → Generation Eval (Gate-8)

### Key Concepts

- **Embeddings**: OpenAI ada-002 (1536-dim), cached in `data/cache/embeddings/`
- **Multi-index**: FAISS (general), Weaviate (dev docs), Pinecone (press/financial)
- **Routing**: Keyword rules → Persona bias → Weighted scoring → Fallback
- **MCP Tools**: Local stubs on ports 7801-7805 (kb.search, web.fetch, link.resolve, crm.lookup, safety.check)
- **LangGraph**: 8 nodes (Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler)

**For detailed design**: See [docs/architecture.md](docs/architecture.md)

## Environment Setup

### Conda Path

**Conda executable**: `/Users/liyunxiao/anaconda3/bin/conda`

### Two Environments (Critical!)

#### `age` (Python 3.13) — Primary
- **Use for**: Gate-1, Gate-3-8, routing, MCP stubs, graph execution
- **DO NOT** install pip `faiss-cpu` (causes OMP Error #15)

#### `ageFaiss` (Python 3.12) — FAISS Only
- **Use for**: Gate-2 (FAISS index builds) ONLY

**Create environments**:
```bash
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
```

**For environment details**: See [docs/envs.md](docs/envs.md)

## Common Commands

### Quality Gates (Run in Sequence)

```bash
# Gate-0: Baseline checks
conda run -n age python scripts/qa_step00_baseline.py

# Gate-1: Generate embeddings (requires OPENAI_API_KEY in .env)
conda run -n age python scripts/qa_step01_embeddings.py

# Gate-2: Build FAISS index (USE ageFaiss environment!)
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate-3: Validate MCP tools
conda run -n age python scripts/qa_step03_mcp.py

# Gate-7: Retrieval evaluation (with relaxed budgets)
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py

# Gate-8: Generation evaluation
conda run -n age python scripts/qa_step08_generation_eval.py
```

**For complete command reference**: See [docs/commands.md](docs/commands.md)

### Execute Graph Workflow

```bash
# LangGraph implementation (recommended)
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session

# Original implementation (for comparison)
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

Both produce identical outputs in `outputs/<session-id>/`.

## Critical Gotchas

### 1. OpenMP Conflicts

**Problem**: `OMP Error #15` or segfault during FAISS operations

**Solution**:
- Always use `ageFaiss` environment for Gate-2
- NEVER install pip `faiss-cpu` in `age` environment
- If already installed: `conda env remove -n age && conda env create -f envs/age.yaml`

### 2. Embedding Consistency

**Problem**: Retrieval recall = 0% despite having indexed documents

**Solution**:
- Both documents AND queries MUST use `embed_text()` from `scripts/embedding_utils.py`
- Never use different embedding functions or random vectors
- Dimension must be 1536 for both

### 3. OpenAI API Key

**Problem**: Embedding generation fails

**Solution**:
- Create `.env` file: `echo "OPENAI_API_KEY=your-key" > .env`
- Verify key works before running Gate-1

**For more troubleshooting**: See [docs/troubleshooting.md](docs/troubleshooting.md)

## File Locations

```
configs/          # YAML/JSON configuration (10 files)
scripts/          # Processing and QA scripts (41 total)
data/
  ├── raw/        # Original documents (7 source types)
  ├── interim/    # Processing artifacts (normalized, chunks, dedup)
  ├── vector/     # Embeddings and indexes (FAISS/Weaviate/Pinecone)
  ├── cache/      # Embedding cache (SHA-256 keys)
  └── final/      # Production-ready artifacts
reports/
  ├── qa/         # Gate reports (JSON + Markdown)
  ├── eval/       # Evaluation traces
  └── router/     # Router decision logs
outputs/          # Generated emails per session
state/            # Persistent state snapshots
docs/             # Documentation (this file + 5 detailed docs)
```

## Configuration Files

All configs in `configs/`:

| File | Purpose |
|------|---------|
| `vector.indexing.yaml` | Embedding model, FAISS params |
| `router.heuristics.yaml` | Query routing rules |
| `mcp.tools.yaml` | MCP service endpoints |
| `langgraph.nodes.yaml` | Agent graph topology |
| `metadata.dictionary.yaml` | Metadata extraction |
| `normalization.rules.yaml` | Text cleaning |
| `chunking.config.json` | Document chunking |

**For config details**: See [docs/configuration.md](docs/configuration.md)

## Quality Metrics

### Gate-7 (Retrieval)

- **recall@10** ≥ 0.80 (proportion of relevant docs in top 10)
- **nDCG@5** ≥ 0.70 (ranking quality)
- **median_latency** ≤ 1000ms × multiplier

### Gate-8 (Generation)

- **structural_pass_rate** == 1.0 (all runs structurally valid)
- **critical_flags_total** == 0 (no compliance violations)
- **length_readability_pass_runs** ≥ 9 (out of 10)
- **persona_keyword_hits_avg** ≥ 2.0

**For metric definitions**: See [docs/evaluation.md](docs/evaluation.md)

## Code Conventions

1. **Embedding consistency**: Always use `embed_text()` from `scripts/embedding_utils.py`
2. **Environment discipline**: Use `age` for most tasks, `ageFaiss` ONLY for Gate-2
3. **Report preservation**: Maintain dual JSON+Markdown format
4. **Config-driven behavior**: Prefer env vars/config options over hardcoding
5. **Minimal dependencies**: Keep system lightweight and portable
6. **No auto-install**: Don't add automatic package installation
7. **Traceability**: Preserve evidence links and provenance chains
8. **Stateless design**: Keep stages independent and replayable

## Quick Start

```bash
# 1. Create environments
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml

# 2. Set up API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Run quality gates
conda run -n age python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
conda run -n age python scripts/qa_step03_mcp.py
conda run -n age AG7_IGNORE_COVERAGE=1 python scripts/qa_step07_retrieval_eval.py

# 4. Execute graph
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id test-session

# 5. Inspect results
cat reports/qa/step07_retrieval_eval.md
cat outputs/test-session/email.json | jq .
```

## Environment Variables

### Gate-1 (Embeddings)
- `AG1_AUTO_CONFIRM=1`: Skip cost confirmation
- `OPENAI_API_KEY`: OpenAI API key (required)

### Gate-7 (Retrieval)
- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets (e.g., 3.0)
- `AG7_DEBUG=1`: Enable debug tracing

**For complete env var list**: See [docs/commands.md](docs/commands.md#environment-variables-reference)

## Documentation Index

- **[docs/architecture.md](docs/architecture.md)** — Detailed system design, pipeline stages, LangGraph orchestration, embedding system, multi-index routing, MCP tools
- **[docs/commands.md](docs/commands.md)** — Complete command reference for all 41 scripts, quality gates, data collection, processing, verification
- **[docs/configuration.md](docs/configuration.md)** — Configuration file deep dive, all 10 config files with examples and tuning guidelines
- **[docs/troubleshooting.md](docs/troubleshooting.md)** — Debug playbook, common issues (OpenMP, recall=0, API errors, port conflicts), debugging tools
- **[docs/evaluation.md](docs/evaluation.md)** — Quality gates and metrics, Gate-7 (retrieval), Gate-8 (generation), thresholds, status colors
- **[docs/envs.md](docs/envs.md)** — Environment setup, package details, conflict resolution

## Additional Resources

- **README.md** — Main project documentation (architecture overview, system design)
- **AGENTS.md** — Agent/automation guidelines (automation-friendly runbook)
- **README_DAY1.md** — Day-1 milestone documentation (initial milestone verification)

## Notes for Claude Code

- **Conda path**: Always use `/Users/liyunxiao/anaconda3/bin/conda`
- This is a research/evaluation system designed for traceability and reproducibility
- All stages emit dual-format reports (JSON + Markdown)
- OpenAI ada-002 embedding requires API key in `.env` with caching to minimize costs
- MCP stubs run locally (no network required for development)
- Two-environment architecture is critical: mixing FAISS pip packages causes OpenMP crashes
- When in doubt about details, refer to the detailed docs in `docs/`
