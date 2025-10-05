# Tools Documentation

User guides for development, debugging, and QA tools in this project.

## Available Tools

### Gate-8 Debug Tool (`qa_step08_debug.py`)

**Purpose**: Deep inspection of single LangGraph pipeline runs with comprehensive state tracking and validation.

**Quick Start**:
```bash
conda run -n age python scripts/qa_step08_debug.py \
  --company Salesforce \
  --persona vp_customer_experience
```

**Key Features**:
- 🔍 Full state inspection at each node
- 🤖 LLM interaction monitoring (prompts, responses, tokens)
- ✅ Per-node quality validation
- 📊 Dual-format reports (JSON + Markdown)
- 🔬 Detailed trace files (JSONL)

**Use Cases**:
- Debug pipeline failures or unexpected outputs
- Analyze LLM prompt/response quality
- Measure node-level performance (timing, token usage)
- Validate insight card enhancement
- Inspect email generation process

**Documentation**: See [`gate8-debug.md`](gate8-debug.md) for complete usage guide

**Development History**: See `docs/features/gate8-debug-tool-2025-10-05/`

---

### Production Gate-8 (`qa_step08_generation_eval.py`)

**Purpose**: End-to-end email generation quality validation across 10 runs and ≥3 personas.

**Quick Start**:
```bash
conda run -n age python scripts/qa_step08_generation_eval.py
```

**Metrics Evaluated**:
- G8-01: Structural pass rate (100% required)
- G8-02: Critical compliance flags (0 required)
- G8-03: Length/readability (≥9/10 runs pass)
- G8-04: Persona keyword hits (avg ≥2.0)

**Outputs**:
- `reports/qa/step08_generation_eval.json` - Machine report
- `reports/qa/step08_generation_eval.md` - Human report
- `reports/eval/generation_metrics.json` - Detailed metrics

---

## Tool Categories

### Quality Gates (Gate-0 to Gate-8)
- **Gate-0**: Baseline checks
- **Gate-1**: Embedding generation
- **Gate-2**: Index building and validation
- **Gate-3**: MCP tool health
- **Gate-4**: Router heuristics
- **Gate-5**: LangGraph orchestration
- **Gate-6**: A2A compliance
- **Gate-7**: Retrieval evaluation
- **Gate-8**: Generation evaluation (production)
- **Gate-8 Debug**: Single-run deep inspection (development)

### Data Processing
- **Collection**: `fetch_*.py`, `ingest_*.py`
- **Normalization**: `qa_verify_normalization.py`
- **Metadata**: `extract_metadata.py`
- **Chunking**: `chunk_documents.py`
- **Deduplication**: `dedupe_chunks.py`

### Utilities
- **Inventory**: `build_inventory_csv.py`
- **Link Health**: `link_health_check.py`
- **Verification**: `qa_verify_*.py` scripts

---

## Adding Tool Documentation

When adding a new tool:

1. **Create usage guide** in this directory:
   ```
   docs/tools/{tool-name}.md
   ```

2. **Required sections**:
   - Quick Start (basic command)
   - Purpose (what it does)
   - Key Features (bullets)
   - Options/Parameters (table)
   - Output Files (with paths)
   - Common Use Cases
   - Troubleshooting

3. **Update this README.md** with:
   - Tool name and script path
   - One-line purpose
   - Quick start command
   - Key features (3-5 bullets)
   - Link to full guide

4. **Optional**: Add development history in `docs/features/`

---

## Related Documentation

- **Main README**: `README.md` - Project overview
- **Claude Guide**: `CLAUDE.md` - AI assistant instructions
- **Agents Guide**: `AGENTS.md` - Automation guidelines
- **Environment Setup**: `docs/envs.md` - Conda environment details
- **Feature Archive**: `docs/features/` - Development histories
- **Fix Archive**: `docs/fixes/` - Bug fix records
