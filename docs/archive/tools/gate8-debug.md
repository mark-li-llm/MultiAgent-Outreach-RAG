# Gate-8 Debug Usage Guide

## Quick Start

The new debug Gate-8 provides deep visibility into a single LangGraph pipeline execution.

### Basic Usage

```bash
# Run with default settings
conda run -n age python scripts/qa_step08_debug.py \
  --company Salesforce \
  --persona vp_customer_experience
```

### With Options

```bash
# Custom session ID and error handling
conda run -n age python scripts/qa_step08_debug.py \
  --company Salesforce \
  --persona cio \
  --session-id debug_001 \
  --stop-on-error
```

### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--company` | Company name | Salesforce |
| `--persona` | Target persona (vp_customer_experience, cio, vp_sales_ops) | vp_customer_experience |
| `--session-id` | Session ID for output tracking | Auto-generated |
| `--stop-on-error` | Stop on first error instead of continuing | Continue |
| `--no-llm-capture` | Disable LLM interaction monitoring | Enabled |
| `--no-validation` | Disable node-level validation | Enabled |

## Prerequisites

### 1. Set OpenAI API Key

The pipeline uses ChatOpenAI for LLM calls. You must set your API key:

```bash
# Option 1: Environment variable
export OPENAI_API_KEY=sk-your-actual-key-here

# Option 2: Create .env file in project root
echo "OPENAI_API_KEY=sk-your-actual-key-here" > .env
```

### 2. Ensure Conda Environment

Make sure the `age` environment is created:

```bash
conda env create -f envs/age.yaml
```

### 3. Run MCP Services (Optional)

For full MCP integration, start stub services (or use external services):

```bash
# Terminal 1: Start MCP stub services
conda run -n age python scripts/qa_step03_mcp.py
```

## Output Files

After running, you'll find:

```
reports/
├── qa/
│   ├── step08_debug.json      # Machine-readable report
│   └── step08_debug.md        # Human-readable report
└── debug/
    ├── node_states.jsonl      # State snapshots per node
    ├── llm_interactions.jsonl # LLM prompts and responses
    └── validation_trace.jsonl # Validation results

outputs/
└── {session_id}/
    ├── email.json             # Final email output
    ├── insights.json          # Enhanced insight cards
    ├── compliance_report.json # A2A compliance results
    └── timing.json            # Timing metrics
```

## Reading the Reports

### JSON Report (`step08_debug.json`)

Machine-readable format with:
- `execution_timeline`: Node-by-node timing
- `node_validations`: Pass/fail checks per node
- `llm_usage`: Token consumption per node
- `quality_metrics`: Structural, compliance, and persona metrics
- `issues_detected`: All errors and warnings

### Markdown Report (`step08_debug.md`)

Human-readable format with:
- Executive summary with quality score
- Visual timeline of node execution
- Node-by-node validation results
- Issues section with emojis
- Quality metrics tables

### Trace Files (`.jsonl`)

Detailed line-by-line logs:
- **node_states.jsonl**: Before/after state snapshots with diffs
- **llm_interactions.jsonl**: Every LLM call with prompts and responses
- **validation_trace.jsonl**: Detailed validation results

## Example Workflow

### Debug a Specific Persona

```bash
# Test CIO persona with full debugging
export OPENAI_API_KEY=sk-...
conda run -n age python scripts/qa_step08_debug.py \
  --persona cio \
  --session-id cio_debug_001

# Check results
cat reports/qa/step08_debug.md
```

### Quick Iteration (No LLM Capture)

For faster debugging without LLM monitoring:

```bash
conda run -n age python scripts/qa_step08_debug.py \
  --no-llm-capture \
  --persona vp_sales_ops
```

### Strict Error Mode

Stop immediately on any error:

```bash
conda run -n age python scripts/qa_step08_debug.py \
  --stop-on-error \
  --persona vp_customer_experience
```

## Interpreting Results

### Quality Score

The score (0-100) is computed from:
- **-20**: Wrong insight count (not 5)
- **-15**: Insufficient source diversity (<4)
- **-10**: Not enough recent insights (<2)
- **-10**: Word count over limit (>160)
- **-10**: Readability too high (>10.0 grade)
- **-15**: Insufficient persona keywords (<2)

Target: **≥85** for production readiness

### Node Validation Status

Each node has checks:
- ✅ **PASS**: All checks passed
- ❌ **FAIL**: One or more checks failed

Common failures:
- **Planner**: Generated <5 queries or low persona relevance
- **Retriever**: <10 chunks or <3 sources
- **Consolidator**: Missing LLM enhancements or <4 domains
- **Stylist**: Missing email fields or persona keywords

### LLM Usage

Token counts per node:
- **Consolidator**: Typically 1200-2000 tokens
- **Stylist**: Typically 1000-1500 tokens

High usage may indicate verbose prompts or long responses.

## Troubleshooting

### Error: "api_key client option must be set"

**Cause**: Missing OpenAI API key

**Fix**:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

### Error: "Port busy" (MCP services)

**Cause**: MCP stub services not running or port conflict

**Fix**:
```bash
# Check if services are running
lsof -i :7801-7805

# Kill if needed
kill -9 <PID>

# Or use external services (edit configs/mcp.tools.yaml)
```

### Empty Timeline / No Nodes Executed

**Cause**: Pipeline failed early before any nodes ran

**Fix**: Check `issues_detected` in JSON report for root cause

### No LLM Interactions Captured

**Cause**: LLM calls happened but weren't monkey-patched

**Fix**: Ensure you're not using `--no-llm-capture` and that langchain_openai is importable

## Comparison with Production Gate-8

| Feature | Debug Gate-8 | Production Gate-8 |
|---------|-------------|-------------------|
| **Runs** | 1 detailed run | 10 runs across personas |
| **Execution** | Direct import | Subprocess isolation |
| **Visibility** | Full state inspection | Output files only |
| **LLM Monitoring** | Yes (optional) | No |
| **Node Validation** | Per-node checks | End-to-end only |
| **Error Handling** | Configurable continue/stop | Always continues |
| **Use Case** | Development/debugging | CI/CD validation |

## Next Steps

After running debug Gate-8:

1. **Review Quality Score**: Target ≥85
2. **Check Node Validations**: All should PASS
3. **Inspect Issues**: Address any errors or warnings
4. **Review LLM Interactions**: Verify prompts and responses
5. **Optimize**: Use timing data to find bottlenecks

Once quality is high, run production Gate-8:
```bash
conda run -n age python scripts/qa_step08_generation_eval.py
```

---
*Gate-8 Debug Implementation - October 2024*
