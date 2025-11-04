# Troubleshooting Guide

Debug playbook for common issues in the ag3 system.

## Critical Issues

### OpenMP Runtime Conflicts

**Symptom**:
```
OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
```
or segfault during FAISS operations.

**Cause**: Mixing pip `faiss-cpu` (which bundles libomp) with conda OpenBLAS+OpenMP in the same environment.

**Fix**:

1. **Always run Gate-2** (FAISS index builds) in the `ageFaiss` environment:
   ```bash
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```

2. **NEVER install pip `faiss-cpu`** in the `age` environment

3. If already installed, recreate the environment from scratch:
   ```bash
   conda env remove -n age
   conda env create -f envs/age.yaml
   ```

**Prevention**: Use the correct environment for each task (see [Environment Discipline](#environment-discipline)).

---

### Recall=0 in Gate-7

**Symptom**: Retrieval evaluation shows 0% recall despite having indexed documents.

**Cause**: Mismatched embeddings between documents and queries (e.g., using different embedding functions or random vectors).

**Diagnosis**:

1. Check embedding dimensions match:
   ```bash
   # Check document embeddings
   python3 -c "import pyarrow.parquet as pq; print(pq.read_table('data/vector/embeddings/embeddings.parquet').schema)"

   # Should show: vector: list<item: float> (1536 items)
   ```

2. Check if embeddings were generated with correct function:
   ```bash
   grep -n "embed_text" scripts/qa_step01_embeddings.py
   ```

**Fix**: Ensure both document indexing (Gate-1) and query processing use `embed_text()` from `scripts/embedding_utils.py` with the same dimensionality (1536):

```python
from scripts.embedding_utils import embed_text

# For documents (Gate-1)
doc_vector = embed_text(doc_text, dim=1536)

# For queries (Gate-7)
query_vector = embed_text(query_text, dim=1536)
```

**Prevention**: Always use the centralized embedding function, never create random vectors or use different embedding models.

---

### OpenAI API Errors

**Symptom**: Embedding generation fails with API errors or rate limits.

**Cause**: Missing or invalid `OPENAI_API_KEY`, network issues, or rate limiting.

**Diagnosis**:

1. Check if `.env` file exists:
   ```bash
   ls -la .env
   ```

2. Check if API key is set:
   ```bash
   grep OPENAI_API_KEY .env
   # Should show: OPENAI_API_KEY=sk-...
   ```

3. Test API key:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $(grep OPENAI_API_KEY .env | cut -d= -f2)"
   ```

**Fix**:

1. Create `.env` file with valid API key:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

2. Verify API key is valid (try test request above)

3. For rate limits, the retry logic will handle transient errors (3 attempts with exponential backoff)

4. Use cached embeddings when possible:
   ```bash
   # Check cache hit rate
   ls -1 data/cache/embeddings/ | wc -l
   ```

**Prevention**: Always set `OPENAI_API_KEY` in `.env` before running Gate-1.

## Environment Issues

### Missing Dependencies

**Symptom**: Import errors for packages like `yaml`, `pyarrow`, `aiohttp`.

**Cause**: Incomplete conda environment or manual package modifications.

**Diagnosis**:

```bash
# Check installed packages
conda run -n age pip list | grep -i "<package_name>"
```

**Fix**: Recreate the conda environment from the YAML file:

```bash
conda env remove -n age
conda env create -f envs/age.yaml
```

**Prevention**: Never manually install packages with pip in conda environments. Always update `envs/age.yaml` and recreate.

---

### Wrong Python Version

**Symptom**: Syntax errors or import failures.

**Cause**: Using wrong Python version (should be 3.13 for `age`, 3.12 for `ageFaiss`).

**Diagnosis**:

```bash
conda run -n age python --version
# Should show: Python 3.13.x

conda run -n ageFaiss python --version
# Should show: Python 3.12.x
```

**Fix**: Recreate environments from YAML files (see above).

---

### Environment Discipline

Always use the correct environment for each task:

| Task | Environment | Command Example |
|------|-------------|-----------------|
| Gate-1 (embeddings) | `age` | `conda run -n age python scripts/qa_step01_embeddings.py` |
| Gate-2 (FAISS) | `ageFaiss` | `conda run -n ageFaiss python scripts/qa_step02_indexes.py` |
| Gate-3 (MCP) | `age` | `conda run -n age python scripts/qa_step03_mcp.py` |
| Gate-7 (retrieval) | `age` | `conda run -n age python scripts/qa_step07_retrieval_eval.py` |
| Gate-8 (generation) | `age` | `conda run -n age python scripts/qa_step08_generation_eval.py` |
| Graph execution | `age` | `conda run -n age python scripts/run_graph.py` |

**Rule of thumb**: Use `ageFaiss` ONLY for Gate-2. Use `age` for everything else.

## Service Issues

### Port Conflicts

**Symptom**: "Port busy" or "Address already in use" errors when starting MCP services.

**Cause**: Another process is using ports 7801-7805.

**Diagnosis**:

```bash
# Check which processes are using MCP ports
lsof -i :7801-7805
```

**Fix**:

1. Stop existing stub services:
   ```bash
   # Kill specific process by PID
   kill <PID>

   # Or kill all Python processes using those ports
   lsof -ti :7801-7805 | xargs kill
   ```

2. Alternative: Update `configs/mcp.tools.yaml` to use different ports:
   ```yaml
   tools:
     kb.search:
       endpoint: http://localhost:8801/search  # Changed from 7801
   ```

**Prevention**: Always stop MCP services after use (Ctrl+C).

---

### MCP Service Timeouts

**Symptom**: MCP tool calls time out or fail.

**Cause**: Services are slow, timeout budgets are too tight, or network latency.

**Diagnosis**:

```bash
# Test service health
curl http://localhost:7801/health
curl http://localhost:7802/health
curl http://localhost:7803/health
curl http://localhost:7804/health
curl http://localhost:7805/health
```

**Fix**:

1. Increase timeouts in `configs/mcp.tools.yaml`:
   ```yaml
   tools:
     kb.search:
       timeout_ms: 5000  # Increased from 2000
   ```

2. Check if services are running:
   ```bash
   ps aux | grep qa_step03_mcp
   ```

3. Restart services:
   ```bash
   conda run -n age python scripts/qa_step03_mcp.py
   ```

## Data Quality Issues

### PDF Glyph Noise

**Symptom**: Chunks contain CID-like tokens (`(cid:123)`) or rendering artifacts from PDF extraction.

**Impact**: May affect retrieval quality and embedding generation.

**Diagnosis**:

```bash
# Check for glyph noise in chunks
grep -r "cid:" data/interim/chunks/ | head -5
```

**Mitigation**:

1. Add normalization rule in `configs/normalization.rules.yaml`:
   ```yaml
   - name: remove_pdf_glyphs
     pattern: '\(cid:\d+\)'
     replacement: ''
   ```

2. Re-run normalization:
   ```bash
   python3 scripts/qa_verify_normalization.py
   ```

3. Alternative: Enhance the reranker to downweight noisy chunks.

**Prevention**: Use better PDF extraction tools or post-process PDFs before ingestion.

---

### JSONL Parse Errors

**Symptom**: Errors reading intermediate JSONL files.

**Cause**: Corrupted JSONL files, non-UTF-8 encoding, or invalid JSON.

**Diagnosis**:

```bash
# Check for invalid JSON lines
python3 -c "
import json
with open('data/interim/eval/eval_seed.jsonl') as f:
    for i, line in enumerate(f):
        try:
            json.loads(line)
        except:
            print(f'Line {i}: {line[:100]}')
"
```

**Fix**:

1. Remove corrupted lines:
   ```bash
   # Backup first
   cp data/interim/eval/eval_seed.jsonl data/interim/eval/eval_seed.jsonl.backup

   # Filter valid JSON lines
   python3 -c "
   import json
   with open('data/interim/eval/eval_seed.jsonl') as f_in:
       with open('data/interim/eval/eval_seed.jsonl.clean', 'w') as f_out:
           for line in f_in:
               try:
                   json.loads(line)
                   f_out.write(line)
               except:
                   pass
   "

   mv data/interim/eval/eval_seed.jsonl.clean data/interim/eval/eval_seed.jsonl
   ```

2. Re-run the stage that generated the file

**Prevention**: Always validate JSONL output before writing to disk.

---

### Low Retrieval Quality

**Symptom**: Low recall, nDCG, or irrelevant results in Gate-7.

**Diagnosis**:

1. Check embedding quality:
   ```bash
   conda run -n age python scripts/qa_step01_embeddings.py
   # Look for embedding coverage and API errors
   ```

2. Check index health:
   ```bash
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   # Look for index stats (size, dimensions)
   ```

3. Inspect failed retrievals:
   ```bash
   cat reports/eval/retrieval_failures.jsonl | jq -r '.query'
   ```

4. Check router trace:
   ```bash
   cat reports/router/step07_retrieval_trace.jsonl | jq -r '.backend'
   ```

**Fix**:

1. If embeddings are mismatched:
   - Re-run Gate-1 with correct embedding function
   - Rebuild indexes (Gate-2)

2. If router is selecting wrong backend:
   - Adjust routing rules in `configs/router.heuristics.yaml`
   - Add query keywords to routing patterns

3. If FAISS parameters are suboptimal:
   - Increase `efSearch` in `configs/vector.indexing.yaml`
   - Rebuild index (Gate-2)

4. If chunks are too large/small:
   - Adjust `chunk_size` in `configs/chunking.config.json`
   - Re-run chunking and downstream stages

## Quality Gate Failures

### Gate-7 Failures

**Common failure modes**:

1. **recall@10 < threshold**:
   - Check embedding consistency (see [Recall=0](#recall0-in-gate-7))
   - Tune FAISS parameters (increase `efSearch`)
   - Review failed queries in `reports/eval/retrieval_failures.jsonl`

2. **nDCG@5 < threshold**:
   - Ranking quality issue, not coverage
   - Check reranking weights in `configs/router.heuristics.yaml`
   - Consider adding a reranker stage

3. **Latency budget exceeded**:
   - Use `AG7_LATENCY_MULTIPLIER` to relax budgets temporarily
   - Profile slow queries in router trace
   - Optimize index size or parameters

**Debug workflow**:

```bash
# Run with full tracing
conda run -n age \
  AG7_DEBUG=1 \
  AG7_TRACE=1 \
  AG7_TRACE_SUCCESSES=1 \
  python scripts/qa_step07_retrieval_eval.py

# Inspect failures
cat reports/eval/retrieval_failures.jsonl | jq .

# Inspect routing decisions
cat reports/router/step07_retrieval_trace.jsonl | jq .
```

---

### Gate-8 Failures

**Common failure modes**:

1. **structural_pass_rate < 1.0**:
   - Not generating exactly 5 insights
   - Missing required metadata (sources, dates)
   - Invalid email schema
   - Fix: Debug graph execution, check node outputs

2. **critical_flags_total > 0**:
   - Compliance violations detected
   - Check `outputs/<session-id>/compliance_report.json`
   - Fix: Update compliance templates or adjust stylist prompts

3. **length_readability_pass_runs < 9**:
   - Emails too long (>160 words) or complex (grade >10.0)
   - Fix: Adjust stylist prompts for conciseness

4. **persona_keyword_hits_avg < 2.0**:
   - Not enough persona-specific keywords
   - Fix: Enhance persona definitions in `icl/persona/`

## Performance Issues

### Slow Embedding Generation

**Symptom**: Gate-1 takes a long time.

**Cause**: Too many API calls, rate limiting, or no cache hits.

**Fix**:

1. Check cache hit rate:
   ```bash
   ls -1 data/cache/embeddings/ | wc -l
   # Compare to chunk count
   find data/interim/dedup/ -name "*.jsonl" | xargs wc -l
   ```

2. Use `AG1_AUTO_CONFIRM=1` to skip confirmation prompts

3. Increase batch size in `configs/vector.indexing.yaml` (if not hitting token limits)

---

### Slow FAISS Index Build

**Symptom**: Gate-2 takes a long time.

**Cause**: Large index or high `efConstruction` parameter.

**Fix**:

1. Decrease `efConstruction` in `configs/vector.indexing.yaml`:
   ```yaml
   faiss:
     efConstruction: 100  # Reduced from 200
   ```

2. Note: This may reduce index quality (lower recall)

---

### Slow LLM Calls

**Symptom**: Consolidator/Stylist nodes are slow (>10s).

**Cause**: LLM latency, large prompts, or network issues.

**Fix**:

1. Check LLM model configuration in `scripts/run_graph.py`:
   ```python
   # Line 176
   model = "gpt-5-nano"  # Fast model for development
   ```

2. Reduce prompt size by limiting context

3. Use local LLM for development (if available)

## Debugging Tools

### Enable Debug Logging

```bash
# Set debug environment variables
export AG7_DEBUG=1
export AG7_TRACE=1
export AG7_TRACE_SUCCESSES=1

# Run with debug output
conda run -n age python scripts/qa_step07_retrieval_eval.py
```

### Inspect JSONL Logs

```bash
# Pretty-print JSONL
cat logs/mcp/step03_probes.jsonl | jq .

# Filter by field
cat reports/router/step07_retrieval_trace.jsonl | jq 'select(.backend == "faiss")'

# Count by field
cat reports/router/step07_retrieval_trace.jsonl | jq -r '.backend' | sort | uniq -c
```

### Validate Configuration

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/vector.indexing.yaml'))"

# Validate JSON syntax
python3 -c "import json; json.load(open('configs/chunking.config.json'))"
```

### Check Index Health

```bash
# Run index health check
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Inspect health report
cat data/final/reports/index_health.json | jq .
```

## Related Documentation

- **[architecture.md](architecture.md)** - System design and component interactions
- **[commands.md](commands.md)** - Command reference for all scripts
- **[configuration.md](configuration.md)** - Configuration file documentation
- **[evaluation.md](evaluation.md)** - Quality gates and evaluation metrics
- **[envs.md](envs.md)** - Environment setup and package details
