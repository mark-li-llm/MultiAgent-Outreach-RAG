# Quality Gates & Evaluation Metrics

Comprehensive documentation for quality gates, evaluation metrics, and output formats.

## Quality Gates Overview

Each quality gate produces dual-format reports (JSON for machines, Markdown for humans) in `reports/qa/`.

| Gate | Purpose | Script | Environment | Est. Time |
|------|---------|--------|-------------|-----------|
| Gate-0 | Baseline checks | `qa_step00_baseline.py` | `age` | <1 min |
| Gate-1 | Generate embeddings | `qa_step01_embeddings.py` | `age` | 5-10 min |
| Gate-2 | Build indexes | `qa_step02_indexes.py` | `ageFaiss` | 2-5 min |
| Gate-3 | Validate MCP tools | `qa_step03_mcp.py` | `age` | <1 min |
| Gate-4 | Test router | `qa_step04_router.py` | `age` | <1 min |
| Gate-5 | Validate graph | `qa_step05_graph.py` | `age` | 2-3 min |
| Gate-6 | A2A compliance | `qa_step06_a2a.py` | `age` | 1-2 min |
| Gate-7 | Retrieval evaluation | `qa_step07_retrieval_eval.py` | `age` | 3-5 min |
| Gate-8 | Generation evaluation | `qa_step08_generation_eval.py` | `age` | 10-15 min |

**Total time**: ~30-40 minutes for full quality gate run

## Output Formats

All quality gates produce dual-format reports:

### JSON Format (Machine-Readable)

Location: `reports/qa/step*.json`

```json
{
  "gate_id": "step07",
  "gate_name": "Retrieval Evaluation",
  "timestamp": "2025-10-09T12:34:56Z",
  "status": "GREEN",
  "metrics": {
    "recall@10": 0.85,
    "ndcg@5": 0.78,
    "median_latency_ms": 450
  },
  "thresholds": {
    "recall@10": 0.80,
    "ndcg@5": 0.70,
    "median_latency_ms": 1000
  },
  "passed": true,
  "errors": []
}
```

### Markdown Format (Human-Readable)

Location: `reports/qa/step*.md`

```markdown
# Gate-7: Retrieval Evaluation

**Status**: GREEN ✓
**Timestamp**: 2025-10-09 12:34:56

## Metrics

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| recall@10 | 0.85 | 0.80 | PASS |
| nDCG@5 | 0.78 | 0.70 | PASS |
| median_latency_ms | 450 | 1000 | PASS |

## Summary

All thresholds passed. Retrieval quality is acceptable.
```

## Gate-0: Baseline Checks

**Purpose**: Validate directory structure, file permissions, and basic system requirements.

**Command**:
```bash
conda run -n age python scripts/qa_step00_baseline.py
```

**Checks**:
- Required directories exist (`data/`, `reports/`, `configs/`)
- Write permissions for output directories
- Python version (3.13 for `age`, 3.12 for `ageFaiss`)
- Required packages installed

**Output**:
- `reports/qa/step00_baseline.json`
- `reports/qa/step00_baseline.md`

**Pass Criteria**: All directory checks pass, no permission errors.

## Gate-1: Generate Embeddings

**Purpose**: Generate OpenAI ada-002 text embeddings for all deduplicated chunks.

**Command**:
```bash
conda run -n age python scripts/qa_step01_embeddings.py
```

**Metrics**:
- `embedding_count`: Number of embeddings generated
- `cache_hit_rate`: Proportion of embeddings from cache (0-1)
- `api_call_count`: Number of OpenAI API calls made
- `total_cost_usd`: Estimated cost in USD

**Output**:
- `data/vector/embeddings/embeddings.parquet` - Embedding vectors (Parquet format)
- `reports/qa/step01_embeddings.json`
- `reports/qa/step01_embeddings.md`

**Pass Criteria**:
- All chunks have embeddings
- All embeddings are 1536-dimensional
- No API errors

**Environment Variables**:
- `AG1_AUTO_CONFIRM=1`: Skip cost confirmation prompt

## Gate-2: Build and Validate Indexes

**Purpose**: Build FAISS index and validate index health.

**Command**:
```bash
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Metrics**:
- `index_size`: Number of vectors in index
- `index_dimension`: Dimensionality (should be 1536)
- `index_type`: Index algorithm (HNSW)
- `build_time_ms`: Time to build index

**Output**:
- `data/vector/faiss/index.faiss` - FAISS index file
- `data/final/reports/index_health.json` - Index health report
- `reports/qa/step02_indexes.json`
- `reports/qa/step02_indexes.md`

**Pass Criteria**:
- Index size matches embedding count
- Index dimension is 1536
- No build errors

**Critical**: Must use `ageFaiss` environment to avoid OpenMP conflicts.

## Gate-3: Validate MCP Tools

**Purpose**: Start MCP stub services and validate tool health and contracts.

**Command**:
```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**Checks**:
- All 5 MCP services are reachable (ports 7801-7805)
- Services respond within timeout budgets
- Response schemas match contracts
- Error handling works correctly

**Output**:
- `reports/qa/step03_mcp.json`
- `reports/qa/step03_mcp.md`
- `logs/mcp/step03_probes.jsonl` - Detailed probe logs

**Pass Criteria**: All services pass health checks, no timeout errors.

## Gate-4: Test Router Heuristics

**Purpose**: Test query routing heuristics across FAISS/Weaviate/Pinecone.

**Command**:
```bash
conda run -n age python scripts/qa_step04_router.py
```

**Metrics**:
- `routing_accuracy`: Proportion of queries routed to expected backend
- `fallback_rate`: Proportion of queries using fallback order
- `avg_routing_time_ms`: Average time to make routing decision

**Output**:
- `reports/qa/step04_router.json`
- `reports/qa/step04_router.md`

**Pass Criteria**: Routing accuracy > 80%, no routing errors.

## Gate-5: Validate LangGraph Orchestration

**Purpose**: Validate LangGraph workflow execution (node transitions, state management).

**Command**:
```bash
conda run -n age python scripts/qa_step05_graph.py
```

**Checks**:
- All 8 nodes execute successfully
- State transitions follow expected topology
- Conditional edges work correctly (A2A routing)
- No node timeouts

**Output**:
- `reports/qa/step05_graph.json`
- `reports/qa/step05_graph.md`

**Pass Criteria**: All nodes pass, state transitions are correct.

## Gate-6: Agent-to-Agent Compliance Checks

**Purpose**: Validate A2A handoff protocols and compliance negotiation.

**Command**:
```bash
conda run -n age python scripts/qa_step06_a2a.py --session-id <SESSION>
```

**Checks**:
- A2A handoff protocol works
- Compliance negotiation rounds execute (up to 2)
- Critical flags are detected and handled
- Revision logic works correctly

**Output**:
- `reports/qa/step06_a2a.json`
- `reports/qa/step06_a2a.md`
- `outputs/<session-id>/compliance_report.json`

**Pass Criteria**: A2A negotiation completes within 2 rounds, no protocol errors.

## Gate-7: Retrieval Evaluation

**Purpose**: End-to-end retrieval quality assessment (recall@10, nDCG@5, latency).

**Command**:
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

### Metrics

#### Recall@k

**Definition**: Proportion of relevant documents retrieved in top k results.

**Formula**:
```
recall@k = (# relevant docs in top k) / (# total relevant docs)
```

**Threshold**: recall@10 ≥ 0.80 (80%)

**Interpretation**:
- 1.0 (100%) = Perfect recall, all relevant docs retrieved
- 0.8 (80%) = Acceptable, most relevant docs retrieved
- <0.5 (50%) = Poor, many relevant docs missed

#### nDCG@k (Normalized Discounted Cumulative Gain)

**Definition**: Ranking quality metric that rewards relevant documents at higher ranks.

**Formula**:
```
DCG@k = Σ (rel_i / log2(i+1))  for i=1 to k
nDCG@k = DCG@k / IDCG@k
```

Where `rel_i` is relevance score at position i, and `IDCG@k` is ideal DCG (perfect ranking).

**Threshold**: nDCG@5 ≥ 0.70

**Interpretation**:
- 1.0 = Perfect ranking
- 0.7-0.9 = Good ranking quality
- <0.5 = Poor ranking

#### Coverage (Optional)

**Definition**: Proportion of indexed documents that are reachable via search.

**Threshold**: coverage ≥ 0.95 (95%)

**Note**: Can be skipped with `AG7_IGNORE_COVERAGE=1`

#### Freshness

**Definition**: Time-based relevance of retrieved documents (prefers recent documents).

**Metric**: Proportion of results within 12 months

**Threshold**: freshness ≥ 0.30 (30% of results)

#### Latency

**Definition**: Response time budgets per backend.

**Metrics**:
- `median_latency_ms`: Median response time
- `p95_latency_ms`: 95th percentile response time
- `p99_latency_ms`: 99th percentile response time

**Thresholds** (relaxed by `AG7_LATENCY_MULTIPLIER`):
- Median: 1000ms × multiplier
- P95: 2000ms × multiplier
- P99: 3000ms × multiplier

### Output

- `reports/qa/step07_retrieval_eval.json`
- `reports/qa/step07_retrieval_eval.md`
- `reports/eval/retrieval_failures.jsonl` - Failed queries with details
- `reports/router/step07_retrieval_trace.jsonl` - Per-query routing trace

### Environment Variables

- `AG7_IGNORE_COVERAGE=1`: Skip coverage gating (recommended for initial runs)
- `AG7_LATENCY_MULTIPLIER=<float>`: Relax latency budgets (e.g., 3.0 for 3x tolerance)
- `AG7_ANALYZE_TOPK=<int>`: Retrieval cut-off for evaluation (default: 10)
- `AG7_TOPK_SLICES="1,3,5,10"`: Additional @k slices for recall curves
- `AG7_NEAR_SEQ_TOL=<int>`: Near-miss tolerance in chunks within same doc (default: 1)
- `AG7_TRACE=1`: Enable per-query trace JSONL (default: enabled)
- `AG7_TRACE_TOPK=<int>`: Number of top-K items to capture in trace (default: 10)
- `AG7_TRACE_SUCCESSES=1`: Include successes in trace (default: enabled; set 0 for misses only)
- `AG7_DEBUG=1`: Umbrella debug switch enabling tracing (default: enabled)

### Pass Criteria

All thresholds must pass:
- recall@10 ≥ 0.80
- nDCG@5 ≥ 0.70
- coverage ≥ 0.95 (if not ignored)
- median_latency_ms ≤ 1000 × multiplier

### Status Colors

- **GREEN**: All thresholds passed
- **AMBER**: Minor threshold violations (e.g., latency only)
- **RED**: Critical threshold violations (recall or nDCG)

## Gate-8: Generation & Compliance Evaluation

**Purpose**: End-to-end email generation quality and compliance validation.

**Command**:
```bash
conda run -n age python scripts/qa_step08_generation_eval.py
```

### Test Design

- **10 runs** across **≥3 personas**
- Each run generates 5 insights + 1 email
- Compliance checks on all outputs

### Metrics

#### G8-01: Structural Pass Rate

**Definition**: Proportion of runs that produce structurally valid outputs.

**Requirements** (all must pass):
- Exactly 5 insights generated
- ≥4 distinct sources cited
- ≥2 recent items (within 12 months)
- Valid email schema (subject, body, tone)
- Resolvable proof points (all links work)

**Threshold**: structural_pass_rate == 1.0 (100%)

**Interpretation**:
- 1.0 = All runs structurally valid
- <1.0 = Some runs have structural issues (CRITICAL)

#### G8-02: Critical Flags Total

**Definition**: Total number of critical compliance violations across all runs.

**Critical violations include**:
- Forward-looking statements without disclaimers
- Missing source attribution
- Incorrect fact claims
- Inappropriate tone for persona

**Threshold**: critical_flags_total == 0

**Interpretation**:
- 0 = No critical violations
- >0 = Compliance issues detected (CRITICAL)

#### G8-03: Length & Readability Pass Runs

**Definition**: Number of runs where email passes length and readability checks.

**Requirements**:
- Email body ≤160 words
- Flesch-Kincaid grade ≤10.0 (10th grade reading level)

**Threshold**: length_readability_pass_runs ≥ 9 (out of 10)

**Interpretation**:
- 10/10 = All emails are concise and readable
- 9/10 = Acceptable (1 outlier allowed)
- <9/10 = Emails too long or complex

#### G8-04: Persona Keyword Hits (Average)

**Definition**: Average number of persona-specific keywords per email.

**Calculation**: Count persona keywords (from `icl/persona/*.yaml`) in email body, average across runs.

**Threshold**: persona_keyword_hits_avg ≥ 2.0

**Interpretation**:
- ≥3.0 = Strong persona alignment
- 2.0-2.9 = Acceptable alignment
- <2.0 = Weak persona alignment

### Output

- `reports/qa/step08_generation_eval.json`
- `reports/qa/step08_generation_eval.md`
- `reports/eval/generation_metrics.json` - Detailed per-run metrics
- `reports/eval/compliance_metrics.json` - Compliance flag breakdown

### Pass Criteria & Status

**Thresholds**:
- G8-01: structural_pass_rate == 1.0 (CRITICAL)
- G8-02: critical_flags_total == 0 (CRITICAL)
- G8-03: length_readability_pass_runs ≥ 9
- G8-04: persona_keyword_hits_avg ≥ 2.0

**Status Colors**:
- **GREEN**: All 4 thresholds pass
- **AMBER**: Only G8-03 or G8-04 fails (non-critical)
- **RED**: G8-01 or G8-02 fails, or multiple failures (CRITICAL)

### Example Output

```json
{
  "structural_pass_rate": 1.0,
  "critical_flags_total": 0,
  "length_readability_pass_runs": 9,
  "persona_keyword_hits_avg": 2.3,
  "status": "GREEN"
}
```

## Report Locations

All reports are stored in `reports/qa/`:

| Gate | JSON Report | Markdown Report | Additional Artifacts |
|------|-------------|-----------------|---------------------|
| Gate-0 | `step00_baseline.json` | `step00_baseline.md` | N/A |
| Gate-1 | `step01_embeddings.json` | `step01_embeddings.md` | `data/vector/embeddings/embeddings.parquet` |
| Gate-2 | `step02_indexes.json` | `step02_indexes.md` | `data/final/reports/index_health.json`, FAISS indexes |
| Gate-3 | `step03_mcp.json` | `step03_mcp.md` | `logs/mcp/step03_probes.jsonl` |
| Gate-4 | `step04_router.json` | `step04_router.md` | N/A |
| Gate-5 | `step05_graph.json` | `step05_graph.md` | N/A |
| Gate-6 | `step06_a2a.json` | `step06_a2a.md` | `outputs/<session-id>/compliance_report.json` |
| Gate-7 | `step07_retrieval_eval.json` | `step07_retrieval_eval.md` | `reports/eval/retrieval_failures.jsonl`, `reports/router/step07_retrieval_trace.jsonl` |
| Gate-8 | `step08_generation_eval.json` | `step08_generation_eval.md` | `reports/eval/generation_metrics.json`, `reports/eval/compliance_metrics.json` |

## Scale & Performance

Current system performance characteristics:

- **Current scale**: Designed and verified on 100+ documents (~1.6k chunks)
- **FAISS latency**: Median sub-second local retrieval
- **Weaviate/Pinecone**: Simulated manifests (no network required for development)
- **Horizontal scaling**: Stateless stages and services; indexes can be sharded externally

### Performance Benchmarks

| Stage | Time (100 docs) | Time (1k docs) | Time (10k docs) |
|-------|-----------------|----------------|-----------------|
| Gate-1 (embeddings) | 5-10 min | 30-60 min | 5-8 hours |
| Gate-2 (indexing) | 2-5 min | 10-15 min | 1-2 hours |
| Gate-7 (retrieval) | 3-5 min | 10-15 min | 30-60 min |
| Gate-8 (generation) | 10-15 min | 10-15 min | 10-15 min |

**Note**: Times are estimates and depend on hardware, network, and API rate limits.

## Debugging Failed Gates

### Gate-7 Debugging

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

# Count failures by query type
cat reports/eval/retrieval_failures.jsonl | jq -r '.query_type' | sort | uniq -c
```

### Gate-8 Debugging

```bash
# Run generation eval
conda run -n age python scripts/qa_step08_generation_eval.py

# Inspect compliance flags
cat reports/eval/compliance_metrics.json | jq '.critical_flags'

# Check per-run metrics
cat reports/eval/generation_metrics.json | jq '.runs[] | {persona, word_count, grade_level}'

# Inspect generated emails
ls -lh outputs/gate8-*/email.json
cat outputs/gate8-01/email.json | jq .
```

## Related Documentation

- **[architecture.md](architecture.md)** - System design and component interactions
- **[commands.md](commands.md)** - Command reference for running quality gates
- **[configuration.md](configuration.md)** - Configuration affecting evaluation
- **[troubleshooting.md](troubleshooting.md)** - Debug guide for gate failures
