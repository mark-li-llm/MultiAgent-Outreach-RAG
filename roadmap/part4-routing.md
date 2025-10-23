---
date: 2025-10-20T16:30:52-04:00
researcher: Claude Code
git_commit: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
branch: agent-weaviate
repository: agent-weaviate
topic: "Multi-Index Routing System: Query Routing to Vector Backends"
tags: [research, codebase, routing, multi-index, vector-search, faiss, weaviate, pinecone, heuristics]
status: complete
last_updated: 2025-10-20
last_updated_by: Claude Code
---

# Research: Multi-Index Routing System - Query Routing to Vector Backends

**Date**: 2025-10-20T16:30:52-04:00
**Researcher**: Claude Code
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

## Research Question

How does the multi-agent RAG system route queries to the appropriate vector backend (FAISS, Weaviate, or Pinecone) based on keywords, personas, and heuristics? This research documents the complete routing architecture, decision logic, configuration, integration points, and quality validation mechanisms.

## Summary

The routing system implements a deterministic three-tier decision hierarchy that selects the optimal vector backend for each query:

1. **Rule-Based Matching** (highest priority): Keywords in the query trigger explicit backend assignments (e.g., "earnings" → Pinecone, "api" → Weaviate)
2. **Persona Bias** (medium priority): User personas have preferred backends (e.g., VP Sales Ops → Pinecone, CIO → Weaviate)
3. **Heuristic Fallback** (lowest priority): Short/definitional queries → FAISS, longer queries → Weaviate

The system is fully config-driven (no hardcoded routing logic), supports automatic fallback when backends return empty results, and includes diversity-aware reranking to improve result variety. All routing decisions are traced with reason codes for audit and debugging. Quality validation occurs at Gate-4 (routing coverage) and Gate-7 (retrieval quality per backend).

**Key Implementation**: The `decide_backend()` function in `router_core.py` makes routing decisions in ~30 lines of code, while the `rerank()` function applies multi-factor scoring (similarity 60%, recency 30%, diversity 10%) with domain-based diversity enforcement.

## Detailed Findings

### 1. Core Components

#### 1.1 Routing Decision Engine

**Location**: `scripts/router_core.py:72-100`

The `decide_backend()` function implements the core routing logic:

```python
def decide_backend(query: str, persona: Optional[str], meta: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """Deterministic router using configs/router.heuristics.yaml.

    Returns (backend, reason_codes).
    """
```

**Three-Tier Decision Tree**:

1. **Rule-Based Matching** (lines 81-89)
   - Iterates through rules from config
   - Checks if any keyword appears in lowercase query
   - Returns backend from first matching rule with custom reason code
   - Example: Keywords `[results, earnings, fiscal]` → `pinecone` (reason: `PR_QUERY`)

2. **Persona Bias** (lines 91-94)
   - Checks persona_bias map from config
   - Returns persona-specific backend preference
   - Example: `vp_sales_ops` → `pinecone` (reason: `PERSONA_BIAS`)

3. **Heuristic Fallback** (lines 96-100)
   - Short queries (≤4 words) or definitional keywords → `faiss` (reason: `DEFAULT_SHORT_FAISS`)
   - All other queries → `weaviate` (reason: `DEFAULT_WEAVIATE`)

**Return Value**: `(backend: str, reasons: List[str])` where backend ∈ {`faiss`, `weaviate`, `pinecone`}

#### 1.2 Reranking Engine

**Location**: `scripts/router_core.py:113-184`

The `rerank()` function implements multi-factor result reordering:

```python
def rerank(
    results: List[Dict[str, Any]],
    docmeta: Dict[str, DocMeta],
    weights: Optional[Dict[str, float]] = None,
    *,
    top_k: int = 10,
    domain_cap: int = 2,
) -> List[Dict[str, Any]]:
```

**Scoring Formula** (lines 127-159):
```
final_score = (similarity × 0.6) + (recency × 0.3) + (diversity × 0.1)
```

Where:
- **Similarity** (lines 134-140): Transforms negative L2 distances to (0,1] range using `1.0 / (1.0 + abs(score))`
- **Recency** (lines 147-153): Linear decay from 1.0 (today) to 0.0 (≥2 years), unknown dates get 0.3
- **Diversity** (lines 155-156): +0.1 bonus for first occurrence of each domain

**Domain-Aware Selection** (lines 165-183):
- Enforces `domain_cap=2` (max 2 results per domain in top_k)
- Returns diversified top_k followed by remainder in score order
- Prevents single domain from dominating results

#### 1.3 Configuration Loader

**Location**: `scripts/router_core.py:27-37`

The `load_router_config()` function provides graceful degradation:

```python
def load_router_config(path: str = ROUTER_CONF) -> Dict[str, Any]:
    if not os.path.exists(path):
        # sensible defaults
        return {
            "weights": {"similarity": 0.6, "recency": 0.3, "diversity": 0.1},
            "persona_bias": {},
            "rules": [],
            "fallback_order": ["faiss", "weaviate", "pinecone"],
            "top_k_default": 10,
        }
    return _load_yaml(path)
```

**Default Behavior** (when config missing):
- Empty persona bias → all personas use heuristic fallback
- Empty rules → all queries use persona bias or heuristic fallback
- Fallback order: FAISS → Weaviate → Pinecone

#### 1.4 Document Metadata Loader

**Location**: `scripts/router_core.py:53-69`

The `load_doc_meta()` function loads document metadata for reranking:

```python
def load_doc_meta() -> Dict[str, DocMeta]:
    m: Dict[str, DocMeta] = {}
    for p in glob.glob(os.path.join(NORM_DIR, "*.json")):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        doc_id = d.get("doc_id")
        if not doc_id:
            continue
        m[doc_id] = DocMeta(
            doc_id=doc_id,
            publish_date=(d.get("publish_date") or "").strip() or None,
            source_domain=(d.get("source_domain") or "").strip() or None,
            url=(d.get("final_url") or d.get("url") or "").strip() or None,
        )
    return m
```

Loads metadata from `data/interim/normalized/*.json` for:
- `publish_date`: ISO date string for recency scoring
- `source_domain`: Domain name for diversity enforcement
- `url`: Reference URL (not used in routing, for traceability)

### 2. Configuration

#### 2.1 Router Heuristics Configuration

**Location**: `configs/router.heuristics.yaml`

**Structure**:

```yaml
weights:
  similarity: 0.5
  recency: 0.3
  diversity: 0.2

persona_bias:
  vp_sales_ops: pinecone
  cio: weaviate
  vp_customer_experience: faiss

rules:
  # Press/financial results → pinecone
  - if:
      has_keywords: [results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k]
    then:
      backend: pinecone
      reason: PR_QUERY

  # Developer/doc/API → weaviate (filtering/schema-friendly)
  - if:
      has_keywords: [api, apis, endpoint, schema, developer, example]
    then:
      backend: weaviate
      reason: FILTER_MATCH

  # Definitional / product overview → faiss
  - if:
      has_keywords: [definition, what is, overview]
    then:
      backend: faiss
      reason: DEFINITION

fallback_order: [faiss, weaviate, pinecone]
top_k_default: 10
```

**Weights Configuration** (lines 1-4):
- `similarity: 0.5` (50%): Vector similarity score
- `recency: 0.3` (30%): Document freshness
- `diversity: 0.2` (20%): Domain diversity bonus

**Persona Bias** (lines 7-10):
- `vp_sales_ops` → `pinecone` (press/financial focus)
- `cio` → `weaviate` (technical/API focus)
- `vp_customer_experience` → `faiss` (general/product focus)

**Routing Rules** (lines 19-39):
- **Rule 1 - Press Queries**: 9 keywords → Pinecone (financial/PR content)
- **Rule 2 - Developer Queries**: 6 keywords → Weaviate (technical docs)
- **Rule 3 - Definitional Queries**: 3 keywords → FAISS (general knowledge)

**Fallback Order** (line 41):
- Primary fails → Try FAISS → Try Weaviate → Try Pinecone

#### 2.2 MCP Tools Configuration

**Location**: `configs/mcp.tools.yaml:2-5`

```yaml
kb.search:
  host: 127.0.0.1
  port: 7801
  timeout_ms: 2000
```

Used by `kb_search()` client to construct endpoint URL and timeout for routing backend parameter to MCP service.

### 3. Integration Points

#### 3.1 LangGraph Retriever Node

**Location**: `scripts/langgraph_nodes.py:214-244`

The `retriever_node()` function is the **primary integration point** for routing:

```python
async def retriever_node(state: AgentState) -> dict:
    """Execute vector search via MCP kb.search (run_graph.py lines 224-441)."""
    tools_cfg = load_mcp_map()           # Line 216
    router_cfg = load_router_config()    # Line 217
    docmeta = load_doc_meta()           # Line 218

    retrieved_chunks = []
    retrieval_logs = []
    route_decisions = []                # Line 222

    connector = aiohttp.TCPConnector(limit_per_host=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        for q in state["queries"]:
            backend, reasons = decide_backend(q, state["persona"], None)  # Line 227
            route_decisions.append({"query": q, "backend": backend, "reasons": reasons})  # Line 228

            # Retrieve
            res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)  # Line 231

            # Re-rank + attach meta
            res = rerank(res, {k: type("DM", (), v) for k, v in docmeta.items()}, top_k=12, domain_cap=2)  # Line 234

            # Log and extend
            retrieval_logs.append({"query": q, "results": res[:10]})
            retrieved_chunks.extend(res[:10])

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_logs": retrieval_logs,
        "route_decisions": route_decisions,  # Line 243
    }
```

**Data Flow**:
1. Line 227: Call `decide_backend()` to get backend selection
2. Line 228: Store decision with query, backend, and reason codes
3. Line 231: Pass selected backend to `kb_search()` for retrieval
4. Line 234: Rerank results using document metadata
5. Line 243: Return decisions for state merge (accumulated via `Annotated[..., add]`)

**Graph Position**:
```
Intake → Planner → [Retriever] → Synthesizer → Consolidator → Stylist → A2A → Assembler
                       ↑
                   Routing happens here only
```

#### 3.2 MCP kb.search Client

**Location**: `scripts/langgraph_nodes.py:144-161`

The `kb_search()` function sends backend parameter via HTTP:

```python
async def kb_search(session: aiohttp.ClientSession, backend: str, query: str, top_k: int, tools_cfg: Dict[str, Any]):
    """MCP kb.search client (copied from run_graph.py lines 114-131)."""
    base = tools_cfg.get("kb.search") or {}
    host = base.get("host", "127.0.0.1")
    port = int(base.get("port", 7801))
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "search", "params": {"query": query, "backend": backend, "top_k": int(top_k)}}  # Line 150
    t0 = datetime.now(timezone.utc)
    try:
        async with session.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
            status = resp.status
            j = await resp.json()
            if status >= 400:
                return [], (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, (j.get("error") or {}).get("code")
            res = j.get("results") or []
            return res, (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, None
    except asyncio.TimeoutError:
        return [], (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, "Timeout"
    except Exception as e:
        return [], (datetime.now(timezone.utc) - t0).total_seconds() * 1000.0, "NetworkError"
```

**Interface Contract**:
- **Request**: `POST /invoke` with payload `{"method": "search", "params": {"query": str, "backend": str, "top_k": int}}`
- **Response**: `{"results": [{"chunk_id": str, "doc_id": str, "score": float, "snippet": str}]}`
- **Return**: `(results: List[Dict], latency_ms: float, error_code: Optional[str])`

**Identical Implementations**:
- `scripts/run_graph.py:114-131` (original graph)
- `scripts/qa_step04_router.py:65-83` (router QA)
- `scripts/qa_step07_retrieval_eval.py:61-80` (retrieval eval)

#### 3.3 MCP Stub Server

**Location**: `scripts/qa_step03_mcp.py:82-156`

The `handle_invoke_kb()` handler processes backend parameter:

```python
async def handle_invoke_kb(request: web.Request) -> web.Response:
    """kb.search MCP stub handler."""
    # Line 84: Parse request
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": {"code": "InvalidJSON", "message": "malformed JSON"}}, status=400)

    # Line 87-89: Validate method
    method = (body.get("method") or "").strip()
    if method != "search":
        return web.json_response({"error": {"code": "InvalidMethod", "message": "only 'search' supported"}}, status=400)

    # Line 90-95: Extract and validate params
    params = body.get("params") or {}
    q = (params.get("query") or "").strip()
    backend = (params.get("backend") or "").strip()  # Line 92
    topk = int(params.get("top_k") or 10)
    if not q or not backend:
        return web.json_response({"error": {"code": "InvalidParams", "message": "query and backend required"}}, status=400)
    if backend not in ("faiss", "weaviate", "pinecone"):  # Line 96
        return web.json_response({"error": {"code": "BackendUnavailable", "message": "unsupported backend"}}, status=503)

    # Line 98-101: Simulate backend-specific latency
    delay_ms = {"faiss": (5, 10), "weaviate": (40, 80), "pinecone": (80, 160)}[backend]
    d = random.uniform(*delay_ms) / 1000.0
    await asyncio.sleep(d)

    # Line 102-119: Perform vector search (identical for all backends)
    qv = embed_query(q, dim)
    scored: List[Tuple[float, int]] = []
    for i, v in enumerate(vectors):
        scored.append((l2(qv, v), i))
    scored.sort(key=lambda x: x[0])
    top = scored[:topk]

    # Line 120-145: Apply lexical reranking
    # ... (keyword boosting logic)

    # Line 136-156: Build and return response
    results = []
    for dist, idx in top:
        ch = chunks_index[idx]
        results.append({
            "chunk_id": ch.get("chunk_id"),
            "doc_id": ch.get("doc_id"),
            "score": float(-dist),
            "snippet": (ch.get("text") or "")[:280],
        })
    return web.json_response({"results": results})
```

**Backend-Specific Behavior**:
- **FAISS**: 5-10ms latency (fastest, in-memory index)
- **Weaviate**: 40-80ms latency (medium, structured query)
- **Pinecone**: 80-160ms latency (slowest, managed cloud service)

**Note**: Search algorithm is identical across all backends in the stub (uses numpy-based L2 distance). Backend parameter only affects simulated latency.

#### 3.4 State Management

**Location**: `scripts/langgraph_state.py:27`

The `route_decisions` field uses accumulation semantics:

```python
route_decisions: Annotated[List[Dict[str, Any]], add]
```

- The `add` operator concatenates lists across node invocations
- Each decision object: `{"query": str, "backend": str, "reasons": List[str]}`
- Accumulated decisions are accessible to downstream nodes and written to trace files

#### 3.5 Trace File Output

**Location**: `scripts/run_graph_langgraph.py:190-198`

After graph completion, routing decisions are persisted:

```python
with open(os.path.join(out_dir, "router_trace.jsonl"), "w", encoding="utf-8") as f:  # Line 191
    for rd in result.get("route_decisions", []):
        f.write(json.dumps({
            "timestamp": result["timestamp"],
            "query_text": rd.get("query"),
            "decision_backend": rd.get("backend"),  # Line 196
            "reason_codes": rd.get("reasons"),     # Line 197
        }) + "\n")
```

**Output**: `outputs/<session-id>/router_trace.jsonl` (one JSON line per query)

### 4. Advanced Routing Features

#### 4.1 Automatic Fallback

**Location**: `scripts/qa_step04_router.py:296-316`

When primary backend returns empty results, system automatically tries fallback backends:

```python
fallback_used = False
if not results:
    empty += 1
    # try fallback
    fallback_used = True
    # next backend in configured order (choose the first different backend)
    try_order = [b for b in fallback_order if b != backend] + [backend]
    for fb in try_order:
        if fb == backend:
            continue
        if not use_offline:
            res2, lat2, err2 = await kb_search(session, fb, q, top_k_default, tools_cfg)
        else:
            # In offline mode, fallback is identical to primary search; reuse results
            res2, lat2, err2 = results, latency_ms, None
        if res2:
            results = res2
            latency_ms = lat2
            reasons = reasons + ["FALLBACK:" + fb]
            retry_success += 1
            break
```

**Behavior**:
- Only triggered when primary backend returns zero results
- Tries backends in `fallback_order` (skipping already-tried primary)
- Stops at first successful retrieval
- Appends reason code: `"FALLBACK:" + backend_name`
- Tracks success rate for QA metrics

#### 4.2 Diversity Merge

**Location**: `scripts/qa_step04_router.py:332-364`

When domain diversity is insufficient, system queries multiple backends and merges results:

```python
def count_unique_domains(res: List[Dict[str, Any]]) -> int:
    doms = set()
    for r in res[: top_k_default]:
        did = r.get("doc_id")
        dm = docmeta.get(did)
        if dm and dm.source_domain:
            doms.add(dm.source_domain)
    return len(doms)

if results and not use_offline:
    uniq_now = count_unique_domains(results)
    target_domains = 3
    if uniq_now < target_domains:
        # Gather from other backends
        merge_pool: List[Dict[str, Any]] = list(results)
        tried_fb = []
        # Prepare ordered list of other backends
        others = [b for b in fallback_order if b != backend]
        # Query each alternate backend and extend pool
        merge_topk = max(top_k_default, 30)
        for fb in others:
            res2, lat2, err2 = await kb_search(session, fb, q, merge_topk, tools_cfg)
            tried_fb.append(fb)
            latency_ms = max(latency_ms, lat2)
            if res2:
                # Deduplicate by chunk_id
                seen = set(r.get("chunk_id") for r in merge_pool)
                for r in res2:
                    cid = r.get("chunk_id")
                    if cid and cid not in seen:
                        merge_pool.append(r)
                        seen.add(cid)
            # Early stop if we have enough diversity after merge
            tmp = rerank(merge_pool, docmeta, weights, top_k=top_k_default, domain_cap=2)
            if count_unique_domains(tmp) >= target_domains:
                results = tmp
                reasons = reasons + ["DIVERSITY_MERGE:" + ",".join(tried_fb)]
                break
        else:
            # If loop completes without break, still apply best diversified ordering
            results = rerank(merge_pool, docmeta, weights, top_k=top_k_default, domain_cap=2)
```

**Behavior**:
- Activates when unique domains in top_k < 3
- Queries other backends with larger top_k (30)
- Deduplicates by chunk_id to avoid redundancy
- Reranks after each merge to check diversity
- Early stops when target diversity (3 domains) achieved
- Appends reason code: `"DIVERSITY_MERGE:" + backend_list`

### 5. Quality Validation

#### 5.1 Gate-4: Router Coverage

**Location**: `scripts/qa_step04_router.py`

**Purpose**: Validate routing decisions across evaluation queries

**Metrics** (lines 411-459):

1. **Backend Coverage** (lines 413-424):
   - `COV-faiss`, `COV-weaviate`, `COV-pinecone`
   - Threshold: ≥10% share OR ≥1 route
   - Ensures all backends are exercised

2. **Empty Result Rate** (lines 426-432):
   - Metric: `empty_result_rate`
   - Threshold: ≤2%
   - Tracks how often backends return zero results

3. **Auto-Retry Success Rate** (lines 433-439):
   - Metric: `auto_retry_success_rate`
   - Threshold: ≥95% (when empty results occur)
   - Validates fallback logic effectiveness

4. **Document Freshness** (lines 441-448):
   - Metric: `avg_doc_age_days`
   - Threshold: ≤365 days (or baseline median)
   - Ensures recent documents are prioritized

5. **Domain Diversity** (lines 451-459):
   - Metric: `mean_unique_domains_top10`
   - Threshold: ≥2.4 (AMBER at ≥2.0)
   - Validates diversity enforcement

**Status Rollup** (lines 462-473):
- **GREEN**: All checks pass
- **AMBER**: Only warnings (no failures)
- **RED**: One or more failures

**Reports**:
- `reports/qa/step04_router.json` (machine-readable)
- `reports/qa/step04_router.md` (human-readable)
- `reports/router/step04_router_trace.jsonl` (per-query trace)

#### 5.2 Gate-7: Retrieval Quality

**Location**: `scripts/qa_step07_retrieval_eval.py`

**Purpose**: Measure retrieval quality per backend

**Per-Backend Metrics** (lines 375-404):
- `total`: Number of queries routed to this backend
- `chunk_hit`: Chunk-level recall (expected chunk in top_k)
- `doc_hit`: Document-level recall (expected doc in top_k)
- `dcg5`: Discounted cumulative gain at rank 5
- `doc_dcg5`: Document-level DCG at rank 5
- `near_miss`: Queries where expected doc is just outside top_k

**Aggregated Metrics**:
- `recall@10`: Proportion of queries with expected chunk in top 10
- `nDCG@5`: Normalized DCG averaged across queries
- `median_latency`: P50 retrieval latency

**Backend Quality Correlation**:
Gate-7 enables analysis of whether routing decisions lead to good retrieval quality. Example: If `pinecone` routes have lower recall than `faiss` routes, this indicates routing rules may need tuning.

**Reports**:
- `reports/qa/step07_retrieval_eval.json` (includes per-backend stats)
- `reports/router/step07_retrieval_trace.jsonl` (includes routing context)

### 6. File Inventory

#### 6.1 Core Implementation
- `scripts/router_core.py` (185 lines) — Routing logic, reranking, config loading
- `scripts/qa_step04_router.py` (582 lines) — Gate-4 router validation

#### 6.2 Integration Files
- `scripts/langgraph_nodes.py:214-244` — Retriever node (primary integration point)
- `scripts/run_graph.py:343-379` — Original graph routing loop
- `scripts/qa_step07_retrieval_eval.py:375-404` — Evaluation with routing context
- `scripts/qa_step03_mcp.py:82-156` — MCP stub server backend handling

#### 6.3 Configuration
- `configs/router.heuristics.yaml` (43 lines) — Routing rules, persona bias, weights
- `configs/mcp.tools.yaml:2-5` — kb.search endpoint configuration

#### 6.4 State Management
- `scripts/langgraph_state.py:27` — `route_decisions` field definition
- `scripts/run_graph_langgraph.py:190-198` — Trace file output

#### 6.5 Reports
- `reports/qa/step04_router.json` — Gate-4 validation results
- `reports/qa/step04_router.md` — Gate-4 human-readable report
- `reports/router/step04_router_trace.jsonl` — Gate-4 routing decisions
- `reports/router/step07_retrieval_trace.jsonl` — Gate-7 retrieval traces with routing
- `outputs/<session-id>/router_trace.jsonl` — Per-session routing log

#### 6.6 Documentation
- `docs/architecture.md` — Routing system architecture
- `docs/configuration.md` — Router configuration details
- `docs/evaluation.md` — Gate-4 and Gate-7 metrics
- `docs/commands.md` — Gate-4 command reference
- `roadmap/issue/issue004-routing.md` — Routing feature planning

### 7. Code Patterns and Conventions

#### 7.1 Always Log Routing Decisions

**Pattern**: Every routing decision is stored with query, backend, and reason codes

**Example** (`langgraph_nodes.py:227-228`):
```python
backend, reasons = decide_backend(q, state["persona"], None)
route_decisions.append({"query": q, "backend": backend, "reasons": reasons})
```

**Rationale**: Enables debugging, audit trails, and correlation with retrieval quality

#### 7.2 Config-Driven Routing

**Pattern**: No hardcoded routing logic; all rules loaded from YAML

**Example** (`router_core.py:77-89`):
```python
cfg = load_router_config()
ql = (query or "").lower()
reasons: List[str] = []
# Rule-based
for rule in cfg.get("rules", []):
    cond = rule.get("if", {})
    kws = [str(x).lower() for x in cond.get("has_keywords", [])]
    if kws and any(kw in ql for kw in kws):
        then = rule.get("then", {})
        backend = str(then.get("backend") or "").strip() or "faiss"
        reason = str(then.get("reason") or "RULE_MATCH").strip()
        reasons.append(reason)
        return backend, reasons
```

**Rationale**: Allows rule tuning without code changes; easier A/B testing

#### 7.3 Fallback Order from Config

**Pattern**: Fallback order specified in config, not code

**Example** (`qa_step04_router.py:189,302`):
```python
fallback_order: List[str] = router_cfg.get("fallback_order", ["faiss", "weaviate", "pinecone"])
# ...
try_order = [b for b in fallback_order if b != backend] + [backend]
```

**Default**: `["faiss", "weaviate", "pinecone"]`

**Rationale**: Allows experimentation with different fallback strategies

#### 7.4 Reason Code Traceability

**Pattern**: Every decision path appends a unique reason code

**Reason Codes**:
- `"PR_QUERY"`, `"FILTER_MATCH"`, `"DEFINITION"` — Rule-based matches (custom per rule)
- `"PERSONA_BIAS"` — Persona-specific routing
- `"DEFAULT_SHORT_FAISS"` — Short/definitional heuristic
- `"DEFAULT_WEAVIATE"` — Longer query heuristic
- `"FALLBACK:" + backend` — Automatic retry
- `"DIVERSITY_MERGE:" + backends` — Diversity enhancement

**Rationale**: Enables debugging ("why was Pinecone chosen?") and analytics ("how often does fallback activate?")

#### 7.5 Deterministic Routing

**Pattern**: No randomness in routing decisions

**Implementation**: `decide_backend()` is a pure function with no side effects or random choices

**Rationale**: Reproducible behavior for testing and debugging

#### 7.6 Immediate Backend Usage

**Pattern**: Backend selection is used immediately in the same iteration

**Example** (`langgraph_nodes.py:227,231`):
```python
backend, reasons = decide_backend(q, state["persona"], None)  # Line 227
# ...
res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)  # Line 231
```

**Rationale**: Tight coupling between routing decision and retrieval reduces state complexity

#### 7.7 Always Rerank After Retrieval

**Pattern**: Every retrieval is followed by rerank() with docmeta

**Example** (`langgraph_nodes.py:234`):
```python
res = rerank(res, {k: type("DM", (), v) for k, v in docmeta.items()}, top_k=12, domain_cap=2)
```

**Rationale**: Ensures consistent quality (recency, diversity) regardless of backend

#### 7.8 DocMeta Wrapper Pattern

**Pattern**: Convert dict-based docmeta to object-like interface using dynamic class

**Example** (`langgraph_nodes.py:234`):
```python
{k: type("DM", (), v) for k, v in docmeta.items()}
```

Creates objects where `dm.publish_date` and `dm.source_domain` are accessible as attributes

**Rationale**: Allows rerank() to use attribute access (cleaner API) while maintaining dict-based metadata storage

### 8. Testing and Verification

#### 8.1 Gate-4 Test Coverage

**Evaluation Seed**: `data/interim/eval/salesforce_eval_seed.jsonl`

**Test Queries** (examples from fallback queries in `qa_step04_router.py:38-42`):
- `"Agentforce product announcement"` (vp_customer_experience persona)
- `"remaining performance obligation definition"` (cio persona)
- `"latest earnings results"` (vp_sales_ops persona)

**Coverage Requirements**:
- All 3 backends must be used (≥10% share or ≥1 route)
- All 3 personas must be tested
- Empty result rate ≤2%
- Auto-retry success rate ≥95%

#### 8.2 Expected Routing Behavior

**Test Case 1**: Press Query
- Query: `"latest earnings results"`
- Expected backend: `pinecone`
- Expected reason: `["PR_QUERY"]`
- Rationale: Keyword "earnings" matches rule

**Test Case 2**: Developer Query
- Query: `"API endpoint documentation"`
- Expected backend: `weaviate`
- Expected reason: `["FILTER_MATCH"]`
- Rationale: Keyword "api" and "endpoint" match rule

**Test Case 3**: Definitional Query
- Query: `"what is RPO"`
- Expected backend: `faiss`
- Expected reason: `["DEFINITION"]`
- Rationale: Keyword "what is" matches rule

**Test Case 4**: Persona Bias
- Query: `"product features"` (no keyword matches)
- Persona: `vp_sales_ops`
- Expected backend: `pinecone`
- Expected reason: `["PERSONA_BIAS"]`
- Rationale: No rule match, persona bias applies

**Test Case 5**: Heuristic Fallback (Short)
- Query: `"revenue growth"` (no keyword matches, no persona bias)
- Expected backend: `faiss`
- Expected reason: `["DEFAULT_SHORT_FAISS"]`
- Rationale: 2 words (≤4), falls through to heuristic

**Test Case 6**: Heuristic Fallback (Long)
- Query: `"tell me about the company's strategic initiatives for customer success"` (no matches)
- Expected backend: `weaviate`
- Expected reason: `["DEFAULT_WEAVIATE"]`
- Rationale: >4 words, no definitional keywords

#### 8.3 Offline Mode Testing

**Location**: `qa_step04_router.py:212-261`

When MCP service is unavailable, Gate-4 switches to offline mode:

```python
if use_offline:
    # Load chunks and build vectors deterministically
    def hash_vec(seed: str, d: int) -> List[float]:
        rnd = random.Random()
        h = 0
        for ch in seed:
            h = (h * 1315423911) ^ ord(ch)
            h &= 0xFFFFFFFFFFFFFFFF
        rnd.seed(h)
        vals = [rnd.uniform(-1.0, 1.0) for _ in range(d)]
        s2 = sum(v*v for v in vals) or 1.0
        inv = 1.0 / math.sqrt(s2)
        return [v*inv for v in vals]
```

**Behavior**:
- Uses deterministic hash-based embeddings (same as Gate-1)
- Performs local numpy-based L2 search
- Returns results in same format as online mode
- Latency measurements still captured

**Rationale**: Allows routing validation without external dependencies

### 9. Known Design Decisions

#### 9.1 Static Keyword Rules (No ML)

**Current Implementation**: Keyword matching with simple substring checks

**Rationale**:
- Deterministic and debuggable
- No training data required
- Fast execution (<1ms)
- Easy to tune via config

**Trade-off**: May miss semantic variations (e.g., "quarterly report" vs "earnings results")

#### 9.2 Hardcoded Persona Biases

**Current Implementation**: Fixed persona → backend mappings in config

**Rationale**:
- Aligns with known personas (VP Sales Ops focuses on financial data)
- Provides personalization without complex user modeling
- Easy to update via config

**Trade-off**: Does not adapt to individual user behavior over time

#### 9.3 Heuristic-Based Fallback

**Current Implementation**: Query length and definitional keywords determine default backend

**Rationale**:
- Simple catch-all for queries not matching rules/personas
- FAISS is fastest for simple queries (5-10ms latency)
- Weaviate supports more complex structured queries

**Trade-off**: Heuristic may not be optimal for all query types

#### 9.4 First-Match Rule Ordering

**Current Implementation**: First matching rule wins; subsequent rules not evaluated

**Rationale**:
- Predictable behavior
- Avoids rule conflicts
- Fast execution (early termination)

**Trade-off**: Rule order matters; must arrange from most specific to most general

#### 9.5 Backend-Agnostic Reranking

**Current Implementation**: Same rerank() formula applied to all backend results

**Rationale**:
- Consistent scoring across backends
- Enables fair quality comparison
- Improves diversity regardless of initial backend

**Trade-off**: May not leverage backend-specific features (e.g., Weaviate filters)

### 10. Execution and Usage

#### 10.1 Command-Line Execution

**Gate-4 Router Validation**:
```bash
conda run -n age python scripts/qa_step04_router.py
```

**Graph Execution with Routing**:
```bash
# LangGraph implementation
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session

# Original implementation
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Gate-7 Retrieval Evaluation** (includes routing metrics):
```bash
conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python scripts/qa_step07_retrieval_eval.py
```

#### 10.2 Environment Variables

**Gate-4**:
- `AG4_TRACE=1`: Enable detailed routing trace (not currently implemented)

**Gate-7**:
- `AG7_TRACE=1`: Enable retrieval trace with routing context
- `AG7_TRACE_TOPK=10`: Number of results to include in trace
- `AG7_TRACE_SUCCESSES=1`: Trace successful retrievals (not just failures)

#### 10.3 Output Files

**Per-Session Outputs** (`outputs/<session-id>/`):
- `router_trace.jsonl` — Routing decisions for this session
- `email.json` — Generated email (routing affects retrieved evidence)
- `trace.jsonl` — Full graph execution trace

**QA Reports** (`reports/qa/`):
- `step04_router.json` — Gate-4 machine-readable report
- `step04_router.md` — Gate-4 human-readable report
- `step07_retrieval_eval.json` — Gate-7 report (includes per-backend quality)

**Router Traces** (`reports/router/`):
- `step04_router_trace.jsonl` — Gate-4 routing decisions
- `step07_retrieval_trace.jsonl` — Gate-7 retrieval traces with routing

#### 10.4 Inspecting Routing Decisions

**View routing decisions for a session**:
```bash
cat outputs/my-session/router_trace.jsonl | jq .
```

**Example output**:
```json
{
  "timestamp": "2025-10-20T16:30:52-04:00",
  "query_text": "Agentforce product announcement",
  "decision_backend": "pinecone",
  "reason_codes": ["PR_QUERY"]
}
```

**Count backend distribution**:
```bash
cat reports/router/step04_router_trace.jsonl | jq -r .decision_backend | sort | uniq -c
```

**View reason code distribution**:
```bash
cat reports/router/step04_router_trace.jsonl | jq -r '.reason_codes[]' | sort | uniq -c
```

### 11. Architecture Diagrams

#### 11.1 Routing Decision Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   decide_backend(query, persona)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Load router.heuristics │
              │  ========================│
              │  • weights              │
              │  • persona_bias         │
              │  • rules                │
              │  • fallback_order       │
              └─────────────┬───────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │ Tier 1: Rule-Based      │
              │ ─────────────────────── │
              │ For each rule:          │
              │   if keyword in query:  │
              │     return rule.backend │
              └─────────┬───────────────┘
                        │ No match
                        ▼
              ┌─────────────────────────┐
              │ Tier 2: Persona Bias    │
              │ ─────────────────────── │
              │ if persona in map:      │
              │   return bias[persona]  │
              └─────────┬───────────────┘
                        │ No persona
                        ▼
              ┌─────────────────────────┐
              │ Tier 3: Heuristic       │
              │ ─────────────────────── │
              │ if short or definition: │
              │   return "faiss"        │
              │ else:                   │
              │   return "weaviate"     │
              └─────────┬───────────────┘
                        │
                        ▼
              ┌─────────────────────────┐
              │ Return: (backend,       │
              │          reason_codes)  │
              └─────────────────────────┘
```

#### 11.2 Retrieval Pipeline with Routing

```
┌────────────────┐
│  Planner Node  │
│  ─────────────│
│  Generates 5   │
│  queries       │
└───────┬────────┘
        │
        ▼
┌────────────────────────────────────────────────────┐
│           Retriever Node (routing happens here)    │
│  ──────────────────────────────────────────────── │
│  For each query:                                   │
│    1. backend, reasons = decide_backend(q, ...)   │
│    2. results = kb_search(backend, q, top_k=12)   │
│    3. results = rerank(results, docmeta, ...)     │
│    4. store: route_decisions.append(...)          │
└───────┬────────────────────────────────────────────┘
        │
        ▼
┌────────────────┐
│ Synthesizer    │
│ ────────────── │
│ Combines       │
│ retrieved      │
│ chunks         │
└───────┬────────┘
        │
        ▼
     (... downstream nodes ...)
```

#### 11.3 Backend Parameter Flow

```
decide_backend()          kb_search()           MCP Stub Server
     │                        │                      │
     │ ("pinecone", [...])    │                      │
     ├────────────────────────▶                      │
     │                        │ POST /invoke         │
     │                        │ {"method": "search"  │
     │                        │  "params": {         │
     │                        │    "backend":        │
     │                        │      "pinecone",     │
     │                        │    "query": "...",   │
     │                        │    "top_k": 12}}     │
     │                        ├──────────────────────▶
     │                        │                      │ Validate backend
     │                        │                      │ Simulate latency
     │                        │                      │ Perform search
     │                        │                      │
     │                        │ {"results": [...]}   │
     │                        ◀──────────────────────┤
     │ (results, latency, err)│                      │
     ◀────────────────────────┤                      │
     │                        │                      │
```

#### 11.4 Fallback and Diversity Merge

```
Primary Retrieval
     │
     ▼
┌──────────────┐
│ Empty result?│───No───▶ Continue
└──────┬───────┘
       │ Yes
       ▼
┌──────────────────┐
│ Automatic        │
│ Fallback         │
│ ───────────────  │
│ Try backends in  │
│ fallback_order   │
│ (skip primary)   │
└──────┬───────────┘
       │ Results found
       ▼
┌──────────────────┐
│ Check diversity  │
│ (unique domains) │
└──────┬───────────┘
       │ <3 domains
       ▼
┌──────────────────┐
│ Diversity Merge  │
│ ───────────────  │
│ Query other      │
│ backends (top_k  │
│ =30), dedupe,    │
│ rerank           │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Final reranked   │
│ results          │
└──────────────────┘
```

### 12. Related Documentation

**Part 3 - Multi-Index System** (`roadmap/part3-indexes/`):
- What indexes are created (FAISS, Weaviate, Pinecone)
- Index build process (Gate-2)
- Index characteristics and formats

**Part 5 - MCP Tools** (`roadmap/part5-mcp/`):
- kb.search tool interface
- MCP stub server implementation
- Service fallback modes (internal stub, external service, offline)

**Part 6 - LangGraph Orchestration** (`roadmap/part6-agents/`):
- 8-node graph topology
- Retriever node implementation
- State management and accumulation

**Part 7 - Quality Gates** (`roadmap/part7-quality/`):
- Gate-4 (routing) validation metrics and thresholds
- Gate-7 (retrieval) quality evaluation
- Status colors and go/no-go criteria

**Architecture Documentation** (`docs/architecture.md`):
- Multi-index routing design rationale
- Backend characteristics (FAISS: speed, Weaviate: filters, Pinecone: scale)
- Routing strategy overview

**Configuration Documentation** (`docs/configuration.md`):
- router.heuristics.yaml schema and tuning guidelines
- Weight adjustment recommendations
- Rule authoring best practices

**Evaluation Documentation** (`docs/evaluation.md`):
- Gate-4 metrics definitions
- Gate-7 retrieval quality metrics
- Per-backend quality analysis

## Code References

**Core Routing Logic**:
- `scripts/router_core.py:72-100` — `decide_backend()` three-tier decision tree
- `scripts/router_core.py:113-184` — `rerank()` multi-factor scoring with diversity
- `scripts/router_core.py:27-37` — `load_router_config()` with defaults

**Integration Points**:
- `scripts/langgraph_nodes.py:214-244` — Retriever node (primary integration)
- `scripts/langgraph_nodes.py:227` — `decide_backend()` invocation
- `scripts/langgraph_nodes.py:231` — Backend usage in `kb_search()`
- `scripts/langgraph_nodes.py:234` — Reranking with docmeta

**MCP Integration**:
- `scripts/langgraph_nodes.py:144-161` — `kb_search()` client
- `scripts/qa_step03_mcp.py:82-156` — MCP stub server handler
- `scripts/qa_step03_mcp.py:92` — Backend parameter extraction
- `scripts/qa_step03_mcp.py:96-97` — Backend validation
- `scripts/qa_step03_mcp.py:99-101` — Backend-specific latency simulation

**Advanced Features**:
- `scripts/qa_step04_router.py:296-316` — Automatic fallback logic
- `scripts/qa_step04_router.py:332-364` — Diversity merge strategy

**Quality Gates**:
- `scripts/qa_step04_router.py:411-459` — Gate-4 metrics and thresholds
- `scripts/qa_step07_retrieval_eval.py:375-404` — Gate-7 per-backend quality

**Configuration**:
- `configs/router.heuristics.yaml:1-43` — Routing rules, persona bias, weights
- `configs/router.heuristics.yaml:20-39` — Keyword rules
- `configs/router.heuristics.yaml:7-10` — Persona mappings
- `configs/mcp.tools.yaml:2-5` — kb.search endpoint

**State and Output**:
- `scripts/langgraph_state.py:27` — `route_decisions` field with `add` operator
- `scripts/run_graph_langgraph.py:190-198` — Router trace file generation

## Open Questions

**None** — This research is comprehensive and documents the routing system as it exists. All major components, integration points, configuration options, and quality validation mechanisms have been analyzed and documented.

If further investigation is needed, potential areas include:
- Performance profiling of routing decision overhead
- A/B testing framework for routing rule variations
- ML-based routing as an alternative to keyword rules
- Dynamic weight adjustment based on retrieval feedback

However, these would be **enhancements** rather than documentation of the current system.

---

**End of Research Document**
