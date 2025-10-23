# Part 6: LangGraph Agent System

**Research Date**: 2025-10-20 16:31:49 EDT
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture & Design](#2-architecture--design)
3. [File Inventory](#3-file-inventory)
4. [Core Components Deep Dive](#4-core-components-deep-dive)
5. [Configuration & Settings](#5-configuration--settings)
6. [Data Structures & Schemas](#6-data-structures--schemas)
7. [External Dependencies](#7-external-dependencies)
8. [Execution & Usage](#8-execution--usage)
9. [Code Patterns & Conventions](#9-code-patterns--conventions)
10. [Testing & Verification](#10-testing--verification)
11. [Known Issues & Limitations](#11-known-issues--limitations)
12. [References](#12-references)

---

## 1. Overview

### Purpose

The **LangGraph Agent System** is an 8-node orchestrated pipeline for generating persona-specific B2B outreach emails. It combines multi-index retrieval, LLM-based synthesis, and compliance checking to produce audit-ready emails with full traceability.

### The 8 Nodes

1. **Intake** - Validates company and persona inputs
2. **Planner** - Generates 5 persona-specific search queries
3. **Retriever** - Executes multi-backend vector search (FAISS/Weaviate/Pinecone)
4. **Synthesizer** - Converts retrieved chunks into insight candidates
5. **Consolidator** - LLM enhancement with persona relevance scoring
6. **Stylist** - Generates email draft from insight cards
7. **A2A** - Agent-to-agent compliance check and revision
8. **Assembler** - Final assembly with proof points and safety defaults

### Technology Stack

- **LangGraph** - Declarative state machine framework for agent orchestration
- **LangChain** - LLM integration (ChatOpenAI with ChatPromptTemplate)
- **OpenAI API** - gpt-5-mini model for consolidation and email generation
- **aiohttp** - Async HTTP client for MCP tool invocation
- **Python 3.13** - Primary runtime environment

### Key Features

- **Conditional Revision Loop** - A2A node can route back to Stylist for up to 2 regeneration rounds
- **Stateful Execution** - TypedDict-based state with field-level accumulation semantics
- **Async/Await** - All nodes are async functions for concurrent operations
- **MCP Tool Integration** - HTTP-based calls to kb.search and safety.check services
- **Dual-Format Outputs** - JSON artifacts + JSONL traces for auditability
- **LLM Retry Logic** - Defensive retry mechanism for ID hallucination recovery

---

## 2. Architecture & Design

### 2.1 Node Flow Diagram

```
┌─────────┐
│ Intake  │ Validates inputs (company, persona)
└────┬────┘
     ↓
┌─────────┐
│ Planner │ Generates 5 queries + persona keywords
└────┬────┘
     ↓
┌──────────┐
│Retriever │ Multi-backend search (FAISS/Weaviate/Pinecone)
└────┬─────┘
     ↓
┌────────────┐
│Synthesizer │ Chunk → Candidate conversion
└─────┬──────┘
     ↓
┌──────────────┐
│Consolidator  │ LLM enhancement (persona relevance)
└──────┬───────┘
     ↓
┌─────────┐
│ Stylist │ Email generation via LLM
└────┬────┘
     ↓
┌──────┐
│ A2A  │ Compliance check
└──┬───┘
   │
   ├─ [CRITICAL flags + rounds < 2] ──┐
   │                                  │
   │                                  ↓
   │                           ┌─────────┐
   │                           │ Stylist │ Revision Round 2
   │                           └────┬────┘
   │                                │
   │                                ↓
   │                           ┌──────┐
   │                           │ A2A  │ Re-check
   │                           └──┬───┘
   │                              │
   └── [No critical OR rounds≥2] ─┴────────┐
                                           ↓
                                    ┌───────────┐
                                    │ Assembler │ Add proof points
                                    └─────┬─────┘
                                          ↓
                                        [END]
```

### 2.2 State Flow Through Nodes

**State Initialization** (run_graph_langgraph.py:84-101)
- All 23 fields initialized with empty lists/dicts or input values
- Timestamp captured at workflow start

**Accumulation Pattern** (langgraph_state.py:24-27, 37, 42)
- 5 fields use `Annotated[List[...], add]` for accumulation:
  - `retrieved_chunks` - Grows across multi-query retrieval
  - `retrieval_logs` - One entry per query
  - `route_decisions` - Routing decision per query
  - `compliance_flags` - Flags from multiple A2A rounds
  - `errors` - Errors from any node

**Replacement Pattern**
- All other fields (18 total) replace values on each update
- Example: `insight_cards` from Consolidator replaces any previous value

**Partial Update Semantics**
- Nodes return dicts with only fields being updated
- LangGraph merges updates into state automatically
- Unmentioned fields remain unchanged

### 2.3 Conditional Revision Loop

**Decision Function** (run_graph_langgraph.py:25-33)

```python
def should_revise_email(state: AgentState) -> str:
    """Route to Stylist for revision or Assembler for final assembly."""
    critical_flags = [f for f in state.get("compliance_flags", []) if f.startswith("CRITICAL:")]
    rounds = state.get("a2a_rounds", 0)

    # If critical flags exist and we haven't exceeded 2 rounds, revise
    if critical_flags and rounds < 2:
        return "revise"
    return "assemble"
```

**Routing Logic**
- **"revise"** path: Routes back to Stylist for email regeneration
- **"assemble"** path: Proceeds to Assembler for final assembly
- **Max rounds**: 2 total (original + 1 retry)

**A2A Round Tracking** (langgraph_nodes.py:554)
- `a2a_rounds` incremented each time A2A node executes
- Conditional edge checks this counter to prevent infinite loops

---

## 3. File Inventory

### 3.1 Core Implementation Files

#### LangGraph Implementation (High Priority)

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/run_graph_langgraph.py` | 222 | Main LangGraph orchestrator (recommended implementation) |
| `scripts/langgraph_nodes.py` | 583 | 8 node implementations (Intake, Planner, Retriever, Synthesizer, Consolidator, Stylist, A2A, Assembler) |
| `scripts/langgraph_state.py` | 43 | State schema definition (AgentState TypedDict with 23 fields) |

#### Configuration

| File | Purpose |
|------|---------|
| `configs/langgraph.nodes.yaml` | Node topology and timeout budgets (2-10 seconds per node) |

#### Supporting Infrastructure

| File | Purpose |
|------|---------|
| `scripts/router_core.py` | Query routing logic (decide_backend, rerank) |
| `scripts/embedding_utils.py` | Embedding generation utilities |
| `scripts/common.py` | Shared utilities (now_iso, ensure_dir, load_yaml) |
| `scripts/tool_safety_check_server.py` | Safety check MCP tool server (local fallback) |

### 3.2 Sample Outputs (6 Session Directories)

Each output directory contains 5 files:

| File | Purpose |
|------|---------|
| `email.json` | Generated email artifact (subject, body, unsubscribe, company_info, proof_points) |
| `insights.json` | 5 insight cards with persona relevance fields |
| `compliance_report.json` | Compliance validation results (critical/warning flags, rounds) |
| `router_trace.jsonl` | Per-query routing decisions (JSONL format) |
| `timing.json` | Total runtime in milliseconds |

**Sample Sessions:**
- `outputs/official-cio/` - CIO persona run
- `outputs/official-vp-cx/` - VP Customer Experience run
- `outputs/official-vp-sales-ops/` - VP Sales Operations run
- `outputs/test-run-cio-2025-1/` - Test CIO run
- `outputs/test-run-vp-cx-2025-1/` - Test VP CX run
- `outputs/test-run-vp-sales-ops-2025-1/` - Test VP Sales Ops run

### 3.3 State Snapshots (6 Files)

Complete state dictionaries written to `state/` for replay/debugging:
- `state/session-official-cio.json`
- `state/session-official-vp-cx.json`
- `state/session-official-vp-sales-ops.json`
- `state/session-test-run-cio-2025-1.json`
- `state/session-test-run-vp-cx-2025-1.json`
- `state/session-test-run-vp-sales-ops-2025-1.json`

### 3.4 Quality Gate Scripts

| Script | Gate | Purpose |
|--------|------|---------|
| `scripts/qa_step05_graph.py` | Gate-5 | Graph validation (node connectivity, state schema) |
| `scripts/qa_step06_a2a.py` | Gate-6 | A2A validation (compliance checking) |
| `scripts/qa_step08_generation_eval.py` | Gate-8 | Generation quality evaluation |
| `scripts/qa_step08_debug.py` | - | Gate-8 debugging tool |

**Gate Reports:**
- `reports/qa/step05_graph.json` + `step05_graph.md`
- `reports/qa/step06_a2a.json` + `step06_a2a.md`
- `reports/qa/step08_generation_eval.json` + `step08_generation_eval.md`

### 3.5 Documentation

| File | Purpose |
|------|---------|
| `docs/architecture.md` | Full system architecture (includes LangGraph section) |
| `docs/langgraph-edge-cases.md` | Edge case handling documentation |
| `docs/langgraph/001-llm-id-hallucination.md` | LLM ID hallucination issue analysis |
| `docs/commands.md` | Command reference (includes graph execution) |

### 3.6 Persona and Template Files

| File | Purpose |
|------|---------|
| `icl/persona/vp_customer_experience.yaml` | VP CX persona definition |
| `icl/templates/email.yaml` | Email generation template |
| `configs/eval.prompts.yaml` | Evaluation prompt configurations (persona keywords) |

### 3.7 Visualization

| File | Purpose |
|------|---------|
| `scripts/visualize_graph.py` | Graph visualization utility |
| `reports/graphs/agent_workflow.mmd` | Mermaid diagram source |
| `reports/graphs/agent_workflow.png` | Rendered workflow diagram |

### 3.8 Original Implementation (For Comparison)

| File | Purpose |
|------|---------|
| `scripts/run_graph.py` | Original non-LangGraph implementation |

**Comparison:**
- Both implementations produce identical outputs in `outputs/<session-id>/`
- LangGraph version uses declarative graph construction
- Original uses procedural async function calls

### 3.9 Environment Files

| File | Purpose |
|------|---------|
| `envs/age.yaml` | Primary environment (Python 3.13, for graph execution) |
| `envs/ageFaiss.yaml` | FAISS-only environment (Python 3.12) |

---

## 4. Core Components Deep Dive

### 4.1 State Management (langgraph_state.py)

**File**: `scripts/langgraph_state.py` (43 lines)

#### Import Dependencies (Lines 3-4)

```python
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
```

- `TypedDict` - Provides static type checking for state schema
- `Annotated` - Enables field-level metadata for accumulator semantics
- `add` - Operator function used to define accumulation behavior

#### GraphState TypedDict (Lines 7-42)

**Complete Schema: 23 Fields Organized in 7 Categories**

**Input Fields (Lines 14-18)**
```python
company: str                  # Target company (e.g., "Salesforce")
persona: str                  # Recipient persona (e.g., "vp_customer_experience")
session_id: str               # Unique session identifier
timestamp: str                # ISO timestamp of workflow start
```

**Planning Fields (Lines 20-22)**
```python
queries: List[str]            # 5 persona-specific search queries
persona_keywords: List[str]   # Persona-relevant keywords
```

**Retrieval Fields (Lines 24-27)** - **ACCUMULATIVE**
```python
retrieved_chunks: Annotated[List[Dict[str, Any]], add]    # All chunks across queries
retrieval_logs: Annotated[List[Dict[str, Any]], add]      # Per-query logs
route_decisions: Annotated[List[Dict[str, Any]], add]     # Routing decisions
```

**Synthesis Fields (Lines 29-31)**
```python
insight_candidates: List[Dict[str, Any]]   # Raw candidates from chunks
insight_cards: List[Dict[str, Any]]        # Top 5 LLM-enhanced cards
```

**Generation Fields (Line 34)**
```python
email_draft: Dict[str, Any]                # Email content (subject, body, etc.)
```

**Compliance Fields (Lines 36-38)**
```python
compliance_flags: Annotated[List[str], add]  # ACCUMULATIVE - CRITICAL:/WARN: flags
a2a_rounds: int                              # A2A negotiation round counter
```

**Observability Fields (Lines 40-42)**
```python
metrics: Dict[str, Any]                   # Execution metrics
errors: Annotated[List[str], add]         # ACCUMULATIVE - Error messages
```

#### Accumulator Pattern (Lines 25-27, 37, 42)

**Mechanism:**
- Fields with `Annotated[List[...], add]` accumulate values across node executions
- LangGraph uses `operator.add` to extend lists rather than replace them
- Equivalent to `list1 + list2` when merging node returns

**Accumulated Fields (5 total):**
1. `retrieved_chunks` - Appends chunks from each query in Retriever
2. `retrieval_logs` - Appends log entry per query
3. `route_decisions` - Appends routing decision per query
4. `compliance_flags` - Appends flags from each A2A round
5. `errors` - Appends errors from any node

**Replaced Fields (18 total):**
- All fields without `Annotated[..., add]` are replaced on each update
- Nodes return new values, LangGraph overwrites previous state

#### Partial Update Semantics

**Example from retriever_node** (langgraph_nodes.py:240-244):
```python
return {
    "retrieved_chunks": retrieved_chunks,
    "retrieval_logs": retrieval_logs,
    "route_decisions": route_decisions,
}
```

**Behavior:**
- Node returns only 3 of 23 fields
- LangGraph appends to accumulated fields (chunks, logs, decisions)
- All other 20 fields remain unchanged from previous state

#### State Initialization (run_graph_langgraph.py:84-101)

```python
initial_state: AgentState = {
    "company": args.company,
    "persona": args.persona,
    "session_id": session_id,
    "timestamp": now_iso(),
    "queries": [],
    "persona_keywords": [],
    "retrieved_chunks": [],
    "retrieval_logs": [],
    "route_decisions": [],
    "insight_candidates": [],
    "insight_cards": [],
    "email_draft": {},
    "compliance_flags": [],
    "a2a_rounds": 0,
    "metrics": {"nodes_executed": [], "timings": {}},
    "errors": [],
}
```

**Initialization Requirements:**
- All 23 fields must be initialized before graph execution
- Lists initialized as `[]`, not `None`
- Dicts initialized as `{}`
- Counters start at `0`
- Prevents KeyError exceptions during node execution

---

### 4.2 Graph Builder (run_graph_langgraph.py)

**File**: `scripts/run_graph_langgraph.py` (222 lines)

#### 4.2.1 build_graph() Function (Lines 36-70)

**Purpose**: Constructs LangGraph StateGraph with 8 nodes and conditional routing

**Line 36-38: Initialization**
```python
def build_graph() -> StateGraph:
    """Construct LangGraph StateGraph with 8 nodes."""
    workflow = StateGraph(AgentState)
```

**Lines 40-48: Node Addition**
```python
# Add nodes
workflow.add_node("Intake", intake_node)
workflow.add_node("Planner", planner_node)
workflow.add_node("Retriever", retriever_node)
workflow.add_node("Synthesizer", synthesizer_node)
workflow.add_node("Consolidator", consolidator_node)
workflow.add_node("Stylist", stylist_node)
workflow.add_node("A2A", a2a_node)
workflow.add_node("Assembler", assembler_node)
```

- Node names are strings (Title case: "Intake", "Planner", etc.)
- Node functions imported from `langgraph_nodes.py`
- All nodes are async callables with signature: `async def node_name(state: AgentState) -> dict`

**Lines 50-57: Sequential Edges**
```python
# Add sequential edges
workflow.set_entry_point("Intake")
workflow.add_edge("Intake", "Planner")
workflow.add_edge("Planner", "Retriever")
workflow.add_edge("Retriever", "Synthesizer")
workflow.add_edge("Synthesizer", "Consolidator")
workflow.add_edge("Consolidator", "Stylist")
workflow.add_edge("Stylist", "A2A")
```

- `set_entry_point("Intake")` designates starting node
- `add_edge(from, to)` creates directed edge between nodes
- Linear flow from Intake → Planner → ... → A2A

**Lines 59-68: Conditional Edge for A2A Revision**
```python
# Phase 3: Conditional edge for A2A revision loop
workflow.add_conditional_edges(
    "A2A",
    should_revise_email,
    {
        "revise": "Stylist",      # Re-generate email (Round 2)
        "assemble": "Assembler",  # Proceed to final assembly
    }
)
workflow.add_edge("Assembler", END)
```

- Source node: `"A2A"`
- Decision function: `should_revise_email` (lines 25-33)
- Route mapping:
  - `"revise"`: loops back to Stylist for regeneration
  - `"assemble"`: proceeds to Assembler for final output
- `END` constant from LangGraph marks terminal state

**Line 70: Return**
```python
return workflow
```

- Returns uncomplied StateGraph instance
- Will be compiled via `workflow.compile()` at execution time

#### 4.2.2 Conditional Routing (Lines 25-33)

```python
def should_revise_email(state: AgentState) -> str:
    """Route to Stylist for revision or Assembler for final assembly."""
    critical_flags = [f for f in state.get("compliance_flags", []) if f.startswith("CRITICAL:")]
    rounds = state.get("a2a_rounds", 0)

    # If critical flags exist and we haven't exceeded 2 rounds, revise
    if critical_flags and rounds < 2:
        return "revise"
    return "assemble"
```

**Decision Logic:**
1. Extract critical flags (prefix: "CRITICAL:")
2. Get current round count from state
3. If critical flags exist AND rounds < 2: return "revise"
4. Otherwise: return "assemble"

**Maximum Rounds:**
- 2 total rounds enforced (original + 1 retry)
- Prevents infinite revision loops

#### 4.2.3 Main Execution Flow (Lines 73-205)

**Graph Compilation (Lines 75-82)**
```python
workflow = build_graph()
app = workflow.compile()

session_id = args.session_id or uuid.uuid4().hex[:12]
out_dir = os.path.join("outputs", session_id)
state_dir = "state"
ensure_dir(out_dir)
ensure_dir(state_dir)
```

- `workflow.compile()` creates executable app
- Session ID from args or generated from UUID
- Output directories created before execution

**Graph Invocation (Lines 103-108)**
```python
t0 = time.perf_counter()

# Invoke graph
result = await app.ainvoke(initial_state)

total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
```

- Async execution via `app.ainvoke(initial_state)`
- Total runtime measured with `time.perf_counter()`
- Returns final merged state

**Post-Processing (Lines 110-167)**

Enforces Gate-8 compliance thresholds:

1. **Word Count Limit (160 words)** (Lines 135-142)
   - Iterative truncation if body > 160 words
   - Max 3 truncation iterations
   - `_shorten_body()` function: truncates lines, limits bullets

2. **Readability Grade (<15)** (Lines 144-167)
   - Flesch-Kincaid grade level calculation
   - Trusts A2A output if no critical flags
   - Safeguarded truncation if A2A also flagged issues
   - Stops if grade gets worse (no infinite degradation)

**Output Generation (Lines 169-202)**

All outputs written to `outputs/<session-id>/`:

```python
# insights.json (lines 170-171)
with open(os.path.join(out_dir, "insights.json"), "w", encoding="utf-8") as f:
    json.dump(result["insight_cards"], f, ensure_ascii=False, indent=2)

# email.json (lines 172-173)
with open(os.path.join(out_dir, "email.json"), "w", encoding="utf-8") as f:
    json.dump(result["email_draft"], f, ensure_ascii=False, indent=2)

# timing.json (lines 174-175)
with open(os.path.join(out_dir, "timing.json"), "w", encoding="utf-8") as f:
    json.dump({"total_runtime_ms": total_ms}, f, indent=2)

# compliance_report.json (lines 177-188)
critical = [f for f in result.get("compliance_flags", []) if f.startswith("CRITICAL:")]
warnings = [f for f in result.get("compliance_flags", []) if f.startswith("WARN:")]
report = {"rounds": result.get("a2a_rounds", 0), "flags": {"critical": critical, "warning": warnings}}
with open(os.path.join(out_dir, "compliance_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# router_trace.jsonl (lines 190-198)
with open(os.path.join(out_dir, "router_trace.jsonl"), "a", encoding="utf-8") as f:
    for dec in result.get("route_decisions", []):
        entry = {
            "timestamp": result["timestamp"],
            "query": dec.get("query"),
            "backend": dec.get("backend"),
            "reasons": dec.get("reasons"),
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# state/session-<id>.json (lines 200-202)
with open(os.path.join(state_dir, f"session-{session_id}.json"), "w", encoding="utf-8") as f:
    json.dump(dict(result), f, ensure_ascii=False, indent=2)
```

**Output Summary (Line 204)**
```python
print(json.dumps({"session_id": session_id, "out_dir": out_dir, "total_ms": total_ms}, indent=2))
```

#### 4.2.4 Command-Line Arguments (Lines 208-213)

```python
def parse_args():
    p = argparse.ArgumentParser(description="Run LangGraph agent workflow")
    p.add_argument("--company", default="Salesforce")
    p.add_argument("--persona", default="vp_customer_experience")
    p.add_argument("--session-id", default=None)
    return p.parse_args()
```

**Defaults:**
- Company: "Salesforce"
- Persona: "vp_customer_experience"
- Session ID: None (generates UUID if omitted)

#### 4.2.5 Entry Point (Lines 216-222)

```python
def main():
    args = parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()
```

- Synchronous `main()` parses args and launches async main
- `asyncio.run(main_async(args))` for async execution

---

### 4.3 Node Implementations (langgraph_nodes.py)

**File**: `scripts/langgraph_nodes.py` (583 lines)

All 8 nodes are async functions with signature:
```python
async def node_name(state: AgentState) -> dict
```

#### 4.3.1 Node 1: intake_node (Lines 166-171)

**Purpose**: Input validation

**State Fields Read:**
- `state.get("company")`
- `state.get("persona")`

**State Fields Updated:**
- `errors`

**Logic:**
```python
async def intake_node(state: AgentState) -> dict:
    """Validate company and persona inputs (run_graph.py lines 186-190)."""
    errors = []
    if not state.get("company") or not state.get("persona"):
        errors.append("missing company/persona")
    return {"errors": errors}
```

- Checks both company and persona are present
- Returns empty errors list if valid
- Appends "missing company/persona" if invalid

**Return Value:**
```python
{"errors": List[str]}
```

#### 4.3.2 Node 2: planner_node (Lines 174-211)

**Purpose**: Generate 5 persona-specific queries and persona keywords

**State Fields Read:**
- `state["persona"]`

**State Fields Updated:**
- `queries`
- `persona_keywords`

**External Calls:**
- File I/O: `data/interim/eval/salesforce_eval_seed.jsonl`
- File I/O: `configs/eval.prompts.yaml`

**Logic:**
1. **Load Seed Queries** (Lines 176-196)
   - Reads eval seed JSONL file
   - Filters by matching persona
   - Deduplicates query_text via set
   - Limits to 5 queries

2. **Fallback Queries** (Lines 198-205)
   - If no seed queries found, uses hardcoded list:
     - "Agentforce product announcement"
     - "latest earnings results"
     - "remaining performance obligation definition"
     - "customer experience AI"
     - "Data Cloud recent updates"

3. **Load Persona Keywords** (Lines 208-209)
   - Reads from `eval.prompts.yaml`
   - Extracts keywords for specified persona

**Return Value:**
```python
{
    "queries": List[str],           # 5 persona-specific queries
    "persona_keywords": List[str]   # Persona keywords
}
```

#### 4.3.3 Node 3: retriever_node (Lines 214-244)

**Purpose**: Execute vector search via MCP kb.search for all queries

**State Fields Read:**
- `state["queries"]`
- `state["persona"]`

**State Fields Updated:**
- `retrieved_chunks` (ACCUMULATIVE)
- `retrieval_logs` (ACCUMULATIVE)
- `route_decisions` (ACCUMULATIVE)

**External Calls:**
- `load_mcp_map()` - MCP tool configuration
- `load_router_config()` - Routing rules
- `load_doc_meta()` - Document metadata
- `decide_backend(query, persona, None)` - Route query to backend
- `kb_search(session, backend, query, 12, tools_cfg)` - Async MCP call
- `rerank(results, docmeta, top_k=12, domain_cap=2)` - Re-rank results

**Logic:**
1. **Initialization** (Lines 216-222)
   - Load configurations
   - Initialize empty result lists
   - Create aiohttp session with TCPConnector (limit_per_host=8)

2. **Per-Query Loop** (Lines 226-238)
   - Decide backend via router (line 227)
   - Record routing decision (line 228)
   - Execute MCP kb.search with top_k=12 (line 231)
   - Re-rank with domain_cap=2 (line 234)
   - Log and extend chunks (lines 237-238)

**Return Value:**
```python
{
    "retrieved_chunks": List[Dict[str, Any]],  # All chunks from all queries
    "retrieval_logs": List[Dict[str, Any]],    # One log per query
    "route_decisions": List[Dict[str, Any]]    # One decision per query
}
```

**Chunk Structure:**
- text, metadata, score, doc_id, source, chunk_id, url, date, snippet

#### 4.3.4 Node 4: synthesizer_node (Lines 247-277)

**Purpose**: Convert retrieved chunks into insight candidates

**State Fields Read:**
- `state["retrieved_chunks"]`

**State Fields Updated:**
- `insight_candidates`

**External Calls:**
- `load_doc_meta()` - Document metadata lookup

**Logic:**
1. **Deduplication** (Lines 250-257)
   - Uses `seen_cids` set to track chunk IDs
   - Skips duplicate chunks

2. **Metadata Extraction** (Lines 258-263)
   - Looks up parent document via doc_id
   - Extracts title, URL, publish date

3. **Candidate Construction** (Lines 264-275)
   - Creates candidate object with fields:
     - `id`: chunk_id
     - `title`: truncated to 120 chars
     - `summary`: snippet or text, truncated to 320 chars
     - `url`, `date`, `evidence_snippet`, `confidence` (0.7)
     - `source_domain`, `doc_id`

**Return Value:**
```python
{
    "insight_candidates": List[Dict[str, Any]]  # One per unique chunk
}
```

**Candidate Structure:**
```python
{
    "id": str,
    "title": str,
    "summary": str,
    "url": str,
    "date": str,
    "evidence_snippet": str,
    "confidence": float,
    "source_domain": str,
    "doc_id": str
}
```

#### 4.3.5 Node 5: consolidator_node (Lines 280-447)

**Purpose**: Select 5 candidates with domain diversity, LLM-enhance with persona relevance

**State Fields Read:**
- `state["insight_candidates"]`
- `state["company"]`
- `state["persona"]`
- `state.get("persona_keywords")`
- `state.get("session_id", "unknown")`

**State Fields Updated:**
- `insight_cards`

**External Calls:**
- `load_doc_meta()` - For synthetic card generation
- `ChatOpenAI(temperature=0.3, model="gpt-5-mini")` - LLM instance
- `llm.ainvoke()` - Async LLM call
- `log_llm_retry_event()` - Retry event logging

**Logic:**

**Phase 1: Card Selection (Lines 284-302)**
1. Initialize empty cards list and used_domains dict
2. Iterate candidates, accept if:
   - Domain count is 0, OR
   - Total domains < 4
3. Fill to 5 cards from remaining candidates

**Phase 2: Domain Diversity Enforcement (Lines 304-369)**
1. Count unique domains in selected cards
2. If < 4 unique domains:
   - Add cards from unused domains (lines 307-325)
   - If still < 4, synthesize cards from docmeta (lines 327-368):
     - Preferred doctypes: press, product, dev_docs, help_docs, wiki
     - Create synth card with id format: `synth::{doc_id}::card`
     - Replace duplicate-domain cards

**Phase 3: Pre-LLM Validation (Lines 370-372)**
```python
if len(cards) != 5:
    raise AssertionError(f"consolidator_node: Expected exactly 5 cards before LLM, got {len(cards)}")
```

**Phase 4: LLM Enhancement with Retry (Lines 374-438)**

**Retry Setup** (Lines 376-381):
```python
input_ids = [c["id"] for c in cards]
synth_count = sum(1 for cid in input_ids if cid.startswith("synth::"))
MAX_ATTEMPTS = 3
cards_final = []

for attempt in range(1, MAX_ATTEMPTS + 1):
```

**LLM Call** (Lines 384-398):
```python
llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
consolidator_tmpl = ChatPromptTemplate.from_messages([
    ("system", CONSOLIDATOR_SYSTEM_PROMPT),
    ("user", CONSOLIDATOR_USER_PROMPT),
])

consolidator_vars = {
    "company": state["company"],
    "persona": state["persona"],
    "persona_keywords": ", ".join(state.get("persona_keywords") or []),
    "candidates_json": json.dumps(cards, ensure_ascii=False),
}

resp = await llm.ainvoke(consolidator_tmpl.format_messages(**consolidator_vars))
cards_llm = json.loads(resp.content)
```

**Merge with KeyError Handling** (Lines 401-413):
```python
by_id = {c["id"]: c for c in cards}
cards_final = []
for item in cards_llm:
    base = by_id[item["id"]]  # KeyError if LLM hallucinated ID
    base["title"] = item.get("title") or base["title"]
    base["summary"] = item.get("summary") or base["summary"]
    base["persona_relevance"] = item.get("persona_relevance")
    base["metric_impact"] = item.get("metric_impact")
    base["action_suggestion"] = item.get("action_suggestion")
    cards_final.append(base)

# Success - break out of retry loop
break
```

**Retry on KeyError** (Lines 415-438):
```python
except KeyError as e:
    hallucinated_id = str(e).strip("'")

    if attempt < MAX_ATTEMPTS:
        # Log retry event
        log_llm_retry_event(
            session_id=state.get("session_id", "unknown"),
            attempt=attempt,
            error_id=hallucinated_id,
            input_ids=input_ids,
            synth_count=synth_count
        )

        # Warn user
        print(f"⚠️  LLM retry {attempt}/{MAX_ATTEMPTS}: ID mismatch detected (hallucinated: {hallucinated_id[:80]}...), retrying...", file=sys.stderr)

        continue  # Retry
    else:
        # All attempts exhausted
        raise AssertionError(
            f"consolidator_node: LLM ID hallucination after {MAX_ATTEMPTS} retries. "
            f"Hallucinated ID: {hallucinated_id}. Expected one of: {input_ids[:3]}..."
        ) from e
```

**Phase 5: Post-LLM Validation (Lines 440-445)**
```python
if len(cards_final) != 5:
    raise AssertionError(f"consolidator_node: Expected 5 cards after LLM, got {len(cards_final)}")
```

**Return Value:**
```python
{
    "insight_cards": List[Dict[str, Any]]  # 5 LLM-enhanced cards
}
```

**Enhanced Card Structure:**
```python
{
    "id": str,
    "title": str,                  # LLM-improved
    "summary": str,                # LLM-improved
    "url": str,
    "date": str,
    "evidence_snippet": str,
    "confidence": float,
    "source_domain": str,
    "doc_id": str,
    "persona_relevance": {         # LLM-added
        "why_it_matters": str,
        "relevance_score": int,    # 1-5
        "keywords_hit": List[str]
    },
    "metric_impact": {             # LLM-added
        "metric": str,
        "direction": str,          # "increase" or "decrease"
        "magnitude": str           # "low", "med", "high"
    },
    "action_suggestion": str       # LLM-added
}
```

#### 4.3.6 Node 6: stylist_node (Lines 450-468)

**Purpose**: Generate email copy via LLM

**State Fields Read:**
- `state["company"]`
- `state["persona"]`
- `state.get("persona_keywords")`
- `state["insight_cards"]`

**State Fields Updated:**
- `email_draft`

**External Calls:**
- `ChatOpenAI(temperature=0.3, model="gpt-5-mini")` - LLM instance
- `llm.ainvoke()` - Async LLM call

**Logic:**
```python
async def stylist_node(state: AgentState) -> dict:
    """Generate email copy via LLM (run_graph.py lines 589-607)."""
    llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
    stylist_tmpl = ChatPromptTemplate.from_messages([
        ("system", STYLIST_SYSTEM_PROMPT),
        ("user", STYLIST_USER_PROMPT),
    ])

    stylist_vars = {
        "company": state["company"],
        "persona": state["persona"],
        "persona_keywords": ", ".join(state.get("persona_keywords") or []),
        "insight_cards": json.dumps(state["insight_cards"], ensure_ascii=False),
    }

    resp = await llm.ainvoke(stylist_tmpl.format_messages(**stylist_vars))
    email_fields = json.loads(resp.content)

    return {"email_draft": email_fields}
```

**Prompt Variables:**
- Company name
- Persona identifier
- Persona keywords (comma-separated string)
- Insight cards (JSON string)

**Return Value:**
```python
{
    "email_draft": {
        "subject": str,
        "body": str,
        "unsubscribe_block": str,
        "company_info_block": str
    }
}
```

#### 4.3.7 Node 7: a2a_node (Lines 471-567)

**Purpose**: Compliance check and email revision

**State Fields Read:**
- `state["email_draft"]`
- `state["insight_cards"]`
- `state.get("a2a_rounds", 0)`

**State Fields Updated:**
- `compliance_flags` (ACCUMULATIVE)
- `a2a_rounds`
- `email_draft` (if critical flags + round 1)

**External Calls:**
- `call_safety()` helper: MCP safety.check service
- `revise_email()` helper: Programmatic fixes

**Helper: call_safety (Lines 476-494)**

**Purpose**: Invoke MCP safety.check service with fallback

```python
async def call_safety(email_fields: Dict[str, Any], cards: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    tools = load_mcp_map()
    base = tools.get("safety.check") or {}
    host = base.get("host", "127.0.0.1")
    port = int(base.get("port", 7805))
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "moderate", "params": {"text": email_fields.get("body"), "email_fields": email_fields, "insight_cards": cards}}
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
                j = await resp.json()
                f = (j.get("flags") or {})
                return f.get("critical", []) or [], f.get("warning", []) or []
    except Exception:
        # Fallback: local checks
        spec = load_yaml(os.path.join("configs", "compliance.template.yaml"))
        from tool_safety_check_server import check_email
        c, w = check_email(email_fields, state["insight_cards"], spec)
        return c, w
```

**Behavior:**
- HTTP POST to safety.check service (default port 7805)
- Method: "moderate"
- Returns tuple: (critical_flags, warning_flags)
- On exception: falls back to local `check_email()` function

**Helper: revise_email (Lines 496-542)**

**Purpose**: Apply programmatic fixes for known violations

**Critical Flag Fixes** (Lines 501-510):
- `MISSING_UNSUBSCRIBE`: Add default unsubscribe block
- `MISSING_COMPANY_INFO`: Add default company info block
- `PROHIBITED_PHRASE`: Replace banned words
- `UNCITED_CLAIM`: Append reference from first insight card

**Warning Flag Fixes** (Lines 513-540):
- `EXCESS_LENGTH`: Keep header + top 3 bullets, truncate to 18 words per bullet
- `READABILITY`: Shorten sentences to 10-12 words per line

**Main A2A Logic (Lines 544-567)**

```python
async def a2a_node(state: AgentState) -> dict:
    """A2A compliance negotiation (run_graph.py lines 609-692)."""
    current_round = state.get("a2a_rounds", 0)

    # Safety check
    critical, warnings = await call_safety(state["email_draft"], state["insight_cards"])

    # Format flags
    all_flags = [f"CRITICAL:{f}" for f in critical] + [f"WARN:{w}" for w in warnings]

    # Increment round
    new_round = current_round + 1

    # Revise if critical flags and this is round 1
    revised_draft = state["email_draft"]
    if critical and current_round < 1:
        revised_draft = revise_email(state["email_draft"], critical, warnings, state["insight_cards"])

    return {
        "compliance_flags": all_flags,
        "a2a_rounds": new_round,
        "email_draft": revised_draft,
    }
```

**Revision Logic:**
- If critical flags present AND current_round < 1: apply `revise_email()` fixes
- Otherwise: return draft unchanged
- Always increment round counter
- Always append flags to state (accumulated)

**Return Value:**
```python
{
    "compliance_flags": List[str],  # Prefixed with "CRITICAL:" or "WARN:"
    "a2a_rounds": int,              # Incremented counter
    "email_draft": Dict[str, Any]   # Potentially revised
}
```

#### 4.3.8 Node 8: assembler_node (Lines 570-582)

**Purpose**: Final assembly with proof points and safety defaults

**State Fields Read:**
- `state.get("email_draft")`
- `state.get("insight_cards")`

**State Fields Updated:**
- `email_draft`

**Logic:**
```python
async def assembler_node(state: AgentState) -> dict:
    """Final assembly with proof points (run_graph.py lines 694-714)."""
    email = dict(state.get("email_draft") or {})

    # Safety defaults
    email.setdefault("unsubscribe_block", "You can unsubscribe at any time by replying 'unsubscribe'.")
    email.setdefault("company_info_block", "Sent by ACME AI, 123 Market St, San Francisco, CA.")

    # Proof points
    cards = state.get("insight_cards") or []
    email["proof_points"] = [{"id": c["id"], "title": c["title"]} for c in cards[:5]]

    return {"email_draft": email}
```

**Safety Defaults:**
- Unsubscribe block if missing
- Company info block if missing

**Proof Points:**
- Extracts id + title from first 5 insight cards
- Attached to email as `proof_points` array

**Return Value:**
```python
{
    "email_draft": {
        "subject": str,
        "body": str,
        "unsubscribe_block": str,
        "company_info_block": str,
        "proof_points": [
            {"id": str, "title": str},
            ...
        ]
    }
}
```

---

### 4.4 Supporting Infrastructure

#### 4.4.1 Helper Functions (langgraph_nodes.py)

**load_yaml (Lines 103-109)**
```python
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
```
- Silent exception handling
- Returns empty dict on any error

**log_llm_retry_event (Lines 112-130)**
```python
def log_llm_retry_event(session_id: str, attempt: int, error_id: str, input_ids: List[str], synth_count: int):
    """Log LLM retry event to JSONL for debugging (see docs/langgraph/001-llm-id-hallucination.md)."""
    from common import ensure_dir
    event = {
        "timestamp": now_iso(),
        "session_id": session_id,
        "node": "consolidator",
        "attempt": attempt,
        "max_attempts": 3,
        "error_type": "KeyError",
        "hallucinated_ids": [error_id],
        "expected_ids": input_ids,
        "synth_card_count": synth_count,
        "retry_reason": "LLM_ID_MISMATCH"
    }
    log_path = os.path.join("logs", "langgraph", "llm_retry_events.jsonl")
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
```
- Logs to `logs/langgraph/llm_retry_events.jsonl`
- JSONL format (one event per line)
- Reference to design doc in docstring

**load_doc_meta (Lines 133-141)**
```python
def load_doc_meta() -> Dict[str, Dict[str, Any]]:
    """Load all normalized document metadata."""
    dm = {}
    norm_dir = os.path.join("data", "interim", "normalized")
    if not os.path.isdir(norm_dir):
        return dm
    for fn in os.listdir(norm_dir):
        if fn.endswith(".json"):
            with open(os.path.join(norm_dir, fn), "r", encoding="utf-8") as f:
                doc = json.load(f)
                dm[doc["doc_id"]] = doc
    return dm
```
- Scans `data/interim/normalized/*.json`
- Returns dict mapping doc_id → metadata

**kb_search (Lines 144-161)**
```python
async def kb_search(session: aiohttp.ClientSession, backend: str, query: str, top_k: int, tools_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, str]:
    """MCP kb.search client (copied from run_graph.py lines 114-131)."""
    base = tools_cfg.get("kb.search") or {}
    host = base.get("host", "127.0.0.1")
    port = int(base.get("port", 7801))
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "search", "params": {"query": query, "backend": backend, "top_k": int(top_k)}}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
            status = resp.status
            j = await resp.json()
            if status >= 400:
                return [], (time.perf_counter() - t0) * 1000.0, (j.get("error") or {}).get("code")
            res = j.get("results") or []
            return res, (time.perf_counter() - t0) * 1000.0, None
    except Exception as e:
        return [], (time.perf_counter() - t0) * 1000.0, "NetworkError"
```
- Async MCP client for kb.search
- Returns tuple: (results, latency_ms, error_code)
- Latency measured even on errors

#### 4.4.2 Prompt Templates (langgraph_nodes.py:22-100)

**CONSOLIDATOR_SYSTEM_PROMPT (Lines 22-30)**
```
You are a B2B research analyst consolidating RAG chunks into persona-aware insight cards.
- Preserve factual grounding strictly to the provided candidates.
- Do NOT invent IDs or sources.
- Write concise, executive-friendly copy.
- Tailor emphasis to the persona:
  * vp_customer_experience: NPS, CSAT, contact center, omnichannel, agent productivity, self-service, first contact resolution
  * cio: data integration, governance, security, TCO, platform, APIs, real-time
  * vp_sales_ops: pipeline, forecast accuracy, win rate, productivity, automation
```

**CONSOLIDATOR_USER_PROMPT (Lines 32-51)**
```
Company: {company}
Persona: {persona}
Persona keywords to weave in naturally (only when relevant): {persona_keywords}

From these candidates (JSON), select exactly the same items (DO NOT add/remove), and for each:
- Improve 'title' (≤ 12 words) and 'summary' (1–2 sentences) with persona relevance.
- Keep 'id' exactly as given to preserve traceability.
- You may rephrase 'summary' but stay within the evidence.
- Add fields:
  persona_relevance: {{ "why_it_matters": str, "relevance_score": 1-5, "keywords_hit": [str] }}
  metric_impact: {{ "metric": str, "direction": "increase|decrease", "magnitude": "low|med|high" }}
  action_suggestion: str (1 actionable step for the recipient)

Return ONLY a JSON array of 5 objects with fields:
[id, title, summary, persona_relevance, metric_impact, action_suggestion]
(The original URL/date/doc_id/source_domain/evidence_snippet/confidence are preserved elsewhere via id.)

Candidates JSON:
{candidates_json}
```

**STYLIST_SYSTEM_PROMPT (Lines 53-79)**
```
You are a B2B email copywriter creating persona-tailored outreach from insight cards.

Critical rules:
- DO NOT open with job title statements ("As a CIO, ..." or "In your role as VP Customer Experience, ...").
- NEVER use "I hope this message finds you well" or similar cliches.
- Start with a natural, insight-driven opening tied to the company context.
- Body should be 100-140 words.
- Use 3-5 short bullets highlighting insights.
- Professional, warm tone (avoid hype or urgency).
- Include `unsubscribe_block` and `company_info_block` in output.

Persona voice:
- vp_customer_experience: Focus on customer satisfaction, contact center efficiency, agent productivity, omnichannel experience.
- cio: Focus on data integration, security, governance, platform capabilities, APIs, real-time processing.
- vp_sales_ops: Focus on pipeline visibility, forecast accuracy, win rate, productivity, automation.

Output must be valid JSON with exactly these fields:
{
  "subject": str,
  "body": str,
  "unsubscribe_block": str,
  "company_info_block": str
}
```

**STYLIST_USER_PROMPT (Lines 81-100)**
```
Company: {company}
Persona: {persona}
Persona keywords to weave in naturally: {persona_keywords}

Generate an email using these insight cards (JSON):
{insight_cards}

Requirements:
- Natural opening that references the company context (NO job title statements).
- Body: 100-140 words with 3-5 bullet points highlighting key insights.
- Optional: 1-2 sentence next steps (soft CTA).
- Subject line: Concise, relevant, no hype.
- Unsubscribe block: Professional opt-out language.
- Company info: Sender details.

Return ONLY the JSON object with: subject, body, unsubscribe_block, company_info_block.
```

---

## 5. Configuration & Settings

### 5.1 Node Configuration (configs/langgraph.nodes.yaml)

**File**: `configs/langgraph.nodes.yaml` (18 lines)

#### Structure

```yaml
nodes:
  - Intake
  - Planner
  - Retriever
  - Synthesizer
  - Consolidator
  - Stylist
  - A2A
  - Assembler

timeouts_ms:
  Intake: 2000
  Planner: 2000
  Retriever: 10000
  Synthesizer: 5000
  Consolidator: 3000
  Stylist: 3000
  A2A: 3000
  Assembler: 2000
```

#### Node List
- 8 nodes in sequential order
- Node names match labels in `build_graph()`

#### Timeout Budgets

| Node | Timeout (ms) | Timeout (sec) | Purpose |
|------|--------------|---------------|---------|
| Intake | 2000 | 2 | Fast validation |
| Planner | 2000 | 2 | File I/O only |
| Retriever | 10000 | 10 | Multi-backend search (slowest) |
| Synthesizer | 5000 | 5 | Chunk processing |
| Consolidator | 3000 | 3 | LLM call |
| Stylist | 3000 | 3 | LLM call |
| A2A | 3000 | 3 | Safety check + revision |
| Assembler | 2000 | 2 | Fast assembly |

**Total Budget**: 32 seconds for single-pass pipeline (without A2A revision loops)

#### Usage

**Loading** (run_graph.py:26, 146-149):
```python
NODES_CONF = os.path.join("configs", "langgraph.nodes.yaml")
nodes_cfg = load_yaml(NODES_CONF)
nodes = nodes_cfg.get("nodes", [
    "Intake","Planner","Retriever","Synthesizer","Consolidator","Stylist","A2A","Assembler"
])
```

**Timeout Conversion** (qa_step08_debug.py:901-908):
```python
self.nodes_config = load_yaml(NODES_CONFIG)
self.node_timeouts = {
    k: v/1000 for k, v in
    self.nodes_config.get("timeouts_ms", {}).items()
}
```

### 5.2 LLM Configuration

**NOT in config files** - Hardcoded in scripts

**Consolidator LLM** (langgraph_nodes.py:384):
```python
llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
```

**Stylist LLM** (langgraph_nodes.py:452):
```python
llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
```

**Parameters:**
- Model: `gpt-5-mini` (OpenAI multimodal model from 2025)
- Temperature: `0.3` (consistent but slightly varied outputs)
- No max_tokens specified (uses model default)

### 5.3 MCP Tool Configuration (configs/mcp.tools.yaml)

**Referenced but not shown in detail**

**Loading** (langgraph_nodes.py:216, 477):
```python
tools_cfg = load_mcp_map()
base = tools.get("safety.check") or {}
```

**Default Endpoints:**
- `kb.search`: `127.0.0.1:7801`
- `safety.check`: `127.0.0.1:7805`

### 5.4 Router Configuration (configs/router.heuristics.yaml)

**Referenced in retriever_node** (langgraph_nodes.py:217):
```python
router_cfg = load_router_config()
```

**Used by** `decide_backend()` function from `router_core.py`

**Backends:**
- FAISS (general knowledge)
- Weaviate (dev docs)
- Pinecone (press/financial)

### 5.5 Persona Configuration (configs/eval.prompts.yaml)

**Referenced in planner_node** (langgraph_nodes.py:208-209):
```python
eval_cfg = load_yaml(os.path.join("configs", "eval.prompts.yaml"))
persona_keywords = (eval_cfg.get("personas", {}) or {}).get(state["persona"], [])
```

**Persona Keywords:**
- `vp_customer_experience`: NPS, CSAT, contact center, omnichannel, agent productivity, self-service, first contact resolution
- `cio`: data integration, governance, security, TCO, platform, APIs, real-time
- `vp_sales_ops`: pipeline, forecast accuracy, win rate, productivity, automation

---

## 6. Data Structures & Schemas

### 6.1 AgentState TypedDict (Complete Schema)

**Defined in**: `scripts/langgraph_state.py:7-43`

```python
class AgentState(TypedDict):
    """
    Shared state across all agent nodes.

    Fields with Annotated[..., add] accumulate across node invocations.
    Fields without annotation are replaced on each update.
    """
    # Input fields
    company: str
    persona: str
    session_id: str
    timestamp: str

    # Planning fields
    queries: List[str]
    persona_keywords: List[str]

    # Retrieval fields (accumulate)
    retrieved_chunks: Annotated[List[Dict[str, Any]], add]
    retrieval_logs: Annotated[List[Dict[str, Any]], add]
    route_decisions: Annotated[List[Dict[str, Any]], add]

    # Synthesis fields
    insight_candidates: List[Dict[str, Any]]
    insight_cards: List[Dict[str, Any]]

    # Generation fields
    email_draft: Dict[str, Any]

    # Compliance fields
    compliance_flags: Annotated[List[str], add]
    a2a_rounds: int

    # Observability fields
    metrics: Dict[str, Any]
    errors: Annotated[List[str], add]
```

### 6.2 Insight Candidate Structure

**Generated by**: synthesizer_node (langgraph_nodes.py:264-275)

```python
{
    "id": str,                    # chunk_id (e.g., "chunk::12345")
    "title": str,                 # Truncated to 120 chars
    "summary": str,               # Snippet or text, truncated to 320 chars
    "url": str,                   # Source URL
    "date": str,                  # Publish date (ISO format or empty)
    "evidence_snippet": str,      # Text excerpt
    "confidence": float,          # Always 0.7
    "source_domain": str,         # Domain (e.g., "salesforce.com")
    "doc_id": str                 # Parent document ID
}
```

### 6.3 Insight Card Structure

**Generated by**: consolidator_node (langgraph_nodes.py:401-410)

```python
{
    "id": str,
    "title": str,                 # LLM-improved (≤ 12 words)
    "summary": str,               # LLM-improved (1-2 sentences)
    "url": str,
    "date": str,
    "evidence_snippet": str,
    "confidence": float,
    "source_domain": str,
    "doc_id": str,

    # LLM-added fields:
    "persona_relevance": {
        "why_it_matters": str,
        "relevance_score": int,    # 1-5
        "keywords_hit": List[str]
    },
    "metric_impact": {
        "metric": str,
        "direction": str,          # "increase" or "decrease"
        "magnitude": str           # "low", "med", "high"
    },
    "action_suggestion": str       # 1 actionable step
}
```

### 6.4 Email Draft Structure

**Generated by**: stylist_node (langgraph_nodes.py:466)
**Revised by**: a2a_node (langgraph_nodes.py:561)
**Finalized by**: assembler_node (langgraph_nodes.py:580)

```python
{
    "subject": str,                # Email subject line
    "body": str,                   # Email body text (100-140 words)
    "unsubscribe_block": str,      # Opt-out text
    "company_info_block": str,     # Sender details
    "proof_points": [              # Added by assembler_node
        {
            "id": str,
            "title": str
        },
        ...  # Up to 5 proof points
    ]
}
```

### 6.5 Compliance Report Structure

**Generated in**: run_graph_langgraph.py:177-188

```python
{
    "rounds": int,                 # Number of A2A negotiation rounds
    "flags": {
        "critical": List[str],     # Critical violations (e.g., "MISSING_UNSUBSCRIBE")
        "warning": List[str]       # Warnings (e.g., "EXCESS_LENGTH")
    }
}
```

### 6.6 Routing Decision Structure

**Generated by**: retriever_node (langgraph_nodes.py:228)

```python
{
    "query": str,                  # Query text
    "backend": str,                # Selected backend ("faiss", "weaviate", "pinecone")
    "reasons": List[str]           # Reason codes for decision
}
```

### 6.7 Retrieval Log Structure

**Generated by**: retriever_node (langgraph_nodes.py:237)

```python
{
    "query": str,                  # Query text
    "results": List[Dict[str, Any]]  # Top 10 chunks
}
```

### 6.8 Router Trace Entry (JSONL)

**Written in**: run_graph_langgraph.py:192-198

```python
{
    "timestamp": str,              # ISO timestamp
    "query": str,                  # Query text
    "backend": str,                # Selected backend
    "reasons": List[str]           # Reason codes
}
```

### 6.9 LLM Retry Event (JSONL)

**Logged by**: log_llm_retry_event (langgraph_nodes.py:112-130)

```python
{
    "timestamp": str,              # ISO timestamp
    "session_id": str,             # Session identifier
    "node": str,                   # Always "consolidator"
    "attempt": int,                # Retry attempt number (1-3)
    "max_attempts": int,           # Always 3
    "error_type": str,             # Always "KeyError"
    "hallucinated_ids": List[str], # IDs invented by LLM
    "expected_ids": List[str],     # Valid IDs from input
    "synth_card_count": int,       # Number of synthetic cards
    "retry_reason": str            # Always "LLM_ID_MISMATCH"
}
```

---

## 7. External Dependencies

### 7.1 LangGraph & LangChain

**LangGraph** - State machine framework
- `langgraph.graph.StateGraph` - Graph construction
- `langgraph.graph.END` - Terminal state constant

**LangChain** - LLM integration
- `langchain_openai.ChatOpenAI` - OpenAI client
- `langchain_core.prompts.ChatPromptTemplate` - Prompt templating

**Usage Example** (langgraph_nodes.py:384-398):
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
consolidator_tmpl = ChatPromptTemplate.from_messages([
    ("system", CONSOLIDATOR_SYSTEM_PROMPT),
    ("user", CONSOLIDATOR_USER_PROMPT),
])

resp = await llm.ainvoke(consolidator_tmpl.format_messages(**consolidator_vars))
cards_llm = json.loads(resp.content)
```

### 7.2 OpenAI API

**Model**: `gpt-5-mini` (multimodal model from 2025)
**API Key**: Loaded from environment or `.env` file

**Used in**:
- consolidator_node (langgraph_nodes.py:384)
- stylist_node (langgraph_nodes.py:452)

**Temperature**: `0.3` (consistent outputs)

### 7.3 MCP Tools

**kb.search** - Knowledge base search
- Default endpoint: `127.0.0.1:7801`
- Method: `"search"`
- Params: `{"query": str, "backend": str, "top_k": int}`
- Returns: `{"results": List[Dict]}`

**safety.check** - Compliance validation
- Default endpoint: `127.0.0.1:7805`
- Method: `"moderate"`
- Params: `{"text": str, "email_fields": dict, "insight_cards": list}`
- Returns: `{"flags": {"critical": [...], "warning": [...]}}`

**Fallback**: Local implementation in `tool_safety_check_server.py`

### 7.4 aiohttp

**Purpose**: Async HTTP client for MCP tool invocation

**Usage Example** (langgraph_nodes.py:224-238):
```python
import aiohttp

connector = aiohttp.TCPConnector(limit_per_host=8)
async with aiohttp.ClientSession(connector=connector) as session:
    for q in state["queries"]:
        res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)
```

**Configuration**:
- TCPConnector with `limit_per_host=8` for connection pooling
- Session reused across loop iterations
- Timeout from config (default 2000ms)

### 7.5 Python Standard Library

**asyncio** - Async runtime
```python
import asyncio
asyncio.run(main_async(args))
```

**time** - Performance measurement
```python
import time
t0 = time.perf_counter()
latency_ms = (time.perf_counter() - t0) * 1000.0
```

**json** - JSON parsing/serialization
```python
import json
cards_llm = json.loads(resp.content)
json.dump(result, f, ensure_ascii=False, indent=2)
```

**argparse** - Command-line parsing
```python
import argparse
p = argparse.ArgumentParser(description="Run LangGraph agent workflow")
```

**uuid** - Session ID generation
```python
import uuid
session_id = args.session_id or uuid.uuid4().hex[:12]
```

### 7.6 Internal Dependencies

**router_core.py**:
- `load_mcp_map()` - Load MCP tool configuration
- `load_router_config()` - Load routing rules
- `decide_backend(query, persona, None)` - Route query to backend
- `rerank(results, docmeta, top_k, domain_cap)` - Re-rank results

**embedding_utils.py**:
- `embed_text()` - Generate embeddings (not used in LangGraph nodes directly)

**common.py**:
- `now_iso()` - ISO timestamp generation
- `ensure_dir()` - Directory creation
- `load_yaml()` - YAML file loading (also defined inline in langgraph_nodes.py)

**tool_safety_check_server.py**:
- `check_email()` - Local fallback for safety.check

---

## 8. Execution & Usage

### 8.1 Running the Graph

**Recommended Command** (LangGraph implementation):
```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Original Command** (for comparison):
```bash
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id my-session
```

**Both produce identical outputs in** `outputs/<session-id>/`

### 8.2 Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--company` | "Salesforce" | Target company name |
| `--persona` | "vp_customer_experience" | Recipient persona identifier |
| `--session-id` | None | Unique session ID (generates 12-char UUID if omitted) |

**Valid Personas**:
- `vp_customer_experience` - VP Customer Experience
- `cio` - Chief Information Officer
- `vp_sales_ops` - VP Sales Operations

### 8.3 Example Sessions

**Official Runs** (production quality):
```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona cio \
  --session-id official-cio
```

Output: `outputs/official-cio/`

```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id official-vp-cx
```

Output: `outputs/official-vp-cx/`

```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_sales_ops \
  --session-id official-vp-sales-ops
```

Output: `outputs/official-vp-sales-ops/`

### 8.4 Output Locations

**Directory Structure**:
```
outputs/<session-id>/
├── email.json              # Final email artifact
├── insights.json           # 5 insight cards
├── compliance_report.json  # Compliance validation
├── router_trace.jsonl      # Routing decisions (JSONL)
└── timing.json             # Runtime metrics

state/
└── session-<session-id>.json  # Complete state snapshot
```

**Inspecting Outputs**:
```bash
# View email
cat outputs/my-session/email.json | jq .

# View insights
cat outputs/my-session/insights.json | jq .

# View compliance report
cat outputs/my-session/compliance_report.json | jq .

# View routing trace
cat outputs/my-session/router_trace.jsonl | jq .

# View complete state
cat state/session-my-session.json | jq .
```

### 8.5 Output File Schemas

**email.json**:
```json
{
  "subject": "...",
  "body": "...",
  "unsubscribe_block": "...",
  "company_info_block": "...",
  "proof_points": [
    {"id": "...", "title": "..."},
    ...
  ]
}
```

**insights.json**:
```json
[
  {
    "id": "...",
    "title": "...",
    "summary": "...",
    "url": "...",
    "date": "...",
    "evidence_snippet": "...",
    "confidence": 0.7,
    "source_domain": "...",
    "doc_id": "...",
    "persona_relevance": {
      "why_it_matters": "...",
      "relevance_score": 4,
      "keywords_hit": [...]
    },
    "metric_impact": {
      "metric": "...",
      "direction": "increase",
      "magnitude": "high"
    },
    "action_suggestion": "..."
  },
  ...  // 5 total
]
```

**compliance_report.json**:
```json
{
  "rounds": 1,
  "flags": {
    "critical": [],
    "warning": ["EXCESS_LENGTH"]
  }
}
```

**timing.json**:
```json
{
  "total_runtime_ms": 12345.67
}
```

**router_trace.jsonl** (one JSON object per line):
```json
{"timestamp": "2025-10-20T16:31:49Z", "query": "...", "backend": "faiss", "reasons": ["..."] }
{"timestamp": "2025-10-20T16:31:49Z", "query": "...", "backend": "weaviate", "reasons": ["..."] }
...
```

### 8.6 Environment Setup

**Conda Environment**: `age` (Python 3.13)

**Create Environment**:
```bash
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
```

**Activate Environment**:
```bash
conda activate age
```

**Verify Installation**:
```bash
python --version    # Should show Python 3.13.x
pip list | grep langgraph
pip list | grep langchain
```

**API Key Setup**:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 8.7 Debugging

**Enable Debug Logging**:
```bash
# Set environment variable
export AG7_DEBUG=1

# Run graph
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id debug-session
```

**View LLM Retry Events**:
```bash
cat logs/langgraph/llm_retry_events.jsonl | jq .
```

**Replay from State Snapshot**:
```python
import json

# Load state
with open("state/session-my-session.json", "r") as f:
    state = json.load(f)

# Inspect specific fields
print(state["queries"])
print(state["compliance_flags"])
print(state["a2a_rounds"])
```

---

## 9. Code Patterns & Conventions

### 9.1 Async Function Patterns

**Pattern**: All nodes are async functions

**Basic Node Signature**:
```python
async def node_name(state: AgentState) -> dict:
    """Docstring with original implementation reference."""
    # Node logic
    return {"field": value}
```

**Example** (intake_node at lines 166-171):
```python
async def intake_node(state: AgentState) -> dict:
    """Validate company and persona inputs (run_graph.py lines 186-190)."""
    errors = []
    if not state.get("company") or not state.get("persona"):
        errors.append("missing company/persona")
    return {"errors": errors}
```

**Key Aspects**:
- `async def` keyword for async execution
- `state: AgentState` parameter (TypedDict)
- Return `dict` with partial state updates
- Docstring references original implementation location

### 9.2 Partial State Return Patterns

**Pattern**: Nodes return only fields being updated

**Single Field Update**:
```python
return {"errors": errors}
```

**Multiple Field Updates**:
```python
return {
    "queries": queries,
    "persona_keywords": persona_keywords
}
```

**Accumulated Fields**:
```python
return {
    "retrieved_chunks": retrieved_chunks,
    "retrieval_logs": retrieval_logs,
    "route_decisions": route_decisions,
}
```

**State Merging**:
- LangGraph merges returned dict into state automatically
- Accumulated fields (`Annotated[List, add]`) append values
- Non-accumulated fields replace previous values
- Unmentioned fields remain unchanged

### 9.3 Logging and Timing Patterns

**Pattern 1: Latency Measurement with perf_counter**

```python
import time

t0 = time.perf_counter()
# ... operation ...
latency_ms = (time.perf_counter() - t0) * 1000.0
```

**Example** (kb_search at lines 144-161):
```python
async def kb_search(...) -> Tuple[List[Dict[str, Any]], float, str]:
    t0 = time.perf_counter()
    try:
        # ... HTTP call ...
        return results, (time.perf_counter() - t0) * 1000.0, None
    except Exception as e:
        return [], (time.perf_counter() - t0) * 1000.0, "NetworkError"
```

**Key Aspects**:
- Start timer before operation
- Calculate latency in all exit paths (success, error, exception)
- Multiply by 1000.0 to convert seconds to milliseconds
- Return latency as part of tuple

**Pattern 2: Event Logging to JSONL**

```python
def log_event(data: dict):
    event = {
        "timestamp": now_iso(),
        **data
    }
    log_path = "logs/events.jsonl"
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
```

**Example** (log_llm_retry_event at lines 112-130):
```python
def log_llm_retry_event(session_id: str, attempt: int, error_id: str, input_ids: List[str], synth_count: int):
    event = {
        "timestamp": now_iso(),
        "session_id": session_id,
        "node": "consolidator",
        "attempt": attempt,
        "max_attempts": 3,
        "error_type": "KeyError",
        "hallucinated_ids": [error_id],
        "expected_ids": input_ids,
        "synth_card_count": synth_count,
        "retry_reason": "LLM_ID_MISMATCH"
    }
    log_path = os.path.join("logs", "langgraph", "llm_retry_events.jsonl")
    ensure_dir(os.path.dirname(log_path))
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
```

**Key Aspects**:
- Structured event as dict with ISO timestamp
- Append mode (`"a"`) for JSONL format
- One JSON object per line
- Directory creation with `ensure_dir()`

### 9.4 Error Handling Patterns

**Pattern 1: Silent Exception Suppression with Fallback**

```python
def load_config(path: str) -> dict:
    try:
        # ... load operation ...
        return config
    except Exception:
        return {}  # Safe fallback
```

**Example** (load_yaml at lines 103-109):
```python
def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
```

**Key Aspects**:
- Broad `except Exception:` catches all errors
- Returns empty dict as safe fallback
- No logging or warning
- Used when file absence is expected

**Pattern 2: Retry Loop with Specific Exception Handling**

```python
MAX_ATTEMPTS = 3
for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        # ... operation ...
        break  # Success - exit loop
    except SpecificError as e:
        if attempt < MAX_ATTEMPTS:
            log_retry(attempt)
            continue  # Retry
        else:
            raise FinalError(...) from e  # Exhausted
```

**Example** (consolidator_node at lines 374-438):
```python
MAX_ATTEMPTS = 3
for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        # LLM enhancement
        resp = await llm.ainvoke(...)
        cards_llm = json.loads(resp.content)

        # Merge (KeyError occurs here if ID mismatch)
        by_id = {c["id"]: c for c in cards}
        cards_final = []
        for item in cards_llm:
            base = by_id[item["id"]]  # KeyError if hallucinated
            # ... merge fields ...
            cards_final.append(base)

        break  # Success

    except KeyError as e:
        hallucinated_id = str(e).strip("'")

        if attempt < MAX_ATTEMPTS:
            log_llm_retry_event(...)
            print(f"⚠️  LLM retry {attempt}/{MAX_ATTEMPTS}: ...", file=sys.stderr)
            continue  # Retry
        else:
            raise AssertionError(f"LLM ID hallucination after {MAX_ATTEMPTS} retries...") from e
```

**Key Aspects**:
- Fixed retry count (MAX_ATTEMPTS)
- Specific exception type (`KeyError`)
- Event logging on each retry
- User warning to stderr
- Re-raise as different exception type after exhaustion
- `from e` preserves exception chain

**Pattern 3: Fallback with Local Implementation**

```python
async def call_remote_service():
    try:
        # ... remote call ...
        return response
    except Exception:
        # Fallback to local
        from local_module import local_function
        return local_function()
```

**Example** (call_safety at lines 476-494):
```python
async def call_safety(email_fields: Dict[str, Any], cards: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    # ... MCP service config ...
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(url, json=payload, timeout=...) as resp:
                j = await resp.json()
                f = (j.get("flags") or {})
                return f.get("critical", []) or [], f.get("warning", []) or []
    except Exception:
        # Fallback: local checks
        spec = load_yaml(os.path.join("configs", "compliance.template.yaml"))
        from tool_safety_check_server import check_email
        c, w = check_email(email_fields, state["insight_cards"], spec)
        return c, w
```

**Key Aspects**:
- Try network service first
- Broad `except Exception:` for any failure
- Import fallback implementation only when needed
- Silent downgrade (no warning logged)
- Returns consistent tuple format

**Pattern 4: Validation with AssertionError**

```python
if not invariant_holds:
    raise AssertionError(f"node_name: Expected X, got Y")
```

**Example** (consolidator_node at lines 370-372):
```python
if len(cards) != 5:
    raise AssertionError(f"consolidator_node: Expected exactly 5 cards before LLM, got {len(cards)}")
```

**Key Aspects**:
- `AssertionError` for invariant violations
- Node name in error message for traceability
- Descriptive message with actual vs expected values

### 9.5 LLM Integration Patterns

**Pattern**: Inline LLM instantiation with ChatPromptTemplate

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", USER_PROMPT),
])

variables = {
    "field1": value1,
    "field2": value2,
}

resp = await llm.ainvoke(template.format_messages(**variables))
result = json.loads(resp.content)
```

**Example** (consolidator_node at lines 384-398):
```python
llm = ChatOpenAI(temperature=0.3, model="gpt-5-mini")
consolidator_tmpl = ChatPromptTemplate.from_messages([
    ("system", CONSOLIDATOR_SYSTEM_PROMPT),
    ("user", CONSOLIDATOR_USER_PROMPT),
])

consolidator_vars = {
    "company": state["company"],
    "persona": state["persona"],
    "persona_keywords": ", ".join(state.get("persona_keywords") or []),
    "candidates_json": json.dumps(cards, ensure_ascii=False),
}

resp = await llm.ainvoke(consolidator_tmpl.format_messages(**consolidator_vars))
cards_llm = json.loads(resp.content)
```

**Key Aspects**:
- LLM instantiated inline (not singleton)
- Fixed temperature (0.3) for consistency
- Model name from CLAUDE.md spec
- ChatPromptTemplate with system + user messages
- Template variables passed as dict
- `format_messages()` to build prompt
- `await llm.ainvoke()` for async call
- Response content parsed as JSON

### 9.6 MCP Tool Integration Patterns

**Pattern 1: Tool Configuration Loading**

```python
tools_cfg = load_mcp_map()
base = tools_cfg.get("tool.name") or {}
host = base.get("host", "127.0.0.1")
port = int(base.get("port", 7801))
```

**Pattern 2: HTTP Client Call with Latency Tracking**

```python
async def call_mcp_tool(session, params):
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "search", "params": params}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=2.0) as resp:
            j = await resp.json()
            if resp.status >= 400:
                return [], (time.perf_counter() - t0) * 1000.0, j.get("error", {}).get("code")
            return j.get("results", []), (time.perf_counter() - t0) * 1000.0, None
    except Exception:
        return [], (time.perf_counter() - t0) * 1000.0, "NetworkError"
```

**Example** (kb_search at lines 144-161):
```python
async def kb_search(session: aiohttp.ClientSession, backend: str, query: str, top_k: int, tools_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, str]:
    base = tools_cfg.get("kb.search") or {}
    host = base.get("host", "127.0.0.1")
    port = int(base.get("port", 7801))
    url = f"http://{host}:{port}/invoke"
    payload = {"method": "search", "params": {"query": query, "backend": backend, "top_k": int(top_k)}}
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=payload, timeout=base.get("timeout_ms", 2000) / 1000.0) as resp:
            status = resp.status
            j = await resp.json()
            if status >= 400:
                return [], (time.perf_counter() - t0) * 1000.0, (j.get("error") or {}).get("code")
            res = j.get("results") or []
            return res, (time.perf_counter() - t0) * 1000.0, None
    except Exception as e:
        return [], (time.perf_counter() - t0) * 1000.0, "NetworkError"
```

**Key Aspects**:
- Session passed as parameter (not created in helper)
- Tool name keys into config dict
- Host/port from config with defaults
- URL pattern: `http://{host}:{port}/invoke`
- Payload structure: `{"method": str, "params": dict}`
- Timeout from config converted from ms to seconds
- Returns tuple: (results, latency_ms, error_code)

**Pattern 3: Session Reuse**

```python
connector = aiohttp.TCPConnector(limit_per_host=8)
async with aiohttp.ClientSession(connector=connector) as session:
    for item in items:
        result = await call_mcp_tool(session, item)
```

**Example** (retriever_node at lines 224-238):
```python
connector = aiohttp.TCPConnector(limit_per_host=8)
async with aiohttp.ClientSession(connector=connector) as session:
    for q in state["queries"]:
        backend, reasons = decide_backend(q, state["persona"], None)
        res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)
```

**Key Aspects**:
- Custom TCPConnector with connection pooling
- Session created once, reused across loop
- Session passed to helper function
- Proper cleanup via `async with` context manager

### 9.7 State Schema Patterns

**Pattern**: TypedDict with Annotated accumulation

```python
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add

class StateSchema(TypedDict):
    # Replaced fields
    field1: str
    field2: List[str]

    # Accumulated fields
    field3: Annotated[List[Dict[str, Any]], add]
```

**Example** (AgentState at lines 7-43):
```python
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add

class AgentState(TypedDict):
    """Shared state across all agent nodes."""

    # Replaced fields
    company: str
    persona: str
    queries: List[str]
    insight_cards: List[Dict[str, Any]]

    # Accumulated fields
    retrieved_chunks: Annotated[List[Dict[str, Any]], add]
    retrieval_logs: Annotated[List[Dict[str, Any]], add]
    compliance_flags: Annotated[List[str], add]
```

**Key Aspects**:
- Inherits from `TypedDict`
- Docstring explains behavior
- `Annotated[List[...], add]` for accumulated fields
- Plain types for replaced fields
- `operator.add` imported at module level

### 9.8 Graph Construction Patterns

**Pattern**: Declarative graph builder function

```python
from langgraph.graph import StateGraph, END

def build_graph() -> StateGraph:
    workflow = StateGraph(StateSchema)

    # Add nodes
    workflow.add_node("Node1", node1_func)
    workflow.add_node("Node2", node2_func)

    # Add edges
    workflow.set_entry_point("Node1")
    workflow.add_edge("Node1", "Node2")
    workflow.add_conditional_edges(
        "Node2",
        decision_func,
        {
            "route1": "Node3",
            "route2": END
        }
    )

    return workflow
```

**Example** (build_graph at lines 36-70):
```python
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("Intake", intake_node)
    workflow.add_node("Planner", planner_node)
    # ... 6 more nodes ...

    workflow.set_entry_point("Intake")
    workflow.add_edge("Intake", "Planner")
    # ... sequential edges ...

    workflow.add_conditional_edges(
        "A2A",
        should_revise_email,
        {
            "revise": "Stylist",
            "assemble": "Assembler",
        }
    )
    workflow.add_edge("Assembler", END)

    return workflow
```

**Key Aspects**:
- Function returns `StateGraph` instance
- `StateGraph(StateSchema)` with state schema
- Node names as strings
- Node functions passed directly
- Sequential edges with `add_edge(from, to)`
- Conditional edge with decision function + route map
- `END` constant for terminal state

---

## 10. Testing & Verification

### 10.1 Gate-5: Graph Validation

**Script**: `scripts/qa_step05_graph.py`

**Purpose**: Validate graph structure and node connectivity

**What It Checks**:
- All 8 nodes are present
- Entry point is "Intake"
- All nodes are reachable from entry point
- State schema has all 23 required fields
- No orphaned nodes (nodes with no incoming edges)

**Run Command**:
```bash
conda run -n age python scripts/qa_step05_graph.py
```

**Output**:
- `reports/qa/step05_graph.json` - Machine-readable results
- `reports/qa/step05_graph.md` - Human-readable report

**Pass Criteria**:
- All nodes present: ✓
- All edges valid: ✓
- State schema complete: ✓

### 10.2 Gate-6: A2A Validation

**Script**: `scripts/qa_step06_a2a.py`

**Purpose**: Validate agent-to-agent compliance checking

**What It Checks**:
- A2A node detects critical violations
- Revision logic applies correct fixes
- Max rounds enforced (2 total)
- Compliance flags formatted correctly
- Local fallback works when service unavailable

**Run Command**:
```bash
conda run -n age python scripts/qa_step06_a2a.py
```

**Output**:
- `reports/qa/step06_a2a.json` - Machine-readable results
- `reports/qa/step06_a2a.md` - Human-readable report

**Pass Criteria**:
- Critical flag detection: ✓
- Revision logic works: ✓
- Max rounds respected: ✓
- Fallback operational: ✓

### 10.3 Gate-8: Generation Evaluation

**Script**: `scripts/qa_step08_generation_eval.py`

**Purpose**: Evaluate email generation quality across 10 runs per persona

**What It Checks**:
- **Structural Validity**: All required fields present (subject, body, etc.)
- **Compliance**: No critical flags
- **Length**: 100-160 words (enforced by post-processing)
- **Readability**: Flesch-Kincaid grade ≤ 15
- **Persona Keywords**: Avg ≥ 2.0 keyword hits per email

**Run Command**:
```bash
conda run -n age python scripts/qa_step08_generation_eval.py
```

**Output**:
- `reports/qa/step08_generation_eval.json` - Detailed metrics
- `reports/qa/step08_generation_eval.md` - Summary report

**Pass Criteria**:
- `structural_pass_rate` == 1.0 (all runs valid)
- `critical_flags_total` == 0 (no compliance violations)
- `length_readability_pass_runs` ≥ 9 (out of 10)
- `persona_keyword_hits_avg` ≥ 2.0

**Debug Tool**: `scripts/qa_step08_debug.py` for interactive debugging

### 10.4 End-to-End Tests

**Test Cases**:
1. **Happy Path**: Valid inputs → 5 insights → compliant email
2. **Missing Input**: Empty company/persona → error in Intake
3. **No Seed Queries**: Missing eval seed → fallback queries used
4. **MCP Service Down**: kb.search unavailable → empty results (graceful degradation)
5. **LLM ID Hallucination**: Consolidator retry logic → max 3 attempts
6. **Critical Compliance Flags**: A2A revision → Stylist regeneration
7. **Max Rounds Exceeded**: 2 rounds → proceed to Assembler despite flags

**Manual Testing**:
```bash
# Test 1: Happy path
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id test-happy

# Test 2: Different persona
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona cio \
  --session-id test-cio

# Test 3: Custom session ID
conda run -n age python scripts/run_graph_langgraph.py \
  --company "Acme Corp" \
  --persona vp_sales_ops \
  --session-id custom-test
```

### 10.5 Visualization

**Script**: `scripts/visualize_graph.py`

**Purpose**: Generate visual representation of graph topology

**Output**: `reports/graphs/agent_workflow.png`

**Run Command**:
```bash
python scripts/visualize_graph.py
```

---

## 11. Known Issues & Limitations

### 11.1 LLM ID Hallucination

**Issue**: Consolidator LLM occasionally invents insight card IDs not in input

**Root Cause**: LLM generates IDs that don't match input cards, causing KeyError during merge

**Mitigation**:
- Retry loop (max 3 attempts) at langgraph_nodes.py:374-438
- Log event to `logs/langgraph/llm_retry_events.jsonl`
- Warn user to stderr on each retry

**Documentation**: `docs/langgraph/001-llm-id-hallucination.md`

**Status**: **Mitigated** via defensive retry mechanism

**Frequency**: Rare (< 1% of runs with synthetic cards)

### 11.2 Max Insights Hardcoded

**Issue**: Insight card count hardcoded to 5

**Location**:
- Planner: 5 queries (langgraph_nodes.py:195)
- Consolidator: 5 cards (langgraph_nodes.py:290-302)
- Assembler: 5 proof points (langgraph_nodes.py:580)

**Limitation**: Cannot configure different insight counts per persona

**Workaround**: Edit hardcoded values and rerun

**Status**: **By Design** - 5 insights is optimal for email length (100-160 words)

### 11.3 A2A Timeout Issues

**Issue**: Safety.check service may timeout under load

**Symptoms**:
- NetworkError in call_safety (langgraph_nodes.py:492)
- Falls back to local check_email (langgraph_nodes.py:492-494)

**Mitigation**:
- Fallback to local implementation
- Default timeout: 2000ms (configurable in mcp.tools.yaml)

**Status**: **Mitigated** via local fallback

**Recommendation**: Increase timeout for production workloads

### 11.4 LLM Latency Bottleneck

**Issue**: LLM calls in Consolidator (3s) and Stylist (3s) are slowest operations

**Impact**: Total runtime 12-20 seconds for single-pass pipeline

**Breakdown**:
- Retriever: 10s (multi-backend search)
- Consolidator: 3s (LLM call)
- Stylist: 3s (LLM call)
- A2A: 3s (safety check)
- Other nodes: <2s each

**Optimization Opportunities**:
- Cache LLM responses for identical inputs
- Use faster model (gpt-4o-mini)
- Parallel LLM calls (not currently supported)

**Status**: **Accepted** - Tradeoff for quality

### 11.5 Post-Processing Readability Logic

**Issue**: Readability truncation at run_graph_langgraph.py:144-167 may degrade email quality

**Behavior**:
- Truncates sentences to 10-12 words if grade > 15
- May break coherence in complex sentences
- Safeguard stops if grade gets worse

**Mitigation**:
- Trusts A2A output if no critical flags (line 147-149)
- Only truncates if A2A also flagged readability

**Status**: **By Design** - Gate-8 compliance requirement

**Alternative**: Improve Stylist prompt to generate simpler text upfront

### 11.6 Domain Diversity Enforcement

**Issue**: Consolidator enforces ≥4 unique domains, may synthesize cards from docmeta

**Behavior** (langgraph_nodes.py:304-369):
- If retrieved chunks lack domain diversity, synthesizes cards
- Synthetic card ID format: `synth::{doc_id}::card`
- Fills to 4 domains by replacing duplicate-domain cards

**Limitation**: Synthetic cards lack evidence snippets

**Frequency**: Rare (only when retrieval yields <4 domains)

**Status**: **By Design** - Ensures topical diversity

### 11.7 Hardcoded Prompt Templates

**Issue**: Prompt templates hardcoded in langgraph_nodes.py:22-100

**Limitation**: Cannot A/B test prompts without code changes

**Workaround**: Edit templates and rerun

**Status**: **Enhancement Opportunity** - Move to config files

### 11.8 Single Persona per Execution

**Issue**: Cannot generate emails for multiple personas in one run

**Limitation**: Must invoke graph separately for each persona

**Workaround**: Script multiple invocations:
```bash
for persona in vp_customer_experience cio vp_sales_ops; do
  conda run -n age python scripts/run_graph_langgraph.py \
    --company Salesforce \
    --persona $persona \
    --session-id "batch-$persona"
done
```

**Status**: **By Design** - Persona-specific workflows

---

## 12. References

### 12.1 Related Documentation

**Part 4: Query Routing** - Referenced by retriever_node
- `decide_backend()` function determines vector backend per query
- Keyword rules → Persona bias → Weighted scoring → Fallback

**Part 5: MCP Tools** - Referenced by retriever_node and a2a_node
- kb.search service (port 7801) - Vector search across FAISS/Weaviate/Pinecone
- safety.check service (port 7805) - Compliance validation

**Part 7: Quality Gates** - Gate-5 and Gate-6 validate graph system
- Gate-5: Graph structure and state schema validation
- Gate-6: A2A compliance checking validation
- Gate-8: End-to-end email generation quality

### 12.2 Architecture Documentation

**docs/architecture.md** - Full system design
- Section on LangGraph orchestration (8-node pipeline)
- State management patterns
- MCP tool integration

**docs/langgraph-edge-cases.md** - Edge case handling
- LLM hallucination recovery
- MCP service fallbacks
- Domain diversity enforcement

**docs/langgraph/001-llm-id-hallucination.md** - LLM ID mismatch analysis
- Root cause analysis
- Retry mechanism design
- Event logging format

### 12.3 Configuration References

**configs/langgraph.nodes.yaml** - Node topology and timeouts
- 8 node names
- Timeout budgets (2-10 seconds)

**configs/eval.prompts.yaml** - Persona keywords
- vp_customer_experience keywords
- cio keywords
- vp_sales_ops keywords

**configs/mcp.tools.yaml** - MCP service endpoints
- kb.search: 127.0.0.1:7801
- safety.check: 127.0.0.1:7805

**configs/compliance.template.yaml** - Compliance rules
- Critical violations (MISSING_UNSUBSCRIBE, PROHIBITED_PHRASE, etc.)
- Warning violations (EXCESS_LENGTH, READABILITY, etc.)

### 12.4 Code References

**Primary Implementation Files**:
- `scripts/run_graph_langgraph.py:36-70` - build_graph() function
- `scripts/run_graph_langgraph.py:25-33` - should_revise_email() conditional routing
- `scripts/run_graph_langgraph.py:73-205` - main_async() execution flow
- `scripts/langgraph_state.py:7-43` - AgentState TypedDict schema
- `scripts/langgraph_nodes.py:166-582` - 8 node implementations
- `scripts/langgraph_nodes.py:22-100` - Prompt templates (CONSOLIDATOR, STYLIST)

**Helper Functions**:
- `scripts/langgraph_nodes.py:103-109` - load_yaml()
- `scripts/langgraph_nodes.py:112-130` - log_llm_retry_event()
- `scripts/langgraph_nodes.py:133-141` - load_doc_meta()
- `scripts/langgraph_nodes.py:144-161` - kb_search() MCP client
- `scripts/langgraph_nodes.py:476-494` - call_safety() helper
- `scripts/langgraph_nodes.py:496-542` - revise_email() helper

**Supporting Infrastructure**:
- `scripts/router_core.py` - Query routing (decide_backend, rerank)
- `scripts/embedding_utils.py` - Embedding generation
- `scripts/common.py` - Shared utilities (now_iso, ensure_dir, load_yaml)
- `scripts/tool_safety_check_server.py` - Safety check fallback

### 12.5 External Resources

**LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- StateGraph construction
- Conditional edges
- State management

**LangChain Documentation**: https://python.langchain.com/
- ChatOpenAI integration
- ChatPromptTemplate usage

**OpenAI API Documentation**: https://platform.openai.com/docs/
- gpt-5-mini model details
- API key setup

**aiohttp Documentation**: https://docs.aiohttp.org/
- Async HTTP client
- TCPConnector configuration

### 12.6 Quality Gate Reports

**Gate-5 Report**: `reports/qa/step05_graph.md`
- Graph structure validation
- Node connectivity check
- State schema verification

**Gate-6 Report**: `reports/qa/step06_a2a.md`
- A2A compliance validation
- Revision logic testing
- Fallback verification

**Gate-8 Report**: `reports/qa/step08_generation_eval.md`
- Generation quality metrics
- 10-run evaluation per persona
- Pass/fail thresholds

### 12.7 Sample Outputs

**Official Runs**:
- `outputs/official-cio/` - CIO persona output
- `outputs/official-vp-cx/` - VP CX persona output
- `outputs/official-vp-sales-ops/` - VP Sales Ops persona output

**Test Runs**:
- `outputs/test-run-cio-2025-1/` - Test CIO run
- `outputs/test-run-vp-cx-2025-1/` - Test VP CX run
- `outputs/test-run-vp-sales-ops-2025-1/` - Test VP Sales Ops run

---

## Appendix A: File Path Reference

**Core Files**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/run_graph_langgraph.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/langgraph_nodes.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/langgraph_state.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/configs/langgraph.nodes.yaml`

**Supporting Files**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/router_core.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/embedding_utils.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/common.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/tool_safety_check_server.py`

**Quality Gates**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_step05_graph.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_step06_a2a.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_step08_generation_eval.py`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/scripts/qa_step08_debug.py`

**Documentation**:
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/architecture.md`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/langgraph-edge-cases.md`
- `/Users/liyunxiao/repo/ag3/worktrees/agent-weaviate/docs/langgraph/001-llm-id-hallucination.md`

---

**End of Document**

Total Lines: ~2,472
