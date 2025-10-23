# Part 5: MCP Tools & Services

**Research Date**: 2025-10-20 16:30:56 EDT
**Researcher**: Claude Code
**Git Commit**: `c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27`
**Branch**: `agent-weaviate`
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

### 1.1 MCP Purpose

**MCP (Model Context Protocol)** is a local service infrastructure that provides specialized tools for the multi-agent RAG system. The protocol enables LangGraph nodes to access external capabilities through standardized HTTP endpoints.

### 1.2 Five Service Overview

The system implements five MCP services, each serving a distinct purpose:

| Service | Port | Purpose | Implementation Status |
|---------|------|---------|----------------------|
| **kb.search** | 7801 | Vector search across FAISS/Weaviate/Pinecone indexes | **Fully implemented** (vector search + lexical reranking) |
| **web.fetch** | 7802 | Web content fetching | **Stub only** (mock responses) |
| **link.resolve** | 7803 | URL resolution | **Stub only** (mock responses) |
| **crm.lookup** | 7804 | CRM term lookup | **Stub only** (mock responses) |
| **safety.check** | 7805 | Email compliance validation | **Fully implemented** (compliance rules engine) |

### 1.3 Local Stub Implementation

All MCP services run as **local HTTP stub servers** on `127.0.0.1` (localhost):
- **No external dependencies** required for development
- **Isolated testing environment** with predictable behavior
- **Fast iteration** without network latency
- **Cost-free development** (no API charges for stubs)

---

## 2. Architecture & Design

### 2.1 Service Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ retriever_  │  │   a2a_node  │  │  (future    │         │
│  │    node     │  │             │  │   nodes)    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         │ aiohttp        │ aiohttp        │ aiohttp         │
│         │ ClientSession  │ ClientSession  │ ClientSession   │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
    ┌─────────────────────────────────────────────────┐
    │           MCP Configuration Layer               │
    │        (configs/mcp.tools.yaml)                 │
    │  ┌───────────────────────────────────────────┐  │
    │  │ tools:                                    │  │
    │  │   kb.search:       {host, port, timeout} │  │
    │  │   web.fetch:       {host, port, timeout} │  │
    │  │   link.resolve:    {host, port, timeout} │  │
    │  │   crm.lookup:      {host, port, timeout} │  │
    │  │   safety.check:    {host, port, timeout} │  │
    │  └───────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ HTTP POST    │ │ HTTP POST    │ │ HTTP POST    │
    │ /invoke      │ │ /invoke      │ │ /invoke      │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
    ┌──────▼───────────────────────────────────▼──────┐
    │       MCP Stub Server Infrastructure            │
    │         (scripts/qa_step03_mcp.py)              │
    │                                                  │
    │  ┌──────────────────────────────────────────┐   │
    │  │  Port 7801: kb.search                    │   │
    │  │  - Vector similarity search              │   │
    │  │  - Lexical reranking (0.7 vec + 0.3 lex) │   │
    │  │  - OpenAI ada-002 embeddings             │   │
    │  └──────────────────────────────────────────┘   │
    │                                                  │
    │  ┌──────────────────────────────────────────┐   │
    │  │  Port 7802-7804: Stub Services           │   │
    │  │  - web.fetch: {"content_length": 1234}   │   │
    │  │  - link.resolve: {"final_url": "..."}    │   │
    │  │  - crm.lookup: {"matches": 1}            │   │
    │  └──────────────────────────────────────────┘   │
    └──────────────────────────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────────────┐
    │    Safety Check Service (Separate Process)       │
    │    (scripts/tool_safety_check_server.py)         │
    │                                                   │
    │  ┌──────────────────────────────────────────┐    │
    │  │  Port 7805: safety.check                 │    │
    │  │  - Critical flags (compliance violations)│    │
    │  │  - Warning flags (quality issues)        │    │
    │  │  - Compliance rules from YAML config     │    │
    │  └──────────────────────────────────────────┘    │
    └───────────────────────────────────────────────────┘
```

### 2.2 HTTP Server Implementation (aiohttp)

All MCP services are implemented using **aiohttp**, Python's asynchronous HTTP framework.

**Key architectural decisions**:
- **Async/await pattern**: Enables concurrent request handling
- **Two-endpoint design**: Every service exposes `/healthz` (GET) and `/invoke` (POST)
- **JSON-based protocol**: All requests/responses use JSON serialization
- **Error code standardization**: Consistent error codes across all services

**Server lifecycle**:
1. **Startup**: `start_stub_servers()` loads data, creates aiohttp apps, binds to ports
2. **Runtime**: Services handle HTTP requests asynchronously
3. **Shutdown**: `stop_stub_servers()` gracefully stops sites and cleans up runners

### 2.3 Service Calling Flow

**Typical request flow**:

```
┌──────────────┐
│ LangGraph    │
│ Node         │
└──────┬───────┘
       │
       │ 1. Load MCP config (load_mcp_map)
       ▼
┌──────────────┐
│ Read YAML    │
│ Get host/port│
└──────┬───────┘
       │
       │ 2. Create aiohttp ClientSession
       ▼
┌──────────────┐
│ HTTP Client  │
│ Session      │
└──────┬───────┘
       │
       │ 3. POST /invoke with JSON payload
       ▼
┌──────────────┐
│ MCP Service  │
│ Port 780X    │
└──────┬───────┘
       │
       │ 4. Validate request (method, params)
       ▼
┌──────────────┐
│ Execute      │
│ Logic        │
└──────┬───────┘
       │
       │ 5. Return JSON response
       ▼
┌──────────────┐
│ Node         │
│ Processes    │
└──────────────┘
```

**Request format** (all services):
```json
{
  "method": "<service-specific-method>",
  "params": {
    "<param1>": "value1",
    "<param2>": "value2"
  }
}
```

**Response format** (success):
```json
{
  "status": "ok",
  "<service-specific-field>": "..."
}
```

**Response format** (error):
```json
{
  "error": {
    "code": "ErrorCodeName",
    "message": "Human-readable description"
  }
}
```

---

## 3. File Inventory

### 3.1 Core Implementation Files

#### MCP Server Infrastructure
- **`scripts/qa_step03_mcp.py`** (432 lines)
  - Main MCP stub server implementation
  - Implements all 5 services in a single process
  - Lines 40-205: `start_stub_servers()` - server initialization
  - Lines 82-156: `handle_invoke_kb()` - kb.search vector search handler
  - Lines 158-179: `handle_invoke_simple()` - generic stub handler
  - Lines 263-421: `main_async()` - Gate-3 validation orchestration

#### Standalone Service
- **`scripts/tool_safety_check_server.py`** (107 lines)
  - Safety check service (runs as separate process)
  - Lines 51-74: `check_email()` - compliance validation logic
  - Lines 77-89: `handle_invoke()` - HTTP request handler
  - Lines 96-107: `main()` - server startup

#### Client Integration
- **`scripts/langgraph_nodes.py`** (567 lines)
  - LangGraph node implementations that call MCP tools
  - Lines 144-162: `kb_search()` - kb.search HTTP client
  - Lines 214-244: `retriever_node()` - uses kb.search for vector retrieval
  - Lines 476-494: `call_safety()` - safety.check HTTP client
  - Lines 471-567: `a2a_node()` - uses safety.check for compliance

#### Configuration Utilities
- **`scripts/router_core.py`** (299 lines)
  - Lines 22-32: `load_mcp_map()` - loads `configs/mcp.tools.yaml`
  - Query routing logic (references MCP context)

- **`scripts/common.py`** (163 lines)
  - Common utilities including MCP helper functions
  - Lines 27-28: `now_iso()` - timestamp generation for reports

### 3.2 Configuration Files

- **`configs/mcp.tools.yaml`** (34 lines)
  - MCP service endpoint definitions
  - Lines 1-21: Tool configurations (5 services × 4 lines each)
  - Lines 23-34: Fallback policy configuration

- **`configs/compliance.template.yaml`** (33 lines)
  - Compliance rules for safety.check service
  - Lines 1-13: Critical rules (4 rules)
  - Lines 15-25: Warning rules (2 rules)
  - Lines 27-33: Prohibited phrases list

### 3.3 Quality Gate Scripts

- **`scripts/qa_step03_mcp.py`** (432 lines)
  - Gate-3: MCP tool health & contract conformance
  - Tests all 5 services for availability, correctness, and performance

### 3.4 Report Files

- **`reports/qa/step03_mcp.json`** - Gate-3 validation report (JSON format)
- **`reports/qa/step03_mcp.md`** - Gate-3 validation report (Markdown format)

### 3.5 Log Files

- **`logs/mcp/step03_probes.jsonl`** - MCP tool probe logs (JSONL format)

### 3.6 Documentation Files

- **`docs/architecture.md`** - System architecture (includes MCP tools section)
- **`docs/commands.md`** - Command reference (includes Gate-3 commands)
- **`docs/configuration.md`** - Configuration deep dive (includes mcp.tools.yaml)
- **`docs/troubleshooting.md`** - Troubleshooting guide (includes MCP port conflicts)
- **`docs/evaluation.md`** - Evaluation metrics (includes Gate-3 criteria)

---

## 4. Core Components Deep Dive

### 4.1 kb.search (Port 7801)

**Purpose**: Vector similarity search across FAISS/Weaviate/Pinecone indexes with lexical reranking.

#### Implementation Details

**File**: `scripts/qa_step03_mcp.py:82-156`

**Request Schema**:
```json
{
  "method": "search",
  "params": {
    "query": "Agentforce product announcement",
    "backend": "faiss",
    "top_k": 10
  }
}
```

**Response Schema**:
```json
{
  "results": [
    {
      "chunk_id": "press_release_001::chunk_003",
      "doc_id": "press_release_001",
      "score": 0.8742,
      "snippet": "Agentforce is Salesforce's new autonomous AI platform..."
    }
  ]
}
```

#### Vector Search Logic

**Embedding Loading** (`scripts/qa_step03_mcp.py:45-53`):
```python
# Load embeddings from parquet
t = pq.read_table("/Users/.../data/vector/embeddings/embeddings.parquet")
vecs = []
rows = []
for i in range(t.num_rows):
    row = {name: cols[name][i].as_py() for name in t.column_names}
    vecs.append([float(x) for x in row["vector"]])
    rows.append(row)
xb = np.array(vecs, dtype="float32")  # Shape: (N, 1536)
```

**Query Embedding** (`scripts/qa_step03_mcp.py:69-76`):
```python
def embed_query(q: str) -> np.ndarray:
    from embedding_utils import embed_text
    dim = xb.shape[1]  # 1536
    v = embed_text(q, dim)  # OpenAI ada-002
    return np.array(v, dtype="float32").reshape(1, -1)
```

**Distance Computation** (`scripts/qa_step03_mcp.py:102-109`):
```python
# L2 squared distance
qv = state["embed_query"](q)  # Shape: (1, 1536)
dists = ((xb - qv)**2).sum(axis=1)  # Shape: (N,)

# Get top candidates (wider set for reranking)
cand_k = max(top_k, 100)
idx = np.argsort(dists)[:cand_k]
```

**Vector Similarity Scoring** (`scripts/qa_step03_mcp.py:111-119`):
```python
res = []
for i in idx:
    r = state["rows"][i]
    chunk_id = r["chunk_id"]
    doc_id = r["doc_id"]
    dist = float(dists[i])
    vec_sim = 1.0 / (1.0 + dist)  # Convert distance to similarity [0,1]
    snippet = state["chunk_text"].get(chunk_id, "")[:280]
    res.append({"chunk_id": chunk_id, "doc_id": doc_id, "_vec_sim": vec_sim, "snippet": snippet})
```

**Lexical Reranking** (`scripts/qa_step03_mcp.py:120-145`):
```python
# Tokenize query
from embedding_utils import tokenize
qset = set(tokenize(q))

def lex_boost(snippet: str) -> float:
    sset = set(tokenize(snippet))
    if not qset:
        return 0.0
    return len(qset & sset) / len(qset)  # Term overlap ratio

# Hybrid scoring: 70% vector + 30% lexical
for r in res:
    vec = r["_vec_sim"]
    lex = lex_boost(r["snippet"])
    r["score"] = 0.7 * vec + 0.3 * lex

# Sort by final score descending
res.sort(key=lambda x: x["score"], reverse=True)
return res[:top_k]
```

#### Latency Simulation

**Backend-Specific Envelopes** (`scripts/qa_step03_mcp.py:98-101`):
```python
# Simulate network latency
envelopes = {"faiss": (5, 10), "weaviate": (40, 80), "pinecone": (80, 160)}
env = envelopes.get(backend, (50, 100))
await asyncio.sleep(random.uniform(env[0], env[1]) / 1000.0)
```

**Purpose**: Mimics realistic latency profiles for different backends in testing.

#### Error Handling

**Validation Errors** (`scripts/qa_step03_mcp.py:84-97`):
- **InvalidJSON** (400): Malformed request body
- **InvalidMethod** (400): Method != "search"
- **InvalidParams** (400): Missing query or backend
- **BackendUnavailable** (503): Backend not in ("faiss", "weaviate", "pinecone")

**Fallback Behavior** (`scripts/qa_step03_mcp.py:145-155`):
```python
except Exception:
    # If lexical reranking fails, return vector-only scores
    for r in res:
        r["score"] = r["_vec_sim"]
    res.sort(key=lambda x: x["score"], reverse=True)
    return res[:top_k]
```

---

### 4.2 web.fetch (Port 7802)

**Purpose**: Web content fetching (stub implementation).

**File**: `scripts/qa_step03_mcp.py:187` (handler binding)

**Implementation**: Generic stub handler (`handle_invoke_simple` with `["url"]`, `"fetch"`)

**Request Schema**:
```json
{
  "method": "fetch",
  "params": {
    "url": "https://example.com"
  }
}
```

**Response Schema**:
```json
{
  "status": "ok",
  "content_length": 1234
}
```

**Contract**:
- **Required parameter**: `url`
- **Response**: Static mock with hardcoded content_length
- **Usage**: Not currently used in graph nodes (available for future expansion)

---

### 4.3 link.resolve (Port 7803)

**Purpose**: URL resolution (stub implementation).

**File**: `scripts/qa_step03_mcp.py:188` (handler binding)

**Implementation**: Generic stub handler (`handle_invoke_simple` with `["url"]`, `"resolve"`)

**Request Schema**:
```json
{
  "method": "resolve",
  "params": {
    "url": "https://short.link/abc"
  }
}
```

**Response Schema**:
```json
{
  "status": "ok",
  "final_url": "https://short.link/abc"
}
```

**Contract**:
- **Required parameter**: `url`
- **Response**: Echoes input URL (no actual resolution)
- **Usage**: Not currently used in graph nodes

---

### 4.4 crm.lookup (Port 7804)

**Purpose**: CRM term lookup (stub implementation).

**File**: `scripts/qa_step03_mcp.py:189` (handler binding)

**Implementation**: Generic stub handler (`handle_invoke_simple` with `["term"]`, `"lookup"`)

**Request Schema**:
```json
{
  "method": "lookup",
  "params": {
    "term": "RPO"
  }
}
```

**Response Schema**:
```json
{
  "status": "ok",
  "matches": 1
}
```

**Contract**:
- **Required parameter**: `term`
- **Response**: Static mock with hardcoded match count
- **Usage**: Not currently used in graph nodes

---

### 4.5 safety.check (Port 7805)

**Purpose**: Email compliance validation with critical and warning flags.

**File**: `scripts/tool_safety_check_server.py` (standalone service)

#### Request Schema

**File**: `scripts/tool_safety_check_server.py:77-89`

```json
{
  "method": "moderate",
  "params": {
    "email_fields": {
      "body": "Dear VP, we guarantee 100% ROI with our solution...",
      "unsubscribe_block": "",
      "company_info_block": ""
    },
    "insight_cards": [
      {"id": "chunk001", "title": "Agentforce Features"}
    ],
    "text": "fallback body text (deprecated)"
  }
}
```

**Required Fields**:
- `method`: Must be `"moderate"`
- `params`: Object containing validation parameters

**Optional Fields**:
- `email_fields`: Dictionary with email components (`body`, `unsubscribe_block`, `company_info_block`)
- `insight_cards`: List of insight dictionaries with `id` field
- `text`: Fallback body text if `email_fields` not provided

#### Response Schema

```json
{
  "status": "ok",
  "flags": {
    "critical": ["MISSING_UNSUBSCRIBE", "PROHIBITED_PHRASE"],
    "warning": ["EXCESS_LENGTH", "READABILITY"]
  }
}
```

**Possible Critical Flags**:
1. **MISSING_UNSUBSCRIBE**: Email lacks `unsubscribe_block` field
2. **MISSING_COMPANY_INFO**: Email lacks `company_info_block` field
3. **UNCITED_CLAIM**: Body contains quantitative claims without citations
4. **PROHIBITED_PHRASE**: Body contains prohibited phrases

**Possible Warning Flags**:
1. **EXCESS_LENGTH**: Body exceeds max word count (default: 160 words)
2. **READABILITY**: Flesch-Kincaid grade exceeds target (default: grade 10)

#### Compliance Rules Implementation

**File**: `scripts/tool_safety_check_server.py:51-74`

**Critical Checks**:

1. **Unsubscribe Block** (lines 59-60):
```python
if not (email.get("unsubscribe_block") or "").strip():
    crit.append("MISSING_UNSUBSCRIBE")
```

2. **Company Info Block** (lines 61-62):
```python
if not (email.get("company_info_block") or "").strip():
    crit.append("MISSING_COMPANY_INFO")
```

3. **Uncited Claims** (lines 63-64):
```python
if has_uncited_claim(body, insight_ids):
    crit.append("UNCITED_CLAIM")

# Helper function (lines 39-43)
def has_uncited_claim(text: str, insight_ids: List[str]) -> bool:
    # Pattern: \b(\d+%|double|guarantee|always|never)\b
    import re
    return bool(re.search(r'\b(\d+%|double|guarantee|always|never)\b', text, re.I))
```

4. **Prohibited Phrases** (lines 65-66):
```python
if prohibited_present(body, spec.get("prohibited_phrases", [])):
    crit.append("PROHIBITED_PHRASE")

# Helper function (lines 46-48)
def prohibited_present(text: str, phrases: List[str]) -> bool:
    tl = text.lower()
    return any(p.lower() in tl for p in phrases)
```

**Warning Checks**:

1. **Excess Length** (lines 68-70):
```python
wr = spec.get("warning_rules", [])
max_w = next((r["max_words"] for r in wr if r.get("id") == "EXCESS_LENGTH"), 160)
if word_count(body) > max_w:
    warn.append("EXCESS_LENGTH")

# Helper function (lines 26-27)
def word_count(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))
```

2. **Readability** (lines 71-73):
```python
max_g = next((r["max_grade"] for r in wr if r.get("id") == "READABILITY"), 10)
if readability_grade(body) > max_g:
    warn.append("READABILITY")

# Helper function (lines 30-36)
def readability_grade(text: str) -> float:
    sents = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    ns = max(len(sents), 1)
    words = max(word_count(text), 1)
    syllables = max(sum(len(re.findall(r'[aeiouyAEIOUY]', w)) for w in text.split()), 1)
    # Flesch-Kincaid Grade Level formula
    return 0.39 * (words / ns) + 11.8 * (syllables / words) - 15.59
```

#### Compliance Configuration

**File**: `configs/compliance.template.yaml`

**Critical Rules** (lines 1-13):
```yaml
critical_rules:
  - id: MISSING_UNSUBSCRIBE
    description: Email must include unsubscribe mechanism
  - id: MISSING_COMPANY_INFO
    description: Email must include company identification
  - id: UNCITED_CLAIM
    description: Quantitative claims must cite source
  - id: PROHIBITED_PHRASE
    description: Email must not contain prohibited phrases
```

**Warning Rules** (lines 15-25):
```yaml
warning_rules:
  - id: EXCESS_LENGTH
    description: Email body should not exceed word limit
    max_words: 160
  - id: READABILITY
    description: Email should maintain target readability grade
    max_grade: 10
```

**Prohibited Phrases** (lines 27-33):
```yaml
prohibited_phrases:
  - guaranteed
  - free money
  - no strings attached
  - risk-free
  - 100% safe
```

#### Standalone Service

**File**: `scripts/tool_safety_check_server.py:96-107`

The safety check service runs as a **separate process** from the main stub servers:

```python
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7805, type=int)
    args = parser.parse_args()
    a = web.Application()
    a.add_routes([web.get("/healthz", handle_health), web.post("/invoke", handle_invoke)])
    web.run_app(a, host=args.host, port=args.port)
```

**Startup**:
```bash
# Standalone service (manual start)
python scripts/tool_safety_check_server.py --host 127.0.0.1 --port 7805
```

**Integration**: Called by a2a_node during agent-to-agent negotiation phase.

---

## 5. Configuration & Settings

### 5.1 mcp.tools.yaml Schema

**File**: `configs/mcp.tools.yaml` (34 lines)

**Structure**:
```yaml
tools:
  <service_name>:
    host: <string>           # IP address or hostname
    port: <integer>          # TCP port number
    timeout_ms: <integer>    # Request timeout in milliseconds

fallback:
  mode: <string>             # Failure handling mode
  policy:
    log_downgrades: <boolean>
    retry_attempts: <integer>
    connection_timeout_ms: <integer>
    warn_on_offline: <boolean>
    warn_on_external: <boolean>
```

### 5.2 Service Endpoints

**Complete Configuration** (lines 1-21):

```yaml
tools:
  kb.search:
    host: 127.0.0.1
    port: 7801
    timeout_ms: 2000

  web.fetch:
    host: 127.0.0.1
    port: 7802
    timeout_ms: 2000

  link.resolve:
    host: 127.0.0.1
    port: 7803
    timeout_ms: 2000

  crm.lookup:
    host: 127.0.0.1
    port: 7804
    timeout_ms: 2000

  safety.check:
    host: 127.0.0.1
    port: 7805
    timeout_ms: 2000
```

**Key aspects**:
- All services on localhost (`127.0.0.1`)
- Sequential port assignment (7801-7805)
- Uniform 2-second timeout

### 5.3 Timeouts

**Request Timeouts**:
- **Per-service**: 2000ms (configured in `mcp.tools.yaml`)
- **Connection timeout**: 2000ms (configured in fallback policy)

**Usage in clients** (`scripts/langgraph_nodes.py:152`):
```python
timeout = base.get("timeout_ms", 2000) / 1000.0  # Convert ms to seconds
async with session.post(url, json=payload, timeout=timeout) as resp:
    # ...
```

### 5.4 Fallback Policies

**File**: `configs/mcp.tools.yaml:23-34`

```yaml
fallback:
  mode: default  # Options: default (silent), warn (log), strict (raise)
  policy:
    log_downgrades: true      # Log service degradation events
    retry_attempts: 1         # Number of retry attempts
    connection_timeout_ms: 2000  # Connection establishment timeout
    warn_on_offline: true     # Warn when services are offline (warn mode)
    warn_on_external: false   # Warn on external service fallbacks (warn mode)
```

**Modes**:
1. **default**: Silent fallback (no exceptions raised)
2. **warn**: Log downgrades (warnings emitted)
3. **strict**: Fail fast (raise exceptions)

**Current Setting**: `mode: default` (graceful degradation)

**Fallback Implementation** (safety.check only):

**File**: `scripts/langgraph_nodes.py:489-494`

```python
except Exception:
    # Fallback: local checks
    spec = load_yaml(os.path.join("configs", "compliance.template.yaml"))
    from tool_safety_check_server import check_email
    c, w = check_email(email_fields, state["insight_cards"], spec)
    return c, w
```

**Key aspects**:
- Only safety.check has local fallback implementation
- Falls back to importing server logic directly
- No distinction between network and local execution in result

---

## 6. Data Structures & Schemas

### 6.1 Request Schemas (All 5 Tools)

#### kb.search Request
```json
{
  "method": "search",
  "params": {
    "query": "string (required)",
    "backend": "faiss|weaviate|pinecone (required)",
    "top_k": "integer (required)"
  }
}
```

#### web.fetch Request
```json
{
  "method": "fetch",
  "params": {
    "url": "string (required)"
  }
}
```

#### link.resolve Request
```json
{
  "method": "resolve",
  "params": {
    "url": "string (required)"
  }
}
```

#### crm.lookup Request
```json
{
  "method": "lookup",
  "params": {
    "term": "string (required)"
  }
}
```

#### safety.check Request
```json
{
  "method": "moderate",
  "params": {
    "email_fields": {
      "body": "string (required)",
      "unsubscribe_block": "string (optional)",
      "company_info_block": "string (optional)"
    },
    "insight_cards": [
      {"id": "string"}
    ],
    "text": "string (fallback if email_fields missing)"
  }
}
```

### 6.2 Response Schemas (All 5 Tools)

#### kb.search Response (Success)
```json
{
  "results": [
    {
      "chunk_id": "string",
      "doc_id": "string",
      "score": "float (0-1)",
      "snippet": "string (max 280 chars)"
    }
  ]
}
```

#### web.fetch Response (Success)
```json
{
  "status": "ok",
  "content_length": "integer (mock)"
}
```

#### link.resolve Response (Success)
```json
{
  "status": "ok",
  "final_url": "string (echoes input)"
}
```

#### crm.lookup Response (Success)
```json
{
  "status": "ok",
  "matches": "integer (mock)"
}
```

#### safety.check Response (Success)
```json
{
  "status": "ok",
  "flags": {
    "critical": ["string (flag ID)"],
    "warning": ["string (flag ID)"]
  }
}
```

### 6.3 Error Formats

**Standard Error Response** (all services):
```json
{
  "error": {
    "code": "string (error code)",
    "message": "string (human-readable)"
  }
}
```

**HTTP Status Codes**:
- **200**: Success
- **400**: Client error (InvalidJSON, InvalidMethod, InvalidParams)
- **503**: Service unavailable (BackendUnavailable)

**Error Codes**:
- **InvalidJSON**: Malformed request body
- **InvalidMethod**: Method name doesn't match expected value
- **InvalidParams**: Missing or invalid required parameters
- **BackendUnavailable**: Requested backend not supported (kb.search only)
- **Timeout**: Request timed out (client-side)
- **NetworkError**: Network connectivity issue (client-side)

---

## 7. External Dependencies

### 7.1 aiohttp (HTTP Server)

**Purpose**: Asynchronous HTTP server framework for MCP services.

**Version**: Specified in `envs/age.yaml`

**Key Components Used**:
- `aiohttp.web.Application` - Web application container
- `aiohttp.web.AppRunner` - Application lifecycle manager
- `aiohttp.web.TCPSite` - TCP server binding
- `aiohttp.web.json_response()` - JSON response helper
- `aiohttp.web.get()` / `web.post()` - Route decorators

**Server Setup Pattern** (`scripts/qa_step03_mcp.py:193-200`):
```python
a = web.Application()
a.add_routes([
    web.get("/healthz", handle_health),
    web.post("/invoke", handler)
])
r = web.AppRunner(a)
await r.setup()
site = web.TCPSite(r, host, port)
await site.start()
```

### 7.2 httpx (HTTP Client)

**Purpose**: HTTP client library for making requests to MCP services.

**Note**: The codebase actually uses **aiohttp.ClientSession** for HTTP client operations, not httpx.

**Client Usage Pattern** (`scripts/langgraph_nodes.py:149-161`):
```python
async with aiohttp.ClientSession(connector=connector) as session:
    async with session.post(url, json=payload, timeout=timeout) as resp:
        status = resp.status
        j = await resp.json()
        if status >= 400:
            return [], latency, j.get("error", {}).get("code")
        return j.get("results", []), latency, None
```

**Key Components Used**:
- `aiohttp.ClientSession` - HTTP client session
- `aiohttp.TCPConnector` - Connection pooling (limit_per_host=8)
- `session.post()` - Async POST requests
- JSON serialization/deserialization

---

## 8. Execution & Usage

### 8.1 Start MCP Servers

**Gate-3 Validation** (includes server startup):
```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**What it does**:
1. Loads embeddings from `data/vector/embeddings/embeddings.parquet`
2. Loads chunk text from `data/interim/chunks/*.chunks.jsonl`
3. Starts 5 HTTP servers on ports 7801-7805
4. Runs health checks (GET /healthz on all services)
5. Runs contract tests (valid/invalid requests)
6. Runs latency sampling (15 kb.search queries)
7. Generates Gate-3 reports (JSON + Markdown)
8. Shuts down servers

**Environment**: Use `age` environment (Python 3.13)

### 8.2 Start Safety Check Service (Standalone)

**Manual Startup**:
```bash
conda run -n age python scripts/tool_safety_check_server.py --host 127.0.0.1 --port 7805
```

**What it does**:
1. Loads compliance rules from `configs/compliance.template.yaml`
2. Starts aiohttp server on port 7805
3. Exposes `/healthz` (GET) and `/invoke` (POST) endpoints
4. Runs until terminated (Ctrl+C)

### 8.3 Call Tools via HTTP POST

**Example: kb.search (curl)**:
```bash
curl -X POST http://127.0.0.1:7801/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "method": "search",
    "params": {
      "query": "Agentforce product announcement",
      "backend": "faiss",
      "top_k": 5
    }
  }'
```

**Example: safety.check (curl)**:
```bash
curl -X POST http://127.0.0.1:7805/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "method": "moderate",
    "params": {
      "email_fields": {
        "body": "Dear VP, we guarantee 100% ROI...",
        "unsubscribe_block": "",
        "company_info_block": ""
      },
      "insight_cards": [{"id": "chunk001"}]
    }
  }'
```

**Example: Health Check (curl)**:
```bash
curl http://127.0.0.1:7801/healthz
# Response: {"status": "ok"}
```

### 8.4 Python Client Example (kb.search)

**File**: `scripts/langgraph_nodes.py:144-162`

```python
import aiohttp
import time

async def kb_search_example():
    tools_cfg = {
        "kb.search": {
            "host": "127.0.0.1",
            "port": 7801,
            "timeout_ms": 2000
        }
    }

    base = tools_cfg["kb.search"]
    url = f"http://{base['host']}:{base['port']}/invoke"
    payload = {
        "method": "search",
        "params": {
            "query": "Agentforce features",
            "backend": "faiss",
            "top_k": 10
        }
    }

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=2.0) as resp:
            status = resp.status
            j = await resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if status >= 400:
                error_code = j.get("error", {}).get("code")
                print(f"Error: {error_code}")
                return []

            results = j.get("results", [])
            print(f"Found {len(results)} results in {latency_ms:.1f}ms")
            return results

# Usage
# results = await kb_search_example()
```

### 8.5 Integration with LangGraph

**retriever_node** (uses kb.search):

**File**: `scripts/langgraph_nodes.py:214-244`

```python
async def retriever_node(state: AgentState) -> dict:
    """Execute vector search via MCP kb.search."""
    tools_cfg = load_mcp_map()
    retrieved_chunks = []
    retrieval_logs = []
    route_decisions = []

    connector = aiohttp.TCPConnector(limit_per_host=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        for q in state["queries"]:
            # Route query to backend
            backend, reasons = decide_backend(q, state["persona"], None)
            route_decisions.append({"query": q, "backend": backend, "reasons": reasons})

            # Retrieve via kb.search
            res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)

            # Re-rank + attach metadata
            res = rerank(res, docmeta, top_k=12, domain_cap=2)

            # Log and extend
            retrieval_logs.append({"query": q, "results": res[:10]})
            retrieved_chunks.extend(res[:10])

    return {
        "retrieved_chunks": retrieved_chunks,
        "retrieval_logs": retrieval_logs,
        "route_decisions": route_decisions,
    }
```

**a2a_node** (uses safety.check):

**File**: `scripts/langgraph_nodes.py:471-567`

```python
async def a2a_node(state: AgentState) -> dict:
    """Compliance negotiation with safety.check."""
    current_round = state.get("a2a_rounds", 0)

    # Call safety check on current email draft
    crit, warn = await call_safety(state["email_draft"], state["insight_cards"])

    # Record flags
    compliance_flags = [f"CRITICAL:{f}" for f in crit] + [f"WARN:{f}" for f in warn]

    # Increment round counter
    new_round = current_round + 1

    # If critical flags present and this is round 1, prepare for revision
    email_draft = state["email_draft"]
    if crit and current_round < 1:
        email_draft = revise_email(dict(state["email_draft"]), state["insight_cards"], crit, warn)

    return {
        "compliance_flags": compliance_flags,
        "a2a_rounds": new_round,
        "email_draft": email_draft,
    }
```

---

## 9. Code Patterns & Conventions

### 9.1 All Tools Return JSON

**Consistency**: Every MCP service returns JSON responses, enabling uniform parsing across clients.

**Success Response Pattern**:
```python
return web.json_response({
    "status": "ok",
    # ... service-specific fields
})
```

**Error Response Pattern**:
```python
return web.json_response({
    "error": {
        "code": "ErrorCodeName",
        "message": "Human-readable description"
    }
}, status=400)  # or 503
```

### 9.2 Consistent Error Handling

**Request Validation Pattern** (`scripts/qa_step03_mcp.py:84-97`):
```python
# 1. Parse JSON
try:
    body = await request.json()
except Exception:
    return web.json_response({"error": {"code": "InvalidJSON", "message": "Malformed JSON"}}, status=400)

# 2. Validate method
m = (body.get("method") or "").strip()
if m != expected_method:
    return web.json_response({"error": {"code": "InvalidMethod", "message": "Unknown method"}}, status=400)

# 3. Validate params
params = body.get("params") or {}
if not params.get("required_field"):
    return web.json_response({"error": {"code": "InvalidParams", "message": "missing required_field"}}, status=400)
```

**Client-Side Error Handling** (`scripts/langgraph_nodes.py:155-161`):
```python
try:
    async with session.post(url, json=payload, timeout=timeout) as resp:
        status = resp.status
        j = await resp.json()
        if status >= 400:
            error_code = (j.get("error") or {}).get("code")
            return [], latency, error_code
        return j.get("results", []), latency, None
except Exception as e:
    return [], latency, "NetworkError"
```

### 9.3 Timeout Enforcement

**Server-Side**: No explicit timeout handling (relies on client timeouts)

**Client-Side Timeout** (`scripts/langgraph_nodes.py:152`):
```python
timeout_ms = base.get("timeout_ms", 2000)
timeout_sec = timeout_ms / 1000.0
async with session.post(url, json=payload, timeout=timeout_sec) as resp:
    # ...
```

**Timeout Error Handling** (`scripts/qa_step03_mcp.py:252-254`):
```python
except asyncio.TimeoutError:
    status = 0
    err_code = "Timeout"
```

### 9.4 Factory Pattern for Handlers

**File**: `scripts/qa_step03_mcp.py:185-191`

```python
bindings = [
    ("kb.search", handle_invoke_kb),
    ("web.fetch", lambda req: handle_invoke_simple(req, ["url"], "fetch")),
    ("link.resolve", lambda req: handle_invoke_simple(req, ["url"], "resolve")),
    ("crm.lookup", lambda req: handle_invoke_simple(req, ["term"], "lookup")),
    ("safety.check", lambda req: handle_invoke_simple(req, ["text"], "moderate")),
]
```

**Purpose**: Parameterized handler creation via lambda closures, enabling code reuse for simple services.

### 9.5 Shared State Pattern

**File**: `scripts/qa_step03_mcp.py:54-67`

```python
# Shared state dictionary passed to handlers
state["xb"] = xb                      # Embeddings matrix (N, 1536)
state["rows"] = rows                  # Metadata rows
state["chunk_text"] = chunk_text      # chunk_id -> snippet mapping
state["embed_query"] = embed_query    # Query embedding function
```

**Purpose**: Load data once, share across all kb.search invocations.

### 9.6 Strategy Pattern (Response Strategies)

**Different response strategies per service**:
- **kb.search**: Complex vector search + reranking
- **web.fetch, link.resolve, crm.lookup**: Simple mock responses
- **safety.check**: Rule-based validation with flag accumulation

**Selection**: Handler binding determines strategy (lines 185-191)

### 9.7 Probe Pattern (Audit Trail)

**File**: `scripts/qa_step03_mcp.py:233, 259`

```python
probes = []  # Audit log

# After each service call
probes.append({
    "tool": name,
    "method": method,
    "request_id": request_id,
    "params_summary": params,
    "status_code": status,
    "error_code": err_code,
    "latency_ms": latency_ms
})

# Export to JSONL
with open(PROBE_LOG, "w", encoding="utf-8") as f:
    for p in probes:
        f.write(json.dumps(p) + "\n")
```

**Purpose**: Complete audit trail of all MCP service interactions for debugging and analysis.

---

## 10. Testing & Verification

### 10.1 Gate-3 MCP Validation

**Purpose**: Validate that all 5 MCP services are healthy, conform to contracts, and meet latency budgets.

**Script**: `scripts/qa_step03_mcp.py`

**Execution**:
```bash
conda run -n age python scripts/qa_step03_mcp.py
```

**Test Phases**:

#### Phase 1: Health Checks (Lines 273-282)
- **Test**: GET `/healthz` on all 5 services
- **Expected**: `{"status": "ok"}` with HTTP 200
- **Metric**: `health_endpoints_ok` (count of healthy services)
- **Threshold**: `== 5 tools`

#### Phase 2: Contract Tests (Lines 284-303)
- **Test**: Send valid and invalid requests to each service
- **Valid Requests**: Should return HTTP 200 with expected response structure
- **Invalid Requests**: Should return HTTP 400 with appropriate error codes
  - Empty params → `InvalidParams`
  - Wrong backend → `BackendUnavailable`
  - Wrong method → `InvalidMethod`
- **Metric**: `contract_ok_rate_{service}` (per service)
- **Threshold**: `== 1.0` (100% of invalid requests correctly rejected)

#### Phase 3: Latency Sampling (Lines 305-346)
- **Test**: 15 kb.search queries (5 per backend: faiss, weaviate, pinecone)
- **Queries**: Loaded from `data/interim/eval/salesforce_eval_seed.jsonl`
- **Metrics**: p50 (median), p95 latency per backend
- **Budgets**:
  - **faiss**: 300ms documented, actual = min(300, p95 * 1.20)
  - **weaviate**: 1000ms documented, actual = min(1000, p95 * 1.20)
  - **pinecone**: 1500ms documented, actual = min(1500, p95 * 1.20)
- **Threshold**: p50 and p95 ≤ budget

#### Phase 4: Stability Check (Line 380)
- **Test**: Count timeout errors across all latency sampling
- **Metric**: `timeout_rate` (timeouts / 15 total queries)
- **Threshold**: `== 0.0` (no timeouts)

### 10.2 Service Health Checks

**Endpoint**: `GET /healthz` (all services)

**Response**:
```json
{"status": "ok"}
```

**Implementation** (`scripts/qa_step03_mcp.py:79-80`):
```python
async def handle_health(request):
    return web.json_response({"status": "ok"})
```

**Usage**:
- Gate-3 validation (line 227)
- External monitoring tools
- Kubernetes liveness probes (if deployed)

### 10.3 Contract Testing

**Test Definitions** (`scripts/qa_step03_mcp.py:285-291`):

```python
methods = {
    "kb.search": ("search",
        {"query": "Agentforce", "backend": "faiss", "top_k": 5},
        [{"backend": "unknown"}, {}]  # Invalid: bad backend, empty params
    ),
    "web.fetch": ("fetch",
        {"url": "https://example.com"},
        [{}, {"bogus": 1}]  # Invalid: empty params, bogus field
    ),
    "link.resolve": ("resolve",
        {"url": "https://example.com"},
        [{}, {"bogus": 1}]
    ),
    "crm.lookup": ("lookup",
        {"term": "RPO"},
        [{}, {"bogus": 1}]
    ),
    "safety.check": ("moderate",
        {"text": "hello world"},
        [{}, {"bogus": 1}]
    ),
}
```

**Execution** (`scripts/qa_step03_mcp.py:293-303`):
```python
for svc, (method, valid, invalid_list) in methods.items():
    # Test valid request (should succeed)
    st, err, lat = await invoke(session, svc, base_url, method, valid, "valid", timeout_ms, probes)

    # Test invalid requests (should fail with 400 or 503)
    ok_invalid = 0
    for params in invalid_list:
        st, err, lat = await invoke(session, svc, base_url, method, params, "invalid", timeout_ms, probes)
        if st in (400, 503) and err in ("InvalidParams", "BackendUnavailable", "InvalidMethod"):
            ok_invalid += 1

    contract_ok_rate = ok_invalid / len(invalid_list)
```

**Expected Error Codes**:
- `InvalidParams` - Missing or invalid required parameters
- `BackendUnavailable` - Unsupported backend (kb.search only)
- `InvalidMethod` - Method name mismatch

### 10.4 Gate-3 Quality Checks

**Check ID: G3-01** (Health Endpoints)
- **Metric**: `health_endpoints_ok`
- **Actual**: Count of services responding to /healthz
- **Threshold**: `== 5 tools`
- **Status**: PASS if 5/5, FAIL otherwise

**Check ID: G3-02** (Contract Conformance - kb.search)
- **Metric**: `contract_ok_rate_kb.search`
- **Actual**: Proportion of invalid requests correctly rejected
- **Threshold**: `== 1.0`
- **Status**: PASS if 1.0, FAIL otherwise

**Check IDs: G3-02-web_fetch, G3-02-link_resolve, G3-02-crm_lookup, G3-02-safety_check**
- Same contract conformance check for remaining 4 services

**Check IDs: G3-03-faiss, G3-03-weaviate, G3-03-pinecone** (Latency Budgets)
- **Metric**: `{backend}_latency_budget`
- **Actual**: JSON with `{p50, p95, budget_p95}`
- **Threshold**: `p50 <= budget AND p95 <= budget`
- **Status**:
  - PASS if both within budget
  - WARN if within 110% of budget
  - FAIL otherwise

**Check ID: G3-04** (Stability)
- **Metric**: `timeout_rate`
- **Actual**: Proportion of queries that timed out
- **Threshold**: `== 0.0`
- **Status**: PASS if 0.0, FAIL otherwise

### 10.5 Gate Status Determination

**File**: `scripts/qa_step03_mcp.py:382-395`

**GREEN** (lines 383-385):
- **Condition**: All checks have `status == "PASS"`
- **Next Action**: `"continue"`
- **Meaning**: All services healthy, proceed to next gate

**AMBER** (lines 387-392):
- **Condition**: No FAIL checks, exactly 1 WARN check, and WARN is latency-related (`G3-03-*`)
- **Next Action**: `"proceed_with_caution"`
- **Meaning**: Services functional but one backend slightly slow

**RED** (lines 394-395):
- **Condition**: Any FAIL checks OR multiple WARNs
- **Next Action**: `"fix_and_rerun"`
- **Meaning**: Critical issues, must fix before proceeding

### 10.6 Report Outputs

**JSON Report**: `reports/qa/step03_mcp.json`

**Structure**:
```json
{
  "step": 3,
  "gate": "G3",
  "status": "GREEN|AMBER|RED",
  "checks": [
    {
      "id": "G3-01",
      "metric": "health_endpoints_ok",
      "actual": 5,
      "threshold": "==5 tools",
      "status": "PASS"
    }
  ],
  "next_action": "continue|proceed_with_caution|fix_and_rerun",
  "timestamp": "2025-10-20T16:30:56-04:00"
}
```

**Markdown Report**: `reports/qa/step03_mcp.md`

**Structure**:
```markdown
# STEP 3 — MCP Tool Health & Contract Conformance (Gate‑3) — GREEN

**Timestamp**: 2025-10-20T16:30:56-04:00

## Checks

- G3-01: health_endpoints_ok = 5 (threshold ==5 tools) -> PASS
- G3-02: contract_ok_rate_kb.search = 1.0 (threshold ==1.0) -> PASS
- G3-02-web_fetch: contract_ok_rate_web.fetch = 1.0 (threshold ==1.0) -> PASS
- G3-02-link_resolve: contract_ok_rate_link.resolve = 1.0 (threshold ==1.0) -> PASS
- G3-02-crm_lookup: contract_ok_rate_crm.lookup = 1.0 (threshold ==1.0) -> PASS
- G3-02-safety_check: contract_ok_rate_safety.check = 1.0 (threshold ==1.0) -> PASS
- G3-03-faiss: faiss_latency_budget = {"p50":8.2,"p95":9.8,"budget_p95":300} (threshold p50,p95<=budget) -> PASS
- G3-03-weaviate: weaviate_latency_budget = {"p50":58.3,"p95":77.1,"budget_p95":1000} (threshold p50,p95<=budget) -> PASS
- G3-03-pinecone: pinecone_latency_budget = {"p50":118.7,"p95":156.2,"budget_p95":1500} (threshold p50,p95<=budget) -> PASS
- G3-04: timeout_rate = 0.0 (threshold ==0.0) -> PASS

## Go/No-Go Decision

**Status**: Go (GREEN)

**Next Action**: continue
```

---

## 11. Known Issues & Limitations

### 11.1 Stubs Only (web.fetch, link.resolve, crm.lookup)

**Issue**: Three services are stub implementations returning mock responses.

**Affected Services**:
- **web.fetch** (port 7802): Returns `{"status": "ok", "content_length": 1234}`
- **link.resolve** (port 7803): Returns `{"status": "ok", "final_url": params.get("url")}`
- **crm.lookup** (port 7804): Returns `{"status": "ok", "matches": 1}`

**Impact**:
- Cannot perform actual web fetching
- Cannot resolve redirect chains
- Cannot query CRM systems
- Only useful for testing service integration

**Workaround**: None (by design for development/testing)

**Future**: Implement full logic when external integrations are ready

### 11.2 No Authentication

**Issue**: All MCP services lack authentication/authorization mechanisms.

**Security Implications**:
- Any process on localhost can access services
- No API key validation
- No rate limiting
- No user/tenant isolation

**Mitigation**:
- Services bind to `127.0.0.1` only (not exposed externally)
- Network-level firewall rules prevent external access
- Local-only deployment model

**Future**: Add API key authentication when deploying to shared environments

### 11.3 Single-Threaded

**Issue**: Services run in asyncio event loop (single-threaded execution).

**Concurrency Model**:
- **Async I/O concurrency**: Enabled (handles multiple requests via async/await)
- **CPU parallelism**: Not enabled (no multiprocessing/threading)

**Impact**:
- CPU-intensive operations (vector search, embedding) block event loop
- Cannot utilize multi-core CPUs for parallel request processing
- Latency may spike under load

**Workaround**:
- kb.search uses numpy vectorized operations (reasonably fast)
- Latency budgets account for single-threaded execution

**Future**: Add multiprocessing worker pool for CPU-intensive operations

### 11.4 No Request Batching

**Issue**: Each request is processed independently (no batching).

**Impact**:
- Inefficient for bulk operations (e.g., embedding 100 queries)
- Cannot amortize fixed costs (model loading, connection setup)
- Higher total latency for batch workloads

**Workaround**: Client-side batching (send multiple requests concurrently)

**Example** (`scripts/langgraph_nodes.py:224-244`):
```python
connector = aiohttp.TCPConnector(limit_per_host=8)
async with aiohttp.ClientSession(connector=connector) as session:
    for q in state["queries"]:  # 5 queries
        res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)
```

**Future**: Add batch endpoint (e.g., `/invoke_batch` accepting array of requests)

### 11.5 No Caching

**Issue**: No response caching (every request re-executes logic).

**Impact**:
- Duplicate queries (same query string) recompute results
- Wastes CPU cycles and increases latency
- Inefficient for common queries

**Workaround**: kb.search uses pre-cached embeddings from parquet file

**Partial Mitigation**:
- Query embeddings call `embed_text()` which has SHA-256 based cache (`scripts/embedding_utils.py:20-21, 34-36`)
- Embeddings cached in `data/cache/embeddings/`

**Future**: Add Redis cache layer for frequently accessed results

### 11.6 Uncited Claim Detection Incomplete

**Issue**: `has_uncited_claim()` in safety.check detects strong quantifiers but doesn't verify citations.

**File**: `scripts/tool_safety_check_server.py:39-43`

```python
def has_uncited_claim(text: str, insight_ids: List[str]) -> bool:
    # TODO: actually check for *cited* claims using insight_ids
    import re
    return bool(re.search(r'\b(\d+%|double|guarantee|always|never)\b', text, re.I))
```

**Current Behavior**: Flags any text with quantitative/absolute language, regardless of citations

**Expected Behavior**: Only flag claims without corresponding citations from `insight_cards`

**Impact**: False positives (flagging properly cited claims)

**Workaround**: Manual review of `UNCITED_CLAIM` flags

**Future**: Implement citation verification logic

### 11.7 Lexical Reranking Fallback

**Issue**: If tokenization fails during lexical reranking, kb.search falls back to vector-only scores.

**File**: `scripts/qa_step03_mcp.py:145-155`

```python
except Exception:
    # If lexical fails, use vector-only scores
    for r in res:
        r["score"] = r["_vec_sim"]
    res.sort(key=lambda x: x["score"], reverse=True)
    return res[:top_k]
```

**Impact**:
- Silent degradation (no error logged)
- Results may differ from expected hybrid ranking
- Difficult to debug when lexical boost is missing

**Workaround**: None (automatic fallback by design)

**Future**: Add logging for fallback events

### 11.8 Port Conflicts

**Issue**: If ports 7801-7805 are already in use, services fail to start.

**Error Example**:
```
OSError: [Errno 48] Address already in use
```

**Mitigation**:
- Check port availability before starting: `lsof -i :7801-7805`
- Kill conflicting processes: `kill $(lsof -t -i:7801)`

**Configuration**: Ports are hardcoded in `configs/mcp.tools.yaml` (not easily changeable)

**Future**: Add auto-detection of available ports or configurable port ranges

---

## 12. References

### 12.1 Part 4 (Routing Used by kb.search)

**Document**: `roadmap/part4-routing/README.md`

**Relevance**: kb.search uses routing logic to select backend (faiss/weaviate/pinecone)

**Key Functions**:
- `decide_backend(query, persona, context)` - Routes query to appropriate backend
- Routing rules in `configs/router.heuristics.yaml`

**Integration Point** (`scripts/langgraph_nodes.py:228-229`):
```python
backend, reasons = decide_backend(q, state["persona"], None)
res, lat, err = await kb_search(session, backend, q, 12, tools_cfg)
```

### 12.2 Part 6 (How retriever_node and a2a_node Call Tools)

**Document**: `roadmap/part6-agents/README.md` (to be created)

**Relevance**: LangGraph nodes implement MCP tool clients

**Key Implementations**:
- **retriever_node** (`scripts/langgraph_nodes.py:214-244`): Calls kb.search for vector retrieval
- **a2a_node** (`scripts/langgraph_nodes.py:471-567`): Calls safety.check for compliance validation

**Tool Usage Patterns**:
- Shared session pattern (retriever_node with connection pooling)
- One-shot session pattern (a2a_node for single request)
- Error handling and fallback strategies

### 12.3 Embedding System

**File**: `scripts/embedding_utils.py`

**Key Functions**:
- `embed_text(text, dim)` - Generate OpenAI ada-002 embedding with caching
- `tokenize(text)` - Tokenize text for lexical reranking

**Caching**:
- SHA-256 based cache keys
- Cache directory: `data/cache/embeddings/`
- MD5 hash validation for cache integrity

**Usage in kb.search**:
- Query embedding (`scripts/qa_step03_mcp.py:71-74`)
- Lexical tokenization (`scripts/qa_step03_mcp.py:122-123`)

### 12.4 Configuration Files

**mcp.tools.yaml**:
- Service endpoints, ports, timeouts
- Fallback policies
- See Section 5 (Configuration & Settings)

**compliance.template.yaml**:
- Critical rules (4 rules)
- Warning rules (2 rules)
- Prohibited phrases (5 phrases)
- See Section 4.5 (safety.check)

### 12.5 Gate-3 Documentation

**Docs**:
- `docs/evaluation.md` - Gate-3 metrics and thresholds
- `docs/commands.md` - Gate-3 execution commands
- `docs/troubleshooting.md` - Common MCP issues and fixes

**Reports**:
- `reports/qa/step03_mcp.json` - JSON report from last Gate-3 run
- `reports/qa/step03_mcp.md` - Markdown report from last Gate-3 run

---

## Summary

This document provides comprehensive documentation of the **5 MCP tool services** and their HTTP server implementations:

1. **kb.search** (port 7801): Fully implemented vector search with lexical reranking
2. **web.fetch** (port 7802): Stub implementation (mock responses)
3. **link.resolve** (port 7803): Stub implementation (mock responses)
4. **crm.lookup** (port 7804): Stub implementation (mock responses)
5. **safety.check** (port 7805): Fully implemented compliance validation service

**Key Findings**:
- MCP services enable modular tool integration via HTTP protocol
- Two services (kb.search, safety.check) have production-ready implementations
- Three services (web.fetch, link.resolve, crm.lookup) are stubs for testing
- All services follow consistent patterns (JSON protocol, error codes, health checks)
- Gate-3 validates service health, contract conformance, and latency budgets
- Integration with LangGraph nodes via aiohttp ClientSession

**Next Steps**: See Part 6 for detailed LangGraph agent implementation and tool orchestration.

---

**Research Completed**: 2025-10-20 16:30:56 EDT
**Total Lines**: ~1200 lines
**Status**: Complete
