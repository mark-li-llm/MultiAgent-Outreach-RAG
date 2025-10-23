# Part 8: Configuration & Operations

**Research Date**: 2025-10-20 16:32:32 EDT
**Git Commit**: c4d22f8e35ca9bf3bd79a3d5f41cc87bd66f4d27
**Branch**: agent-weaviate
**Repository**: agent-weaviate

---

## 1. Overview

This document provides comprehensive documentation of all configuration files, operational procedures, and troubleshooting guidance for the multi-agent RAG system.

### Configuration System Summary

The system uses a **distributed configuration loading pattern** where YAML and JSON files in `configs/` are loaded on-demand by individual scripts. There is no central configuration manager—each script loads only the configs it needs using utility functions.

**10 Core Configuration Files**:
1. `vector.indexing.yaml` - Embedding model and vector index parameters
2. `router.heuristics.yaml` - Query routing rules and persona biases
3. `mcp.tools.yaml` - MCP service endpoints and timeouts
4. `langgraph.nodes.yaml` - Agent graph topology and node timeouts
5. `metadata.dictionary.yaml` - Metadata field definitions
6. `normalization.rules.yaml` - HTML normalization selectors
7. `eval.prompts.yaml` - Evaluation prompts and persona keywords
8. `agents.schema.yaml` - Agent schema definitions
9. `compliance.template.yaml` - Compliance templates
10. `chunking.config.json` - Document chunking parameters

**2 Environment Configuration Files**:
- `envs/age.yaml` - Primary Python 3.13 environment (Gate-1, Gate-3-8, graph execution)
- `envs/ageFaiss.yaml` - FAISS-specific Python 3.12 environment (Gate-2 only)

**Runtime Configuration**:
- `.env` file - API keys and secrets (git-ignored)
- Environment variables - Runtime behavior overrides (AG1_*, AG7_*, etc.)

### Environment Setup Summary

The system requires **two separate conda environments** to avoid OpenMP runtime conflicts:

1. **`age`** (Python 3.13) - Primary environment for most operations
2. **`ageFaiss`** (Python 3.12) - Isolated FAISS environment to avoid OpenMP library conflicts

**Critical**: Never install pip `faiss-cpu` in the `age` environment. This causes OMP Error #15 due to duplicate OpenMP libraries.

---

## 2. Architecture & Design

### Configuration Loading Architecture

The system uses **lazy, on-demand configuration loading**:

- **No centralized config manager**: Each script loads configs independently
- **Load-time validation**: Configs validated when accessed, not at startup
- **Defensive defaults**: Extensive use of `.get(key, default)` prevents crashes
- **Environment variable overrides**: Runtime behavior tuning via env vars

#### Configuration Loading Flow

```
Script Execution
    ↓
Load Config File (YAML/JSON)
    ↓
Extract Needed Sections
    ↓
Apply Defaults via .get()
    ↓
Check Environment Variable Overrides
    ↓
Use Config Values
```

#### Key Design Principles

1. **Stateless**: Scripts don't share config state
2. **Fail Fast**: Required fields validated early (e.g., embedding.dim)
3. **Graceful Degradation**: Optional configs use sensible defaults
4. **Late Binding**: Configs loaded when needed, not at import time
5. **No Hot Reload**: Config changes require script restart

### Environment Isolation Strategy

The two-environment architecture exists to prevent OpenMP runtime conflicts:

**Problem**: FAISS (via pip) bundles `libomp.dylib`, which conflicts with conda's OpenBLAS+OpenMP libraries, causing:
```
OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
```

**Solution**:
- `age` environment - Uses conda OpenBLAS+OpenMP, NO pip faiss-cpu
- `ageFaiss` environment - Uses conda faiss-cpu=1.9.* (manages OpenMP correctly)

**Usage**:
- Run Gate-2 (FAISS indexing) in `ageFaiss` environment ONLY
- Run all other operations in `age` environment

### Configuration Override Hierarchy

**Precedence Order** (highest to lowest):
1. Environment variables (e.g., `AG7_LATENCY_MULTIPLIER`)
2. Configuration files (e.g., `configs/router.heuristics.yaml`)
3. Hardcoded defaults in code (e.g., `.get("batch_size", 100)`)

**Example** (`scripts/common.py:480`):
```python
mode = os.getenv("AG_MCP_FALLBACK_MODE") or \
       config.get("fallback", {}).get("mode") or \
       "default"
```

---

## 3. File Inventory

### Primary Configuration Files (`configs/` directory)

All 10 configuration files:

| File | Format | Purpose | Used By |
|------|--------|---------|---------|
| `vector.indexing.yaml` | YAML | Embedding model, FAISS/Pinecone/Weaviate parameters | Gate-1, Gate-2 |
| `router.heuristics.yaml` | YAML | Query routing rules, persona bias, backend selection | Gate-4, Gate-7, graph execution |
| `mcp.tools.yaml` | YAML | MCP service endpoints (ports 7801-7805), timeouts | Gate-3, gate execution |
| `langgraph.nodes.yaml` | YAML | Agent graph node names and timeout configuration | Graph execution |
| `metadata.dictionary.yaml` | YAML | Metadata field schema and extraction rules | Stage 3 (extract_metadata.py) |
| `normalization.rules.yaml` | YAML | HTML normalization: remove/preserve selectors | Stage 2 (normalize_html.py) |
| `eval.prompts.yaml` | YAML | Evaluation prompts, persona keyword mappings | Gate-8, metadata extraction |
| `agents.schema.yaml` | YAML | Agent schema definitions | Agent configuration |
| `compliance.template.yaml` | YAML | Compliance templates | Compliance checks |
| `chunking.config.json` | JSON | Document chunking: target_tokens, overlap, boundaries | Stage 4 (chunk_documents.py) |

### Environment Configuration Files (`envs/` directory)

| File | Format | Purpose | Python Version |
|------|--------|---------|----------------|
| `envs/age.yaml` | YAML | Primary conda environment | 3.13 |
| `envs/ageFaiss.yaml` | YAML | FAISS-specific conda environment | 3.12 |

### Runtime Environment Files

| File | Format | Purpose | Status |
|------|--------|---------|--------|
| `.env` | ENV | API keys (OPENAI_API_KEY) | Git-ignored, user-created |
| `.gitignore` | Text | Git ignore patterns | Version-controlled |

### Data Directory Configuration Copies

These are processed copies in the data pipeline:

| File | Original Source |
|------|----------------|
| `data/final/dictionaries/metadata.dictionary.yaml` | `configs/metadata.dictionary.yaml` |
| `data/final/rules/normalization.rules.yaml` | `configs/normalization.rules.yaml` |

### Total Configuration Files: 19 files

- 10 core system configs (`configs/`)
- 2 environment configs (`envs/`)
- 1 runtime config (`.env`)
- 1 version control (`.gitignore`)
- 2 data directory copies
- 3 utility tool configs (`hack/linear/`: package.json, package-lock.json, tsconfig.json)

---

## 4. Core Components Deep Dive

### 4.1 Vector Indexing Configuration (`vector.indexing.yaml`)

**Location**: `configs/vector.indexing.yaml`
**Format**: YAML
**Used By**: Gate-1 (embeddings), Gate-2 (indexing)

**Full Schema**:
```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20
  notes: OpenAI text-embedding-ada-002 with caching and retry logic

faiss:
  type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128

pinecone:
  index_name: demo-index
  namespace: default
  metric: cosine
  notes: simulated manifest only (no network)

weaviate:
  class_name: Doc
  notes: schema-only manifest (simulated)
```

**Field Descriptions**:

**`embedding` section**:
- `model` (string) - Embedding model identifier (always `openai-ada-002`)
- `dim` (integer) - Vector dimension, MUST be 1536 for ada-002
- `batch_size` (integer) - API batch size (20 to avoid 8192 token limit)
- `notes` (string) - Implementation notes

**`faiss` section**:
- `type` (string) - Index type (`HNSW` for Hierarchical Navigable Small World)
- `metric` (string) - Distance metric (`L2` or `INNER_PRODUCT`)
- `M` (integer) - HNSW parameter: number of bi-directional links per node (default 32)
- `efConstruction` (integer) - HNSW build-time search depth (default 200)
- `efSearch` (integer) - HNSW query-time search depth (default 128)

**`pinecone` section**:
- `index_name` (string) - Pinecone index name
- `namespace` (string) - Pinecone namespace
- `metric` (string) - Distance metric (`cosine`)
- `notes` (string) - Implementation notes

**`weaviate` section**:
- `class_name` (string) - Weaviate class name
- `notes` (string) - Implementation notes

**Loading Patterns**:

**Gate-1** (`scripts/qa_step01_embeddings.py:31-39`):
```python
def read_yaml_dim(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dim = int(cfg.get("embedding", {}).get("dim") or 0)
    if not dim:
        raise ValueError("embedding.dim missing or invalid")
    return dim
```

**Gate-2** (`scripts/qa_step02_indexes.py:117-127`):
```python
faiss_cfg = cfg.get("faiss", {})
metric = (faiss.METRIC_L2 if str(faiss_cfg.get("metric", "L2")).upper() == "L2"
          else faiss.METRIC_INNER_PRODUCT)
M = int(faiss_cfg.get("M", 32))
efC = int(faiss_cfg.get("efConstruction", 200))
efS = int(faiss_cfg.get("efSearch", 128))
```

**Validation**:
- `embedding.dim` - REQUIRED, raises ValueError if missing
- `embedding.dim` - MUST equal 1536, validated in `embedding_utils.py:97`
- All FAISS params - Optional with defaults

**Tuning Guidelines**:
- **FAISS M**: Higher = better recall, more memory (16-128 typical)
- **efConstruction**: Higher = better index quality, slower build (100-500 typical)
- **efSearch**: Higher = better recall, slower search (64-256 typical)

---

### 4.2 Router Heuristics Configuration (`router.heuristics.yaml`)

**Location**: `configs/router.heuristics.yaml`
**Format**: YAML
**Used By**: Gate-4 (router testing), Gate-7 (retrieval eval), graph execution

**Full Schema**:
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
  - if:
      has_keywords: [results, earnings, fiscal, guidance, gaap, non-gaap, rpo, 10-k, 10-q, 8-k]
    then:
      backend: pinecone
      reason: PR_QUERY
  - if:
      has_keywords: [api, apis, endpoint, schema, developer, example]
    then:
      backend: weaviate
      reason: FILTER_MATCH
  - if:
      has_keywords: [definition, what is, overview]
    then:
      backend: faiss
      reason: DEFINITION

fallback_order: [faiss, weaviate, pinecone]
top_k_default: 10
```

**Field Descriptions**:

**`weights` section** (used for future weighted scoring):
- `similarity` (float) - Weight for semantic similarity (0.5 = 50%)
- `recency` (float) - Weight for document recency (0.3 = 30%)
- `diversity` (float) - Weight for result diversity (0.2 = 20%)

**`persona_bias` section**:
- Maps persona keys to preferred backend
- Persona keys: `vp_sales_ops`, `cio`, `vp_customer_experience`
- Backend values: `faiss`, `weaviate`, `pinecone`

**`rules` section** (list of rule objects):
- Each rule has `if` condition and `then` action
- **`if.has_keywords`** (list of strings) - Lowercase keywords to match in query
- **`then.backend`** (string) - Backend to route to (`faiss` | `weaviate` | `pinecone`)
- **`then.reason`** (string) - Reason code for logging (e.g., `PR_QUERY`, `FILTER_MATCH`)

**`fallback_order` (list of strings)**:
- Ordered list of backends to try if preferred backend fails
- Default: `[faiss, weaviate, pinecone]`

**`top_k_default` (integer)**:
- Default number of results to retrieve
- Default: 10

**Loading Pattern** (`scripts/router_core.py:27-37`):
```python
def load_router_config(path: str = ROUTER_CONF) -> Dict[str, Any]:
    if not os.path.exists(path):
        # Sensible defaults if file missing
        return {
            "weights": {"similarity": 0.6, "recency": 0.3, "diversity": 0.1},
            "persona_bias": {},
            "rules": [],
            "fallback_order": ["faiss", "weaviate", "pinecone"],
            "top_k_default": 10,
        }
    return _load_yaml(path)
```

**Usage Pattern** (`scripts/router_core.py:72-100`):
```python
cfg = load_router_config()

# Check keyword rules
for rule in cfg.get("rules", []):
    cond = rule.get("if", {})
    kws = [str(x).lower() for x in cond.get("has_keywords", [])]
    if any(kw in query_lower for kw in kws):
        return rule.get("then", {}).get("backend"), rule.get("then", {}).get("reason")

# Check persona bias
pb = cfg.get("persona_bias", {})
if persona in pb:
    return pb[persona], f"PERSONA({persona})"

# Use fallback order
return cfg.get("fallback_order", ["faiss"])[0], "DEFAULT"
```

**Decision Flow**:
1. Check keyword rules (first match wins)
2. Check persona bias
3. Use first backend in fallback_order

---

### 4.3 MCP Tools Configuration (`mcp.tools.yaml`)

**Location**: `configs/mcp.tools.yaml`
**Format**: YAML
**Used By**: Gate-3 (MCP validation), graph execution

**Full Schema**:
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

fallback:
  mode: default
  policy:
    log_downgrades: true
    retry_attempts: 1
    connection_timeout_ms: 2000
    warn_on_offline: true
    warn_on_external: false
```

**Field Descriptions**:

**`tools` section** (dictionary of tool configs):
- **Tool names**: `kb.search`, `web.fetch`, `link.resolve`, `crm.lookup`, `safety.check`
- Each tool has:
  - `host` (string) - Server bind address (default `127.0.0.1`)
  - `port` (integer) - Server port (7801-7805)
  - `timeout_ms` (integer) - Request timeout in milliseconds (default 2000)

**`fallback` section**:
- **`mode`** (string) - Fallback mode: `default` | `warn` | `strict`
  - `default` - Silent fallback to offline mode
  - `warn` - Log warnings on downgrades
  - `strict` - Fail fast on service unavailability
- **`policy` subsection**:
  - `log_downgrades` (boolean) - Log when falling back to simpler modes
  - `retry_attempts` (integer) - Number of connection retry attempts
  - `connection_timeout_ms` (integer) - Connection timeout
  - `warn_on_offline` (boolean) - Warn when falling back to offline mode
  - `warn_on_external` (boolean) - Warn when using external services

**Loading Pattern** (`scripts/router_core.py:40-42`):
```python
def load_mcp_map(path: str = MCP_CONF) -> Dict[str, Dict[str, Any]]:
    cfg = _load_yaml(path)
    return cfg.get("tools", {})
```

**Usage Pattern** (`scripts/qa_step03_mcp.py:196-199`):
```python
cfg_t = cfg["tools"][tool]
site = web.TCPSite(r, cfg_t["host"], int(cfg_t["port"]))
await site.start()
```

**Environment Variable Override** (`scripts/common.py:480`):
```python
mode_str = os.getenv("AG_MCP_FALLBACK_MODE") or \
           (config.get("fallback") or {}).get("mode") or \
           "default"
```

**Port Assignments**:
- **7801** - `kb.search` - Knowledge base vector search
- **7802** - `web.fetch` - Web content fetching
- **7803** - `link.resolve` - URL canonicalization
- **7804** - `crm.lookup` - CRM term lookup
- **7805** - `safety.check` - Content moderation

---

### 4.4 LangGraph Nodes Configuration (`langgraph.nodes.yaml`)

**Location**: `configs/langgraph.nodes.yaml`
**Format**: YAML
**Used By**: Graph execution (run_graph_langgraph.py)

**Full Schema**:
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

**Field Descriptions**:

**`nodes` section** (list of strings):
- Ordered list of node names in execution sequence
- 8 nodes total in LangGraph workflow

**`timeouts_ms` section** (dictionary):
- Per-node timeout in milliseconds
- Node names must match `nodes` list

**Node Execution Flow**:
```
Intake → Planner → Retriever → Synthesizer → Consolidator → Stylist → A2A → Assembler
```

**Node Responsibilities**:
1. **Intake** (2s) - Parse user request, extract company/persona
2. **Planner** (2s) - Generate retrieval plan with search queries
3. **Retriever** (10s) - Execute searches across backends (longest timeout)
4. **Synthesizer** (5s) - Synthesize retrieved content into draft
5. **Consolidator** (3s) - Consolidate and refine content
6. **Stylist** (3s) - Apply persona-specific styling
7. **A2A** (3s) - Agent-to-agent compliance review
8. **Assembler** (2s) - Final assembly and formatting

**Loading Pattern**: Loaded by graph execution script, config defines topology

**Timeout Tuning**:
- **Retriever** has longest timeout (10s) due to multi-backend searches
- **Intake/Planner/Assembler** shortest (2s) as they're lightweight
- **Synthesizer** moderate (5s) for LLM generation

---

### 4.5 Metadata Dictionary Configuration (`metadata.dictionary.yaml`)

**Location**: `configs/metadata.dictionary.yaml`
**Format**: YAML
**Used By**: Stage 3 (extract_metadata.py)

**Full Schema**:
```yaml
fields:
  doc_id: {type: string}
  company: {type: string}
  doctype: {type: string}
  title: {type: string}
  publish_date: {type: string, nullable: true}
  url: {type: string}
  final_url: {type: string}
  source_domain: {type: string}
  section: {type: string}
  topic: {type: string}
  persona_tags: {type: array}
  language: {type: string}
  text: {type: string}
  word_count: {type: integer}
  token_count: {type: integer}
  ingestion_ts: {type: string}
  hash_sha256: {type: string}
  html_title: {type: string, optional: true}
  meta_published_time: {type: string, optional: true}
  last_modified_http: {type: string, optional: true}
  byline: {type: string, optional: true}
  press_location: {type: string, optional: true}
  ticker_mentions: {type: array, optional: true}
  pdf_page_map: {type: any, optional: true}
```

**Field Types**:
- `string` - Text field
- `integer` - Numeric field
- `array` - List field
- `any` - Arbitrary JSON structure

**Field Attributes**:
- `nullable: true` - Field can be null
- `optional: true` - Field may be absent

**Required Fields** (no nullable/optional):
- `doc_id`, `company`, `doctype`, `title`, `url`, `final_url`, `source_domain`
- `section`, `topic`, `persona_tags`, `language`, `text`
- `word_count`, `token_count`, `ingestion_ts`, `hash_sha256`

**Optional Fields**:
- `html_title` - Title extracted from HTML <title> tag
- `meta_published_time` - Publish time from meta tags
- `last_modified_http` - Last-Modified HTTP header
- `byline` - Article author/byline
- `press_location` - Press release location (e.g., "SAN FRANCISCO")
- `ticker_mentions` - Stock ticker symbols mentioned (e.g., ["CRM"])
- `pdf_page_map` - PDF page mapping for multi-page documents

**Usage**: Schema defines required and optional fields for normalized documents

---

### 4.6 Normalization Rules Configuration (`normalization.rules.yaml`)

**Location**: `configs/normalization.rules.yaml`
**Format**: YAML
**Used By**: Stage 2 (normalize_html.py)

**Full Schema**:
```yaml
remove_selectors:
  - nav
  - footer
  - '[aria-label*="cookie"]'
  - '.share'
  - '.social'
  - '.newsletter'
  - '.breadcrumb'
  - '.sidebar'
  - script
  - style
  # XBRL Inline format metadata removal
  - 'ix\3A header'
  - 'ix\3A hidden'
  - 'ix\3A resources'
  - 'ix\3A references'
  - 'xbrli\3A context'
  - 'xbrli\3A unit'

preserve_selectors:
  - article
  - main
  - '.content'
  - '.entry-content'
  - '#content'

newline_blocks:
  - p
  - div
  - section
  - li

heading_levels:
  - h1
  - h2
  - h3
```

**Field Descriptions**:

**`remove_selectors` (list of CSS selectors)**:
- Elements to remove from HTML before text extraction
- Includes navigation, footer, social media widgets, newsletters
- XBRL selectors for SEC filing metadata (uses `\3A` for colon escape)

**`preserve_selectors` (list of CSS selectors)**:
- Primary content containers to preserve
- Used to identify main content area

**`newline_blocks` (list of HTML tags)**:
- Block elements that should create line breaks in text extraction
- Ensures proper paragraph/section separation

**`heading_levels` (list of HTML tags)**:
- Heading tags to preserve (h1, h2, h3)
- Maintains document structure in extracted text

**XBRL Handling**:
SEC filings use XBRL Inline format with namespaced tags (`ix:header`, `xbrli:context`). CSS selectors escape colons as `\3A` (hexadecimal code) for compatibility.

---

### 4.7 Evaluation Prompts Configuration (`eval.prompts.yaml`)

**Location**: `configs/eval.prompts.yaml`
**Format**: YAML
**Used By**: Gate-8 (generation eval), metadata extraction

**Full Schema**:
```yaml
personas:
  vp_customer_experience:
    - nps
    - csat
    - contact center
    - omnichannel
    - agent productivity
    - self-service
    - first contact resolution
  cio:
    - data integration
    - governance
    - security
    - tco
    - platform
    - apis
    - real-time
  vp_sales_ops:
    - pipeline
    - forecast accuracy
    - win rate
    - productivity
    - automation
```

**Field Descriptions**:

**`personas` section** (dictionary):
- Maps persona names to keyword lists
- Used for:
  1. Persona-specific keyword detection in generated emails (Gate-8)
  2. Metadata tagging during document processing

**Persona Keywords**:

**vp_customer_experience**:
- Focus: Customer experience metrics and support operations
- Keywords: NPS, CSAT, contact center, omnichannel, agent productivity, self-service, FCR

**cio**:
- Focus: Technical platform and integration concerns
- Keywords: Data integration, governance, security, TCO, platform, APIs, real-time

**vp_sales_ops**:
- Focus: Sales performance and efficiency
- Keywords: Pipeline, forecast accuracy, win rate, productivity, automation

**Usage in Metadata Extraction** (`scripts/extract_metadata.py:168-178`):
```python
persona_cfg = load_yaml("configs/eval.prompts.yaml")
personas = persona_cfg.get("personas", {})
for key, kws in personas.items():
    for kw in kws:
        if kw.lower() in text_lower:
            persona_tags.append(key)
            break
```

**Usage in Gate-8**: Checks if generated emails contain persona-specific keywords

---

### 4.8 Agents Schema Configuration (`agents.schema.yaml`)

**Location**: `configs/agents.schema.yaml`
**Format**: YAML
**Used By**: Agent configuration (not directly loaded in examined scripts)

**Purpose**: Defines agent schema structures for agent-to-agent communication and state management.

**Note**: Not directly loaded in the main pipeline scripts examined during research. Likely used for advanced agent coordination features.

---

### 4.9 Compliance Template Configuration (`compliance.template.yaml`)

**Location**: `configs/compliance.template.yaml`
**Format**: YAML
**Used By**: Compliance checks (Gate-6, A2A node)

**Purpose**: Defines compliance rules and templates for email generation validation.

**Note**: Not directly loaded in the main pipeline scripts examined during research. Used during agent-to-agent compliance review in Gate-6.

---

### 4.10 Chunking Configuration (`chunking.config.json`)

**Location**: `configs/chunking.config.json`
**Format**: JSON
**Used By**: Stage 4 (chunk_documents.py)

**Full Schema**:
```json
{
  "tokenizer": "cl100k_base",
  "target_tokens": 800,
  "overlap_tokens": 120,
  "short_doc_threshold_tokens": 350,
  "boundary_tolerance_chars": 50
}
```

**Field Descriptions**:

- **`tokenizer`** (string) - Tokenizer identifier (`cl100k_base` for GPT-3.5/4)
- **`target_tokens`** (integer) - Target chunk size in tokens (default 800)
- **`overlap_tokens`** (integer) - Overlap between chunks (default 120 = 15%)
- **`short_doc_threshold_tokens`** (integer) - Threshold for short documents (default 350)
  - Documents below this are kept as single chunk
- **`boundary_tolerance_chars`** (integer) - Character tolerance for finding chunk boundaries (default 50)
  - Allows +/- 50 chars to find natural boundaries (sentence/paragraph)

**Loading Pattern** (`scripts/chunk_documents.py:240`):
```python
cfg = load_config("configs/chunking.config.json")
target = int(cfg.get("target_tokens", 800))
overlap = int(cfg.get("overlap_tokens", 120))
short_thresh = int(cfg.get("short_doc_threshold_tokens", 350))
tol_chars = int(cfg.get("boundary_tolerance_chars", 50))
```

**Chunking Strategy**:
1. If document < `short_doc_threshold_tokens`, keep as single chunk
2. Otherwise, split into `target_tokens` chunks with `overlap_tokens` overlap
3. Use `boundary_tolerance_chars` to find natural boundaries (prefer sentence/paragraph breaks)

**Tuning Guidelines**:
- **target_tokens**: 600-1000 typical (balance between context and granularity)
- **overlap_tokens**: 10-20% of target (ensures no information loss at boundaries)
- **short_doc_threshold**: ~40-50% of target (avoid over-chunking short docs)

---

## 5. Configuration & Settings

### 5.1 Complete Configuration Schema Reference

This section provides the complete schema for each configuration file with all possible fields, types, and defaults.

#### Vector Indexing Schema

**File**: `configs/vector.indexing.yaml`

```yaml
embedding:
  model: string               # Required, default: "openai-ada-002"
  dim: integer                # Required, must be 1536
  batch_size: integer         # Optional, default: 20
  notes: string               # Optional

faiss:
  type: string                # Optional, default: "HNSW"
  metric: string              # Optional, default: "L2", options: ["L2", "INNER_PRODUCT"]
  M: integer                  # Optional, default: 32
  efConstruction: integer     # Optional, default: 200
  efSearch: integer           # Optional, default: 128

pinecone:
  index_name: string          # Optional, default: "demo-index"
  namespace: string           # Optional, default: "default"
  metric: string              # Optional, default: "cosine"
  notes: string               # Optional

weaviate:
  class_name: string          # Optional, default: "Doc"
  notes: string               # Optional
```

#### Router Heuristics Schema

**File**: `configs/router.heuristics.yaml`

```yaml
weights:
  similarity: float           # Optional, default: 0.6
  recency: float              # Optional, default: 0.3
  diversity: float            # Optional, default: 0.1

persona_bias:
  <persona_key>: string       # Optional, value: "faiss"|"weaviate"|"pinecone"

rules:
  - if:
      has_keywords: [string]  # Required, list of lowercase keywords
    then:
      backend: string         # Required, "faiss"|"weaviate"|"pinecone"
      reason: string          # Required, reason code

fallback_order: [string]      # Optional, default: ["faiss", "weaviate", "pinecone"]
top_k_default: integer        # Optional, default: 10
```

#### MCP Tools Schema

**File**: `configs/mcp.tools.yaml`

```yaml
tools:
  <tool_name>:
    host: string              # Required, default: "127.0.0.1"
    port: integer             # Required, must be unique
    timeout_ms: integer       # Optional, default: 2000

fallback:
  mode: string                # Optional, default: "default", options: ["default", "warn", "strict"]
  policy:
    log_downgrades: boolean   # Optional, default: true
    retry_attempts: integer   # Optional, default: 1
    connection_timeout_ms: integer  # Optional, default: 2000
    warn_on_offline: boolean  # Optional, default: true
    warn_on_external: boolean # Optional, default: false
```

### 5.2 Default Values Summary

| Config File | Field | Default Value | Override Method |
|------------|-------|---------------|-----------------|
| `vector.indexing.yaml` | `embedding.batch_size` | 100 (code default) | Config file |
| `vector.indexing.yaml` | `faiss.M` | 32 | Config file |
| `vector.indexing.yaml` | `faiss.efConstruction` | 200 | Config file |
| `vector.indexing.yaml` | `faiss.efSearch` | 128 | Config file |
| `router.heuristics.yaml` | `weights.similarity` | 0.6 | Config file |
| `router.heuristics.yaml` | `fallback_order` | `["faiss", "weaviate", "pinecone"]` | Config file |
| `router.heuristics.yaml` | `top_k_default` | 10 | Config file |
| `mcp.tools.yaml` | `timeout_ms` | 2000 | Config file |
| `mcp.tools.yaml` | `fallback.mode` | "default" | Env var `AG_MCP_FALLBACK_MODE` |
| `chunking.config.json` | `target_tokens` | 800 | Config file |
| `chunking.config.json` | `overlap_tokens` | 120 | Config file |
| `chunking.config.json` | `short_doc_threshold_tokens` | 350 | Config file |
| `chunking.config.json` | `boundary_tolerance_chars` | 50 | Config file |

### 5.3 Override Mechanisms

**Three levels of configuration overrides**:

1. **Environment Variables** (highest priority)
   - Override runtime behavior
   - Examples: `AG7_LATENCY_MULTIPLIER=3.0`, `AG2_DISABLE_FAISS=1`
   - See Section 7 for complete list

2. **Configuration Files** (medium priority)
   - Persistent settings in `configs/`
   - Loaded at script runtime

3. **Code Defaults** (lowest priority)
   - Hardcoded in `.get(key, default)` calls
   - Last resort if config missing

**Example** (`scripts/qa_step01_embeddings.py:124`):
```python
batch_size = int(cfg.get("embedding", {}).get("batch_size") or 100)
```
Priority: Config file value → 100 (code default)

---

## 6. Data Structures & Schemas

### 6.1 Configuration File Formats

**YAML Files** (9 files):
- Use `yaml.safe_load()` for security
- Return empty dict `{}` if file empty
- Support comments with `#`

**JSON Files** (1 file):
- Use `json.load()` for parsing
- No comments allowed
- Strict format validation

### 6.2 Validation Schemas

The system does NOT use explicit JSON Schema or similar validation. Instead, it uses **runtime validation patterns**:

**Pattern 1: Required Field Validation**
```python
# Raise error if required field missing
dim = int(cfg.get("embedding", {}).get("dim") or 0)
if not dim:
    raise ValueError("embedding.dim missing or invalid")
```

**Pattern 2: Existence Check with Default**
```python
# Return default config if file missing
if not os.path.exists(path):
    return {"weights": {...}, "rules": [], ...}
```

**Pattern 3: Type Coercion with Fallback**
```python
# Try to parse, use default on failure
try:
    latency_multiplier = float(os.getenv("AG7_LATENCY_MULTIPLIER", "1.0"))
except ValueError:
    latency_multiplier = 1.0
```

### 6.3 Document Metadata Schema

Defined in `configs/metadata.dictionary.yaml` (see Section 4.5).

**Required Fields**: 14 fields (doc_id, company, doctype, title, url, final_url, source_domain, section, topic, persona_tags, language, text, word_count, token_count, ingestion_ts, hash_sha256)

**Optional Fields**: 7 fields (html_title, meta_published_time, last_modified_http, byline, press_location, ticker_mentions, pdf_page_map)

**Quality Gate Check** (Gate-2):
- Checks percentage of documents missing required metadata
- Threshold: < 2% missing allowed
- Check ID: G2-04

---

## 7. External Dependencies

### 7.1 Conda

**Executable Path**: `/Users/liyunxiao/anaconda3/bin/conda`

**Two Environments Required**:

1. **`age`** (Python 3.13) - Primary environment
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
   ```

2. **`ageFaiss`** (Python 3.12) - FAISS-only environment
   ```bash
   /Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
   ```

**Package Dependencies**:

**`age` environment** (25 packages):
- **Python**: 3.13
- **Core**: aiohttp, pyyaml, pyarrow>=21, numpy>=2.3, certifi, openblas, llvm-openmp
- **OpenAI**: openai>=1.0.0, python-dotenv>=1.0.0, tenacity>=8.2.0
- **LangGraph**: langgraph>=0.2.20, langgraph-checkpoint-sqlite>=1.0.0, langchain-core>=0.3.0, langchain-openai>=0.2.0, langsmith>=0.1.0, aiosqlite>=0.19.0
- **IMPORTANT**: NO pip `faiss-cpu` package

**`ageFaiss` environment** (11 packages):
- **Python**: 3.12 (older version for FAISS compatibility)
- **FAISS**: faiss-cpu=1.9.*
- **Core**: numpy=1.26.*, scipy, pyarrow=21.*, openblas, llvm-openmp, aiohttp, pyyaml, certifi

### 7.2 OpenAI API

**Required For**: Gate-1 (embedding generation)

**Setup**:
1. Create `.env` file in project root:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

2. Verify key format:
   ```bash
   grep OPENAI_API_KEY .env
   # Should show: OPENAI_API_KEY=sk-...
   ```

**Model Used**: `text-embedding-ada-002`
- Dimension: 1536
- Cost: $0.0001 per 1K tokens (as of 2024)

**Retry Logic**:
- 3 attempts with exponential backoff (4-10 seconds)
- Retries on: `APIError`, `APIConnectionError`, `RateLimitError`
- Implementation: `scripts/embedding_utils.py:70-84`

### 7.3 Vector Databases

**FAISS** (Facebook AI Similarity Search):
- **Environment**: `ageFaiss` ONLY
- **Version**: 1.9.* (conda-forge)
- **Usage**: Gate-2 (index building), Gate-7 (retrieval)
- **Bypass**: Set `AG2_DISABLE_FAISS=1` to skip

**Pinecone**:
- **Status**: Simulated (manifest-only, no network)
- **Purpose**: Demonstrates multi-backend support

**Weaviate**:
- **Status**: Simulated (manifest-only, no network)
- **Purpose**: Demonstrates multi-backend support

### 7.4 Additional Tools

**BeautifulSoup4**: HTML parsing (normalize_html.py)
**tiktoken**: Token counting (chunking, text processing)
**langdetect**: Language detection (metadata extraction)
**pdfminer.six**: PDF text extraction (SEC filings)

---

## 8. Execution & Usage

### 8.1 Environment Setup

**Step 1: Create Conda Environments**
```bash
# Create primary environment (age)
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml

# Create FAISS environment (ageFaiss)
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
```

**Step 2: Set Up API Key**
```bash
# Create .env file with OpenAI API key
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env

# Verify .env file
cat .env
```

**Step 3: Verify Environments**
```bash
# Check age environment packages
/Users/liyunxiao/anaconda3/bin/conda list -n age

# Check ageFaiss environment packages
/Users/liyunxiao/anaconda3/bin/conda list -n ageFaiss

# Verify no faiss-cpu in age environment
/Users/liyunxiao/anaconda3/bin/conda list -n age | grep faiss
# Should return nothing

# Verify faiss-cpu in ageFaiss environment
/Users/liyunxiao/anaconda3/bin/conda list -n ageFaiss | grep faiss
# Should show faiss-cpu=1.9.*
```

### 8.2 Running Quality Gates (Sequential)

**Gate-0: Baseline Checks**
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step00_baseline.py
```
- Checks: Inventory, chunks, eval seed
- Output: `reports/qa/step00_baseline.{json,md}`
- No arguments required

**Gate-1: Generate Embeddings**
```bash
# With manual confirmation
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step01_embeddings.py

# With auto-confirmation (skip prompt)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG1_AUTO_CONFIRM=1 \
  python scripts/qa_step01_embeddings.py
```
- Requires: `OPENAI_API_KEY` in `.env`
- Output: `data/vector/embeddings/embeddings.parquet`, `reports/qa/step01_embeddings.{json,md}`
- Environment: `age` (NOT ageFaiss)

**Gate-2: Build Indexes**
```bash
# CRITICAL: Use ageFaiss environment
/Users/liyunxiao/anaconda3/bin/conda run -n ageFaiss python scripts/qa_step02_indexes.py

# To skip FAISS (avoid OpenMP conflicts)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG2_DISABLE_FAISS=1 \
  python scripts/qa_step02_indexes.py
```
- Requires: `data/vector/embeddings/embeddings.parquet` from Gate-1
- Output: `data/vector/faiss/`, `data/vector/pinecone/`, `data/vector/weaviate/`, `reports/qa/step02_indexes.{json,md}`
- **Environment**: `ageFaiss` ONLY (unless FAISS disabled)

**Gate-3: Validate MCP Tools**
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step03_mcp.py
```
- Starts stub servers on ports 7801-7805
- Output: `reports/qa/step03_mcp.{json,md}`
- Servers automatically shut down after validation

**Gate-4: Test Router**
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step04_router.py
```
- Tests routing logic from `configs/router.heuristics.yaml`
- Output: `reports/qa/step04_router.{json,md}`

**Gate-5: Validate Graph**
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step05_graph.py
```
- Validates LangGraph workflow
- Output: `reports/qa/step05_graph.{json,md}`

**Gate-6: Test A2A & Compliance**
```bash
# Requires session-id from previous graph execution
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/qa_step06_a2a.py --session-id <session-id>
```
- Requires: Completed graph execution with session ID
- Output: `reports/qa/step06_a2a.{json,md}`

**Gate-7: Retrieval Evaluation**
```bash
# Standard run (strict)
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step07_retrieval_eval.py

# Relaxed settings (recommended for dev)
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  AG7_DEBUG=1 \
  python scripts/qa_step07_retrieval_eval.py
```
- Requires: Indexes from Gate-2, embeddings from Gate-1
- Output: `reports/qa/step07_retrieval_eval.{json,md}`, `reports/router/step07_retrieval_trace.jsonl`
- Metrics: recall@10, nDCG@5, median_latency

**Gate-8: Generation Evaluation**
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age python scripts/qa_step08_generation_eval.py
```
- Runs 10 complete graph executions
- Output: `reports/qa/step08_generation_eval.{json,md}`, individual session outputs
- Metrics: structural_pass_rate, critical_flags_total, persona_keyword_hits

### 8.3 Graph Execution

**LangGraph Implementation (Recommended)**
```bash
/Users/liyunxiao/anaconda3/bin/conda run -n age \
  python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id test-session-001
```

**Original Implementation (For Comparison)**
```bash
python3 scripts/run_graph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id test-session-001
```

**Arguments**:
- `--company` (optional, default: `Salesforce`) - Target company name
- `--persona` (optional, default: `vp_customer_experience`) - Target persona
  - Options: `vp_customer_experience`, `cio`, `vp_sales_ops`
- `--session-id` (optional, auto-generated if not provided) - Session identifier

**Output Location**: `outputs/<session-id>/`
- `email.json` - Complete email with metadata
- `email.txt` - Plain text email
- `trace.json` - Execution trace
- `state_*.json` - Intermediate state snapshots

### 8.4 Environment Variables Reference

Comprehensive table of all environment variables:

| Variable | Type | Default | Purpose | Used In |
|----------|------|---------|---------|---------|
| `OPENAI_API_KEY` | String | None (required) | OpenAI API authentication | Gate-1, graph execution |
| `AG1_AUTO_CONFIRM` | Boolean | Not set | Skip cost confirmation | Gate-1 |
| `AG2_DISABLE_FAISS` | Boolean | `"0"` | Skip FAISS to avoid OpenMP | Gate-2 |
| `AG7_IGNORE_COVERAGE` | Boolean | `"0"` | Skip coverage gating | Gate-7 |
| `AG7_LATENCY_MULTIPLIER` | Float | `1.0` | Relax latency budgets | Gate-7 |
| `AG7_ANALYZE_TOPK` | Integer | `10` | Retrieval depth | Gate-7 |
| `AG7_NEAR_SEQ_TOL` | Integer | `1` | Near-miss tolerance | Gate-7 |
| `AG7_TOPK_SLICES` | String | `"1,3,5,10"` | Recall curve k-values | Gate-7 |
| `AG7_DEBUG` | Boolean | `"1"` | Master debug switch | Gate-7 |
| `AG7_TRACE` | Boolean | Inherits from DEBUG | Enable trace logging | Gate-7 |
| `AG7_TRACE_TOPK` | Integer | `10` | Trace depth | Gate-7 |
| `AG7_TRACE_SUCCESSES` | Boolean | Inherits from DEBUG | Include successes in trace | Gate-7 |
| `AR_USER_AGENT` | String | `"AccountResearchMVP/1.0"` | HTTP User-Agent | Data collection |
| `AR_GLOBAL_RPS` | Float | `6.0` | HTTP rate limit | Data collection |
| `AG_MCP_FALLBACK_MODE` | String | `"default"` | MCP fallback behavior | Gate-3, graph |

**Environment Variable Usage Examples**:

```bash
# Gate-1: Auto-confirm embeddings
AG1_AUTO_CONFIRM=1 conda run -n age python scripts/qa_step01_embeddings.py

# Gate-2: Skip FAISS
AG2_DISABLE_FAISS=1 conda run -n age python scripts/qa_step02_indexes.py

# Gate-7: Relaxed evaluation
AG7_IGNORE_COVERAGE=1 \
AG7_LATENCY_MULTIPLIER=3.0 \
AG7_DEBUG=1 \
AG7_TRACE=1 \
conda run -n age python scripts/qa_step07_retrieval_eval.py

# Data collection: Custom rate limit
AR_GLOBAL_RPS=10 python scripts/fetch_investor_news.py --limit 50
```

### 8.5 Data Collection & Processing Pipeline

**Phase A: SEC & Financial Documents**

```bash
# Step 1: Fetch SEC filings
python scripts/fetch_sec_filings.py --limit 10

# Step 2: Fetch investor news
python scripts/fetch_investor_news.py --since 2024-01-01 --limit 20

# Step 3: Normalize Phase A
python scripts/normalize_html.py --phase A

# Step 4: Extract metadata Phase A
python scripts/extract_metadata.py --phase A
```

**Phase B: Other Documents**

```bash
# Step 1: Fetch developer docs
python scripts/fetch_dev_docs.py --limit 4

# Step 2: Fetch help docs
python scripts/fetch_help_docs.py

# Step 3: Fetch product docs
python scripts/fetch_product_docs.py --limit 3

# Step 4: Fetch newsroom
python scripts/fetch_newsroom_rss.py --since 2024-01-01 --limit 30

# Step 5: Fetch Wikipedia
python scripts/fetch_wikipedia.py

# Step 6: Normalize Phase B
python scripts/normalize_html.py --phase B

# Step 7: Extract metadata Phase B
python scripts/extract_metadata.py --phase B
```

**Unified Processing (After Both Phases)**

```bash
# Step 8: Chunk documents
python scripts/chunk_documents.py

# Step 9: Deduplicate chunks
python scripts/dedupe_chunks.py

# Step 10: Build eval seed
python scripts/build_eval_seed.py

# Step 11: Verify all stages
python scripts/qa_verify_collection.py
python scripts/qa_verify_normalization.py
python scripts/qa_verify_metadata.py
python scripts/qa_verify_chunking.py
python scripts/qa_verify_dedupe.py
python scripts/qa_verify_eval_seed.py
```

---

## 9. Code Patterns & Conventions

### 9.1 Configuration Loading Patterns

**Pattern 1: YAML Loading with Import Guard**
```python
# At top of file
try:
    import yaml
except Exception:
    yaml = None

# In loader function
def load_yaml(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
```

**Used in**:
- `scripts/router_core.py:9-24`
- `scripts/qa_step01_embeddings.py:17-39`
- `scripts/qa_step02_indexes.py:13-43`
- `scripts/normalize_html.py:20-26`

**Pattern 2: JSON Loading**
```python
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
```

**Used in**:
- `scripts/chunk_documents.py:16-18`
- `scripts/common.py:309-311`

**Pattern 3: Config with Defaults on Missing File**
```python
def load_router_config(path: str = ROUTER_CONF) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "weights": {"similarity": 0.6, "recency": 0.3, "diversity": 0.1},
            "persona_bias": {},
            "rules": [],
            "fallback_order": ["faiss", "weaviate", "pinecone"],
            "top_k_default": 10,
        }
    return _load_yaml(path)
```

**Used in**: `scripts/router_core.py:27-37`

### 9.2 Defensive Access Pattern

**Standard Pattern Throughout Codebase**:
```python
# Always use .get() with defaults
value = config.get("key", default_value)

# For nested access
nested = config.get("parent", {}).get("child", default)

# With type coercion
int_value = int(config.get("number", 100))
```

**Examples**:
```python
# scripts/qa_step01_embeddings.py:124
batch_size = int(cfg.get("embedding", {}).get("batch_size") or 100)

# scripts/router_core.py:81
for rule in cfg.get("rules", []):
    cond = rule.get("if", {})
    kws = [str(x).lower() for x in cond.get("has_keywords", [])]

# scripts/chunk_documents.py:118-121
target = int(cfg.get("target_tokens", 800))
overlap = int(cfg.get("overlap_tokens", 120))
```

### 9.3 Environment Variable Patterns

**Pattern 1: Boolean Environment Variables**
```python
# Check for truthy string values
auto_confirm = os.getenv("AG1_AUTO_CONFIRM", "").lower() in ["1", "true", "yes", "y"]

# Simple "1" check
debug_flag = os.getenv("AG7_DEBUG", "1") == "1"
```

**Pattern 2: Numeric Environment Variables with Fallback**
```python
try:
    latency_multiplier = float(os.getenv("AG7_LATENCY_MULTIPLIER", "1.0"))
except ValueError:
    latency_multiplier = 1.0
```

**Pattern 3: Required Environment Variables**
```python
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in .env file:\n"
        "cp .env.example .env && edit .env"
    )
```

### 9.4 Error Handling Patterns

**Pattern 1: Graceful Import Failure**
```python
try:
    import faiss
except Exception:
    # Write disabled manifest and continue
    manifest = {"disabled": True, "reason": "faiss_import_failed", ...}
    with open(FAISS_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f)
    return len(vecs), 0.0
```

**Pattern 2: Retry Logic with Tenacity**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError)),
    reraise=True
)
def _call_openai_api(text: str) -> List[float]:
    response = client.embeddings.create(...)
    return response.data[0].embedding
```

**Pattern 3: Error Wrapping with Context**
```python
try:
    embedding = _call_openai_api(text)
except Exception as e:
    print(f"ERROR: OpenAI API failed: {e}")
    raise RuntimeError(
        f"OpenAI API call failed: {type(e).__name__}: {e}\n"
        f"Text length: {len(text)} chars\n"
        f"Check your API key and network connection."
    ) from e
```

### 9.5 Quality Gate Patterns

**Pattern 1: Three-Tier Status (GREEN/AMBER/RED)**
```python
# All checks pass → GREEN
if all(c["status"] == "PASS" for c in checks):
    status = "GREEN"
    next_action = "continue"
# Specific warning patterns → AMBER
elif len(warns) == 1 and warns[0]["id"].startswith("G3-03-"):
    status = "AMBER"
    next_action = "proceed_with_caution"
# Everything else → RED
else:
    status = "RED"
    next_action = "fix_and_rerun"
```

**Pattern 2: Dual Report Format (JSON + Markdown)**
```python
# Machine-readable JSON
machine = {
    "step": "step02_indexes",
    "gate": "Gate-2",
    "status": status,
    "checks": checks,
    "next_action": next_action,
    "timestamp": now_iso(),
}
with open("reports/qa/step02_indexes.json", "w") as f:
    json.dump(machine, f, indent=2)

# Human-readable Markdown
lines = [
    f"# STEP 2 — Index Build & Integrity (Gate‑2) — {status}",
    "",
    "Checks:",
    *[f"- {c['id']}: {c['metric']} = {c['actual']} -> {c['status']}" for c in checks],
    "",
    f"Go/No-Go: {'Go' if status in ('GREEN','AMBER') else 'No-Go'}"
]
with open("reports/qa/step02_indexes.md", "w") as f:
    f.write("\n".join(lines))
```

### 9.6 Convention Summary

| Convention | Pattern | Example |
|-----------|---------|---------|
| Config loading | Lazy, on-demand | Each script loads its own configs |
| Default values | `.get(key, default)` | `cfg.get("batch_size", 100)` |
| Nested access | Chained `.get()` | `cfg.get("faiss", {}).get("M", 32)` |
| Environment vars | Prefixed with `AG*` | `AG7_DEBUG`, `AG1_AUTO_CONFIRM` |
| Boolean env vars | String comparison | `== "1"` or `in ["1", "true", "yes"]` |
| Error messages | User-facing, actionable | "Check .env file", "Update config" |
| Validation | Runtime, fail-fast | Validate on access, not at startup |
| Reporting | JSON + Markdown | Dual format for machine + human |

---

## 10. Testing & Verification

### 10.1 Environment Verification

**Verify Conda Environments Exist**:
```bash
# List all conda environments
/Users/liyunxiao/anaconda3/bin/conda env list

# Should show:
# age                      /Users/liyunxiao/anaconda3/envs/age
# ageFaiss                 /Users/liyunxiao/anaconda3/envs/ageFaiss
```

**Verify Package Installation**:
```bash
# Check age environment
/Users/liyunxiao/anaconda3/bin/conda list -n age | grep -E 'python|openai|langgraph|pyarrow|numpy'

# Expected output includes:
# python                    3.13.*
# openai                    1.*
# langgraph                 0.2.20 or higher
# pyarrow                   21 or higher
# numpy                     2.3 or higher

# Check ageFaiss environment
/Users/liyunxiao/anaconda3/bin/conda list -n ageFaiss | grep -E 'python|faiss|numpy|pyarrow'

# Expected output includes:
# python                    3.12.*
# faiss-cpu                 1.9.*
# numpy                     1.26.*
# pyarrow                   21.*
```

**Verify NO faiss-cpu in age environment**:
```bash
/Users/liyunxiao/anaconda3/bin/conda list -n age | grep faiss
# Should return NOTHING (empty output)
```

### 10.2 Configuration File Verification

**Check All Config Files Exist**:
```bash
# List all config files
ls -la configs/

# Should show 10 files:
# agents.schema.yaml
# chunking.config.json
# compliance.template.yaml
# eval.prompts.yaml
# langgraph.nodes.yaml
# mcp.tools.yaml
# metadata.dictionary.yaml
# normalization.rules.yaml
# router.heuristics.yaml
# vector.indexing.yaml
```

**Validate YAML Syntax**:
```bash
# Check YAML files can be parsed
python3 -c "import yaml; yaml.safe_load(open('configs/vector.indexing.yaml'))"
python3 -c "import yaml; yaml.safe_load(open('configs/router.heuristics.yaml'))"
python3 -c "import yaml; yaml.safe_load(open('configs/mcp.tools.yaml'))"

# No output = success
# SyntaxError = invalid YAML
```

**Validate JSON Syntax**:
```bash
# Check JSON files can be parsed
python3 -c "import json; json.load(open('configs/chunking.config.json'))"

# No output = success
# JSONDecodeError = invalid JSON
```

### 10.3 API Key Verification

**Check .env File Exists**:
```bash
ls -la .env

# Should show:
# -rw-r--r--  1 user  staff  45 Oct 20 16:00 .env
```

**Verify API Key Format**:
```bash
grep OPENAI_API_KEY .env

# Should show:
# OPENAI_API_KEY=sk-...
```

**Test API Key Works**:
```bash
# Test OpenAI API connection
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $(grep OPENAI_API_KEY .env | cut -d= -f2)" \
  | head -n 20

# Expected: JSON response with model list
# Error: "Incorrect API key" = invalid key
```

**Test Embedding Generation**:
```python
# Test script: test_api_key.py
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in .env")
    exit(1)

try:
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input="test"
    )
    print(f"✅ API key valid. Dimension: {len(response.data[0].embedding)}")
except Exception as e:
    print(f"❌ API error: {e}")
```

Run: `conda run -n age python test_api_key.py`

### 10.4 Port Availability Check

**Check MCP Ports Free**:
```bash
# Check if ports 7801-7805 are in use
lsof -i :7801-7805

# Expected: No output (ports free)
# If output shows processes, kill them:
lsof -ti :7801-7805 | xargs kill
```

### 10.5 Quality Gate Verification

**Run Baseline Gate (Gate-0)**:
```bash
conda run -n age python scripts/qa_step00_baseline.py

# Check output
cat reports/qa/step00_baseline.md | grep "Go/No-Go"

# Expected: "Go/No-Go: Go" (GREEN or AMBER)
```

**Verify Embedding Cache**:
```bash
# After running Gate-1
ls -1 data/cache/embeddings/ | wc -l

# Expected: Number of cached embeddings (>0)
```

**Verify Indexes Built**:
```bash
# After running Gate-2
ls -la data/vector/faiss/
ls -la data/vector/pinecone/
ls -la data/vector/weaviate/

# Each should contain:
# - manifest.json
# - Index files
```

**Verify Report Format**:
```bash
# Check JSON report structure
cat reports/qa/step02_indexes.json | jq '.status, .checks[0].id'

# Expected: Status (GREEN/AMBER/RED) and first check ID
```

---

## 11. Known Issues & Limitations

### 11.1 OpenMP Error #15 (Critical)

**Symptom**:
```
OMP: Error #15: Initializing libomp.dylib, but found libiomp5.dylib already initialized.
```
Or segfault during FAISS operations.

**Cause**: Mixing pip `faiss-cpu` (which bundles `libomp`) with conda OpenBLAS+OpenMP in the same environment.

**Fix**:
1. **Always run Gate-2 in `ageFaiss` environment**:
   ```bash
   conda run -n ageFaiss python scripts/qa_step02_indexes.py
   ```

2. **NEVER install pip `faiss-cpu` in `age` environment**

3. If already installed, recreate environment:
   ```bash
   conda env remove -n age
   conda env create -f envs/age.yaml
   ```

**Prevention**: Use correct environment for each task. Gate-2 → `ageFaiss`, everything else → `age`.

**Workaround**: Set `AG2_DISABLE_FAISS=1` to skip FAISS entirely:
```bash
conda run -n age AG2_DISABLE_FAISS=1 python scripts/qa_step02_indexes.py
```

**Code Reference**: `scripts/qa_step02_indexes.py:83-113`

---

### 11.2 Recall = 0% (Embedding Mismatch)

**Symptom**: Retrieval evaluation (Gate-7) shows 0% recall despite having indexed documents.

**Cause**: Mismatched embeddings between documents and queries (using different embedding functions or random vectors).

**Diagnosis**:
```bash
# Check embedding dimensions
python3 -c "import pyarrow.parquet as pq; print(pq.read_table('data/vector/embeddings/embeddings.parquet').schema)"

# Should show: vector: list<item: float> (1536 items)
```

**Fix**: Ensure both document indexing (Gate-1) and query processing use `embed_text()` from `scripts/embedding_utils.py` with dimension 1536:

```python
from scripts.embedding_utils import embed_text

# For documents (Gate-1)
doc_vector = embed_text(doc_text, dim=1536)

# For queries (Gate-7)
query_vector = embed_text(query_text, dim=1536)
```

**Prevention**: Always use the centralized `embed_text()` function. Never:
- Use different embedding functions
- Generate random vectors
- Use different embedding models

**Code Reference**: `scripts/embedding_utils.py:86-128`

---

### 11.3 API Key Errors

**Symptom**: Embedding generation fails with authentication or rate limit errors.

**Cause**: Missing/invalid `OPENAI_API_KEY`, network issues, or API rate limiting.

**Diagnosis**:
```bash
# Check .env file exists
ls -la .env

# Check key format
grep OPENAI_API_KEY .env
# Should show: OPENAI_API_KEY=sk-...

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $(grep OPENAI_API_KEY .env | cut -d= -f2)"
```

**Fix**:
1. Create `.env` file with valid API key:
   ```bash
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

2. Verify key format (should start with `sk-`)

3. For rate limits, retry logic handles transient errors automatically (3 attempts with exponential backoff)

**Prevention**: Always set `OPENAI_API_KEY` in `.env` before running Gate-1.

**Code Reference**: `scripts/embedding_utils.py:23-29`, `scripts/embedding_utils.py:70-84`

---

### 11.4 Port Conflicts

**Symptom**: "Port busy" or "Address already in use" errors when starting MCP services.

**Cause**: Another process is using ports 7801-7805.

**Diagnosis**:
```bash
# Check which processes are using MCP ports
lsof -i :7801-7805
```

**Fix**:
1. Kill existing processes:
   ```bash
   # Kill by PID
   kill <PID>

   # Or kill all processes on MCP ports
   lsof -ti :7801-7805 | xargs kill
   ```

2. Alternative: Update `configs/mcp.tools.yaml` to use different ports

**Prevention**: Always stop MCP services after use (Ctrl+C).

**Code Reference**: `scripts/qa_step03_mcp.py:196-199`

---

### 11.5 Gate-7 Latency Timeouts

**Symptom**: Gate-7 fails latency checks on slower hardware.

**Cause**: Default latency budgets are tight (1000ms median).

**Fix**: Use `AG7_LATENCY_MULTIPLIER` to relax budgets:
```bash
conda run -n age \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py
```

**Multiplier Values**:
- `1.0` - Default (strict)
- `2.0` - 2× latency budget (moderate)
- `3.0` - 3× latency budget (relaxed, recommended for dev)

**Code Reference**: `scripts/qa_step07_retrieval_eval.py:262`

---

### 11.6 Configuration Not Hot-Reloadable

**Limitation**: Configuration file changes require script restart. No dynamic reloading.

**Impact**: After editing configs, must:
1. Stop running scripts
2. Re-run scripts to pick up changes

**Workaround**: Use environment variables for runtime tuning (no restart needed):
```bash
# Instead of editing config, use env var
AG7_LATENCY_MULTIPLIER=2.0 conda run -n age python scripts/qa_step07_retrieval_eval.py
```

---

### 11.7 No Schema Validation

**Limitation**: Configuration files have no JSON Schema or formal validation. Invalid configs cause runtime errors.

**Impact**: Typos or incorrect types in configs only discovered when that config section is accessed.

**Mitigation**:
1. Test config changes by running relevant scripts
2. Use defensive `.get(key, default)` patterns
3. Check logs for config-related errors

**Example Error**:
```python
# If faiss.M is "thirty-two" (string) instead of 32 (integer)
M = int(faiss_cfg.get("M", 32))  # Raises ValueError
```

---

### 11.8 System Limitations

**Network Isolation**:
- Pinecone and Weaviate backends are simulated (manifest-only)
- No actual network calls to cloud services
- FAISS is the only functional vector backend

**Embedding Caching**:
- Cache is file-based (no distributed cache)
- Cache invalidation is manual (delete `data/cache/embeddings/`)
- No cache expiration or size limits

**Concurrency**:
- Most scripts use sequential processing with limited parallelism
- `--concurrency` flags available but parallel processing is basic
- No distributed execution support

**Platform Support**:
- Tested on macOS (Darwin)
- Linux support expected but untested
- Windows support unknown

---

## 12. References

### 12.1 Cross-References to Other Research Parts

This document (Part 8: Configuration & Operations) connects to:

- **Part 1: Data Collection** - Uses configuration for fetch scripts
- **Part 2: Normalization** - Uses `normalization.rules.yaml`
- **Part 3: Metadata Extraction** - Uses `metadata.dictionary.yaml`
- **Part 4: Chunking** - Uses `chunking.config.json`
- **Part 5: Embeddings & Indexing** - Uses `vector.indexing.yaml`, critical environment setup
- **Part 6: Routing & Retrieval** - Uses `router.heuristics.yaml`, `mcp.tools.yaml`
- **Part 7: LangGraph & Generation** - Uses `langgraph.nodes.yaml`, `eval.prompts.yaml`

### 12.2 Configuration Usage Map

**Where Each Config is Used**:

| Config File | Primary User Scripts | Quality Gates | Graph Execution |
|------------|---------------------|---------------|-----------------|
| `vector.indexing.yaml` | `qa_step01_embeddings.py`, `qa_step02_indexes.py` | Gate-1, Gate-2 | ✓ (via embedding_utils) |
| `router.heuristics.yaml` | `router_core.py`, `qa_step04_router.py`, `qa_step07_retrieval_eval.py` | Gate-4, Gate-7 | ✓ |
| `mcp.tools.yaml` | `qa_step03_mcp.py`, `common.py` | Gate-3 | ✓ |
| `langgraph.nodes.yaml` | `run_graph_langgraph.py`, `qa_step05_graph.py` | Gate-5 | ✓ |
| `metadata.dictionary.yaml` | `extract_metadata.py`, `qa_verify_metadata.py` | - | - |
| `normalization.rules.yaml` | `normalize_html.py`, `qa_verify_normalization.py` | - | - |
| `eval.prompts.yaml` | `extract_metadata.py`, `qa_step08_generation_eval.py` | Gate-8 | ✓ (persona keywords) |
| `agents.schema.yaml` | (Not directly used in examined scripts) | - | - |
| `compliance.template.yaml` | `qa_step06_a2a.py` | Gate-6 | ✓ (A2A node) |
| `chunking.config.json` | `chunk_documents.py`, `qa_verify_chunking.py` | - | - |

### 12.3 Documentation References

**Internal Documentation**:
- `README.md` - Main project documentation
- `CLAUDE.md` - Project-specific instructions (this file is the canonical source)
- `AGENTS.md` - Agent automation guidelines
- `docs/architecture.md` - Detailed system design
- `docs/commands.md` - Complete command reference
- `docs/configuration.md` - Configuration deep dive
- `docs/troubleshooting.md` - Debug playbook
- `docs/evaluation.md` - Quality gates and metrics
- `docs/envs.md` - Environment setup details

**Configuration Files** (Self-Documenting):
- `configs/*.yaml` - Contain inline comments and notes
- `envs/*.yaml` - Include package version requirements and critical warnings

### 12.4 Code References

**Configuration Loading**:
- `scripts/common.py` - Shared utility functions
- `scripts/embedding_utils.py` - Embedding generation with config validation
- `scripts/router_core.py` - Router config loading and decision logic

**Quality Gates**:
- `scripts/qa_step00_baseline.py` through `scripts/qa_step08_generation_eval.py` - 9 gate scripts
- All gates emit dual-format reports: `reports/qa/*.{json,md}`

**Data Pipeline**:
- `scripts/fetch_*.py` - 7 data collection scripts
- `scripts/normalize_html.py` - Stage 2 (uses normalization.rules.yaml)
- `scripts/extract_metadata.py` - Stage 3 (uses metadata.dictionary.yaml)
- `scripts/chunk_documents.py` - Stage 4 (uses chunking.config.json)
- `scripts/dedupe_chunks.py` - Stage 5

**Graph Execution**:
- `scripts/run_graph_langgraph.py` - LangGraph implementation (recommended)
- `scripts/run_graph.py` - Original implementation
- `scripts/langgraph_nodes.py` - Node implementations
- `scripts/langgraph_state.py` - State definitions

### 12.5 External References

**OpenAI API**:
- Model: `text-embedding-ada-002`
- Docs: https://platform.openai.com/docs/guides/embeddings
- Pricing: https://openai.com/pricing

**FAISS**:
- GitHub: https://github.com/facebookresearch/faiss
- Conda Package: https://anaconda.org/conda-forge/faiss-cpu
- Index Types: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

**LangGraph**:
- Docs: https://langchain-ai.github.io/langgraph/
- GitHub: https://github.com/langchain-ai/langgraph

**Conda**:
- Installation: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
- Environment Management: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

---

## Appendix A: Quick Reference

### Environment Setup (Copy-Paste)

```bash
# 1. Create environments
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml

# 2. Set up API key
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Verify setup
/Users/liyunxiao/anaconda3/bin/conda env list | grep -E 'age|ageFaiss'
cat .env | grep OPENAI_API_KEY
```

### Run All Quality Gates (Copy-Paste)

```bash
# Gate-0: Baseline
conda run -n age python scripts/qa_step00_baseline.py

# Gate-1: Embeddings (with auto-confirm)
conda run -n age AG1_AUTO_CONFIRM=1 python scripts/qa_step01_embeddings.py

# Gate-2: Indexes (CRITICAL: use ageFaiss)
conda run -n ageFaiss python scripts/qa_step02_indexes.py

# Gate-3: MCP
conda run -n age python scripts/qa_step03_mcp.py

# Gate-4: Router
conda run -n age python scripts/qa_step04_router.py

# Gate-5: Graph
conda run -n age python scripts/qa_step05_graph.py

# Gate-7: Retrieval (with relaxed settings)
conda run -n age \
  AG7_IGNORE_COVERAGE=1 \
  AG7_LATENCY_MULTIPLIER=3.0 \
  python scripts/qa_step07_retrieval_eval.py

# Gate-8: Generation
conda run -n age python scripts/qa_step08_generation_eval.py
```

### Run Graph Workflow (Copy-Paste)

```bash
conda run -n age python scripts/run_graph_langgraph.py \
  --company Salesforce \
  --persona vp_customer_experience \
  --session-id test-$(date +%Y%m%d-%H%M%S)
```

### Check Gate Status (Copy-Paste)

```bash
# View all gate statuses
for gate in step0{0..8}; do
  if [ -f "reports/qa/${gate}_*.md" ]; then
    echo "=== $(ls reports/qa/${gate}_*.md | head -1) ==="
    grep -E "^# |Go/No-Go:" reports/qa/${gate}_*.md | head -2
    echo
  fi
done
```

### Troubleshooting Quick Checks (Copy-Paste)

```bash
# Check environment setup
echo "=== Conda Environments ==="
/Users/liyunxiao/anaconda3/bin/conda env list | grep -E 'age|ageFaiss'

echo -e "\n=== API Key ==="
[ -f .env ] && echo "✓ .env exists" || echo "✗ .env missing"
grep -q "OPENAI_API_KEY=sk-" .env 2>/dev/null && echo "✓ API key format OK" || echo "✗ API key invalid"

echo -e "\n=== Config Files ==="
ls configs/*.{yaml,json} 2>/dev/null | wc -l | xargs echo "Config files:"

echo -e "\n=== MCP Ports ==="
lsof -i :7801-7805 2>/dev/null && echo "⚠ Ports in use" || echo "✓ Ports free"

echo -e "\n=== FAISS Check ==="
/Users/liyunxiao/anaconda3/bin/conda list -n age 2>/dev/null | grep -q faiss && echo "✗ faiss in age (BAD!)" || echo "✓ No faiss in age"
/Users/liyunxiao/anaconda3/bin/conda list -n ageFaiss 2>/dev/null | grep -q faiss && echo "✓ faiss in ageFaiss" || echo "✗ No faiss in ageFaiss"
```

---

## Appendix B: Configuration Templates

### Minimal vector.indexing.yaml
```yaml
embedding:
  model: openai-ada-002
  dim: 1536
  batch_size: 20

faiss:
  type: HNSW
  metric: L2
  M: 32
  efConstruction: 200
  efSearch: 128
```

### Minimal router.heuristics.yaml
```yaml
persona_bias:
  vp_sales_ops: pinecone
  cio: weaviate
  vp_customer_experience: faiss

rules: []

fallback_order: [faiss, weaviate, pinecone]
top_k_default: 10
```

### Minimal mcp.tools.yaml
```yaml
tools:
  kb.search:
    host: 127.0.0.1
    port: 7801
    timeout_ms: 2000

fallback:
  mode: default
```

### Minimal .env
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

---

**End of Part 8: Configuration & Operations**

**Document Statistics**:
- Total Lines: ~1800+
- Sections: 12 major sections + 2 appendices
- Config Files Documented: 10 core + 2 environment
- Environment Variables: 16 documented
- Known Issues: 8 documented
- Code References: 50+ file paths

**Next Steps**:
1. Review configuration files for accuracy
2. Test all command examples in clean environment
3. Validate all file paths and line numbers
4. Cross-reference with other parts (1-7) for consistency
