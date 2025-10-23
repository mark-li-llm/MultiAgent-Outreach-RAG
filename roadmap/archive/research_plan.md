# Roadmap Research Plan: Complete Codebase Documentation

## Executive Summary

This document outlines a comprehensive plan to create a **reference documentation roadmap** for the multi-agent RAG system. The goal is to document the CURRENT codebase in such detail that other coding agents can read the roadmap and understand everything WITHOUT needing to explore the code themselves.

**Total Scope**: 8 parts, 34-50 hours of effort, ~10,000-15,000 lines of documentation

**Documentation Philosophy**:
- Document what EXISTS (not what should exist)
- Include specific file paths and line numbers
- Provide code snippets and examples
- Enable coding agents to find answers independently

---

## Overview of Division Strategy

**8 Parts** based on **logical subsystems** that follow natural data/execution flow:

```
Part 1: Foundation (what/why/how big picture)
   “
Part 2: Data Pipeline (raw docs ’ processed data)
   “
Part 3: Vector System (text ’ embeddings ’ indexes)
   “
Part 4: Routing (query ’ backend selection)
   “
Part 5: MCP Services (tool contracts & implementations)
   “
Part 6: Agent System (LangGraph orchestration)
   “
Part 7: Quality Gates (validation & metrics)
   “
Part 8: Operations (config, execution, troubleshooting)
```

Each part is **independently readable** but builds on previous parts. A coding agent can start at Part 1 (overview) and drill down, OR jump to Part 6 (agents) if they already understand the foundation.

---

## Documentation Structure Template

### 12-Section Template (Used for Each Part)

Every part follows this consistent structure:

1. **Overview** - Executive summary, purpose, scope, key concepts, quick stats
2. **Architecture & Design** - System diagrams, data flow, design patterns, decisions
3. **File Inventory** - Complete file listing with purposes, line counts, dependencies
4. **Core Components Deep Dive** - Detailed code walkthroughs with line numbers
5. **Configuration & Settings** - All config files, schemas, defaults, overrides
6. **Data Structures & Schemas** - All data types, example payloads, validation
7. **External Dependencies** - APIs, libraries, authentication, retry logic
8. **Execution & Usage** - Commands, arguments, workflows, output locations
9. **Code Patterns & Conventions** - Naming, organization, patterns, anti-patterns
10. **Testing & Verification** - Test files, commands, coverage, quality gates
11. **Known Issues & Limitations** - Bugs, constraints, bottlenecks, workarounds
12. **References** - Links to other parts, external docs, key commits

---

## Part 1: System Overview & Architecture =Ð

### Research Goal
Understand the **big picture**: what this system does, how it's organized, and how all components fit together.

### Key Questions to Answer
1. What problem does this system solve? (Sales/IR/PR outreach)
2. What are the major subsystems? (pipeline, routing, agents, evaluation)
3. How do components interact? (data flow through 13 stages)
4. What's the technology stack? (Python, LangGraph, OpenAI, conda)
5. What's the directory structure? (scripts/, configs/, data/, etc.)
6. What are the main entry points? (run_graph_langgraph.py)

### Files to Analyze (Priority Order)

**High Priority** (must read):
- `README.md` - main documentation
- `CLAUDE.md` - project instructions
- `docs/architecture.md` - detailed architecture
- `scripts/run_graph_langgraph.py` - main entry point
- Top-level directory structure

**Medium Priority**:
- `AGENTS.md` - automation guidelines
- `README_DAY1.md` - milestone documentation
- `docs/commands.md` - command reference

**Low Priority**:
- Other docs/* files for context

### Research Agent Strategy

Launch **3 agents in parallel**:

**Agent 1: codebase-locator**
```
Prompt: "Find all main entry points, README files, and documentation in this repository.
Focus on understanding what this system does at a high level.
Look for: README.md, CLAUDE.md, docs/*, main execution scripts (run_*.py, main.py, etc.)"
```

**Agent 2: Explore (quick mode)**
```
Prompt: "Perform a quick exploration of the directory structure.
Map out all top-level directories (scripts/, configs/, data/, etc.) and explain their purposes.
Identify how many files exist in each major directory.
Look for patterns in file naming and organization."
```

**Agent 3: codebase-analyzer**
```
Prompt: "Analyze the main orchestration scripts to understand the execution flow.
Focus on: scripts/run_graph_langgraph.py and scripts/run_graph.py
Extract: what inputs they take, what they orchestrate, what outputs they produce.
Trace the high-level data flow from input to output."
```

### What to Write (12 Sections)

**1. Overview**
- Executive summary (2-3 paragraphs)
- Problem statement (what it solves)
- Key capabilities (what it can do)
- Quick stats (# of scripts, stages, nodes, etc.)

**2. Architecture & Design**
- System architecture diagram (ASCII art or mermaid)
- Major subsystems (pipeline, routing, agents, evaluation)
- Component relationships
- Data flow (end-to-end)

**3. File Inventory**
- Directory structure with purposes
- File counts per directory
- Key files and their roles

**4. Core Components Deep Dive**
- Not applicable for overview (defer to later parts)

**5. Configuration & Settings**
- List all config files (defer details to Part 8)

**6. Data Structures & Schemas**
- High-level data types (defer to specific parts)

**7. External Dependencies**
- Technology stack (Python 3.13, LangGraph, OpenAI, conda)
- External services (OpenAI API, vector databases)

**8. Execution & Usage**
- Main entry points (run_graph_langgraph.py)
- Quick start guide (how to run end-to-end)

**9. Code Patterns & Conventions**
- File naming conventions
- Script organization patterns

**10. Testing & Verification**
- Overview of quality gates (defer details to Part 7)

**11. Known Issues & Limitations**
- System-level constraints

**12. References**
- Links to other roadmap parts
- External documentation

### Output Deliverable
**File**: `roadmap/part1-overview/README.md` (~800-1200 lines)

**Estimated Effort**: 3-5 hours (1-2 hours research, 2-3 hours writing)

---

## Part 2: Data Pipeline & Storage =°

### Research Goal
Document the **complete data pipeline** from raw document collection through processing to final artifacts.

### Key Questions to Answer
1. What are the 13 pipeline stages? (names, purposes, sequence)
2. What scripts implement each stage? (file paths, line counts)
3. What data formats are used? (HTML, JSON, Parquet, etc.)
4. Where is data stored at each stage? (data/raw/, data/interim/, etc.)
5. How do stages connect? (dependencies, outputs ’ inputs)
6. What transformations occur? (HTML ’ text, text ’ chunks, etc.)

### Files to Analyze

**High Priority**:
- All `scripts/fetch_*.py` (7 collection scripts)
- `scripts/normalize_html.py`
- `scripts/extract_metadata.py`
- `scripts/chunk_documents.py`
- `scripts/dedupe_chunks.py`
- `data/` directory structure
- Sample files in `data/raw/`, `data/interim/`, `data/final/`

**Medium Priority**:
- `scripts/parse_sec_structures.py`
- `scripts/ingest_manual_html.py`
- `configs/normalization.rules.yaml`
- `configs/metadata.dictionary.yaml`
- `configs/chunking.config.json`

### Research Agent Strategy

Launch **4 agents in parallel**:

**Agent 1: codebase-pattern-finder**
```
Prompt: "Find all data collection scripts (fetch_*.py) in the scripts/ directory.
For each script:
- Extract what data source it fetches from (SEC, docs, newsroom, etc.)
- Identify where it saves data (output directory)
- Document what format it outputs (HTML, JSON, etc.)
- Note any configuration it uses

List all 7 collection scripts with their purposes."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Analyze the data processing pipeline scripts:
- scripts/normalize_html.py
- scripts/extract_metadata.py
- scripts/chunk_documents.py
- scripts/dedupe_chunks.py

For each script:
- What does it do? (transformation logic)
- What's the input? (file paths, formats)
- What's the output? (file paths, formats)
- What configuration does it use?
- How long does it take to run?

Trace the data flow from raw HTML to deduplicated chunks."
```

**Agent 3: Explore (medium mode)**
```
Prompt: "Explore the data/ directory structure thoroughly.
Map out all subdirectories: raw/, interim/, vector/, cache/, final/
For each subdirectory:
- What files exist? (sample 2-3 files)
- What format are they? (HTML, JSON, Parquet, etc.)
- How many files typically exist?
- What's the total size?

Identify the data lifecycle: raw ’ interim ’ vector ’ final"
```

**Agent 4: codebase-locator**
```
Prompt: "Find all configuration files related to data processing:
- normalization rules
- metadata extraction patterns
- chunking parameters
- deduplication settings

For each config, identify:
- File path
- What it configures
- What script uses it (via imports or loading)"
```

### What to Write (12 Sections)

**1. Overview**
- Data pipeline summary
- 13 stages listed
- Purpose of each stage

**2. Architecture & Design**
- Pipeline flow diagram
- Stage dependencies
- Data transformations

**3. File Inventory**
- All pipeline scripts (file paths, purposes, line counts)
- All data directories
- Sample data files

**4. Core Components Deep Dive**
- **Collection Stage**: 7 fetch scripts
  - Each script detailed (what it fetches, how, where it saves)
- **Normalization Stage**: normalize_html.py
  - Transformation logic
  - Input/output formats
- **Metadata Stage**: extract_metadata.py
  - Extraction patterns
  - Metadata schema
- **Chunking Stage**: chunk_documents.py
  - Chunking algorithm
  - Chunk size/overlap
- **Deduplication Stage**: dedupe_chunks.py
  - Dedup strategy
  - Hash function used

**5. Configuration & Settings**
- normalization.rules.yaml
- metadata.dictionary.yaml
- chunking.config.json

**6. Data Structures & Schemas**
- Raw HTML schema
- Normalized JSON schema
- Chunk schema
- Metadata fields

**7. External Dependencies**
- Web scraping libraries
- HTML parsers
- SEC Edgar API

**8. Execution & Usage**
- How to run collection scripts
- How to run processing scripts
- Full pipeline execution

**9. Code Patterns & Conventions**
- fetch_*.py naming pattern
- Data output conventions

**10. Testing & Verification**
- qa_verify_collection.py
- qa_verify_normalization.py
- qa_verify_metadata.py
- qa_verify_chunking.py
- qa_verify_dedupe.py

**11. Known Issues & Limitations**
- Rate limiting on SEC Edgar
- HTML parsing edge cases

**12. References**
- Links to Part 3 (embeddings) and Part 7 (quality gates)

### Output Deliverable
**File**: `roadmap/part2-pipeline/README.md` (~1200-1800 lines)

**Estimated Effort**: 5-7 hours (2-3 hours research, 3-4 hours writing)

---

## Part 3: Vector & Embedding System ="

### Research Goal
Document how **text is converted to vectors** and how **vector indexes are built**.

### Key Questions to Answer
1. How are embeddings generated? (OpenAI API, model, dimension)
2. What caching strategy is used? (SHA-256 keys, cache location)
3. What vector indexes exist? (FAISS, Weaviate, Pinecone)
4. How are indexes built? (scripts, parameters, formats)
5. What are the index schemas? (metadata, namespaces)
6. What's the performance? (latency, cost, cache hit rate)

### Files to Analyze

**High Priority**:
- `scripts/embedding_utils.py` (core embedding logic)
- `scripts/qa_step01_embeddings.py` (Gate-1)
- `scripts/qa_step02_indexes.py` (Gate-2)
- `configs/vector.indexing.yaml`
- `data/cache/embeddings/` (cache structure)
- `data/vector/` (index files)

**Medium Priority**:
- Any FAISS/Weaviate/Pinecone integration code
- Sample cached embeddings
- Index manifest files

### Research Agent Strategy

Launch **3 agents in parallel**:

**Agent 1: codebase-analyzer**
```
Prompt: "Deep dive on scripts/embedding_utils.py:
- How does embed_text() work?
- What OpenAI model is used?
- What's the vector dimension?
- How is caching implemented? (cache key generation, lookup, storage)
- What retry logic exists?
- How are costs tracked?

Provide code snippets with line numbers for key functions."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Analyze the index building process:
- scripts/qa_step01_embeddings.py (Gate-1: generate embeddings)
- scripts/qa_step02_indexes.py (Gate-2: build indexes)

For each script:
- What does it do step-by-step?
- What inputs does it require?
- What outputs does it produce?
- What configuration does it use?
- Why are there two separate conda environments (age vs ageFaiss)?

Trace the flow: chunks ’ embeddings ’ FAISS/Weaviate/Pinecone indexes"
```

**Agent 3: Explore (medium mode)**
```
Prompt: "Explore the vector storage structure:
- data/cache/embeddings/ (how are cached embeddings stored?)
- data/vector/ (what index files exist?)

For each location:
- File formats
- Naming conventions
- File sizes
- Sample contents (if readable)

Also explore configs/vector.indexing.yaml to understand configuration."
```

### What to Write (12 Sections)

**1. Overview**
- Embedding system purpose
- Vector dimension (1536)
- Index types (FAISS, Weaviate, Pinecone)

**2. Architecture & Design**
- Embedding generation flow
- Caching architecture
- Index building process

**3. File Inventory**
- embedding_utils.py
- qa_step01_embeddings.py
- qa_step02_indexes.py
- vector.indexing.yaml
- Cache directory structure
- Index files

**4. Core Components Deep Dive**
- **Embedding Generation** (embedding_utils.py)
  - embed_text() function (line numbers)
  - OpenAI API integration
  - Retry logic
  - Cost tracking
- **Caching System**
  - Cache key generation (SHA-256)
  - Cache lookup/storage
  - Cache hit rate optimization
- **FAISS Index**
  - HNSW parameters
  - Index building script
  - Index file format
- **Weaviate/Pinecone** (manifests)
  - Schema definitions
  - Current status (mock/real)

**5. Configuration & Settings**
- vector.indexing.yaml schema
- Embedding model configuration
- FAISS parameters
- Index build settings

**6. Data Structures & Schemas**
- Embedding format (1536-dim float array)
- Cached embedding structure
- FAISS index metadata
- Weaviate/Pinecone schemas

**7. External Dependencies**
- OpenAI API (text-embedding-ada-002)
- FAISS library (conda)
- Weaviate client
- Pinecone client

**8. Execution & Usage**
- Generate embeddings: `conda run -n age python scripts/qa_step01_embeddings.py`
- Build indexes: `conda run -n ageFaiss python scripts/qa_step02_indexes.py`
- Why two environments? (OpenMP conflict)

**9. Code Patterns & Conventions**
- Always use embed_text() (never random vectors)
- Cache before API call
- Dimension must be 1536

**10. Testing & Verification**
- Gate-1 validation
- Gate-2 validation
- Embedding consistency checks

**11. Known Issues & Limitations**
- OpenMP Error #15 (FAISS in wrong env)
- API rate limits
- Cost accumulation

**12. References**
- Part 2 (where chunks come from)
- Part 4 (how indexes are queried)
- Part 7 (Gate-1 and Gate-2 details)

### Output Deliverable
**File**: `roadmap/part3-vectors/README.md` (~1000-1400 lines)

**Estimated Effort**: 4-6 hours (2-3 hours research, 2-3 hours writing)

---

## Part 4: Multi-Index Routing >í

### Research Goal
Document how **queries are routed to the right vector backend** based on keywords, personas, and heuristics.

### Key Questions to Answer
1. What routing strategies exist? (keyword rules, persona bias, weighted scoring)
2. How are backends selected? (FAISS, Weaviate, Pinecone)
3. What are the routing rules? (press ’ Pinecone, API ’ Weaviate, etc.)
4. How do persona biases work? (VP CX ’ Pinecone, CIO ’ Weaviate)
5. What's the fallback logic? ([FAISS, Weaviate, Pinecone])
6. How are routing decisions logged?

### Files to Analyze

**High Priority**:
- `scripts/router_core.py` (routing logic)
- `scripts/qa_step04_router.py` (Gate-4 testing)
- `configs/router.heuristics.yaml` (routing rules)

**Medium Priority**:
- Routing decision logs in `reports/router/`
- Any persona-specific configs

### Research Agent Strategy

Launch **2 agents in parallel**:

**Agent 1: codebase-analyzer**
```
Prompt: "Deep dive on scripts/router_core.py - the routing logic:
- What's the main function? (route_query()?)
- What inputs does it take? (query text, persona, etc.)
- What's the routing algorithm? (step-by-step)
- How are keyword rules applied?
- How are persona biases applied?
- How is weighted scoring calculated?
- What's the fallback logic?
- How are decisions logged?

Provide code snippets with line numbers for the routing algorithm."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Analyze the routing configuration and testing:
- configs/router.heuristics.yaml (what rules are defined?)
- scripts/qa_step04_router.py (how is routing tested?)

For router.heuristics.yaml:
- What's the schema?
- What keyword rules exist?
- What persona biases exist?
- What scoring weights are defined?

For qa_step04_router.py:
- What test cases exist?
- How is routing validated?
- What are the pass criteria?"
```

### What to Write (12 Sections)

**1. Overview**
- Routing purpose (select best backend per query)
- 3 backends (FAISS, Weaviate, Pinecone)
- Routing strategies (keywords, persona, scoring, fallback)

**2. Architecture & Design**
- Routing flow diagram
- Decision tree
- Weighting algorithm

**3. File Inventory**
- router_core.py
- qa_step04_router.py
- router.heuristics.yaml
- Routing logs

**4. Core Components Deep Dive**
- **Routing Algorithm** (router_core.py)
  - route_query() function (line numbers)
  - Keyword matching logic
  - Persona bias application
  - Weighted scoring formula
  - Fallback logic
- **Routing Rules** (router.heuristics.yaml)
  - Keyword rules (press ’ Pinecone, API ’ Weaviate)
  - Persona biases (VP CX ’ Pinecone, CIO ’ Weaviate)
  - Scoring weights (similarity 0.5, recency 0.3, diversity 0.2)

**5. Configuration & Settings**
- router.heuristics.yaml schema
- Rule definitions
- Weight tuning

**6. Data Structures & Schemas**
- Routing decision structure
- Backend metadata

**7. External Dependencies**
- Vector backends (FAISS, Weaviate, Pinecone)

**8. Execution & Usage**
- How routing is called (from retriever_node)
- Example routing decisions

**9. Code Patterns & Conventions**
- Always log routing decisions
- Fallback order: [FAISS, Weaviate, Pinecone]

**10. Testing & Verification**
- Gate-4 router testing
- Test cases and expected outputs

**11. Known Issues & Limitations**
- Static keyword rules (no ML)
- Hardcoded persona biases

**12. References**
- Part 3 (what indexes are routed to)
- Part 5 (how MCP kb.search uses routing)
- Part 6 (how retriever_node calls router)

### Output Deliverable
**File**: `roadmap/part4-routing/README.md` (~800-1000 lines)

**Estimated Effort**: 3-5 hours (1-2 hours research, 2-3 hours writing)

---

## Part 5: MCP Tools & Services =

### Research Goal
Document the **5 MCP tool services** and their HTTP server implementations.

### Key Questions to Answer
1. What MCP tools exist? (kb.search, web.fetch, link.resolve, crm.lookup, safety.check)
2. How are they implemented? (aiohttp HTTP servers)
3. What are the service contracts? (request/response schemas)
4. How are they called? (HTTP POST with JSON)
5. What ports do they run on? (7801-7805)
6. How are they started? (start_stub_servers() function)

### Files to Analyze

**High Priority**:
- `scripts/qa_step03_mcp.py` (MCP stub servers)
- `scripts/tool_safety_check_server.py` (safety check implementation)
- `configs/mcp.tools.yaml` (service endpoints)

**Medium Priority**:
- Any MCP client code (how tools are called)
- MCP service logs

### Research Agent Strategy

Launch **2 agents in parallel**:

**Agent 1: codebase-analyzer**
```
Prompt: "Deep dive on scripts/qa_step03_mcp.py - the MCP stub servers:
- How many services are implemented? (list all 5)
- How is each service implemented? (HTTP handlers)
- What's the HTTP server technology? (aiohttp, Flask, etc.)
- How are servers started? (start_stub_servers() function?)
- What ports are used? (7801-7805)
- What are the request/response schemas for each service?

Provide code snippets with line numbers for each service handler."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Analyze MCP tool configuration and safety checks:
- configs/mcp.tools.yaml (service definitions)
- scripts/tool_safety_check_server.py (safety check implementation)

For mcp.tools.yaml:
- What's the schema?
- What services are defined?
- What are the endpoints?
- What are the timeouts?

For tool_safety_check_server.py:
- What safety checks are implemented?
- What's the request/response contract?
- What compliance rules are checked?"
```

### What to Write (12 Sections)

**1. Overview**
- MCP purpose (Model Context Protocol tools)
- 5 services overview
- Local stub implementation

**2. Architecture & Design**
- Service architecture diagram
- HTTP server implementation (aiohttp)
- Service calling flow

**3. File Inventory**
- qa_step03_mcp.py
- tool_safety_check_server.py
- mcp.tools.yaml

**4. Core Components Deep Dive**
- **kb.search (port 7801)**
  - Implementation (line numbers)
  - Request schema
  - Response schema
  - Vector search logic
- **web.fetch (port 7802)**
  - Stub implementation
  - Contract
- **link.resolve (port 7803)**
  - Stub implementation
  - Contract
- **crm.lookup (port 7804)**
  - Stub implementation
  - Contract
- **safety.check (port 7805)**
  - Full implementation (tool_safety_check_server.py)
  - Compliance rules
  - Request/response schemas

**5. Configuration & Settings**
- mcp.tools.yaml schema
- Service endpoints
- Timeouts
- Fallback policies

**6. Data Structures & Schemas**
- Request schemas (all 5 tools)
- Response schemas (all 5 tools)
- Error formats

**7. External Dependencies**
- aiohttp (HTTP server)
- httpx (HTTP client)

**8. Execution & Usage**
- Start MCP servers: `conda run -n age python scripts/qa_step03_mcp.py`
- Call tools via HTTP POST
- Example curl commands

**9. Code Patterns & Conventions**
- All tools return JSON
- Consistent error handling
- Timeout enforcement

**10. Testing & Verification**
- Gate-3 MCP validation
- Service health checks
- Contract testing

**11. Known Issues & Limitations**
- Stubs only (web.fetch, link.resolve, crm.lookup)
- No authentication
- Single-threaded

**12. References**
- Part 4 (routing used by kb.search)
- Part 6 (how retriever_node and a2a_node call tools)

### Output Deliverable
**File**: `roadmap/part5-mcp/README.md` (~1000-1200 lines)

**Estimated Effort**: 3-5 hours (1-2 hours research, 2-3 hours writing)

---

## Part 6: LangGraph Agent System >

### Research Goal
Document the **8-node LangGraph orchestration system** that generates persona-specific emails.

### Key Questions to Answer
1. What are the 8 nodes? (Intake, Planner, Retriever, Synthesizer, Consolidator, Stylist, A2A, Assembler)
2. How does state management work? (TypedDict, partial updates)
3. What's the execution flow? (sequential + conditional)
4. How are LLMs integrated? (ChatOpenAI, gpt-5-nano)
5. What outputs are generated? (insights.json, email.json)
6. How does A2A revision work? (up to 2 rounds)

### Files to Analyze

**High Priority**:
- `scripts/run_graph_langgraph.py` (orchestrator)
- `scripts/langgraph_nodes.py` (8 node implementations)
- `scripts/langgraph_state.py` (state schema)
- `configs/langgraph.nodes.yaml` (node topology)
- Sample outputs in `outputs/*/`

**Medium Priority**:
- `scripts/run_graph.py` (original implementation for comparison)
- Persona files in `icl/personas/`
- `configs/eval.prompts.yaml`

### Research Agent Strategy

Launch **4 agents in parallel**:

**Agent 1: codebase-analyzer**
```
Prompt: "Deep dive on scripts/langgraph_state.py - the state schema:
- What's the TypedDict structure?
- What are all the fields? (list all ~23 fields)
- What are the field categories? (inputs, accumulators, final outputs, metadata)
- How is state updated? (partial updates, merge semantics)
- Why TypedDict instead of Pydantic? (design decision)

Provide the complete GraphState schema with line numbers."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Deep dive on scripts/run_graph_langgraph.py - the orchestrator:
- How is the StateGraph built? (build_graph() function)
- How are the 8 nodes added?
- What are the sequential edges?
- What are the conditional edges? (A2A revision loop)
- How is the graph compiled?
- How is the graph executed?
- What are the CLI arguments?

Provide code snippets with line numbers for graph construction."
```

**Agent 3: codebase-analyzer**
```
Prompt: "Deep dive on scripts/langgraph_nodes.py - the 8 node implementations:
This is a long file (~582 lines). For EACH of the 8 nodes, extract:
- Node name and purpose
- Input (what state fields it reads)
- Output (what state fields it updates)
- Key logic (what it does)
- External calls (APIs, MCP tools)
- Line number range

The 8 nodes are:
1. intake_node
2. planner_node
3. retriever_node
4. synthesizer_node
5. consolidator_node
6. stylist_node
7. a2a_node
8. assembler_node

Provide a structured breakdown of all 8 nodes."
```

**Agent 4: Explore (medium mode)**
```
Prompt: "Explore the agent system outputs and configuration:
- outputs/*/ (sample session outputs)
- icl/personas/ (persona-specific prompts)
- configs/langgraph.nodes.yaml (node topology)

For outputs:
- What files are generated per session?
- What's the structure of insights.json?
- What's the structure of email.json?
- Sample content

For personas:
- What personas exist?
- What's the persona file format?

For langgraph.nodes.yaml:
- What's the schema?"
```

### What to Write (12 Sections)

**1. Overview**
- Agent system purpose (email generation)
- 8 nodes listed
- LangGraph technology

**2. Architecture & Design**
- Node flow diagram
- State flow through nodes
- Conditional revision loop (A2A)

**3. File Inventory**
- run_graph_langgraph.py
- langgraph_nodes.py
- langgraph_state.py
- langgraph.nodes.yaml
- Persona files
- Sample outputs

**4. Core Components Deep Dive**
- **State Management** (langgraph_state.py)
  - GraphState TypedDict (all 23 fields)
  - Accumulator pattern
  - Partial update semantics
- **Graph Builder** (run_graph_langgraph.py)
  - build_graph() function (line numbers)
  - Node additions
  - Edge definitions
  - Conditional edges (A2A)
- **8 Nodes** (langgraph_nodes.py)
  - **intake_node** (line X-Y): Validates inputs
  - **planner_node** (line X-Y): Generates 5 queries
  - **retriever_node** (line X-Y): Calls MCP kb.search
  - **synthesizer_node** (line X-Y): Creates insight candidates
  - **consolidator_node** (line X-Y): LLM refinement
  - **stylist_node** (line X-Y): Email generation
  - **a2a_node** (line X-Y): Compliance checks
  - **assembler_node** (line X-Y): Proof points

**5. Configuration & Settings**
- langgraph.nodes.yaml schema
- Node timeouts
- LLM parameters

**6. Data Structures & Schemas**
- GraphState TypedDict (complete schema)
- Insight structure
- Email structure
- Compliance report structure

**7. External Dependencies**
- LangGraph library
- LangChain ChatOpenAI
- OpenAI API (gpt-5-nano)
- MCP tools (kb.search, safety.check)

**8. Execution & Usage**
- Run graph: `conda run -n age python scripts/run_graph_langgraph.py --company Salesforce --persona vp_customer_experience --session-id test`
- Example sessions
- Output locations

**9. Code Patterns & Conventions**
- All nodes are async
- All nodes return partial state
- Always log to state["node_timings"]

**10. Testing & Verification**
- Gate-5 (graph validation)
- Gate-6 (A2A testing)
- End-to-end tests

**11. Known Issues & Limitations**
- A2A timeout issues
- Max 5 insights (hardcoded)
- LLM latency bottleneck

**12. References**
- Part 4 (routing called by retriever_node)
- Part 5 (MCP tools called by retriever/a2a nodes)
- Part 7 (Gate-5 and Gate-6 details)

### Output Deliverable
**File**: `roadmap/part6-agents/README.md` (~2000-2500 lines - this is the biggest part)

**Estimated Effort**: 7-9 hours (3-4 hours research, 4-5 hours writing)

---

## Part 7: Quality Gates & Evaluation <¯

### Research Goal
Document the **9 quality gates** that validate the system at each stage.

### Key Questions to Answer
1. What are the 9 gates? (Gate-0 through Gate-8)
2. What does each gate validate? (baseline, embeddings, indexes, MCP, router, graph, A2A, retrieval, generation)
3. What are the pass criteria? (metrics, thresholds)
4. How are metrics calculated? (recall@10, nDCG@5, etc.)
5. What reports are generated? (JSON + Markdown)
6. How do you run the gates?

### Files to Analyze

**High Priority**:
- `scripts/qa_step00_baseline.py` through `scripts/qa_step08_generation_eval.py` (9 scripts)
- Sample reports in `reports/qa/`
- `configs/eval.prompts.yaml`

**Medium Priority**:
- `scripts/qa_verify_*.py` (verification scripts)
- Evaluation traces in `reports/eval/`

### Research Agent Strategy

Launch **3 agents in parallel**:

**Agent 1: codebase-pattern-finder**
```
Prompt: "Find all quality gate scripts (qa_step*.py) in scripts/:
- List all 9 gates (Gate-0 through Gate-8)
- For each gate, extract:
  - What it validates
  - What the pass criteria are
  - What outputs it produces

Provide a structured table of all 9 gates."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Deep dive on evaluation metrics in Gate-7 and Gate-8:
- scripts/qa_step07_retrieval_eval.py (Gate-7: retrieval quality)
- scripts/qa_step08_generation_eval.py (Gate-8: generation quality)

For Gate-7:
- What metrics are calculated? (recall@10, nDCG@5, latency)
- How are they calculated? (code logic)
- What are the thresholds? (e0.80, e0.70, d1000ms)
- What environment variables control behavior? (AG7_IGNORE_COVERAGE, AG7_LATENCY_MULTIPLIER)

For Gate-8:
- What metrics are calculated? (structural_pass_rate, critical_flags, readability, persona_keywords)
- What are the thresholds?

Provide code snippets with line numbers."
```

**Agent 3: Explore (medium mode)**
```
Prompt: "Explore the quality gate reports:
- reports/qa/ (gate reports)
- reports/eval/ (evaluation traces)

For each location:
- What files exist?
- What's the format? (JSON, Markdown)
- What's the structure?
- Sample contents

Identify the dual-format pattern (JSON + Markdown)."
```

### What to Write (12 Sections)

**1. Overview**
- Quality gate purpose (validate each pipeline stage)
- 9 gates listed
- Dual report format

**2. Architecture & Design**
- Gate execution flow
- Validation strategy
- Reporting architecture

**3. File Inventory**
- All 9 qa_step*.py scripts
- All qa_verify*.py scripts
- Report files

**4. Core Components Deep Dive**
- **Gate-0: Baseline** (qa_step00_baseline.py)
  - What it checks
  - Pass criteria
- **Gate-1: Embeddings** (qa_step01_embeddings.py)
  - Embedding generation validation
  - Cache verification
  - Pass criteria
- **Gate-2: Indexes** (qa_step02_indexes.py)
  - Index build validation
  - FAISS structure checks
  - Pass criteria
- **Gate-3: MCP** (qa_step03_mcp.py)
  - Service health checks
  - Contract validation
  - Pass criteria
- **Gate-4: Router** (qa_step04_router.py)
  - Routing test cases
  - Decision validation
  - Pass criteria
- **Gate-5: Graph** (qa_step05_graph.py)
  - Graph construction checks
  - State schema validation
  - Pass criteria
- **Gate-6: A2A** (qa_step06_a2a.py)
  - A2A compliance checks
  - Revision loop testing
  - Pass criteria
- **Gate-7: Retrieval Eval** (qa_step07_retrieval_eval.py)
  - Metrics: recall@10, nDCG@5, latency
  - Thresholds: e0.80, e0.70, d1000ms
  - Environment overrides
  - Pass criteria
- **Gate-8: Generation Eval** (qa_step08_generation_eval.py)
  - Metrics: structural_pass_rate, critical_flags, readability, persona_keywords
  - Thresholds: 1.0, 0, e9/10, e2.0
  - Pass criteria

**5. Configuration & Settings**
- eval.prompts.yaml schema
- Gate-specific configs

**6. Data Structures & Schemas**
- Gate report structure (JSON)
- Evaluation metrics structure

**7. External Dependencies**
- Evaluation libraries
- Metrics calculation tools

**8. Execution & Usage**
- Run all gates: `bash scripts/run_all_gates.sh`
- Run individual gates: `conda run -n age python scripts/qa_step07_retrieval_eval.py`
- Environment variables

**9. Code Patterns & Conventions**
- All gates emit dual reports (JSON + Markdown)
- Consistent exit codes (0 = pass, 1 = fail)

**10. Testing & Verification**
- How gates are tested
- Validation test cases

**11. Known Issues & Limitations**
- Gate-7 requires relaxed budgets (AG7_IGNORE_COVERAGE=1)
- Gate-2 requires ageFaiss environment

**12. References**
- Part 2 (pipeline stages being validated)
- Part 3 (Gate-1 and Gate-2)
- Part 5 (Gate-3)
- Part 6 (Gate-5 and Gate-6)

### Output Deliverable
**File**: `roadmap/part7-quality/README.md` (~1500-2000 lines)

**Estimated Effort**: 5-7 hours (2-3 hours research, 3-4 hours writing)

---

## Part 8: Configuration & Operations ™

### Research Goal
Document **all configuration files** and **operational procedures** (setup, execution, troubleshooting).

### Key Questions to Answer
1. What config files exist? (10 files in configs/)
2. What can be configured? (embeddings, routing, MCP, nodes, etc.)
3. How do you set up the environment? (conda envs, API keys)
4. How do you run the system? (commands, arguments)
5. What environment variables exist? (AG1_AUTO_CONFIRM, AG7_IGNORE_COVERAGE, etc.)
6. What are common issues? (OpenMP errors, recall=0, API failures)

### Files to Analyze

**High Priority**:
- All `configs/*.yaml` and `configs/*.json` (10 files)
- `envs/age.yaml`, `envs/ageFaiss.yaml`
- `.env` (if exists) or `.env.example`
- `docs/troubleshooting.md`
- `docs/commands.md`

**Medium Priority**:
- All main execution scripts (for CLI args)
- Error logs (if available)

### Research Agent Strategy

Launch **3 agents in parallel**:

**Agent 1: Explore (medium mode)**
```
Prompt: "Explore all configuration files:
- configs/ directory (all 10 files)

For each config file:
- File name and path
- Format (YAML or JSON)
- Purpose (what it configures)
- Schema (what settings exist)
- Sample values

Create a comprehensive inventory of all config files."
```

**Agent 2: codebase-analyzer**
```
Prompt: "Analyze environment setup:
- envs/age.yaml (primary environment)
- envs/ageFaiss.yaml (FAISS-only environment)

For each environment:
- Python version
- All dependencies (with versions)
- Why two separate environments?
- What's the OpenMP conflict?

Also analyze .env configuration:
- What environment variables are used?
- Where are they loaded?
- What's required vs optional?"
```

**Agent 3: codebase-analyzer**
```
Prompt: "Analyze operational procedures and troubleshooting:
- docs/troubleshooting.md (if exists)
- docs/commands.md (if exists)
- Main execution scripts (CLI argument parsing)

Extract:
- Setup steps
- Common commands
- Environment variables
- Common errors and fixes
- Debug procedures"
```

### What to Write (12 Sections)

**1. Overview**
- Configuration system overview
- 10 config files listed
- Environment setup summary

**2. Architecture & Design**
- Configuration loading architecture
- Environment isolation (age vs ageFaiss)
- Config override hierarchy

**3. File Inventory**
- All 10 config files (paths, purposes)
- Environment files
- .env file

**4. Core Components Deep Dive**
- **vector.indexing.yaml**
  - Schema
  - Embedding model settings
  - FAISS parameters
- **router.heuristics.yaml**
  - Routing rules
  - Persona biases
  - Scoring weights
- **mcp.tools.yaml**
  - Service endpoints
  - Ports
  - Timeouts
- **langgraph.nodes.yaml**
  - Node topology
  - Execution order
- **metadata.dictionary.yaml**
  - Extraction patterns
- **normalization.rules.yaml**
  - Text cleaning rules
- **eval.prompts.yaml**
  - Evaluation templates
- **agents.schema.yaml**
  - Agent definitions
- **compliance.template.yaml**
  - Compliance rules
- **chunking.config.json**
  - Chunk size/overlap

**5. Configuration & Settings**
- Complete schema for each config
- Default values
- Override mechanisms

**6. Data Structures & Schemas**
- Config file formats
- Validation schemas

**7. External Dependencies**
- Conda
- OpenAI API key
- Vector databases

**8. Execution & Usage**
- **Environment Setup**:
  ```bash
  /Users/liyunxiao/anaconda3/bin/conda env create -f envs/age.yaml
  /Users/liyunxiao/anaconda3/bin/conda env create -f envs/ageFaiss.yaml
  echo "OPENAI_API_KEY=..." > .env
  ```
- **Run Quality Gates**:
  ```bash
  conda run -n age python scripts/qa_step01_embeddings.py
  conda run -n ageFaiss python scripts/qa_step02_indexes.py
  ...
  ```
- **Run Graph**:
  ```bash
  conda run -n age python scripts/run_graph_langgraph.py \
    --company Salesforce \
    --persona vp_customer_experience \
    --session-id test
  ```
- **Environment Variables**:
  - AG1_AUTO_CONFIRM=1
  - AG7_IGNORE_COVERAGE=1
  - AG7_LATENCY_MULTIPLIER=3.0
  - OPENAI_API_KEY

**9. Code Patterns & Conventions**
- Config loading via common.py
- Environment variable naming (AG*)

**10. Testing & Verification**
- Verify environments: `conda list -n age`
- Test API key: `python -c "import openai; print(openai.api_key)"`

**11. Known Issues & Limitations**
- **OpenMP Error #15**: Never install pip faiss-cpu in age env
- **Recall = 0%**: Must use embed_text() for both docs and queries
- **API key errors**: Check .env file exists
- **Port conflicts**: MCP ports 7801-7805 must be free

**12. References**
- All previous parts (cross-references to where configs are used)

### Output Deliverable
**File**: `roadmap/part8-operations/README.md` (~1200-1600 lines)

**Estimated Effort**: 4-6 hours (2-3 hours research, 2-3 hours writing)

---

## Master Execution Plan

### Directory Structure Setup

```bash
# Create all part directories
mkdir -p roadmap/{part1-overview,part2-pipeline,part3-vectors,part4-routing,part5-mcp,part6-agents,part7-quality,part8-operations}
```

### Execution Workflow (For Each Part)

```
Step 1: Research Phase
    Launch 2-4 specialized agents in parallel (see each part above)
    Wait for ALL agents to complete
    Review agent findings

Step 2: Verification Phase
    Read all key files identified by agents (use Read tool)
    Verify agent findings are accurate
    Identify any gaps in agent research

Step 3: Documentation Phase
    Write roadmap/partN-name/README.md
    Follow 12-section structure
    Include specific file paths and line numbers
    Include code snippets
    Add cross-references to other parts

Step 4: Review Phase
    Verify all sections are complete
    Check for missing information
    Ensure coding agents can understand from docs alone
```

### Estimated Effort Summary

| Part | Research | Writing | Total |
|------|----------|---------|-------|
| Part 1: Overview | 1-2 hours | 2-3 hours | 3-5 hours |
| Part 2: Pipeline | 2-3 hours | 3-4 hours | 5-7 hours |
| Part 3: Vectors | 2-3 hours | 2-3 hours | 4-6 hours |
| Part 4: Routing | 1-2 hours | 2-3 hours | 3-5 hours |
| Part 5: MCP | 1-2 hours | 2-3 hours | 3-5 hours |
| Part 6: Agents | 3-4 hours | 4-5 hours | 7-9 hours |
| Part 7: Quality | 2-3 hours | 3-4 hours | 5-7 hours |
| Part 8: Operations | 2-3 hours | 2-3 hours | 4-6 hours |
| **Total** | **14-22 hours** | **20-28 hours** | **34-50 hours** |

### Dependencies & Recommended Order

**Critical Path** (must do in order):
```
Part 1 (Overview) ’ MUST BE FIRST
  “
Parts 2, 3, 4, 5 (can be parallel, all reference Part 1)
  “
Part 6 (Agents - depends on 2, 3, 4, 5)
  “
Part 7 (Quality - depends on all previous parts)
  “
Part 8 (Operations - depends on all previous parts)
```

**Recommended Sequence**:
1. **Part 1** (foundation) - establishes context
2. **Parts 2 + 3** (parallel) - pipeline + vectors form data layer
3. **Parts 4 + 5** (parallel) - routing + MCP form service layer
4. **Part 6** (agents) - uses all previous layers
5. **Part 7** (quality) - validates all previous parts
6. **Part 8** (operations) - ties everything together

### Parallel vs Sequential Execution

**Can be done in parallel**:
- Parts 2, 3, 4, 5 (after Part 1 is complete)
- Within each part: research agents can run in parallel

**Must be sequential**:
- Part 1 before all others
- Part 6 after Parts 2, 3, 4, 5
- Part 7 after Parts 2-6
- Part 8 after Parts 2-7

---

## Success Criteria

For the roadmap to be considered complete:

1. **Completeness**: All 8 parts written, all 12 sections filled
2. **Specificity**: File paths, line numbers, code snippets included
3. **Clarity**: Coding agents can understand without code exploration
4. **Cross-references**: Parts link to each other appropriately
5. **Examples**: Real examples from codebase included
6. **Accuracy**: All information verified against actual code

---

## Next Steps

1. **Create directory structure**: `mkdir -p roadmap/{part1-overview,...,part8-operations}`
2. **Start with Part 1**: Foundation for all other parts
3. **Work sequentially** through recommended order
4. **Verify each part** before moving to next
5. **Create master index** after all parts complete

---

## Notes

- This is a **living document** - update as codebase evolves
- Estimated effort: 34-50 hours total
- Each part is independently useful
- Coding agents should start with Part 1, then jump to relevant parts
- All file paths are absolute for clarity
