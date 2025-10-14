---
date: 2025-10-13 14:00:00 EDT
researcher: Claude Code
git_commit: eae269f404986786dcd3fb6fdfa9e859a0cb0907
branch: agent-faiss
repository: agent-faiss
topic: "Web Demo for Investor Showcase - Pre-generated Email Outputs Inventory"
tags: [research, web-demo, investor-showcase, langgraph, outputs, dependencies]
status: complete
last_updated: 2025-10-13
last_updated_by: Claude Code
---

# Research: Web Demo for Investor Showcase - Pre-generated Email Outputs Inventory

**Date**: 2025-10-13 14:00:00 EDT
**Researcher**: Claude Code
**Git Commit**: eae269f404986786dcd3fb6fdfa9e859a0cb0907
**Branch**: agent-faiss
**Repository**: agent-faiss

## Research Question

Research the codebase to support building a minimal web demo for investor showcase at an entrepreneur conference. Create a simple web interface where users can select a persona from a dropdown and instantly see a pre-generated personalized outreach email. No real-time processing - just serve cached results. The results currently exist for only one company (Salesforce).

## Summary

The codebase contains **34 pre-generated email outputs** for Salesforce across 3 personas (vp_customer_experience: 16 sessions, cio: 9 sessions, vp_sales_ops: 9 sessions). All outputs are stored in `outputs/<session-id>/` with a consistent 5-file JSON structure. The most recent outputs are from October 4, 2025.

**⚠️ CRITICAL QUALITY ISSUE IDENTIFIED**: All 34 emails have **identical subjects** ("Ideas for improving CX at Salesforce") and **identical body introductions** ("Based on recent updates, here are a few insights that may help your CX agenda"), regardless of persona. The `persona_keywords` field is **missing from all state files** (0 out of 34 sessions), preventing proper persona differentiation in email generation. While retrieval varies by persona (different documents retrieved), the email styling/tone does NOT adapt to persona, making all emails appear generic and CX-focused.

No web UI currently exists, but MCP HTTP stub servers provide backend API infrastructure on ports 7801-7805. A minimal demo requires only Flask/FastAPI (~50-80 MB) versus the full pipeline environment (~4.5 GB with 24-38 dependencies).

## Detailed Findings

### 1. Pre-Generated Email Outputs Inventory

#### Available Sessions

**Location**: `outputs/` directory
**Total Sessions**: 34 directories
**Company**: Salesforce (ticker: `crm`)
**Most Recent**: October 4, 2025

**Session ID Patterns**:
- Auto-generated: 12-character hex (e.g., `05b1a905f6f9`, `f2a5101bdc32`)
- Named session: `step6demo`

**Recent Sessions** (Last modified Oct 4, 2025):
```
outputs/f2a5101bdc32/
outputs/e6656d176e87/
outputs/d0b6533b2b6a/
outputs/be978b58c274/
outputs/bb6cfff2899b/
outputs/ad611d5b00f3/
outputs/aa90d8df3f86/
outputs/9d3b7c866501/
outputs/99a18fec12c0/
outputs/90f19c32fc96/
outputs/89de3d334cec/
outputs/84bdb2946d6b/
outputs/76e474712834/
outputs/75218bdc694e/
outputs/step6demo/
outputs/05b1a905f6f9/
outputs/285932fdd50c/
outputs/4573ab70af60/
...
```

#### Output File Structure (Per Session)

Each session directory contains 5 JSON files:

1. **`email.json`** - Final generated email with proof points
   - Fields: `subject`, `body`, `unsubscribe_block`, `company_info_block`, `proof_points[]`
   - Example subject: "Ideas for improving CX at Salesforce"
   - Body length: 80-160 words (post-processed)
   - Proof points: 5 chunk IDs with titles linking to source documents

2. **`insights.json`** - 5 LLM-enhanced insight cards
   - Fields per card: `id`, `title`, `summary`, `url`, `date`, `evidence_snippet`, `confidence`, `source_domain`, `doc_id`
   - Additional fields (when present): `persona_relevance`, `metric_impact`, `action_suggestion`

3. **`compliance_report.json`** - A2A negotiation results
   - Fields: `rounds` (1 or 2), `flags.critical[]`, `flags.warning[]`
   - Critical flags must be 0 for Gate-8 pass

4. **`timing.json`** - Performance metrics
   - Total runtime and per-node execution times

5. **`router_trace.jsonl`** - Query routing decisions (JSONL format)
   - Per-query entries with backend decisions, latency, domain diversity

**State Snapshots**: `state/session-<session-id>.json` (34 files)
- Full AgentState dictionary with company, persona, all intermediate data

#### Available Personas

**Configuration**: `configs/eval.prompts.yaml`

Three personas with distinct keyword sets:

1. **vp_customer_experience**
   - Keywords: nps, csat, contact center, omnichannel, agent productivity, self-service, first contact resolution
   - Focus: Customer experience metrics and service quality

2. **cio**
   - Keywords: data integration, governance, security, tco, platform, apis, real-time
   - Focus: Technical architecture and IT operations

3. **vp_sales_ops**
   - Keywords: pipeline, forecast accuracy, win rate, productivity, automation
   - Focus: Sales operations and revenue metrics

**Additional Personas in Eval Seed** (`data/interim/eval/salesforce_eval_seed.jsonl`):
- cfo, vp_product, investor_relations, treasurer, chief_legal_officer, vp_sales, vp_corp_dev, equity_analyst, product_marketing, vp_collaboration, vp_commerce, vp_customer_success, economist, ai_researcher, public_sector_vp, industry_vp, data_architect, developer, researcher, admin, infrastructure_architect, new_employee, vp_strategy, business_analyst

**Currently Used**: The 34 generated sessions use the 3 primary personas (vp_customer_experience, cio, vp_sales_ops)

#### Persona Distribution (Verified)

**Actual counts from state files**:
- **vp_customer_experience**: 16 sessions
- **cio**: 9 sessions
- **vp_sales_ops**: 9 sessions
- **Total**: 34 sessions

**Sample session IDs by persona**:
- vp_customer_experience: 05b1a905f6f9, step6demo, 0a2582768652, 0bbdd468b4a4, 1cc01cdaa9b5, ...
- cio: 0fefad3e63ce, 24a53427fa17, 252c7579af51, f2a5101bdc32, 4ac2b75838ec, ...
- vp_sales_ops: e6656d176e87, d0b6533b2b6a, aa90d8df3f86, 0927ce454c14, 4f20b1ccb013, ...

#### Output Quality Assessment

**⚠️ CRITICAL ISSUE: Lack of Persona Differentiation**

All 34 sessions exhibit identical email structure regardless of persona:

**Subject Line** (100% identical across all personas):
```
"Ideas for improving CX at Salesforce"
```

**Body Introduction** (100% identical across all personas):
```
"Hi there,\n\nBased on recent updates, here are a few insights that may help your CX agenda:"
```

**Analysis**:
- ❌ **No persona customization in email generation**: All emails are CX-focused regardless of target persona
- ❌ **Missing persona_keywords field**: 0 out of 34 state files contain `persona_keywords` (should be loaded by Planner node from `configs/eval.prompts.yaml`)
- ❌ **Generic tone**: vp_sales_ops emails say "CX agenda" instead of "pipeline" or "win rates"
- ❌ **Generic tone**: cio emails say "CX agenda" instead of "integration" or "security"
- ✅ **Queries DO vary**: Different search queries generated per persona (5 queries for most, 1 query for some cio sessions)
- ✅ **Content DOES vary**: Different documents retrieved per session (but same email framing)
- ✅ **Structural compliance**: All emails have required compliance blocks

**Root Cause**:
The Stylist node (email generation) is not receiving or not using persona customization parameters. The `persona_keywords` field that should guide tone/focus is absent from all state files, causing emails to default to generic CX messaging.

**Impact on Demo**:
For an investor showcase, this creates a **poor user experience** - selecting different personas shows nearly identical emails with only bullet point content varying. The demo will not effectively demonstrate persona-aware email generation unless this is addressed or the limitation is disclosed upfront.

**Example Output**: `outputs/05b1a905f6f9/email.json` (vp_customer_experience)

```json
{
  "subject": "Ideas for improving CX at Salesforce",
  "body": "Hi there,\n\nBased on recent updates, here are a few insights that may help your CX agenda:\n\n- Salesforce Signs Definitive Agreement to Acquire Informatica (2025-05-27) — https://www.salesforce.com/news/press-releases/2025/05/27/salesforce-signs-definitive-agreement-to-acquire-informatica/?bc=OTH\n- crm-20250430 (2025-04-30) — https://www.sec.gov/Archives/edgar/data/1108524/000110852425000030/crm-20250430.htm\n- Salesforce.com, Inc. - Salesforce Announces Fourth Quarter Fiscal 2025 Results (2025-01-31) — https://investor.salesforce.com/news/news-details/2025/Salesforce-Announces-Fourth-Quarter-and-Fiscal-Year-2025-Results\n- Fy25 Annual Report Pdf (2025-09-07) — https://www.sec.gov/Archives/edgar/data/1108524/000110852425000019/salesforce_fy25annualreport.pdf\n- Fy25 Annual Report Pdf (2025-09-07) — https://www.sec.gov/Archives/edgar/data/1108524/000110852425000019/salesforce_fy25annualreport.pdf\n\nWould you be open to a quick chat to explore?\n",
  "unsubscribe_block": "You can unsubscribe at any time by replying 'unsubscribe'.",
  "company_info_block": "Sent by ACME AI, 123 Market St, San Francisco, CA.",
  "proof_points": [
    {"id": "crm::press::2025-05-27::salesforce-signs-definitive-agreement-to-acquire-informatica::a8a077f6::chunk0003", "title": "Salesforce Signs Definitive Agreement to Acquire Informatica"},
    {"id": "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0078", "title": "crm-20250430"},
    {"id": "crm::press::2025-01-31::news-details::9711c8f6::chunk0009", "title": "Salesforce.com, Inc. - Salesforce Announces Fourth Quarter Fiscal 2025 Results"},
    {"id": "crm::ars_pdf::unknown::fy25-annual-report-pdf::1b31e86f::chunk0372", "title": "Fy25 Annual Report Pdf"},
    {"id": "crm::ars_pdf::unknown::fy25-annual-report-pdf::1b31e86f::chunk0487", "title": "Fy25 Annual Report Pdf"}
  ]
}
```

**Quality Characteristics**:
- ✅ **Readable**: Grade ≤15 Flesch-Kincaid (enforced)
- ✅ **Compliant**: All sessions have unsubscribe + company info blocks
- ✅ **Traceable**: 5 proof points with chunk IDs linking to sources
- ✅ **Diverse sources**: Press releases, SEC filings (10-Q, 10-K), annual reports (PDF)
- ✅ **Recent data**: Dates ranging from 2025-01-31 to 2025-05-27
- ✅ **Valid URLs**: All URLs point to sec.gov, salesforce.com, investor.salesforce.com

**Document Type Coverage** (from proof points):
- Press releases (`crm::press::`)
- SEC 10-Q filings (`crm::10-Q::`)
- SEC 10-K filings (`crm::10-K::`)
- Annual report PDFs (`crm::ars_pdf::`)
- Likely also: dev_docs, help_docs, product, wiki (not in this specific example)

### 2. Email JSON Structure for Parsing and Display

#### Core Schema

**File**: `outputs/<session-id>/email.json`

```typescript
interface Email {
  subject: string;              // ≤12 words
  body: string;                 // 80-160 words, 1-3 bullets
  unsubscribe_block: string;    // Fixed CAN-SPAM compliance
  company_info_block: string;   // Fixed sender identification
  proof_points: ProofPoint[];   // Exactly 5 items
}

interface ProofPoint {
  id: string;      // Format: ticker::doctype::date::slug::hash::chunkN
  title: string;   // Max 120 chars
}
```

#### Insight Card Schema

**File**: `outputs/<session-id>/insights.json`

```typescript
interface InsightCard {
  id: string;                   // Chunk ID (same format as ProofPoint.id)
  title: string;                // Max 120 chars
  summary: string;              // Max 320 chars
  url: string;                  // Full document URL
  date: string;                 // ISO date (YYYY-MM-DD)
  evidence_snippet: string;     // Text excerpt
  confidence: number;           // Always 0.7
  source_domain: string;        // Domain (e.g., "www.salesforce.com")
  doc_id: string;               // Document ID (without chunk suffix)

  // Optional LLM-enhanced fields
  persona_relevance?: {
    why_it_matters: string;
    relevance_score: number;    // 1-5
    keywords_hit: string[];
  };
  metric_impact?: {
    metric: string;
    direction: string;          // "increase" | "decrease"
    magnitude: string;          // "low" | "medium" | "high"
  };
  action_suggestion?: string;
}
```

#### Display-Ready Data Structure

For the demo, combine email + insights for a complete view:

```typescript
interface DemoPayload {
  session_id: string;
  company: string;              // From state file
  persona: string;              // From state file
  email: Email;
  insights: InsightCard[];
  compliance: {
    rounds: number;
    critical_flags: string[];
    warning_flags: string[];
  };
  timing: {
    total_runtime_ms: number;
  };
}
```

#### Field Sources

**Persona & Company** (not in email.json):
- Source: `state/session-<session-id>.json`
- Fields: `state.company`, `state.persona`, `state.persona_keywords[]`

**Execution Metadata**:
- Source: `outputs/<session-id>/timing.json`
- Source: `outputs/<session-id>/compliance_report.json`

#### Chunk ID Decoding

**Format**: `<ticker>::<doctype>::<date>::<slug>::<hash>::<chunk_index>`

**Example**: `crm::press::2025-05-27::salesforce-signs-definitive-agreement-to-acquire-informatica::a8a077f6::chunk0003`

**Components**:
- `crm` = Salesforce ticker
- `press` = Document type (press, 10-Q, 10-K, ars_pdf, dev_docs, help_docs, product, wiki, 8-K)
- `2025-05-27` = Publish date
- `salesforce-signs...` = URL slug (normalized)
- `a8a077f6` = 8-char document hash
- `chunk0003` = Chunk index (zero-padded)

### 3. Existing Web/API Components

#### MCP HTTP Stub Servers

**Primary Implementation**: `scripts/qa_step03_mcp.py`
**Auxiliary Server**: `scripts/tool_safety_check_server.py`

**5 Local Services** (localhost only):

| Service | Port | Endpoint | Purpose |
|---------|------|----------|---------|
| `kb.search` | 7801 | `/invoke` | Vector search (FAISS/Weaviate/Pinecone) |
| `web.fetch` | 7802 | `/invoke` | Web content fetching |
| `link.resolve` | 7803 | `/invoke` | URL resolution |
| `crm.lookup` | 7804 | `/invoke` | CRM data lookup |
| `safety.check` | 7805 | `/invoke` | Compliance validation |

**HTTP Endpoints** (all services):
- `GET /healthz` - Health check
- `POST /invoke` - JSON-RPC style invocation

**Configuration**: `configs/mcp.tools.yaml`

**Usage in Graph**:
- Retriever node calls `kb.search` (port 7801)
- A2A node calls `safety.check` (port 7805)

**NOT User-Facing**: These are backend stubs for agent-to-agent communication, not web UI endpoints.

#### What Does NOT Exist

- ❌ No Flask or FastAPI applications
- ❌ No web-based dashboard or UI
- ❌ No HTML templates or frontend code
- ❌ No Express.js or Node web servers
- ❌ No REST API for external consumption
- ❌ No authentication or user-facing endpoints

#### Graph Visualization (CLI only)

**Python Viz**: `scripts/visualize_graph.py` - Generates graph diagrams (not web-based)
**TypeScript Viz**: `hack/visualize.ts` - Terminal JSONL log viewer (not web-based)

#### Conclusion

The codebase has **backend API infrastructure** (MCP stubs) but **zero web UI**. All interaction is via CLI scripts. The demo will need to be built from scratch using Flask or FastAPI.

### 4. Minimal Dependencies for Read-Only Demo

#### Current Environments (Full Pipeline)

**`age` environment** (Python 3.13):
- 24 packages, ~2.5 GB
- Includes: openai, langgraph, langchain, aiohttp, numpy, pyarrow, tenacity

**`ageFaiss` environment** (Python 3.12):
- 14 packages, ~2.0 GB
- Includes: faiss-cpu, numpy, scipy, pyarrow, openblas

**Total**: ~4.5 GB, 38 packages across 2 environments

#### Minimal Demo Environment

**Purpose**: Read JSON files and serve via HTTP. No embeddings, no FAISS, no LLM calls.

**Option A: Flask-Based Demo**

```yaml
# envs/demo_flask.yaml
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - flask==3.0.0
```

**Size**: ~50 MB
**Reduction**: 97% smaller than full pipeline

**Option B: FastAPI-Based Demo**

```yaml
# envs/demo_fastapi.yaml
name: demo
channels:
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
      - fastapi==0.104.0
      - uvicorn[standard]==0.24.0
      - jinja2==3.1.2  # For HTML templates (optional)
```

**Size**: ~80 MB
**Reduction**: 96% smaller than full pipeline

#### Excluded Dependencies (NOT Needed)

| Dependency | Purpose | Why Excluded |
|------------|---------|--------------|
| `openai` | OpenAI API client | No embedding generation in demo |
| `langgraph*` | LangGraph orchestration | No graph execution in demo |
| `langchain*` | LangChain framework | No LLM operations in demo |
| `faiss-cpu` | Vector search | No indexing or retrieval in demo |
| `numpy` | Numerical computing | No vector operations in demo |
| `scipy` | Scientific computing | No calculations in demo |
| `pyarrow` | Apache Arrow format | Not used in JSON files |
| `aiohttp` | Async HTTP client | No external HTTP calls in demo |
| `tenacity` | Retry logic | No retryable operations in demo |
| `python-dotenv` | Environment variables | No API keys in demo |
| `aiosqlite` | Async SQLite | No database operations in demo |
| `openblas` | Linear algebra | No matrix operations in demo |
| `llvm-openmp` | OpenMP runtime | No parallel processing in demo |

#### Standard Library Modules (Free)

No external dependencies needed beyond Flask/FastAPI:

```python
import json           # Parse JSON files
import os            # File system operations
from pathlib import Path  # Path manipulation
import glob          # File pattern matching
from typing import List, Dict  # Type hints
```

### 5. LangGraph Execution Details

#### Main Scripts

**LangGraph Implementation** (Recommended):
- **Script**: `scripts/run_graph_langgraph.py`
- **Framework**: LangGraph StateGraph with 8 nodes
- **Output files**: 5 JSON files per session (no `a2a_transcript.jsonl`)

**Original Implementation** (For comparison):
- **Script**: `scripts/run_graph.py`
- **Framework**: Procedural async workflow
- **Output files**: 6 JSON files per session (includes `a2a_transcript.jsonl`)

**Both produce identical email.json and insights.json structures.**

#### Command-Line Interface

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

**Arguments**:
- `--company`: Target company (default: "Salesforce")
- `--persona`: Recipient role (default: "vp_customer_experience")
- `--session-id`: Unique identifier (default: auto-generated 12-char hex)

#### 8-Node Pipeline

**Node Configuration**: `configs/langgraph.nodes.yaml`
**Node Implementation**: `scripts/langgraph_nodes.py`

**Workflow**:
1. **Intake** - Validates inputs (company, persona)
2. **Planner** - Generates 5 persona-specific queries from eval seed
3. **Retriever** - Executes vector search via kb.search MCP service, routes queries to FAISS/Weaviate/Pinecone
4. **Synthesizer** - Deduplicates chunks, extracts metadata (title, URL, date)
5. **Consolidator** - Selects 5 insights with LLM enhancement, ensures domain diversity (≥4 domains)
6. **Stylist** - Generates email copy via gpt-5-nano LLM
7. **A2A** - Agent-to-agent compliance validation via safety.check MCP service, conditional revision loop
8. **Assembler** - Attaches proof points, finalizes email structure

**Conditional Routing**: A2A → {Stylist (if critical flags), Assembler (if clean)}

**Post-Processing**: Word count (≤160) and readability (grade ≤15) enforcement after graph execution

#### State Management

**State Schema**: `scripts/langgraph_state.py`

**AgentState TypedDict**:
- Input: `company`, `persona`, `session_id`, `timestamp`
- Accumulating: `retrieved_chunks`, `retrieval_logs`, `route_decisions`, `compliance_flags`, `errors`
- Replacing: `queries`, `persona_keywords`, `insight_candidates`, `insight_cards`, `email_draft`, `a2a_rounds`, `metrics`

**State Persistence**: `state/session-<session-id>.json` (full state snapshot after execution)

## Code References

### Output Generation
- `scripts/run_graph_langgraph.py:169-202` - Output file writing (5 JSON files + state)
- `scripts/langgraph_nodes.py:432-450` - Stylist node (email generation via LLM)
- `scripts/langgraph_nodes.py:552-564` - Assembler node (proof points attachment)
- `scripts/run_graph_langgraph.py:110-167` - Post-processing (readability enforcement)

### Output Locations
- `outputs/<session-id>/email.json` - Final email structure
- `outputs/<session-id>/insights.json` - 5 insight cards
- `outputs/<session-id>/compliance_report.json` - A2A results
- `outputs/<session-id>/timing.json` - Performance metrics
- `outputs/<session-id>/router_trace.jsonl` - Routing decisions
- `state/session-<session-id>.json` - Full AgentState snapshot

### Persona Configuration
- `configs/eval.prompts.yaml:1-24` - 3 personas with keywords
- `data/interim/eval/salesforce_eval_seed.jsonl:1-46` - 46 persona-tagged queries
- `scripts/langgraph_nodes.py:190-191` - Persona keyword loading

### MCP Services
- `scripts/qa_step03_mcp.py:40-208` - MCP stub server implementation
- `configs/mcp.tools.yaml` - Service endpoints and configuration
- `scripts/langgraph_nodes.py:126-143` - kb.search client
- `scripts/langgraph_nodes.py:458-476` - safety.check client

### Environment Configuration
- `envs/age.yaml` - Primary environment (Python 3.13, 24 packages)
- `envs/ageFaiss.yaml` - FAISS environment (Python 3.12, 14 packages)

## Demo Implementation Guidance

### What You Have

✅ **34 pre-generated sessions** with complete JSON outputs
✅ **Consistent data structure** (5 files per session)
✅ **3 personas** with distinct keyword profiles
✅ **1 company** (Salesforce) with diverse document sources
✅ **Quality-assured outputs** (Gates 0-8 validation)
✅ **Traceability** (proof points link to source documents)

### What You Need to Build

#### 1. Minimal Web Server

**Framework Choice**: Flask (simpler) or FastAPI (more features)

**Example Flask Server** (`demo_server.py`):
```python
from flask import Flask, jsonify, render_template
import json
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

@app.route('/api/sessions')
def list_sessions():
    """List all available session IDs"""
    outputs_dir = BASE_DIR / "outputs"
    sessions = [p.name for p in outputs_dir.iterdir() if p.is_dir()]
    return jsonify({"sessions": sessions})

@app.route('/api/session/<session_id>')
def get_session(session_id: str):
    """Get all files for a session"""
    session_dir = BASE_DIR / "outputs" / session_id
    email = json.load(open(session_dir / "email.json"))
    insights = json.load(open(session_dir / "insights.json"))
    compliance = json.load(open(session_dir / "compliance_report.json"))
    state = json.load(open(BASE_DIR / "state" / f"session-{session_id}.json"))

    return jsonify({
        "session_id": session_id,
        "company": state.get("company", "Unknown"),
        "persona": state.get("persona", "Unknown"),
        "email": email,
        "insights": insights,
        "compliance": compliance
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

#### 2. Frontend UI

**Simple HTML Template** (`templates/demo.html`):
- Persona dropdown (3 options)
- Display area for:
  - Email subject
  - Email body (formatted with line breaks)
  - Proof points list with titles
  - Insight cards with URLs
  - Compliance status

**JavaScript**:
- Fetch sessions on page load
- Filter by persona (read from state files)
- Display first matching session or randomly select

#### 3. Environment Setup

```bash
# Create minimal demo environment
/Users/liyunxiao/anaconda3/bin/conda env create -f envs/demo_flask.yaml

# Run demo server
conda run -n demo python demo_server.py

# Access at http://localhost:5000
```

### Recommended Demo Flow

1. **Landing Page**: Show persona dropdown
2. **User Selection**: User picks persona (vp_customer_experience, cio, vp_sales_ops)
3. **Email Display**: System displays a pre-generated email for that persona
   - Subject line
   - Email body with bullet points
   - Source links from proof points
   - "These insights are based on real Salesforce documents from 2025"
4. **Optional Details**: Expandable sections for:
   - Full insight cards
   - Compliance report (show "Passed validation" if no critical flags)
   - Performance metrics (e.g., "Generated in 1.2 seconds")

### Data Mapping for Demo

**Persona → Sessions Mapping**:
Read all 34 state files, extract persona field, group by persona:

```python
import json
from pathlib import Path
from collections import defaultdict

sessions_by_persona = defaultdict(list)
for state_file in Path("state").glob("session-*.json"):
    state = json.load(open(state_file))
    persona = state.get("persona", "unknown")
    session_id = state_file.stem.replace("session-", "")
    sessions_by_persona[persona].append(session_id)

# Result:
# sessions_by_persona["vp_customer_experience"] = [list of session IDs]
# sessions_by_persona["cio"] = [list of session IDs]
# sessions_by_persona["vp_sales_ops"] = [list of session IDs]
```

**Display Logic**:
- On persona selection, pick first/random session from that persona's list
- Load `outputs/{session_id}/email.json` and `outputs/{session_id}/insights.json`
- Render in UI

## Critical Recommendations for Demo

Given the persona differentiation issue, here are strategic options:

### Option 1: Regenerate Emails with Persona Customization (Recommended if Time Permits)

**Action**: Fix the Planner node to load `persona_keywords`, then re-run LangGraph for 3-6 representative sessions.

**Steps**:
1. Verify `scripts/langgraph_nodes.py:190-191` loads keywords correctly
2. Run new sessions with explicit persona parameter
3. Verify `persona_keywords` appears in state files
4. Confirm email subjects/intros vary by persona

**Time estimate**: 2-4 hours (debugging + regeneration + QA)
**Benefit**: Demo shows true persona-aware capability

### Option 2: Demo with Current Data + Disclaimer (Fastest)

**Action**: Use existing 34 sessions but be transparent about limitations.

**Demo messaging**:
- "This demo shows our document retrieval and email assembly pipeline"
- Focus on: Source diversity, compliance validation, traceability (proof points)
- De-emphasize: Persona customization ("Coming soon: Advanced persona-specific tone")

**Time estimate**: < 1 hour
**Benefit**: No risk, honest showcase of current capability

### Option 3: Manual Email Editing (Medium Effort)

**Action**: Hand-edit 3-6 email.json files to show persona variation.

**Edit examples**:
- vp_sales_ops: Change subject to "Boost pipeline visibility at Salesforce", intro to "revenue growth opportunities"
- cio: Change subject to "Data integration insights from Salesforce", intro to "IT modernization"
- Keep vp_customer_experience as-is (already CX-focused)

**Time estimate**: 1-2 hours
**Benefit**: Demo looks polished without code changes
**Risk**: Not authentic to actual system output

### Option 4: Hybrid - Show Content Variety, Not Persona Switching

**Action**: Remove persona dropdown, show email variations based on "different outreach contexts."

**Demo flow**:
- "Our system generates personalized emails based on recent company data"
- Show 3-5 examples with different bullet points/sources
- Highlight: Document types (press, SEC, PDF), date range, compliance blocks

**Time estimate**: < 1 hour
**Benefit**: Sidesteps persona issue entirely, focuses on working features

## Open Questions

1. **Multiple Companies**: Do you want to expand beyond Salesforce? If yes, need to run full data collection + indexing pipeline for new companies (Gates 0-2).

2. **Deployment Target**: Should the demo run:
   - Locally on laptop during conference?
   - On a cloud platform (Heroku, Vercel, AWS)?
   - As a static HTML file with embedded JSON?

3. **UI Sophistication**: How polished should the UI be?
   - Simple Bootstrap/Tailwind styling?
   - Custom design with animations?
   - Mobile-responsive?

4. **Session Selection**: For each persona, should the demo:
   - Always show the same session (deterministic)?
   - Randomly pick from available sessions?
   - Allow cycling through multiple examples?

5. **Conference Timeline**: When is the entrepreneur conference? This determines urgency and scope.

6. **⭐ CRITICAL**: Which recommendation above (1-4) aligns with your timeline and demo goals?

## Related Research

- `CLAUDE.md` - Project overview and environment setup
- `docs/architecture.md` - Detailed system design
- `docs/commands.md` - Complete command reference
- `docs/evaluation.md` - Quality gates and metrics
- `README.md` - Main project documentation

## Conclusion

All necessary pre-generated data exists to build a minimal web demo. The codebase contains 34 high-quality email outputs for Salesforce across 3 personas, with consistent JSON structure and full traceability to source documents. No web UI currently exists, but the backend MCP infrastructure provides a reference architecture. A Flask/FastAPI-based demo requires only 1-3 packages (~50-80 MB) versus the full pipeline environment (~4.5 GB), enabling rapid deployment for investor showcase.

**Next Steps**:
1. Create `envs/demo_flask.yaml` or `envs/demo_fastapi.yaml`
2. Implement `demo_server.py` with 2-3 API endpoints
3. Create `templates/demo.html` with persona dropdown + email display
4. Test locally on `localhost:5000`
5. Deploy to target platform (if needed)
