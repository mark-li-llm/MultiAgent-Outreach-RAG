# Backend Architecture Design

**Date**: 2025-11-03
**Status**: Planning

---

## Background

**Current State**:
- Multi-agent RAG system implemented in 41 Python scripts
- LangGraph orchestration, FAISS/Weaviate/Pinecone vector retrieval
- Command-line only, no user interface

**Goals**:
- Productionize for web deployment
- Support multiple users
- Preserve existing Python implementation

---

## Tech Stack Decisions

### Frontend
- **Choice**: Next.js 14

### Backend
- **Choice**: FastAPI (Python)

**Why FastAPI?**
1. Existing system is Python-native (LangGraph, FAISS, embedding_utils.py, etc.)
2. Zero migration cost, directly `import` existing modules
3. Async support matches existing aiohttp/MCP code
4. 2-3 days to deploy

**Why NOT Next.js for Backend?**
1. LangGraph has no JavaScript version (core dependency)
2. FAISS JS bindings are immature
3. Rewriting 10,000+ lines of code requires 2-3 months
4. Subprocess Python calls have poor performance and debugging issues

**Core Principle**: Use Python framework for Python system, separate frontend/backend

---

## Architecture Plan

### Progressive Implementation

**Phase 1: Minimal Backend (1-2 days)**
```
api/
└── main.py          # FastAPI entry point, ~50 lines

scripts/             # Existing AI system (no changes)
├── run_graph_langgraph.py
├── langgraph_nodes.py
└── ...
```

Capabilities:
- ✅ HTTP API endpoints
- ✅ Call existing Python functions
- ✅ Return JSON results

Not Included:
- ❌ User authentication
- ❌ Database persistence
- ❌ Task queue

**Phase 2: Production Backend (when needed)**
```
backend/
├── api/
│   ├── routes/      # Multiple route modules
│   └── main.py
├── core/            # Config, database
├── models/          # Data models
├── services/        # Business logic
└── middleware/      # Auth, logging, rate limiting

ai_system/           # AI core (migrated from scripts/)
├── scripts/
├── configs/
└── data/
```

Additional Features:
- User authentication & authorization
- PostgreSQL + Redis
- Task queue (Celery)
- Monitoring & alerting

---

## Core Understanding

### Role of FastAPI
- Adds HTTP interface layer to existing Python functions
- Analogy: Installing an "order window" for the kitchen, kitchen itself unchanged

### Definition of "Backend"
- **Minimal Backend** = API layer (receive requests, call functions, return results)
- **Full Backend** = API + auth + database + queue + monitoring + ...

### Implementation Strategy
- Start with Phase 1 to validate feasibility
- Decide on Phase 2 based on actual needs
- Avoid over-engineering

---

## Minimal Backend Conceptual Design

**Pseudocode Structure**:
```python
# api/main.py (conceptual, not verified)

# 1. Initialize FastAPI app
app = create_fastapi_app(title="RAG Email Generator")

# 2. Import existing workflow function
from existing_scripts import workflow_execution_function

# 3. Define HTTP endpoint
@app.post("/api/generate")
def generate_endpoint(company, persona, session_id):
    # Call existing Python function
    result = workflow_execution_function(company, persona, session_id)
    return wrap_response(result)
```

**Setup Steps**:
```bash
# Install dependencies
install fastapi and uvicorn in conda environment

# Run server
start uvicorn server with app on port 8000

# Access auto-generated docs
http://localhost:8000/docs
```

**Note**: Actual implementation requires verification of existing code structure and may differ from pseudocode above.

---

## Next Steps

- [ ] Examine `run_graph_langgraph.py` for importable functions
- [ ] Create `api/main.py` with actual implementation
- [ ] Test basic invocation
- [ ] Decide whether Phase 2 is needed
