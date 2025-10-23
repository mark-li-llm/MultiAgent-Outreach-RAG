
## Part 4: Multi-Index Routing =🔀

### Research Goal
Document how **queries are routed to the right vector backend** based on keywords, personas, and heuristics.

### Key Questions to Answer
1. What routing strategies exist? (keyword rules, persona bias, weighted scoring)
2. How are backends selected? (FAISS, Weaviate, Pinecone)
3. What are the routing rules? (press → Pinecone, API → Weaviate, etc.)
4. How do persona biases work? (VP CX → Pinecone, CIO → Weaviate)
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
  - Keyword rules (press → Pinecone, API → Weaviate)
  - Persona biases (VP CX → Pinecone, CIO → Weaviate)
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
