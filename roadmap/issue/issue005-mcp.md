
## Part 5: MCP Tools & Services =🔧

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
