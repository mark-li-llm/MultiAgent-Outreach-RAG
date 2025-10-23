
## Part 6: LangGraph Agent System =🤖

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
