
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