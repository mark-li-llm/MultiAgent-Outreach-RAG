
## Part 7: Quality Gates & Evaluation =✅

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
  - Thresholds: ≥0.80, ≥0.70, ≤1000ms
  - Environment overrides
  - Pass criteria
- **Gate-8: Generation Eval** (qa_step08_generation_eval.py)
  - Metrics: structural_pass_rate, critical_flags, readability, persona_keywords
  - Thresholds: 1.0, 0, ≥9/10, ≥2.0
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
