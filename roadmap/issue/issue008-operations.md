
## Part 8: Configuration & Operations =⚙️

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
