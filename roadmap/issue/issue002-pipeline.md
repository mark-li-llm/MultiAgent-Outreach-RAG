
## Part 2: Data Pipeline & Storage =📦

### Research Goal
Document the **complete data pipeline** from raw document collection through processing to final artifacts.

### Key Questions to Answer
1. What are the 13 pipeline stages? (names, purposes, sequence)
2. What scripts implement each stage? (file paths, line counts)
3. What data formats are used? (HTML, JSON, Parquet, etc.)
4. Where is data stored at each stage? (data/raw/, data/interim/, etc.)
5. How do stages connect? (dependencies, outputs → inputs)
6. What transformations occur? (HTML → text, text → chunks, etc.)


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
