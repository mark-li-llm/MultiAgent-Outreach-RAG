
## Part 3: Vector & Embedding System =📐

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
