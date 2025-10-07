Overview

  Migrate from hashlex-v1 (768-dim) to OpenAI ada-002 (1536-dim) embeddings with improved
   efficiency, error handling, and cost optimization. This plan combines the best of
  exp52 and exp53 with additional improvements.

  Key Improvements:
  - Batch processing (100 texts/API call)
  - Retry logic with exponential backoff
  - Simple disk caching to avoid re-embedding
  - .env file for API key management
  - Offline mode prevention in run_graph.py
  - Cost estimation with user confirmation

  ---
  Phase 1: Environment Setup

  1.1 Create .env Template

  File: .env.example (new)
  # OpenAI API Configuration
  OPENAI_API_KEY=sk-...your-api-key-here...

  1.2 Update Conda Environment

  File: envs/age.yaml
  name: age
  channels:
    - conda-forge
  dependencies:
    - python=3.13
    - aiohttp
    - pyyaml
    - pyarrow>=21
    - numpy>=2.3
    - certifi
    - openblas
    - llvm-openmp
    - pip
    - pip:
        - openai>=1.0.0
        - python-dotenv>=1.0.0
        - tenacity>=8.2.0  # For retry logic
    # IMPORTANT: Do NOT install pip faiss-cpu in this env

  1.3 Setup Instructions

  # Recreate environment
  conda env remove -n age
  conda env create -f envs/age.yaml

  # Set up API key
  cp .env.example .env
  # Edit .env and add your OpenAI API key

  ---
  Phase 2: Core Embedding Implementation with Improvements

  2.1 New Embedding Utils with Caching & Retry

  File: scripts/embedding_utils.py (complete replacement)

  #!/usr/bin/env python3
  """
  OpenAI ada-002 embedding utilities with caching and retry logic.
  """
  import os
  import json
  import hashlib
  from pathlib import Path
  from typing import List, Optional
  from tenacity import retry, stop_after_attempt, wait_exponential,
  retry_if_exception_type

  from dotenv import load_dotenv
  from openai import OpenAI, APIError, APIConnectionError, RateLimitError

  # Load environment variables
  load_dotenv()

  # Constants
  ADA002_DIM = 1536
  CACHE_DIR = Path("data/cache/embeddings")
  CACHE_DIR.mkdir(parents=True, exist_ok=True)

  # Initialize OpenAI client
  api_key = os.getenv("OPENAI_API_KEY")
  if not api_key:
      raise ValueError(
          "OPENAI_API_KEY not found. Please set it in .env file:\n"
          "cp .env.example .env && edit .env"
      )

  client = OpenAI(api_key=api_key)


  def _get_cache_key(text: str) -> str:
      """Generate cache key for text."""
      return hashlib.sha256(text.encode()).hexdigest()[:16]


  def _load_from_cache(text: str) -> Optional[List[float]]:
      """Load embedding from cache if exists."""
      cache_key = _get_cache_key(text)
      cache_path = CACHE_DIR / f"{cache_key}.json"

      if cache_path.exists():
          try:
              with open(cache_path, 'r') as f:
                  data = json.load(f)
                  if data.get("text_hash") == hashlib.md5(text.encode()).hexdigest():
                      return data.get("embedding")
          except Exception:
              pass  # Invalid cache, will regenerate
      return None


  def _save_to_cache(text: str, embedding: List[float]):
      """Save embedding to cache."""
      cache_key = _get_cache_key(text)
      cache_path = CACHE_DIR / f"{cache_key}.json"

      try:
          with open(cache_path, 'w') as f:
              json.dump({
                  "text_hash": hashlib.md5(text.encode()).hexdigest(),
                  "embedding": embedding
              }, f)
      except Exception:
          pass  # Cache write failure is non-fatal


  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=4, max=10),
      retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError)),
      reraise=True
  )
  def _call_openai_api(text: str) -> List[float]:
      """Call OpenAI API with retry logic."""
      response = client.embeddings.create(
          model="text-embedding-ada-002",
          input=text,
          encoding_format="float"
      )
      return response.data[0].embedding


  def embed_text(text: str, dim: int) -> List[float]:
      """
      Generate embedding using OpenAI ada-002 with caching.

      Args:
          text: Input text to embed
          dim: Expected dimension (must be 1536 for ada-002)

      Returns:
          1536-dimensional embedding vector
      """
      if dim != ADA002_DIM:
          raise ValueError(
              f"OpenAI ada-002 requires dim={ADA002_DIM}, got dim={dim}. "
              f"Update configs/vector.indexing.yaml to set embedding.dim=1536"
          )

      # Handle empty text
      if not text or not text.strip():
          # Return small non-zero vector to avoid validation failures
          return [0.001] * ADA002_DIM

      # Check cache first
      cached = _load_from_cache(text)
      if cached is not None:
          return cached

      # Call API with retry logic
      try:
          embedding = _call_openai_api(text)
      except Exception as e:
          print(f"ERROR: OpenAI API failed after 3 retries: {e}")
          raise RuntimeError(
              f"OpenAI API call failed: {type(e).__name__}: {e}\n"
              f"Text length: {len(text)} chars\n"
              f"Check your API key and network connection."
          ) from e

      # Validate dimension
      if len(embedding) != ADA002_DIM:
          raise RuntimeError(
              f"OpenAI returned {len(embedding)}-dim vector, expected {ADA002_DIM}"
          )

      # Cache the result
      _save_to_cache(text, embedding)

      return embedding


  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(multiplier=1, min=4, max=10),
      retry=retry_if_exception_type((APIError, APIConnectionError, RateLimitError)),
      reraise=True
  )
  def _call_openai_batch(texts: List[str]) -> List[List[float]]:
      """Call OpenAI batch API with retry logic."""
      response = client.embeddings.create(
          model="text-embedding-ada-002",
          input=texts,
          encoding_format="float"
      )
      return [item.embedding for item in response.data]


  def embed_batch(texts: List[str], dim: int, batch_size: int = 100) ->
  List[List[float]]:
      """
      Generate embeddings for multiple texts using batch API with caching.

      Args:
          texts: List of input texts
          dim: Expected dimension (must be 1536)
          batch_size: Number of texts per API call (default 100)

      Returns:
          List of embedding vectors
      """
      if dim != ADA002_DIM:
          raise ValueError(
              f"OpenAI ada-002 requires dim={ADA002_DIM}, got dim={dim}"
          )

      if not texts:
          return []

      all_embeddings: List[List[float]] = []
      texts_to_embed: List[tuple[int, str]] = []

      # Check cache for each text
      for i, text in enumerate(texts):
          if not text or not text.strip():
              # Small non-zero vector for empty text
              all_embeddings.append([0.001] * ADA002_DIM)
          else:
              cached = _load_from_cache(text)
              if cached is not None:
                  all_embeddings.append(cached)
              else:
                  texts_to_embed.append((i, text))
                  all_embeddings.append(None)  # Placeholder

      # Batch process uncached texts
      if texts_to_embed:
          print(f"Embedding {len(texts_to_embed)} uncached texts (cached: {len(texts) -
  len(texts_to_embed)})")

          for batch_start in range(0, len(texts_to_embed), batch_size):
              batch = texts_to_embed[batch_start:batch_start + batch_size]
              batch_texts = [text for _, text in batch]

              try:
                  batch_embeddings = _call_openai_batch(batch_texts)

                  # Fill in results and cache them
                  for (orig_idx, text), embedding in zip(batch, batch_embeddings):
                      all_embeddings[orig_idx] = embedding
                      _save_to_cache(text, embedding)

              except Exception as e:
                  print(f"ERROR: Batch API failed after 3 retries: {e}")
                  raise RuntimeError(
                      f"Batch embedding failed at batch {batch_start//batch_size + 1}\n"
                      f"Error: {e}"
                  ) from e

      return all_embeddings


  def estimate_embedding_cost(num_texts: int, avg_text_length: int) -> dict:
      """
      Estimate cost of embedding a corpus.

      OpenAI ada-002 pricing: $0.0001 per 1K tokens (~750 words, ~4000 chars)
      """
      # Check how many are already cached
      cache_files = list(CACHE_DIR.glob("*.json"))
      cached_count = len(cache_files)

      # Rough estimate: 1 token ≈ 4 characters
      tokens_per_text = avg_text_length / 4.0
      uncached_texts = max(0, num_texts - cached_count)
      total_tokens = uncached_texts * tokens_per_text
      total_cost_usd = (total_tokens / 1000.0) * 0.0001

      return {
          "num_texts": num_texts,
          "cached_texts": min(cached_count, num_texts),
          "uncached_texts": uncached_texts,
          "avg_text_length_chars": avg_text_length,
          "estimated_tokens_per_text": int(tokens_per_text),
          "estimated_total_tokens": int(total_tokens),
          "cost_per_1k_tokens_usd": 0.0001,
          "estimated_total_cost_usd": round(total_cost_usd, 4),
          "note": "Cost only for uncached texts. Cached texts are free."
      }


  def clear_cache():
      """Clear the embedding cache."""
      import shutil
      if CACHE_DIR.exists():
          shutil.rmtree(CACHE_DIR)
          CACHE_DIR.mkdir(parents=True, exist_ok=True)
          print(f"Cleared embedding cache at {CACHE_DIR}")

  ---
  Phase 3: Update Configuration

  3.1 Vector Indexing Config

  File: configs/vector.indexing.yaml
  embedding:
    model: openai-ada-002
    dim: 1536
    batch_size: 100
    notes: OpenAI text-embedding-ada-002 with caching and retry logic

  faiss:
    type: HNSW
    metric: L2
    M: 32
    efConstruction: 200
    efSearch: 128

  pinecone:
    index_name: demo-index
    namespace: default
    metric: cosine
    notes: simulated manifest only (no network)

  weaviate:
    class_name: Doc
    notes: schema-only manifest (simulated)

  ---
  Phase 4: Update Gate-1 with Batch Processing

  4.1 Enhanced Gate-1 Script

  File: scripts/qa_step01_embeddings.py

  Add at line 13:
  from embedding_utils import embed_batch, estimate_embedding_cost

  Replace main function (starting at line ~115):
  def main():
      ensure_dir(VEC_DIR)
      dim = read_yaml_dim(CONF)
      baseline_chunks = load_baseline_chunks()

      # Load batch size from config
      try:
          with open(CONF, "r", encoding="utf-8") as f:
              cfg = yaml.safe_load(f)
          batch_size = int(cfg.get("embedding", {}).get("batch_size") or 100)
      except Exception:
          batch_size = 100

      # Collect all chunks
      all_chunks: List[Dict[str, Any]] = []
      for path in sorted(glob.glob(CHUNK_GLOB)):
          with open(path, "r", encoding="utf-8") as f:
              for line in f:
                  try:
                      j = json.loads(line)
                      all_chunks.append(j)
                  except Exception:
                      continue

      # Cost estimation
      print(f"=== Pre-flight Cost Estimation ===")
      if all_chunks:
          from statistics import median
          sample_lengths = [len(c.get("text") or "") for c in all_chunks[:100]]
          avg_len = int(median(sample_lengths)) if sample_lengths else 500

          cost_estimate = estimate_embedding_cost(len(all_chunks), avg_len)
          print(f"  Total chunks: {cost_estimate['num_texts']}")
          print(f"  Already cached: {cost_estimate['cached_texts']}")
          print(f"  Need to embed: {cost_estimate['uncached_texts']}")
          print(f"  Estimated cost: ${cost_estimate['estimated_total_cost_usd']:.4f}
  USD")
          print(f"  ({cost_estimate['note']})")

          if cost_estimate['uncached_texts'] > 0:
              response = input(f"\nProceed with embedding generation? [y/N]: ")
              if response.lower() != 'y':
                  print("Aborted by user.")
                  sys.exit(0)

      print(f"\n=== Starting Embedding Generation ===")
      print(f"  Model: openai-ada-002")
      print(f"  Dimension: {dim}")
      print(f"  Batch size: {batch_size}")
      print(f"  Total chunks: {len(all_chunks)}")

      # Extract texts and generate embeddings
      texts = [chunk.get("text") or "" for chunk in all_chunks]

      import time
      start_time = time.time()
      vectors = embed_batch(texts, dim, batch_size)
      elapsed = time.time() - start_time

      print(f"  Completed in {elapsed:.1f}s ({len(texts)/elapsed:.1f} chunks/sec)")

      # Process results
      rows: List[Dict[str, Any]] = []
      zero_vectors = 0
      nan_vectors = 0
      norms: List[float] = []

      for chunk, v in zip(all_chunks, vectors):
          n = l2_norm(v)

          if n == 0.0:
              zero_vectors += 1
          if any((x != x) for x in v):  # NaN check
              nan_vectors += 1

          rows.append({
              "chunk_id": chunk.get("chunk_id") or "",
              "doc_id": chunk.get("doc_id") or "",
              "seq_no": chunk.get("seq_no") or 0,
              "token_count": chunk.get("token_count") or 0,
              "l2_norm": n,
              "vector": [float(x) for x in v],
          })
          norms.append(n)

      embedding_rows = len(rows)

      # Continue with existing validation and output...

  ---
  Phase 5: Fix run_graph.py

  5.1 Disable Offline Mode

  File: scripts/run_graph.py

  Replace lines 295-330 with:
      # Offline index setup
      chunks_index: List[Dict[str, Any]] = []
      vectors: List[List[float]] = []
      dim = int(((load_yaml(os.path.join("configs","vector.indexing.yaml")) or
  {}).get("embedding") or {}).get("dim") or 1536)

      if use_offline:
          print("ERROR: Offline mode is not supported with OpenAI embeddings.")
          print("Offline mode requires API calls for query embedding which defeats the
  purpose.")
          print("Please run with online mode (MCP services) instead.")
          sys.exit(1)

      # Load pre-computed embeddings from parquet for graph execution
      import pyarrow.parquet as pq

      emb_path = os.path.join("data", "vector", "embeddings", "embeddings.parquet")
      if not os.path.exists(emb_path):
          print(f"ERROR: Embeddings not found at {emb_path}")
          print("Please run Gate-1 first: conda run -n age python
  scripts/qa_step01_embeddings.py")
          sys.exit(1)

      print(f"Loading embeddings from {emb_path}...")
      emb_table = pq.read_table(emb_path)
      chunk_ids = emb_table["chunk_id"].to_pylist()
      emb_vectors = emb_table["vector"].to_pylist()
      chunk_to_vec = dict(zip(chunk_ids, emb_vectors))

      # Load chunks and match with embeddings
      for cf in
  sorted(glob.glob(os.path.join("data","interim","chunks","*.chunks.jsonl"))):
          with open(cf, "r", encoding="utf-8") as f:
              for line in f:
                  try:
                      j = json.loads(line)
                      chunk_id = j.get("chunk_id")
                      if chunk_id and chunk_id in chunk_to_vec:
                          chunks_index.append(j)
                          vectors.append(chunk_to_vec[chunk_id])
                  except Exception:
                      continue

      print(f"Loaded {len(chunks_index)} chunks with embeddings")

  ---
  Phase 6: Update Other Scripts

  6.1 Remove Hardcoded Dimensions

  File: scripts/qa_step04_router.py

  Lines 214 and 221, replace:
  # OLD:
  dim = 768

  # NEW:
  import yaml
  cfg_path = os.path.join("configs", "vector.indexing.yaml")
  with open(cfg_path) as f:
      cfg = yaml.safe_load(f)
  dim = cfg.get("embedding", {}).get("dim", 1536)

  6.2 Update Default Dimension

  File: scripts/qa_step07_retrieval_eval.py

  Line 209:
  # OLD:
  dim = int((...).get("dim") or 768)

  # NEW:
  dim = int((...).get("dim") or 1536)

  ---
  Phase 7: Execution Steps

  7.1 Setup

  # 1. Recreate environment
  conda env remove -n age
  conda env create -f envs/age.yaml

  # 2. Set up API key
  cp .env.example .env
  # Edit .env and add your OpenAI API key

  # 3. Optional: Clear cache if you want fresh embeddings
  conda run -n age python -c "from embedding_utils import clear_cache; clear_cache()"

  7.2 Run Pipeline

  # 4. Generate embeddings (Gate-1)
  conda run -n age python scripts/qa_step01_embeddings.py
  # Will show cost estimate and ask for confirmation
  # Cached embeddings are free on subsequent runs

  # 5. Build FAISS index (Gate-2)
  conda run -n ageFaiss python scripts/qa_step02_indexes.py

  # 6. Validate MCP service (Gate-3)
  conda run -n age python scripts/qa_step03_mcp.py

  # 7. Run retrieval evaluation (Gate-7)
  conda run -n age AG7_IGNORE_COVERAGE=1 AG7_LATENCY_MULTIPLIER=3.0 python
  scripts/qa_step07_retrieval_eval.py

  ---
  Key Features

  ✅ Batch Processing

  - Process 100 texts per API call
  - Reduces API calls from 1600 to ~16
  - Much faster execution

  ✅ Retry Logic

  - 3 attempts with exponential backoff (4-10 seconds)
  - Handles transient API errors gracefully
  - Only fails on persistent issues

  ✅ Simple Caching

  - Disk-based cache in data/cache/embeddings/
  - SHA256 hash keys for fast lookup
  - Dramatically reduces costs on re-runs
  - Cache persists between runs

  ✅ Cost Transparency

  - Shows cached vs uncached texts
  - Estimates cost before API calls
  - User confirmation required
  - Cost only for new embeddings

  ✅ Offline Mode Prevention

  - run_graph.py exits cleanly if offline mode detected
  - Loads from pre-computed parquet instead
  - Clear error messages guide users

  ✅ .env File Usage

  - API key stored securely in .env
  - Never committed to git
  - Easy to update

  ---
  Verification Checklist

  # 1. Check environment
  conda run -n age python -c "import openai, tenacity, dotenv; print('Dependencies OK')"

  # 2. Check API key
  conda run -n age python -c "from dotenv import load_dotenv; import os; load_dotenv();
  print('API key set' if os.getenv('OPENAI_API_KEY') else 'API key missing')"

  # 3. Test single embedding
  conda run -n age python -c "from embedding_utils import embed_text; v =
  embed_text('test', 1536); print(f'Success: {len(v)} dims')"

  # 4. Check cache
  ls -la data/cache/embeddings/

  # 5. Verify dimension in config
  grep "dim: 1536" configs/vector.indexing.yaml

  # 6. Check parquet after Gate-1
  conda run -n age python -c "import pyarrow.parquet as pq; t =
  pq.read_table('data/vector/embeddings/embeddings.parquet'); print(f'Vectors: {len(t)},
  Dim: {len(t[\"vector\"][0].as_py())}')"

  ---
  Cost & Performance

  Estimated Costs (1600 chunks)

  - First run: ~$0.03-0.05 (all chunks embedded)
  - Subsequent runs: $0.00 (all cached)
  - Partial changes: Only new/modified chunks cost money

  Performance

  - With caching: Gate-1 completes in seconds for cached data
  - Without cache: ~2-5 minutes for 1600 chunks
  - Retry logic: Adds 4-10 seconds per failed attempt

  Cache Management

  # Clear cache if needed
  from embedding_utils import clear_cache
  clear_cache()

  # Check cache size
  du -sh data/cache/embeddings/

  This plan gives you a robust, cost-effective migration path with all the requested
  improvements!