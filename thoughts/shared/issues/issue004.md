Retrieval System Recall Investigation — Gate-7 Status: 
  RED

  Key Metrics:
  - Chunk-level recall@10 = 65.22% (target ≥ 80%) — FAIL
  - Doc-level recall@10 = 82.61% (passing but chunk-level
   fails)
  - nDCG@5 = 34.41% (target ≥ 60%) — FAIL (ranking
  quality issue)
  - Doc-level nDCG@5 = 69.33%
  - Near miss rate = 17.39% (relevant chunks exist but
  ranked too low)

  Backend Performance Differences:
  - FAISS: chunk_recall@10 = 80%, doc_recall@10 = 100%,
  nDCG@5 = 51.31%
  - Weaviate: chunk_recall@10 = 65.38%, doc_recall@10 =
  73.08%, nDCG@5 = 39.22%
  - Pinecone: chunk_recall@10 = 50%, doc_recall@10 = 90%,
   nDCG@5 = 5% ⚠️ (critical ranking issue)

  Document Type Performance (chunk-level hits):
  - Product docs: 100% (6/6) ✓
  - Press releases: 76.92% (20/26)
  - Wikipedia: 50% (1/2)
  - 10-Q filings: 0% (0/6) ⚠️
  - 10-K filings: 0% (0/3) ⚠️
  - 8-K: 100% (1/1)
  - dev_docs: 100% (1/1)
  - help_docs: 100% (1/1)

  Critical Observations:
  1. SEC quarterly/annual filings (10-Q, 10-K) have 0% 
  chunk-level recall despite 50% and 33% doc-level recall
   → ranking/chunking issue
  2. Pinecone has 5% nDCG@5 despite 90% doc-level recall
  → severe ranking disorder
  3. Median chunk rank = 3 (when found) suggests ranking
  quality problems beyond simple recall
  4. 17.39% near-miss rate → relevant content exists but
  isn't surfaced in top-10
  5. ⚠️ Embedding model discrepancy detected: Report
  shows openai-ada-002 (dim=1536), but CLAUDE.md
  specifies hashlex-v1 (dim=768)

  Additional Issues:
  - All backends fail latency budgets (p95 exceeds
  targets)
  - Low soft recall@10 (4.35%) suggests adjacent-chunk
  context rarely retrieved

  Investigation Scope:
  1. ⚠️ PRIORITY: Verify actual embedding model in use
  (hashlex-v1 vs openai-ada-002)
  2. SEC filing chunking/metadata issues (10-Q, 10-K
  failures)
  3. Pinecone ranking algorithm problems (nDCG@5 = 5%)
  4. Reranking logic effectiveness (17.39% near-miss
  rate)
  5. Query-document semantic alignment for financial
  documents

  Evidence Available:
  - reports/eval/retrieval_failures.jsonl — failed query
  details
  - reports/router/step07_retrieval_trace.jsonl —
  per-query routing and results