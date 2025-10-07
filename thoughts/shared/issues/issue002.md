ultrathink
The current retrieval system's recall rate is below standard, and we need to
identify the root cause:

Key Data:
- Chunk-level recall@10 = 52.17% (target ≥ 80%)
- Doc-level recall@10 = 71.74%
- Backend differences: FAISS (70%) > Weaviate (57.69%) > Pinecone (20%)
- Document type differences: Press releases (16/26) vs. Wikipedia/SEC 8-K (0/N)
- Ruled Out: Data quality issues
- Scope of Investigation: From data processing to retrieval results + rationality
  of evaluation design

Please investigate the fundamental reasons for the low recall rate.