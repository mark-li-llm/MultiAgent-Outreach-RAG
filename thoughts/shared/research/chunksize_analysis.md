 Chunk Size Analysis

  Configuration

  - Target: 800 tokens
  - Overlap: 120 tokens
  - Short doc threshold: 350 tokens (no chunking if smaller)
  - Boundary tolerance: 50 chars

  Current Statistics (396 total chunks across 75 docs)

  Token Distribution:
  - Mean: 1113 tokens (39% over target)
  - Median: 817 tokens (close to target)
  - Std Dev: 1699.7 (high variance)
  - Range: 15 - 9220 tokens

  Percentiles:
  - P50: 814 tokens ✓
  - P75: 843 tokens ✓
  - P90: 892 tokens ✓
  - P95: 1373 tokens ⚠️
  - P99: 9211 tokens ❌

  Size Buckets:
  - <400: 10.4%
  - 400-600: 3.6%
  - 600-800: 18.8%
  - 800-1000: 61.4% (majority)
  - ≥1000: 5.8%

  Critical Issue: Title Boost Explosion

  Root Cause: The Wikipedia article crm::wiki::unknown::salesforce-wikipedia::6b727edd has 13 chunks ALL at
  ~9000 tokens due to title boost mechanism.

  How it happens (scripts/chunk_documents.py:126-138):
  # Title boost: title + first H1
  h1 = get_first_h1(text)
  boost_lines = []
  if title:
      boost_lines.append(title)
  if h1 and h1 != title:
      boost_lines.append(h1)
  boost = "\n".join(boost_lines).strip()

  chunk_text = (boost + "\n\n" + body).strip() if boost else body

  The problem:
  - Wikipedia H1 contains the entire infobox (~8000 tokens of structured company data)
  - This H1 is prepended to EVERY chunk
  - Each 800-token body slice gets +8000 token boost = 9000 total tokens

  Impact

  Retrieval quality:
  - 9000-token chunks severely violate the 800-token design budget
  - Dilutes signal-to-noise ratio (10% actual content, 90% repeated context)
  - May cause embedding quality degradation (hashlex-v1 wasn't tested at 9K tokens)

  Statistics pollution:
  - Without Wikipedia outliers, mean would be ~820 tokens (perfectly on target)
  - 13 outlier chunks skew the mean by +293 tokens (+36%)

  Recommendations

  1. Cap title boost size to prevent context explosion:
  if h1 and len(h1) > 500:  # ~100 tokens
      h1 = h1[:500] + "..."
  2. Wikipedia-specific handling: Strip infobox content from H1 before using as boost
  3. Monitor boost ratio: Warn/cap when boost > body size
  4. Alternative: Use only title for boost, not H1 (would fix this immediately)

  The remaining 383 chunks (97%) are well-behaved with median 814 tokens, validating that the chunking logic
  itself is sound.