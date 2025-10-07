#!/usr/bin/env python3
"""
Debug SEC filing retrieval issue.
Test specific query and see what gets retrieved vs expected.
"""

import json
import sys
from pathlib import Path

# Add to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from embedding_utils import embed_text

def main():
    # Test query from eval seed
    query = "What was Salesforce's total revenue for Q1 FY26?"
    expected_chunk_id = "crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866::chunk0015"

    print("=" * 80)
    print("SEC Retrieval Debug Test")
    print("=" * 80)
    print(f"Query: {query}")
    print(f"Expected chunk: {expected_chunk_id}")
    print()

    # Load expected chunk
    chunks_file = Path("data/interim/chunks/crm::10-Q::2025-04-30::fy26-q1-form-10-q::e16f2866.chunks.jsonl")
    expected_chunk = None

    with open(chunks_file, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            if chunk['chunk_id'] == expected_chunk_id:
                expected_chunk = chunk
                break

    if not expected_chunk:
        print(f"ERROR: Expected chunk not found in {chunks_file}")
        return 1

    print("Expected chunk content:")
    print("-" * 80)
    print(expected_chunk['text'][:500])
    print("-" * 80)
    print()

    # Generate query embedding
    print("Generating query embedding...")
    query_vec = embed_text(query, 1536)
    print(f"Query vector norm: {sum(x*x for x in query_vec)**0.5:.4f}")
    print()

    # Load embeddings and compute similarities
    print("Loading embeddings and computing similarities...")
    import pyarrow.parquet as pq

    emb_table = pq.read_table('data/vector/embeddings/embeddings.parquet')
    chunk_ids = emb_table.column('chunk_id').to_pylist()
    embeddings = emb_table.column('vector').to_pylist()

    # Compute cosine similarity with all chunks
    def cosine_sim(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(x*x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if (norm_a * norm_b) > 0 else 0.0

    similarities = []
    expected_idx = None

    for i, (cid, emb) in enumerate(zip(chunk_ids, embeddings)):
        sim = cosine_sim(query_vec, emb)
        similarities.append((sim, i, cid))
        if cid == expected_chunk_id:
            expected_idx = i

    similarities.sort(reverse=True)

    # Find expected chunk rank
    expected_rank = None
    for rank, (sim, idx, cid) in enumerate(similarities, 1):
        if cid == expected_chunk_id:
            expected_rank = rank
            expected_sim = sim
            break

    print(f"Expected chunk rank: {expected_rank}/{len(similarities)}")
    print(f"Expected chunk similarity: {expected_sim:.4f}")
    print()

    # Show top 10 results
    print("Top 10 retrieval results:")
    print("-" * 80)
    for rank, (sim, idx, cid) in enumerate(similarities[:10], 1):
        # Get chunk text
        chunk_text = ""
        for cf in Path("data/interim/chunks").glob("*.chunks.jsonl"):
            with open(cf, 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    if chunk['chunk_id'] == cid:
                        chunk_text = chunk['text'][:100]
                        break
                if chunk_text:
                    break

        marker = " ← EXPECTED" if cid == expected_chunk_id else ""
        print(f"{rank}. {cid[:70]}")
        print(f"   Similarity: {sim:.4f}{marker}")
        print(f"   Text: {chunk_text}...")
        print()

    # Analyze expected chunk
    print("=" * 80)
    print("Analysis:")
    print("=" * 80)

    if expected_rank and expected_rank <= 10:
        print(f"✓ Expected chunk found in top 10 (rank {expected_rank})")
        print(f"  This should have been a HIT in Gate-7")
    elif expected_rank and expected_rank <= 50:
        print(f"⚠ Expected chunk found but low rank: {expected_rank}")
        print(f"  Similarity: {expected_sim:.4f}")
        print(f"  This explains the miss")
    else:
        print(f"✗ Expected chunk very low rank: {expected_rank}")
        print(f"  Similarity: {expected_sim:.4f}")
        print(f"  Semantic mismatch between query and chunk text")

    # Check top result doc type
    top_sim, top_idx, top_cid = similarities[0]
    if "10-Q" in top_cid:
        print(f"\n✓ Top result IS from 10-Q")
    elif "10-K" in top_cid or "8-K" in top_cid:
        print(f"\n⚠ Top result is from other SEC filing")
    else:
        print(f"\n✗ Top result is NOT from SEC filing")
        print(f"  Top: {top_cid[:70]}")

    return 0

if __name__ == "__main__":
    exit(main())
