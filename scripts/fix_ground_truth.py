#!/usr/bin/env python3
"""
Script to verify and fix ground truth in evaluation seed.
Reads chunks and finds the correct chunk for each query.
"""
import json
import re
from typing import Dict, List, Any, Optional

def load_chunks(chunks_file: str) -> List[Dict[str, Any]]:
    """Load all chunks from a JSONL file."""
    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except:
                continue
    return chunks

def search_chunks(chunks: List[Dict[str, Any]], keywords: List[str], context_words: int = 100) -> List[Dict[str, Any]]:
    """Search for chunks containing keywords."""
    results = []
    for chunk in chunks:
        text = chunk.get('text', '').lower()
        # Check if any keyword appears
        matches = sum(1 for kw in keywords if kw.lower() in text)
        if matches > 0:
            # Extract snippet
            snippet = text[:500]
            results.append({
                'chunk_id': chunk.get('chunk_id'),
                'seq_no': chunk.get('seq_no'),
                'matches': matches,
                'snippet': snippet,
                'full_text': chunk.get('text')
            })
    # Sort by number of matches
    results.sort(key=lambda x: x['matches'], reverse=True)
    return results

def verify_query(query_obj: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify if the expected_chunk_id contains the answer."""
    query_text = query_obj.get('query_text', '')
    expected_chunk_id = query_obj.get('expected_chunk_id', '')
    expected_keyphrases = query_obj.get('expected_answer_keyphrases', [])

    # Find the expected chunk
    expected_chunk = None
    for chunk in chunks:
        if chunk.get('chunk_id') == expected_chunk_id:
            expected_chunk = chunk
            break

    if not expected_chunk:
        return {
            'status': 'ERROR',
            'message': 'Expected chunk not found',
            'query': query_text,
            'expected_chunk_id': expected_chunk_id
        }

    # Check if keyphrases are in the chunk
    chunk_text = expected_chunk.get('text', '').lower()
    matches = [kp for kp in expected_keyphrases if kp.lower() in chunk_text]

    if len(matches) < len(expected_keyphrases) * 0.5:  # Less than 50% keyphrases found
        # Search for better chunks
        better_chunks = search_chunks(chunks, expected_keyphrases, context_words=100)
        return {
            'status': 'WRONG',
            'message': f'Only {len(matches)}/{len(expected_keyphrases)} keyphrases found',
            'query': query_text,
            'expected_chunk_id': expected_chunk_id,
            'expected_snippet': chunk_text[:300],
            'matches': matches,
            'keyphrases': expected_keyphrases,
            'suggested_chunks': better_chunks[:3]
        }

    return {
        'status': 'OK',
        'query': query_text,
        'expected_chunk_id': expected_chunk_id,
        'matches': matches
    }

def main():
    # Load eval seed
    seed_file = 'data/interim/eval/salesforce_eval_seed.jsonl'
    queries = []
    with open(seed_file, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries")

    # Process first 10 (10-K/10-Q queries)
    chunks_cache = {}
    results = []

    for i, query in enumerate(queries[:10]):
        print(f"\n--- Query {i+1}: {query.get('eval_id')} ---")

        # Get doc_id
        doc_id = query.get('expected_doc_id')
        chunks_file = f"data/interim/chunks/{doc_id}.chunks.jsonl"

        # Load chunks if not cached
        if doc_id not in chunks_cache:
            try:
                chunks_cache[doc_id] = load_chunks(chunks_file)
                print(f"Loaded {len(chunks_cache[doc_id])} chunks from {doc_id}")
            except Exception as e:
                print(f"Error loading chunks: {e}")
                continue

        # Verify
        result = verify_query(query, chunks_cache[doc_id])
        results.append(result)

        print(f"Status: {result['status']}")
        if result['status'] == 'WRONG':
            print(f"Query: {result['query']}")
            print(f"Expected chunk: {result['expected_chunk_id']}")
            print(f"Expected snippet: {result['expected_snippet'][:200]}...")
            print(f"Keyphrases: {result['keyphrases']}")
            print(f"Matches: {result['matches']}")
            print(f"\nSuggested chunks:")
            for j, sugg in enumerate(result.get('suggested_chunks', [])[:3]):
                print(f"  {j+1}. {sugg['chunk_id']} (seq_no={sugg['seq_no']}, matches={sugg['matches']})")
                print(f"     {sugg['snippet'][:150]}...")

    # Summary
    print(f"\n=== SUMMARY ===")
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    wrong_count = sum(1 for r in results if r['status'] == 'WRONG')
    error_count = sum(1 for r in results if r['status'] == 'ERROR')
    print(f"OK: {ok_count}, WRONG: {wrong_count}, ERROR: {error_count}")

if __name__ == '__main__':
    main()
