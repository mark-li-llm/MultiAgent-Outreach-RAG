#!/usr/bin/env python3
"""
OpenAI ada-002 embedding utilities with caching and retry logic.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


def embed_batch(texts: List[str], dim: int, batch_size: int = 100) -> List[List[float]]:
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
        print(f"Embedding {len(texts_to_embed)} uncached texts (cached: {len(texts) - len(texts_to_embed)})")

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
