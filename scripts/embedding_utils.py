#!/usr/bin/env python3
import math
import re
import unicodedata
from typing import List


def normalize_text(text: str) -> str:
    t = (text or "").lower()
    # Unicode -> ASCII
    t = unicodedata.normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode("ascii", errors="ignore")
    # Separate hyphens/slashes to avoid gluing tokens
    t = t.replace("-", " ").replace("/", " ")
    # Collapse digits to a canonical form to generalize numbers/dates
    t = re.sub(r"\d+", "0", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(text: str) -> List[str]:
    t = normalize_text(text)
    # Keep alnum tokens 2..20 chars to avoid noise
    toks = re.findall(r"[a-z0-9]{2,20}", t)
    # Add bigrams to capture short local context
    bigrams: List[str] = []
    for i in range(len(toks) - 1):
        bigrams.append(f"bg:{toks[i]}_{toks[i+1]}")
    return toks + bigrams


def _stable_hash(seed: int, s: str) -> int:
    # 64-bit FNV-1a style with a custom seed mixed in
    h = (0xCBF29CE484222325 ^ (seed & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def hashlex_embed(tokens: List[str], dim: int, seed: int = 0x9E3779B9) -> List[float]:
    if dim <= 0:
        return []
    vec = [0.0] * dim
    if not tokens:
        # small non-zero uniform vector
        val = 1.0 / math.sqrt(dim)
        return [val] * dim
    for tok in tokens:
        h = _stable_hash(seed, tok)
        idx = int(h % dim)
        # sign bit from another simple mix
        sign = 1.0 if (h & 1) == 0 else -1.0
        vec[idx] += sign
    # L2-normalize
    s2 = sum(v * v for v in vec)
    if s2 <= 0:
        val = 1.0 / math.sqrt(dim)
        return [val] * dim
    inv = 1.0 / math.sqrt(s2)
    return [float(v * inv) for v in vec]


def embed_text(text: str, dim: int) -> List[float]:
    return hashlex_embed(tokenize(text), dim)

