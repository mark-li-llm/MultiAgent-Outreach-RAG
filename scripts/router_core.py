#!/usr/bin/env python3
import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone, date
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


ROUTER_CONF = os.path.join("configs", "router.heuristics.yaml")
MCP_CONF = os.path.join("configs", "mcp.tools.yaml")
NORM_DIR = os.path.join("data", "interim", "normalized")


def _load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML configs")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_router_config(path: str = ROUTER_CONF) -> Dict[str, Any]:
    if not os.path.exists(path):
        # sensible defaults
        return {
            "weights": {"similarity": 0.6, "recency": 0.3, "diversity": 0.1},
            "persona_bias": {},
            "rules": [],
            "fallback_order": ["faiss", "weaviate", "pinecone"],
            "top_k_default": 10,
        }
    return _load_yaml(path)


def load_mcp_map(path: str = MCP_CONF) -> Dict[str, Dict[str, Any]]:
    cfg = _load_yaml(path)
    return cfg.get("tools", {})


@dataclass
class DocMeta:
    doc_id: str
    publish_date: Optional[str]
    source_domain: Optional[str]
    url: Optional[str]


def load_doc_meta() -> Dict[str, DocMeta]:
    m: Dict[str, DocMeta] = {}
    for p in glob.glob(os.path.join(NORM_DIR, "*.json")):
        try:
            d = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        doc_id = d.get("doc_id")
        if not doc_id:
            continue
        m[doc_id] = DocMeta(
            doc_id=doc_id,
            publish_date=(d.get("publish_date") or "").strip() or None,
            source_domain=(d.get("source_domain") or "").strip() or None,
            url=(d.get("final_url") or d.get("url") or "").strip() or None,
        )
    return m


def decide_backend(query: str, persona: Optional[str], meta: Optional[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
    """Deterministic router using configs/router.heuristics.yaml.

    Returns (backend, reason_codes).
    """
    cfg = load_router_config()
    ql = (query or "").lower()
    reasons: List[str] = []
    # Rule-based
    for rule in cfg.get("rules", []):
        cond = rule.get("if", {})
        kws = [str(x).lower() for x in cond.get("has_keywords", [])]
        if kws and any(kw in ql for kw in kws):
            then = rule.get("then", {})
            backend = str(then.get("backend") or "").strip() or "faiss"
            reason = str(then.get("reason") or "RULE_MATCH").strip()
            reasons.append(reason)
            return backend, reasons
    # Persona bias
    pb = cfg.get("persona_bias", {})
    if persona and persona in pb:
        reasons.append("PERSONA_BIAS")
        return pb[persona], reasons
    # Heuristic fallback: short/definitional => faiss, else weaviate
    if len(ql.split()) <= 4 or any(kw in ql for kw in ["what is", "define", "definition", "overview"]):
        reasons.append("DEFAULT_SHORT_FAISS")
        return "faiss", reasons
    reasons.append("DEFAULT_WEAVIATE")
    return "weaviate", reasons


def _days_since(iso_date: Optional[str]) -> Optional[int]:
    if not iso_date:
        return None
    try:
        d = date.fromisoformat(iso_date)
        return (datetime.now(timezone.utc).date() - d).days
    except Exception:
        return None


def rerank(
    results: List[Dict[str, Any]],
    docmeta: Dict[str, DocMeta],
    weights: Optional[Dict[str, float]] = None,
    *,
    top_k: int = 10,
    domain_cap: int = 2,
) -> List[Dict[str, Any]]:
    """Re-rank with recency and domain-aware diversity.

    - Scores each result: similarity + recency + diversity bonus.
    - Enforces a per-domain cap within the top_k window (default 2) to increase diversity.
    - Returns reordered list: diversified top_k followed by remaining results in score order.
    """
    w = {"similarity": 0.6, "recency": 0.3, "diversity": 0.1}
    if weights:
        w.update(weights)

    seen_domains: set = set()
    rescored: List[Tuple[float, Dict[str, Any]]] = []

    def sim_from_score(sc: Any) -> float:
        try:
            s = float(sc)
        except Exception:
            s = -1.0
        # Stub returns negative L2 distances; transform to (0,1]
        return 1.0 / (1.0 + abs(s))

    for r in results:
        did = r.get("doc_id")
        dm = docmeta.get(did) if did else None
        # similarity
        sim = sim_from_score(r.get("score"))
        # recency
        days = _days_since(dm.publish_date if dm else None)
        if days is None:
            rec = 0.3  # small neutral value when unknown
        else:
            # 0 at >= 2 years, 1 at today
            rec = max(0.0, 1.0 - (days / 730.0))
        # diversity bonus: + if new domain
        dom = (dm.source_domain or "") if dm else ""
        div = 0.1 if (dom and dom not in seen_domains) else 0.0
        # compute score
        final = (w["similarity"] * sim) + (w["recency"] * rec) + (w["diversity"] * div)
        rescored.append((final, r))
        if dom:
            seen_domains.add(dom)

    rescored.sort(key=lambda x: x[0], reverse=True)

    # Domain-aware selection for top_k
    selected: List[Dict[str, Any]] = []
    used_idx: set = set()
    domain_counts: Dict[str, int] = {}
    for i, (_s, r) in enumerate(rescored):
        if len(selected) >= max(1, top_k):
            break
        did = r.get("doc_id")
        dm = docmeta.get(did)
        dom = (dm.source_domain or "") if dm else ""
        cnt = domain_counts.get(dom, 0)
        if domain_cap <= 0 or cnt < domain_cap:
            selected.append(r)
            used_idx.add(i)
            domain_counts[dom] = cnt + 1

    # Append remainder in score order
    tail = [r for i, (_s, r) in enumerate(rescored) if i not in used_idx]
    return selected + tail

