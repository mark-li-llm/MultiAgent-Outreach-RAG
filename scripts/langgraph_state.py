#!/usr/bin/env python3
"""LangGraph state schema for multi-agent RAG system."""
from typing import TypedDict, Annotated, List, Dict, Any
from operator import add


class AgentState(TypedDict):
    """
    Shared state across all agent nodes.

    Fields with Annotated[..., add] accumulate across node invocations.
    Fields without annotation are replaced on each update.
    """
    # Input fields
    company: str
    persona: str
    session_id: str
    timestamp: str

    # Planning fields
    queries: List[str]
    persona_keywords: List[str]

    # Retrieval fields (accumulate)
    retrieved_chunks: Annotated[List[Dict[str, Any]], add]
    retrieval_logs: Annotated[List[Dict[str, Any]], add]
    route_decisions: Annotated[List[Dict[str, Any]], add]

    # Synthesis fields
    insight_candidates: List[Dict[str, Any]]
    insight_cards: List[Dict[str, Any]]

    # Generation fields
    email_draft: Dict[str, Any]

    # Compliance fields
    compliance_flags: Annotated[List[str], add]
    a2a_rounds: int  # Track number of A2A negotiation rounds

    # Observability fields
    metrics: Dict[str, Any]
    errors: Annotated[List[str], add]
