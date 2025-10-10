# LangGraph Edge Cases & Retry Mechanisms

This document tracks known edge cases, quirks, and defensive mechanisms in the LangGraph implementation (`scripts/run_graph_langgraph.py`).

## Quick Reference

| ID | Issue | Frequency | Mitigation | Details |
|----|-------|-----------|------------|---------|
| LG-001 | LLM ID Hallucination in Consolidator | 1/5 runs (~20%) | 3x retry with backoff, logged to `logs/langgraph/llm_retry_events.jsonl` | [→](langgraph/001-llm-id-hallucination.md) |

## When to Check This Document

- **Debugging**: Strange KeyErrors or LLM-related failures in LangGraph nodes
- **Monitoring**: Review `logs/langgraph/llm_retry_events.jsonl` periodically to track retry frequency
- **Contributing**: Adding new retry mechanisms? Document here first

## Adding New Entries

1. Create detailed doc: `docs/langgraph/NNN-title.md` (use next sequential ID)
2. Add row to table above with: ID, Issue, Frequency, Mitigation, Link
3. Keep title concise (≤8 words)
4. Include actual frequency data when known

## Log Files

All LangGraph-specific runtime logs are stored in:
- **Retry events**: `logs/langgraph/llm_retry_events.jsonl` (one line per retry)

Query logs: `cat logs/langgraph/llm_retry_events.jsonl | jq .`

## Related Documentation

- **Main architecture**: [docs/architecture.md](architecture.md)
- **Troubleshooting**: [docs/troubleshooting.md](troubleshooting.md)
- **Commands**: [docs/commands.md](commands.md)
