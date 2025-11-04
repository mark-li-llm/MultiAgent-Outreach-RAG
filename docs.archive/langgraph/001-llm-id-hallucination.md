# LG-001: LLM ID Hallucination in Consolidator Node

## TL;DR
LLM rarely drops `synth::` prefix from insight card IDs (~20% failure rate). **Solution**: Retry up to 3x with backoff. Check `logs/langgraph/llm_retry_events.jsonl` for occurrences.

---

## Symptom

```
KeyError: 'crm::press::2025-09-03::salesforce-reports-record-second-quarter-fiscal-2026-results::d558c08c::card'
Location: scripts/langgraph_nodes.py:408 (consolidator_node merge logic)
```

The KeyError occurs when trying to merge LLM-enhanced fields back to base cards, because the LLM returned an ID that doesn't exist in the input dictionary.

## When It Occurs

**Trigger**: Domain diversity enforcement creates synthesized cards

- **Condition**: When retrieved chunks don't span 4+ distinct source domains
- **Frequency**: ~20% of runs (observed 1/5 in testing)
- **Temperature**: 0.3 (allows nondeterminism in LLM responses)
- **Node**: consolidator_node in scripts/langgraph_nodes.py

## Root Cause

The LLM is instructed: *"Keep 'id' exactly as given to preserve traceability"*

However, input JSON includes both `id` and `doc_id` fields:

```json
{
  "id": "synth::crm::press::...::d558c08c::card",
  "doc_id": "crm::press::...::d558c08c",
  "title": "...",
  "summary": "...",
  ...8 other fields
}
```

**What the LLM does**:
1. Sees both `id` and `doc_id` fields
2. Gets confused by the `doc_id` value (which lacks `synth::` prefix)
3. Combines `doc_id` + `::card` suffix
4. Returns hybrid: `crm::press::...::d558c08c::card` (invalid)

**Valid ID patterns**:
- Chunk IDs: `crm::doctype::...::d558c08c::chunk0027`
- **Synth IDs**: `synth::crm::doctype::...::d558c08c::card` ← correct
- **Hallucinated**: `crm::doctype::...::d558c08c::card` ← missing `synth::` prefix

## Solution

**Defensive retry with exponential backoff** (scripts/langgraph_nodes.py:335-420)

```python
MAX_ATTEMPTS = 3
for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        # LLM call
        resp = await llm.ainvoke(...)
        cards_llm = json.loads(resp.content)

        # ID validation
        by_id = {c["id"]: c for c in cards}

        # Merge LLM fields back (this line throws KeyError on mismatch)
        for item in cards_llm:
            base = by_id[item["id"]]  # KeyError if hallucinated
            ...

        break  # Success - exit retry loop

    except KeyError as e:
        hallucinated_id = str(e).strip("'")

        if attempt < MAX_ATTEMPTS:
            # Log retry event
            log_llm_retry_event(
                session_id=state["session_id"],
                attempt=attempt,
                error_id=hallucinated_id,
                input_ids=input_ids,
                synth_count=synth_count
            )

            # Warn user
            print(f"⚠️  LLM retry {attempt}/{MAX_ATTEMPTS}: ID mismatch detected, retrying...", file=sys.stderr)

            continue  # Retry
        else:
            # All attempts exhausted
            raise AssertionError(
                f"consolidator_node: LLM ID hallucination after {MAX_ATTEMPTS} retries. "
                f"Hallucinated ID: {hallucinated_id}"
            ) from e
```

## Monitoring

**Runtime logs**: `logs/langgraph/llm_retry_events.jsonl`

Each retry event is logged as one JSON line:

```json
{
  "timestamp": "2025-10-10T12:34:56Z",
  "session_id": "test-smoke-01",
  "node": "consolidator",
  "attempt": 2,
  "max_attempts": 3,
  "error_type": "KeyError",
  "hallucinated_ids": ["crm::press::2025-09-03::salesforce-reports-record-second-quarter-fiscal-2026-results::d558c08c::card"],
  "expected_ids": [
    "synth::crm::press::2025-09-03::salesforce-reports-record-second-quarter-fiscal-2026-results::d558c08c::card",
    "synth::crm::press::2024-04-25::news-details::65e3c0b7::card",
    ...
  ],
  "synth_card_count": 5,
  "retry_reason": "LLM_ID_MISMATCH"
}
```

**Query retry logs**:
```bash
# View all retry events
cat logs/langgraph/llm_retry_events.jsonl | jq .

# Count retries by session
cat logs/langgraph/llm_retry_events.jsonl | jq -r '.session_id' | sort | uniq -c

# Check if any sessions exhausted all 3 attempts
cat logs/langgraph/llm_retry_events.jsonl | jq 'select(.attempt == 3)'
```

## Investigation History

**Date**: 2025-10-10

**Failed run**: `test-smoke-01`
- Error: KeyError on `crm::press::2025-09-03::...::card` (missing `synth::` prefix)
- All 5 input cards were synth cards
- LLM hallucinated at least one ID

**Successful runs after adding debug logging**: `test-debug-01`, `test-debug-02`, `test-debug-03`, `test-debug-04`
- **4/4 passed** (100% success rate with same input conditions)
- All used same 5 synth cards
- LLM correctly preserved all IDs with `synth::` prefix

**Conclusion**: Issue is nondeterministic, occurring ~20% of the time (1/5 observed runs).

**Full investigation**: `thoughts/shared/eval/2025-10-10-langgraph-test-session.md`

## Alternatives Considered

### ❌ Option A: Lower temperature to 0.0
- **Pro**: Eliminates nondeterminism, more deterministic output
- **Con**: Reduces output quality and creativity
- **Verdict**: Rejected - sacrifices quality for stability

### ❌ Option B: Simplify LLM input (send only id, title, summary)
- **Pro**: Removes confusing `doc_id` field
- **Con**: Diverges from original implementation design
- **Verdict**: Rejected - prefer maintaining design consistency

### ❌ Option C: Accept as rare edge case (no mitigation)
- **Pro**: No code changes needed
- **Con**: 20% failure rate unacceptable for production
- **Verdict**: Rejected - too frequent to ignore

### ✅ Option D: Defensive retry mechanism (chosen)
- **Pro**: Robust, maintains quality (temp=0.3), production-ready
- **Pro**: Provides visibility via logs
- **Con**: Slight complexity increase
- **Verdict**: **Accepted** - best balance of robustness and simplicity

## Related Files

- **Implementation**: `scripts/langgraph_nodes.py:335-420` (consolidator_node)
- **Logging helper**: `scripts/langgraph_nodes.py:XX-XX` (log_llm_retry_event function)
- **Runtime logs**: `logs/langgraph/llm_retry_events.jsonl`
- **Test session**: `thoughts/shared/eval/2025-10-10-langgraph-test-session.md`
- **Summary index**: `docs/langgraph-edge-cases.md`
