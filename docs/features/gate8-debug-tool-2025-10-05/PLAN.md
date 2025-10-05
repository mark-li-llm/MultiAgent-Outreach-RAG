# Gate-8 Debug Implementation Plan (v2.0)

## Executive Summary

This document outlines the implementation plan for a new debug-oriented Gate-8 that provides deep visibility into the LangGraph pipeline through **non-invasive instrumentation**. Unlike the current Gate-8 which runs 10 black-box tests via subprocess, this debug version will execute a single run with comprehensive state inspection, node-level validation, and detailed tracing - all while **fully utilizing the existing LangGraph system** without reimplementation.

**Key Update (v2.0)**: Based on reviewer feedback, this plan now emphasizes wrapping and observing the existing `run_graph.py` rather than reimplementing nodes, with enhanced async handling and proper context management.

## Problem Statement

### Current Gate-8 Limitations
1. **Black-box execution**: Runs `run_graph.py` as subprocess, losing internal state visibility
2. **Batch-oriented**: Designed for 10 runs across personas, not detailed single-run debugging
3. **Limited diagnostics**: Only sees final outputs, can't inspect intermediate transformations
4. **Fallback complexity**: 3-mode MCP fallback obscures integration issues
5. **No LLM visibility**: Can't inspect prompts, responses, or token usage
6. **Post-mortem only**: Issues discovered after full pipeline execution

### Requirements for Debug Gate-8
1. **White-box execution**: Direct access to all internal state
2. **Single detailed run**: One execution with comprehensive instrumentation
3. **Node-level inspection**: Monitor each LangGraph node individually
4. **No fallback**: Strict mode to expose integration issues immediately
5. **LLM transparency**: Capture all LLM interactions
6. **Real-time validation**: Validate state at each node transition

## Proposed Architecture

### Core Components

```text
qa_step08_debug.py
├── LangGraphDebugger (Non-invasive orchestrator)
│   ├── Monkey-patching instrumentation
│   ├── Async context management
│   └── Existing config integration
├── NodeWrapper (Enhanced async wrapper)
│   ├── Pre/post execution hooks
│   ├── Timeout management
│   └── Exception handling
├── ValidationEngine (Quality checks)
│   ├── Node-specific validators
│   ├── State consistency checks
│   └── Output quality metrics
├── LLMMonitor (LLM interaction tracking)
│   ├── Prompt capture
│   ├── Response logging
│   └── Token usage metrics
└── DebugReporter (Output generation)
    ├── JSON machine report
    ├── Markdown human report
    └── JSONL trace files
```

### Integration Philosophy

**Key Principle**: Observe, don't modify. The debug gate should add instrumentation without changing the behavior of the existing LangGraph pipeline.

```python
# Non-invasive integration approach
class LangGraphDebugger:
    def __init__(self):
        # Load existing configurations
        self.nodes_config = load_yaml("configs/langgraph.nodes.yaml")
        self.node_timeouts = {
            k: v/1000 for k, v in
            self.nodes_config.get("timeouts_ms", {}).items()
        }

    async def run_instrumented(self, args):
        """Run existing pipeline with observation hooks"""
        from scripts.run_graph import main_async

        # Monkey-patch timing and state capture
        with self.instrumentation_context():
            result = await main_async(args)

        return result
```

## Implementation Details

### 1. Enhanced Async Node Wrapper (Reviewer Feedback)

```python
class NodeWrapper:
    """Enhanced wrapper with proper async handling and context management"""

    def __init__(self, debugger):
        self.debugger = debugger
        self.node_timeouts = debugger.node_timeouts
        self.stop_on_error = False  # Configurable

    async def wrap_node(self, node_name: str, func, state):
        """Enhanced wrapper with proper async handling"""
        async with self.node_context(node_name) as ctx:
            try:
                # Pre-execution hooks
                await self.pre_execute(node_name, state)

                # Execute with timeout from existing config
                result = await asyncio.wait_for(
                    func(state),
                    timeout=self.node_timeouts.get(node_name, 30)
                )

                # Post-execution hooks
                await self.post_execute(node_name, state, result)

                return result
            except asyncio.TimeoutError:
                ctx.mark_timeout()
                self.debugger.record_issue(
                    "TIMEOUT", node_name,
                    f"Exceeded {self.node_timeouts.get(node_name)}s"
                )
                raise
            except Exception as e:
                ctx.capture_exception(e)
                if self.stop_on_error:
                    raise
                return self.handle_node_failure(node_name, e, state)

    @asynccontextmanager
    async def node_context(self, node_name: str):
        """Context manager for resource management and timing"""
        start_time = time.perf_counter()
        ctx = NodeContext(node_name, self.debugger)

        try:
            yield ctx
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            ctx.finalize(duration_ms)
            await ctx.cleanup()

    async def pre_execute(self, node_name: str, state):
        """Pre-execution hook for state capture"""
        self.debugger.capture_state(node_name, "before", state)
        self.debugger.transitions.append({
            "entering": node_name,
            "timestamp": now_iso(),
            "state_keys": list(state.keys())
        })

    async def post_execute(self, node_name: str, state, result):
        """Post-execution hook for validation and capture"""
        self.debugger.capture_state(node_name, "after", state)
        await self.debugger.validate_node(node_name, state)
```

### 2. State Instrumentation (Non-Invasive)

```python
class StateInstrumentor:
    """Tracks state changes without modifying pipeline behavior"""

    def __init__(self):
        self.snapshots = {}
        self.transitions = []
        self.original_mark = None  # Store original timing function

    def instrument_mark_function(self, state_dict):
        """Monkey-patch the mark() function to capture timing"""
        # Capture original mark function if it exists
        if "mark" in dir(state_dict):
            self.original_mark = state_dict.mark

        def instrumented_mark(node: str, start: float, end: float):
            # Call original if it exists
            if self.original_mark:
                self.original_mark(node, start, end)

            # Add our instrumentation
            self.capture_timing(node, start, end)
            self.validate_node_output(node, state_dict)

        # Replace with instrumented version
        state_dict.mark = instrumented_mark

    def capture_state(self, node_name: str, phase: str, state: Dict):
        """Non-invasive state capture using deep copy"""
        import copy
        snapshot = {
            "timestamp": now_iso(),
            "keys": list(state.keys()),
            "sizes": {k: self._get_size(v) for k, v in state.items()},
            # Deep copy to prevent mutation
            "samples": copy.deepcopy(self._extract_samples(state))
        }

        if phase == "before":
            self.snapshots[node_name] = {"before": snapshot}
        else:
            self.snapshots[node_name]["after"] = snapshot
            self._compute_diff(node_name)
```

### 2. Node-Level Validation

```python
class NodeValidator:
    """Validates output quality at each node"""

    def __init__(self):
        self.validators = {
            "Planner": self.validate_planner,
            "Retriever": self.validate_retriever,
            "Synthesizer": self.validate_synthesizer,
            "Consolidator": self.validate_consolidator,
            "Stylist": self.validate_stylist,
            "A2A": self.validate_a2a,
            "Assembler": self.validate_assembler
        }
        self.results = []

    def validate_planner(self, state: Dict) -> ValidationResult:
        """Validate Planner output"""
        queries = state.get("queries", [])
        persona = state.get("persona", "")

        checks = []

        # Check 1: Query count
        checks.append({
            "check": "query_count",
            "expected": 5,
            "actual": len(queries),
            "pass": len(queries) == 5
        })

        # Check 2: Persona relevance
        persona_keywords = self._get_persona_keywords(persona)
        relevant_queries = sum(
            1 for q in queries
            if any(kw in q.lower() for kw in persona_keywords)
        )
        checks.append({
            "check": "persona_relevance",
            "expected": ">=3",
            "actual": relevant_queries,
            "pass": relevant_queries >= 3
        })

        return ValidationResult(
            node="Planner",
            checks=checks,
            status="PASS" if all(c["pass"] for c in checks) else "FAIL"
        )

    def validate_retriever(self, state: Dict) -> ValidationResult:
        """Validate Retriever output"""
        chunks = state.get("retrieved_chunks", [])
        route_decisions = state.get("route_decisions", [])

        checks = []

        # Check 1: Minimum chunks retrieved
        checks.append({
            "check": "min_chunks",
            "expected": ">=10",
            "actual": len(chunks),
            "pass": len(chunks) >= 10
        })

        # Check 2: Source diversity
        sources = set(c.get("doc_id", "").split("::")[0] for c in chunks)
        checks.append({
            "check": "source_diversity",
            "expected": ">=3",
            "actual": len(sources),
            "pass": len(sources) >= 3
        })

        # Check 3: Routing decisions made
        checks.append({
            "check": "routing_decisions",
            "expected": 5,
            "actual": len(route_decisions),
            "pass": len(route_decisions) == 5
        })

        return ValidationResult(
            node="Retriever",
            checks=checks,
            status="PASS" if all(c["pass"] for c in checks) else "FAIL"
        )

    def validate_consolidator(self, state: Dict) -> ValidationResult:
        """Validate Consolidator output with LLM enhancement"""
        cards = state.get("insight_cards", [])
        persona = state.get("persona", "")

        checks = []

        # Check 1: Card count
        checks.append({
            "check": "card_count",
            "expected": 5,
            "actual": len(cards),
            "pass": len(cards) == 5
        })

        # Check 2: LLM enhancement fields present
        required_fields = ["persona_relevance", "metric_impact", "action_suggestion"]
        enhanced_cards = sum(
            1 for c in cards
            if all(f in c for f in required_fields)
        )
        checks.append({
            "check": "llm_enhancement",
            "expected": 5,
            "actual": enhanced_cards,
            "pass": enhanced_cards == 5
        })

        # Check 3: Source domain diversity
        domains = set(c.get("source_domain", "") for c in cards)
        checks.append({
            "check": "domain_diversity",
            "expected": ">=4",
            "actual": len(domains),
            "pass": len(domains) >= 4
        })

        # Check 4: Persona alignment in enhancements
        persona_aligned = sum(
            1 for c in cards
            if c.get("persona_relevance", {}).get("relevance_score", 0) >= 3
        )
        checks.append({
            "check": "persona_alignment",
            "expected": ">=3",
            "actual": persona_aligned,
            "pass": persona_aligned >= 3
        })

        return ValidationResult(
            node="Consolidator",
            checks=checks,
            status="PASS" if all(c["pass"] for c in checks) else "FAIL"
        )
```

### 3. LLM Interaction Monitor (Non-Invasive)

```python
class LLMMonitor:
    """Tracks LLM interactions through monkey-patching"""

    def __init__(self):
        self.interactions = []
        self.token_usage = {}
        self.original_ainvoke = None

    def instrument_llm(self):
        """Monkey-patch ChatOpenAI.ainvoke for monitoring"""
        from langchain_openai import ChatOpenAI

        # Store original method
        self.original_ainvoke = ChatOpenAI.ainvoke

        # Create instrumented version
        async def instrumented_ainvoke(llm_self, *args, **kwargs):
            # Determine which node is calling based on stack inspection
            node_name = self._detect_calling_node()

            interaction = {
                "node": node_name,
                "timestamp": now_iso(),
                "model": llm_self.model_name,
                "temperature": llm_self.temperature,
            }

            # Capture prompt
            if args and len(args) > 0:
                interaction["prompt"] = self._extract_prompt(args[0])

            t0 = time.perf_counter()
            try:
                # Call original method
                response = await self.original_ainvoke(llm_self, *args, **kwargs)
                interaction["duration_ms"] = (time.perf_counter() - t0) * 1000

                # Capture response without modifying it
                if hasattr(response, 'content'):
                    interaction["response"] = response.content[:500]  # Sample
                if hasattr(response, 'response_metadata'):
                    interaction["tokens"] = response.response_metadata.get(
                        "token_usage", {}
                    )

                self.interactions.append(interaction)
                return response  # Return unmodified response

            except Exception as e:
                interaction["error"] = str(e)
                interaction["duration_ms"] = (time.perf_counter() - t0) * 1000
                self.interactions.append(interaction)
                raise  # Re-raise unmodified exception

        # Replace with instrumented version
        ChatOpenAI.ainvoke = instrumented_ainvoke

    def restore_llm(self):
        """Restore original LLM methods"""
        if self.original_ainvoke:
            from langchain_openai import ChatOpenAI
            ChatOpenAI.ainvoke = self.original_ainvoke

    def _detect_calling_node(self) -> str:
        """Detect which node is making the LLM call from stack"""
        import inspect
        stack = inspect.stack()
        for frame in stack:
            # Look for known node markers in the code
            local_vars = frame[0].f_locals
            if "node" in local_vars:
                return local_vars["node"]
            # Check for specific function names
            func_name = frame[3]
            if "consolidator" in func_name.lower():
                return "Consolidator"
            elif "stylist" in func_name.lower():
                return "Stylist"
        return "Unknown"
```

### 4. Debug Report Structure

#### Machine Report (`step08_debug.json`)

```json
{
    "gate": "Gate-8-Debug",
    "timestamp": "2024-10-05T10:30:00Z",
    "configuration": {
        "mode": "debug_single_run",
        "mcp_mode": "strict",
        "company": "Salesforce",
        "persona": "vp_customer_experience",
        "session_id": "debug_001"
    },
    "execution_timeline": [
        {
            "node": "Planner",
            "start": "2024-10-05T10:30:00.000Z",
            "end": "2024-10-05T10:30:00.050Z",
            "duration_ms": 50,
            "status": "SUCCESS"
        },
        {
            "node": "Retriever",
            "start": "2024-10-05T10:30:00.050Z",
            "end": "2024-10-05T10:30:01.250Z",
            "duration_ms": 1200,
            "status": "SUCCESS"
        }
    ],
    "node_validations": {
        "Planner": {
            "status": "PASS",
            "checks": [
                {"check": "query_count", "expected": 5, "actual": 5, "pass": true},
                {"check": "persona_relevance", "expected": ">=3", "actual": 4, "pass": true}
            ]
        },
        "Retriever": {
            "status": "PASS",
            "checks": [
                {"check": "min_chunks", "expected": ">=10", "actual": 48, "pass": true},
                {"check": "source_diversity", "expected": ">=3", "actual": 5, "pass": true}
            ]
        }
    },
    "state_transitions": [
        {
            "from": "Intake",
            "to": "Planner",
            "keys_added": ["queries"],
            "keys_modified": ["metrics"],
            "data_flow": {
                "input_size": 256,
                "output_size": 1024
            }
        }
    ],
    "llm_usage": {
        "Consolidator": {
            "calls": 1,
            "prompt_tokens": 1200,
            "completion_tokens": 450,
            "total_tokens": 1650,
            "avg_latency_ms": 820
        },
        "Stylist": {
            "calls": 1,
            "prompt_tokens": 980,
            "completion_tokens": 320,
            "total_tokens": 1300,
            "avg_latency_ms": 650
        }
    },
    "quality_metrics": {
        "structural": {
            "insights_count": 5,
            "distinct_sources": 4,
            "recent_count": 2,
            "email_schema_valid": true,
            "proof_points_resolved": true
        },
        "compliance": {
            "critical_flags": [],
            "warning_flags": ["READABILITY"],
            "word_count": 135,
            "readability_grade": 9.8
        },
        "persona_alignment": {
            "keyword_hits": 3,
            "relevance_scores": [4, 5, 3, 4, 4],
            "avg_relevance": 4.0
        }
    },
    "issues_detected": [
        {
            "severity": "WARNING",
            "node": "Retriever",
            "issue": "Backend fallback occurred",
            "details": "Weaviate timeout, fell back to FAISS",
            "timestamp": "2024-10-05T10:30:00.800Z"
        }
    ],
    "artifacts": {
        "session_dir": "outputs/debug_001/",
        "trace_files": [
            "reports/debug/debug_001/node_states.jsonl",
            "reports/debug/debug_001/llm_interactions.jsonl",
            "reports/debug/debug_001/validation_trace.jsonl"
        ]
    }
}
```

#### Human Report (`step08_debug.md`)

```markdown
# Gate-8 Debug Report

**Session**: debug_001
**Timestamp**: 2024-10-05T10:30:00Z
**Company**: Salesforce
**Persona**: vp_customer_experience

## 🎯 Executive Summary

✅ **Pipeline Status**: SUCCESS with 1 warning
⏱️ **Total Runtime**: 3.2s
🤖 **LLM Tokens Used**: 2,950
📊 **Quality Score**: 92/100

## 📈 Execution Timeline

```
[00.000s] ━━━ Start
[00.050s] ✅ Planner        (50ms)   - Generated 5 persona queries
[01.250s] ✅ Retriever      (1200ms) - Retrieved 48 chunks from 5 sources
[01.450s] ✅ Synthesizer    (200ms)  - Created 12 candidate insights
[02.050s] ✅ Consolidator   (600ms)  - Enhanced 5 insights with LLM
[02.650s] ✅ Stylist        (600ms)  - Generated 135-word email
[03.150s] ⚠️  A2A           (500ms)  - 1 revision for compliance
[03.200s] ✅ Assembler      (50ms)   - Packaged final email
[03.200s] ━━━ Complete
```

## 🔍 Node-by-Node Analysis

### Planner Node
**Input**: persona="vp_customer_experience", company="Salesforce"
**Output**: 5 queries focusing on CX metrics
**Validation**: ✅ All checks passed

Generated Queries:
1. "Salesforce customer experience AI" ✅ (persona-aligned)
2. "Agentforce product announcement" ✅ (recent)
3. "Customer service automation metrics" ✅ (persona-aligned)
4. "NPS improvement strategies" ✅ (persona-aligned)
5. "Data Cloud customer insights" ✅ (product-focused)

### Retriever Node
**Routing Decisions**:
- Query 1 → Weaviate (keyword: "AI")
- Query 2 → Pinecone (keyword: "announcement")
- Query 3 → FAISS (fallback)
- Query 4 → Weaviate (keyword: "NPS")
- Query 5 → FAISS (keyword: "Data Cloud")

**Retrieved**: 48 chunks total
- salesforce.com: 18 chunks
- wiki: 8 chunks
- press: 12 chunks
- product: 7 chunks
- dev_docs: 3 chunks

**Issues**: ⚠️ Weaviate timeout on query 1 (fell back to FAISS)

### Consolidator Node (LLM-Enhanced)
**LLM Model**: gpt-5-nano
**Tokens**: 1,650 (prompt: 1,200, completion: 450)
**Latency**: 820ms

**Selected Insights**:
1. ✅ "AI-powered service automation" (relevance: 5/5)
2. ✅ "Agentforce reduces response time by 47%" (relevance: 4/5)
3. ✅ "NPS increased 12 points with Einstein" (relevance: 5/5)
4. ✅ "Data Cloud enables 360° customer view" (relevance: 4/5)
5. ✅ "Self-service adoption up 63%" (relevance: 4/5)

**Persona Enhancements Added**:
- persona_relevance: ✅ All 5 insights enhanced
- metric_impact: ✅ Business metrics identified
- action_suggestion: ✅ Actionable recommendations

### Stylist Node (LLM-Generated)
**LLM Model**: gpt-5-nano
**Tokens**: 1,300 (prompt: 980, completion: 320)
**Latency**: 650ms

**Email Generated**:
- Subject: "Transform Your CX with AI-Driven Insights" (9 words) ✅
- Body: 135 words ✅
- Persona keywords: 3 hits ("customer experience", "NPS", "first contact resolution") ✅
- Bullets: 3 ✅
- Compliance blocks: Present ✅

### A2A Compliance Node
**Rounds**: 1
**Initial Flags**: ["READABILITY"]
**Final Flags**: [] (resolved)
**Revisions**: Simplified 2 complex sentences

## 📊 Quality Validation

### Structural Requirements
| Requirement | Target | Actual | Status |
|------------|--------|--------|--------|
| Insights count | 5 | 5 | ✅ |
| Distinct sources | ≥4 | 4 | ✅ |
| Recent insights | ≥2 | 2 | ✅ |
| Email schema | Valid | Valid | ✅ |
| Proof points | Resolved | Resolved | ✅ |

### Persona Alignment
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Keyword hits | ≥2 | 3 | ✅ |
| Avg relevance | ≥3.0 | 4.0 | ✅ |
| CX terminology | Present | Yes | ✅ |

### Compliance & Readability
| Check | Target | Actual | Status |
|-------|--------|--------|--------|
| Critical flags | 0 | 0 | ✅ |
| Word count | ≤160 | 135 | ✅ |
| Grade level | ≤10.0 | 9.8 | ✅ |
| Unsubscribe | Present | Yes | ✅ |

## 🐛 Issues & Warnings

### ⚠️ Warning: Weaviate Timeout
- **Node**: Retriever
- **Query**: "Salesforce customer experience AI"
- **Impact**: Minimal (FAISS fallback succeeded)
- **Recommendation**: Check Weaviate service health

## 📁 Debug Artifacts

Generated debug files for detailed analysis:
- `reports/debug/debug_001/node_states.jsonl` - State snapshots
- `reports/debug/debug_001/llm_interactions.jsonl` - LLM prompts/responses
- `reports/debug/debug_001/validation_trace.jsonl` - All validation results
- `outputs/debug_001/email.json` - Final email output
- `outputs/debug_001/insights.json` - Enhanced insight cards

## 🎯 Recommendations

1. **Weaviate Performance**: Investigate timeout issues
2. **Query Optimization**: Consider caching frequent persona queries
3. **LLM Efficiency**: Batch Consolidator and Stylist calls could save 400ms

---
*Debug run completed successfully with comprehensive tracing enabled*
```

## Benefits of Non-Invasive Approach (v2.0)

### Why This Approach is Superior

| Aspect | Original Plan | Updated Plan (v2.0) |
|--------|--------------|---------------------|
| **Code Changes** | Reimplements nodes | Zero changes to run_graph.py |
| **Maintenance** | Must track pipeline changes | Automatically adapts |
| **Debugging** | Modifies behavior | Pure observation |
| **Async Handling** | Basic wrapper | Robust context management |
| **Error Recovery** | Limited | Configurable (stop/continue) |
| **Resource Cleanup** | Manual | Automatic via context managers |

### Key Advantages

1. **Zero Modification Risk**: The pipeline runs exactly as it would in production
2. **Automatic Adaptation**: Changes to run_graph.py are immediately reflected
3. **Clean Separation**: Debug logic is completely isolated from business logic
4. **Reversible**: Can enable/disable debugging without any trace
5. **Production-Safe**: Could theoretically run in production for diagnostics

## Implementation Phases (Updated)

### Phase 1: Non-Invasive Framework (3-4 hours)
1. Create `qa_step08_debug.py` with instrumentation context
2. Implement enhanced `NodeWrapper` with async context management
3. Design monkey-patching strategy for mark() function
4. Load existing configs (timeouts, nodes, etc.)

### Phase 2: State & Transition Tracking (2-3 hours)
1. Implement `StateInstrumentor` with deep-copy snapshots
2. Create transition tracking without modifying state
3. Add diff computation for state changes
4. Build non-invasive validation hooks

### Phase 3: LLM Instrumentation (2-3 hours)
1. Monkey-patch `ChatOpenAI.ainvoke` method
2. Implement stack-based node detection
3. Capture prompts/responses without modification
4. Track token usage across nodes

### Phase 4: Validation Engine (3-4 hours)
1. Create validators that observe (not modify) state
2. Load persona keywords from existing configs
3. Implement quality metrics based on state observation
4. Build issue detection without throwing exceptions

### Phase 5: Debug Reporting (2-3 hours)
1. Generate comprehensive JSON report
2. Create visual Markdown timeline
3. Write JSONL traces for deep debugging
4. Ensure all paths are relative to existing structure

### Phase 6: Integration Testing (2-3 hours)
1. Test with existing run_graph.py (no modifications)
2. Verify monkey-patching and restoration
3. Validate async timeout handling
4. Test error scenarios (timeout, LLM failure, etc.)

## Success Criteria

1. **Single Run Execution**: Completes one detailed run in <10 seconds
2. **Full State Visibility**: Captures state at every node transition
3. **LLM Transparency**: Records all prompts, responses, and token usage
4. **Validation Coverage**: Validates each node's output quality
5. **Debug Artifacts**: Generates comprehensive trace files
6. **No Fallback**: Uses strict MCP mode only (fails fast on issues)
7. **Direct Integration**: No subprocess isolation, full white-box access

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Import conflicts | High | Use explicit imports, namespace isolation |
| State mutation | Medium | Deep copy states before modification |
| Async complexity | Medium | Proper async/await handling, no blocking calls |
| Memory usage | Low | Stream large outputs to JSONL, limit in-memory storage |
| LLM API failures | High | Capture exceptions, provide detailed error context |

## Developer Testing Checklist

- [ ] Runs without modifying original `run_graph.py`
- [ ] Captures all node transitions
- [ ] Validates each node's output
- [ ] Records LLM interactions
- [ ] Generates all report formats
- [ ] Handles errors gracefully
- [ ] Completes in <10 seconds
- [ ] No resource leaks
- [ ] Clear debug output

## Questions for Review

1. Should we add interactive debugging (breakpoints)?
2. Do we need real-time streaming of debug events?
3. Should we support multiple personas in one debug run?
4. Do we want to mock LLM responses for deterministic testing?
5. Should debug mode be toggleable in production run_graph.py?

## Appendix: Code Integration Points

Key files to integrate with:
- `scripts/run_graph.py` - Main pipeline (lines 137-761)
- `scripts/router_core.py` - Routing logic
- `scripts/embedding_utils.py` - Embedding functions
- `scripts/qa_step03_mcp.py` - MCP stub services
- `configs/eval.prompts.yaml` - Persona keywords
- `configs/langgraph.nodes.yaml` - Node definitions

## Key Changes from Reviewer Feedback

### 1. Enhanced Async Wrapper
- ✅ Proper async context management with `async with`
- ✅ Timeout handling using existing config values
- ✅ Configurable error behavior (stop vs continue)
- ✅ Resource cleanup guaranteed via context managers

### 2. Non-Invasive Instrumentation
- ✅ Zero modifications to run_graph.py
- ✅ Monkey-patching for observation only
- ✅ Original behavior preserved exactly
- ✅ Clean separation of debug and business logic

### 3. Full Config Integration
- ✅ Uses existing `langgraph.nodes.yaml` for timeouts
- ✅ Reads persona keywords from `eval.prompts.yaml`
- ✅ Respects MCP configuration from `mcp.tools.yaml`
- ✅ Follows existing file structure and conventions

## Example Usage

```python
# Run debug Gate-8
async def main():
    debugger = LangGraphDebugger()

    # Configure debug settings
    debugger.stop_on_error = False  # Continue on errors
    debugger.capture_llm = True     # Monitor LLM calls
    debugger.validate_nodes = True  # Run validators

    # Run with instrumentation
    args = argparse.Namespace(
        company="Salesforce",
        persona="vp_customer_experience",
        session_id="debug_001"
    )

    # Execute with full observability
    result = await debugger.run_instrumented(args)

    # Generate debug reports
    debugger.generate_reports()

    print(f"Debug complete: reports/qa/step08_debug.json")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

1. **Review this updated plan** focusing on non-invasive approach
2. **Confirm async handling** meets requirements
3. **Approve implementation** of Phase 1
4. **Begin development** with enhanced wrapper

---
*Document updated with reviewer feedback - October 2024 (v2.0)*
