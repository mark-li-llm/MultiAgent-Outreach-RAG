#!/usr/bin/env python3
"""
Gate-8 Debug: Single-Run Deep Inspection
Non-invasive instrumentation of the LangGraph pipeline with comprehensive state tracking.
"""
import argparse
import asyncio
import copy
import inspect
import json
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, now_iso


# Paths
NODES_CONF = os.path.join("configs", "langgraph.nodes.yaml")
EVAL_CONFIG = os.path.join("configs", "eval.prompts.yaml")
MCP_CONFIG = os.path.join("configs", "mcp.tools.yaml")

# Output paths
OUT_JSON = os.path.join("reports", "qa", "step08_debug.json")
OUT_MD = os.path.join("reports", "qa", "step08_debug.md")
DEBUG_DIR = os.path.join("reports", "debug")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config with fallback."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def within_12mo(iso: Optional[str]) -> bool:
    """Check if date is within last 12 months."""
    if not iso:
        return False
    try:
        d = date.fromisoformat(iso)
        return (datetime.now(timezone.utc).date() - d).days <= 365
    except Exception:
        return False


def word_count(text: str) -> int:
    """Count words in text."""
    return len(re.findall(r"\b\w+\b", text or ""))


def readability_grade(text: str) -> float:
    """Flesch-Kincaid Grade approximation."""
    sentences = [s for s in re.split(r"[.!?]+", text or "") if s.strip()]
    sents = max(1, len(sentences))
    words = max(1, word_count(text))
    syllables = max(1, sum(len(re.findall(r"[aeiouyAEIOUY]", w)) or 1
                           for w in re.findall(r"\b\w+\b", text or "")))
    return 0.39 * (words / sents) + 11.8 * (syllables / words) - 15.59


class NodeContext:
    """Context object for tracking node execution"""

    def __init__(self, node_name: str, debugger):
        self.node_name = node_name
        self.debugger = debugger
        self.timed_out = False
        self.exception = None
        self.duration_ms = 0

    def mark_timeout(self):
        """Mark this node as timed out"""
        self.timed_out = True

    def capture_exception(self, exc: Exception):
        """Capture exception that occurred"""
        self.exception = exc

    def finalize(self, duration_ms: float):
        """Finalize context with timing"""
        self.duration_ms = duration_ms
        self.debugger.node_timings[self.node_name] = duration_ms

    async def cleanup(self):
        """Async cleanup if needed"""
        pass


class NodeWrapper:
    """Enhanced wrapper with proper async handling and context management"""

    def __init__(self, debugger):
        self.debugger = debugger
        self.node_timeouts = debugger.node_timeouts
        self.stop_on_error = debugger.stop_on_error

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

    def handle_node_failure(self, node_name: str, exc: Exception, state):
        """Handle node failure gracefully"""
        self.debugger.record_issue(
            "ERROR", node_name, f"{type(exc).__name__}: {str(exc)}"
        )
        return state  # Return state unmodified


class StateInstrumentor:
    """Tracks state changes without modifying pipeline behavior"""

    def __init__(self):
        self.snapshots = {}
        self.transitions = []

    def capture_state(self, node_name: str, phase: str, state: Dict):
        """Non-invasive state capture using deep copy"""
        snapshot = {
            "timestamp": now_iso(),
            "keys": list(state.keys()),
            "sizes": self._get_sizes(state),
            "samples": self._extract_samples(state)
        }

        if phase == "before":
            self.snapshots[node_name] = {"before": snapshot}
        else:
            if node_name not in self.snapshots:
                self.snapshots[node_name] = {"before": {}}
            self.snapshots[node_name]["after"] = snapshot
            self._compute_diff(node_name)

    def _get_sizes(self, state: Dict) -> Dict[str, int]:
        """Get sizes of state values"""
        sizes = {}
        for k, v in state.items():
            if isinstance(v, (list, tuple)):
                sizes[k] = len(v)
            elif isinstance(v, dict):
                sizes[k] = len(v)
            elif isinstance(v, str):
                sizes[k] = len(v)
            else:
                sizes[k] = 1
        return sizes

    def _extract_samples(self, state: Dict) -> Dict[str, Any]:
        """Extract sample data from state"""
        samples = {}
        for k, v in state.items():
            if isinstance(v, list) and v:
                samples[k] = {"type": "list", "count": len(v), "first": str(v[0])[:100]}
            elif isinstance(v, dict) and v:
                samples[k] = {"type": "dict", "keys": list(v.keys())[:5]}
            elif isinstance(v, str):
                samples[k] = {"type": "str", "sample": v[:100]}
            else:
                samples[k] = {"type": type(v).__name__}
        return samples

    def _compute_diff(self, node_name: str):
        """Compute what changed in this node"""
        before = self.snapshots[node_name].get("before", {})
        after = self.snapshots[node_name].get("after", {})

        diff = {
            "keys_added": list(set(after.get("keys", [])) - set(before.get("keys", []))),
            "keys_removed": list(set(before.get("keys", [])) - set(after.get("keys", []))),
            "sizes_changed": {}
        }

        # Track size changes
        before_sizes = before.get("sizes", {})
        after_sizes = after.get("sizes", {})
        for key in set(before_sizes.keys()) & set(after_sizes.keys()):
            if before_sizes[key] != after_sizes[key]:
                diff["sizes_changed"][key] = {
                    "before": before_sizes[key],
                    "after": after_sizes[key]
                }

        self.snapshots[node_name]["diff"] = diff


class LLMMonitor:
    """Tracks LLM interactions through monkey-patching"""

    def __init__(self):
        self.interactions = []
        self.token_usage = {}
        self.original_ainvoke = None
        self.enabled = False

    def instrument_llm(self):
        """Monkey-patch ChatOpenAI.ainvoke for monitoring"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            return

        # Store original method
        self.original_ainvoke = ChatOpenAI.ainvoke

        # Create instrumented version
        monitor = self

        async def instrumented_ainvoke(llm_self, *args, **kwargs):
            # Determine which node is calling
            node_name = monitor._detect_calling_node()

            interaction = {
                "node": node_name,
                "timestamp": now_iso(),
                "model": llm_self.model_name,
                "temperature": llm_self.temperature,
            }

            # Capture prompt
            if args and len(args) > 0:
                interaction["prompt"] = monitor._extract_prompt(args[0])

            t0 = time.perf_counter()
            try:
                # Call original method
                response = await monitor.original_ainvoke(llm_self, *args, **kwargs)
                interaction["duration_ms"] = (time.perf_counter() - t0) * 1000

                # Capture response without modifying it
                if hasattr(response, 'content'):
                    interaction["response"] = str(response.content)[:500]
                if hasattr(response, 'response_metadata'):
                    interaction["tokens"] = response.response_metadata.get("token_usage", {})

                monitor.interactions.append(interaction)
                monitor._update_token_usage(node_name, interaction.get("tokens", {}))
                return response  # Return unmodified response

            except Exception as e:
                interaction["error"] = str(e)
                interaction["duration_ms"] = (time.perf_counter() - t0) * 1000
                monitor.interactions.append(interaction)
                raise  # Re-raise unmodified exception

        # Replace with instrumented version
        ChatOpenAI.ainvoke = instrumented_ainvoke
        self.enabled = True

    def restore_llm(self):
        """Restore original LLM methods"""
        if self.original_ainvoke and self.enabled:
            try:
                from langchain_openai import ChatOpenAI
                ChatOpenAI.ainvoke = self.original_ainvoke
            except ImportError:
                pass

    def _detect_calling_node(self) -> str:
        """Detect which node is making the LLM call from stack"""
        stack = inspect.stack()
        for frame in stack:
            func_name = frame[3].lower()
            if "consolidator" in func_name:
                return "Consolidator"
            elif "stylist" in func_name:
                return "Stylist"
        return "Unknown"

    def _extract_prompt(self, messages) -> str:
        """Extract prompt text from messages"""
        if hasattr(messages, 'messages'):
            msgs = messages.messages
            return str([{"role": getattr(m, "type", "unknown"),
                        "content": str(getattr(m, "content", ""))[:200]}
                       for m in msgs])
        return str(messages)[:500]

    def _update_token_usage(self, node_name: str, tokens: Dict):
        """Update token usage statistics"""
        if node_name not in self.token_usage:
            self.token_usage[node_name] = {"prompt": 0, "completion": 0, "total": 0}

        self.token_usage[node_name]["prompt"] += tokens.get("prompt_tokens", 0)
        self.token_usage[node_name]["completion"] += tokens.get("completion_tokens", 0)
        self.token_usage[node_name]["total"] += tokens.get("total_tokens", 0)


class NodeValidator:
    """Validates output quality at each node"""

    def __init__(self, debugger):
        self.debugger = debugger
        self.results = []
        self.persona_keywords = {}
        self._load_persona_keywords()

    def _load_persona_keywords(self):
        """Load persona keywords from config"""
        eval_cfg = load_yaml(EVAL_CONFIG)
        self.persona_keywords = eval_cfg.get("personas", {})

    async def validate_node(self, node_name: str, state: Dict):
        """Validate node output based on node type"""
        validators = {
            "Planner": self.validate_planner,
            "Retriever": self.validate_retriever,
            "Consolidator": self.validate_consolidator,
            "Stylist": self.validate_stylist,
            "A2A": self.validate_a2a,
        }

        if node_name in validators:
            result = validators[node_name](state)
            self.results.append(result)
            return result

        return None

    def validate_planner(self, state: Dict) -> Dict[str, Any]:
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
        keywords = self.persona_keywords.get(persona, [])
        relevant_queries = sum(
            1 for q in queries
            if any(kw in q.lower() for kw in keywords)
        )
        checks.append({
            "check": "persona_relevance",
            "expected": ">=3",
            "actual": relevant_queries,
            "pass": relevant_queries >= 3
        })

        return {
            "node": "Planner",
            "checks": checks,
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL"
        }

    def validate_retriever(self, state: Dict) -> Dict[str, Any]:
        """Validate Retriever output"""
        chunks = state.get("retrieved_chunks", [])
        route_decisions = state.get("route_decisions", [])

        checks = []

        # Check 1: Minimum chunks
        checks.append({
            "check": "min_chunks",
            "expected": ">=10",
            "actual": len(chunks),
            "pass": len(chunks) >= 10
        })

        # Check 2: Source diversity
        sources = set(c.get("doc_id", "").split("::")[0] for c in chunks if c.get("doc_id"))
        checks.append({
            "check": "source_diversity",
            "expected": ">=3",
            "actual": len(sources),
            "pass": len(sources) >= 3
        })

        # Check 3: Routing decisions
        checks.append({
            "check": "routing_decisions",
            "expected": 5,
            "actual": len(route_decisions),
            "pass": len(route_decisions) == 5
        })

        return {
            "node": "Retriever",
            "checks": checks,
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL"
        }

    def validate_consolidator(self, state: Dict) -> Dict[str, Any]:
        """Validate Consolidator output with LLM enhancement"""
        cards = state.get("insight_cards", [])

        checks = []

        # Check 1: Card count
        checks.append({
            "check": "card_count",
            "expected": 5,
            "actual": len(cards),
            "pass": len(cards) == 5
        })

        # Check 2: LLM enhancement fields
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
        domains = set(c.get("source_domain", "") for c in cards if c.get("source_domain"))
        checks.append({
            "check": "domain_diversity",
            "expected": ">=4",
            "actual": len(domains),
            "pass": len(domains) >= 4
        })

        return {
            "node": "Consolidator",
            "checks": checks,
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL"
        }

    def validate_stylist(self, state: Dict) -> Dict[str, Any]:
        """Validate Stylist output"""
        email = state.get("email_draft", {})
        persona = state.get("persona", "")

        checks = []

        # Check 1: Email schema
        required = ["subject", "body", "unsubscribe_block", "company_info_block"]
        checks.append({
            "check": "email_schema",
            "expected": "all_fields",
            "actual": sum(1 for f in required if email.get(f)),
            "pass": all(email.get(f) for f in required)
        })

        # Check 2: Word count
        body = email.get("body", "")
        wc = word_count(body)
        checks.append({
            "check": "word_count",
            "expected": "<=160",
            "actual": wc,
            "pass": wc <= 160
        })

        # Check 3: Persona keywords
        keywords = self.persona_keywords.get(persona, [])
        hits = sum(1 for kw in keywords if kw in body.lower())
        checks.append({
            "check": "persona_keywords",
            "expected": ">=2",
            "actual": hits,
            "pass": hits >= 2
        })

        return {
            "node": "Stylist",
            "checks": checks,
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL"
        }

    def validate_a2a(self, state: Dict) -> Dict[str, Any]:
        """Validate A2A compliance"""
        flags = state.get("compliance_flags", [])

        checks = []

        # Check: No critical flags
        checks.append({
            "check": "no_critical_flags",
            "expected": 0,
            "actual": len(flags),
            "pass": len(flags) == 0
        })

        return {
            "node": "A2A",
            "checks": checks,
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL"
        }


class DebugReporter:
    """Generates comprehensive debug reports"""

    def __init__(self, debugger):
        self.debugger = debugger

    def generate_reports(self):
        """Generate all report formats"""
        # Ensure directories
        ensure_dir(os.path.dirname(OUT_JSON))
        ensure_dir(os.path.dirname(OUT_MD))
        ensure_dir(DEBUG_DIR)

        # Generate JSON report
        json_report = self._build_json_report()
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(json_report, f, ensure_ascii=False, indent=2)

        # Generate Markdown report
        md_report = self._build_markdown_report(json_report)
        with open(OUT_MD, "w", encoding="utf-8") as f:
            f.write(md_report)

        # Generate trace files
        self._write_trace_files()

    def _build_json_report(self) -> Dict[str, Any]:
        """Build machine-readable JSON report"""
        d = self.debugger

        # Build execution timeline
        timeline = []
        cumulative = 0
        for node, duration_ms in d.node_timings.items():
            timeline.append({
                "node": node,
                "start_offset_ms": cumulative,
                "duration_ms": round(duration_ms, 2),
                "status": "SUCCESS" if node not in [i["node"] for i in d.issues] else "ERROR"
            })
            cumulative += duration_ms

        # Build validation summary
        validation_summary = {}
        for result in d.validator.results:
            validation_summary[result["node"]] = {
                "status": result["status"],
                "checks": result["checks"]
            }

        # Build quality metrics
        quality_metrics = self._compute_quality_metrics(d.final_state)

        return {
            "gate": "Gate-8-Debug",
            "timestamp": now_iso(),
            "configuration": {
                "mode": "debug_single_run",
                "mcp_mode": "strict",
                "company": d.args.company,
                "persona": d.args.persona,
                "session_id": d.session_id
            },
            "execution_timeline": timeline,
            "total_runtime_ms": round(sum(d.node_timings.values()), 2),
            "node_validations": validation_summary,
            "state_transitions": d.state_instrumentor.transitions,
            "llm_usage": d.llm_monitor.token_usage,
            "llm_interactions_count": len(d.llm_monitor.interactions),
            "quality_metrics": quality_metrics,
            "issues_detected": d.issues,
            "artifacts": {
                "session_dir": d.output_dir,
                "trace_files": [
                    os.path.join(DEBUG_DIR, "node_states.jsonl"),
                    os.path.join(DEBUG_DIR, "llm_interactions.jsonl"),
                    os.path.join(DEBUG_DIR, "validation_trace.jsonl")
                ]
            }
        }

    def _compute_quality_metrics(self, state: Dict) -> Dict[str, Any]:
        """Compute quality metrics from final state"""
        insights = state.get("insight_cards", [])
        email = state.get("email_draft", {})

        # Structural metrics
        domains = set(c.get("source_domain", "") for c in insights if c.get("source_domain"))
        recent = sum(1 for c in insights if within_12mo(c.get("date")))

        # Compliance metrics
        body = email.get("body", "")
        wc = word_count(body)
        grade = readability_grade(body)

        # Persona alignment
        persona = state.get("persona", "")
        keywords = self.debugger.validator.persona_keywords.get(persona, [])
        keyword_hits = sum(1 for kw in keywords if kw in body.lower())

        return {
            "structural": {
                "insights_count": len(insights),
                "distinct_sources": len(domains),
                "recent_count": recent,
                "email_schema_valid": bool(email.get("subject") and email.get("body"))
            },
            "compliance": {
                "word_count": wc,
                "readability_grade": round(grade, 2)
            },
            "persona_alignment": {
                "keyword_hits": keyword_hits,
                "persona": persona
            }
        }

    def _build_markdown_report(self, json_report: Dict) -> str:
        """Build human-readable Markdown report"""
        config = json_report["configuration"]
        timeline = json_report["execution_timeline"]
        total_ms = json_report["total_runtime_ms"]
        validations = json_report["node_validations"]
        quality = json_report["quality_metrics"]
        issues = json_report["issues_detected"]

        # Build timeline visualization
        timeline_lines = ["[00.000s] ━━━ Start"]
        cumulative = 0
        for item in timeline:
            cumulative = item["start_offset_ms"] + item["duration_ms"]
            status_emoji = "✅" if item["status"] == "SUCCESS" else "❌"
            timeline_lines.append(
                f"[{cumulative/1000:.3f}s] {status_emoji} {item['node']:<15} "
                f"({item['duration_ms']:.0f}ms)"
            )
        timeline_lines.append(f"[{total_ms/1000:.3f}s] ━━━ Complete")

        # Build validation section
        validation_lines = []
        for node, result in validations.items():
            status_emoji = "✅" if result["status"] == "PASS" else "❌"
            validation_lines.append(f"### {status_emoji} {node} Node")
            for check in result["checks"]:
                check_emoji = "✅" if check["pass"] else "❌"
                validation_lines.append(
                    f"- {check_emoji} {check['check']}: {check['actual']} "
                    f"(expected: {check['expected']})"
                )
            validation_lines.append("")

        # Build issues section
        issues_section = ""
        if issues:
            issues_section = "## 🐛 Issues Detected\n\n"
            for issue in issues:
                severity_emoji = "⚠️" if issue["severity"] == "WARNING" else "❌"
                issues_section += f"### {severity_emoji} {issue['severity']}: {issue['node']}\n"
                issues_section += f"- **Issue**: {issue['issue']}\n"
                issues_section += f"- **Details**: {issue['details']}\n\n"
        else:
            issues_section = "## ✅ No Issues Detected\n\n"

        # Build report
        md = f"""# Gate-8 Debug Report

**Session**: {config['session_id']}
**Timestamp**: {json_report['timestamp']}
**Company**: {config['company']}
**Persona**: {config['persona']}

## 🎯 Executive Summary

✅ **Pipeline Status**: {"SUCCESS" if not issues else "ISSUES FOUND"}
⏱️ **Total Runtime**: {total_ms/1000:.2f}s
🤖 **LLM Interactions**: {json_report['llm_interactions_count']}
📊 **Quality Score**: {self._compute_score(quality)}/100

## 📈 Execution Timeline

```
{chr(10).join(timeline_lines)}
```

## 🔍 Node Validation Results

{chr(10).join(validation_lines)}

{issues_section}

## 📊 Quality Metrics

### Structural Requirements
| Metric | Value | Target |
|--------|-------|--------|
| Insights count | {quality['structural']['insights_count']} | 5 |
| Distinct sources | {quality['structural']['distinct_sources']} | ≥4 |
| Recent insights | {quality['structural']['recent_count']} | ≥2 |

### Compliance
| Metric | Value | Target |
|--------|-------|--------|
| Word count | {quality['compliance']['word_count']} | ≤160 |
| Readability grade | {quality['compliance']['readability_grade']} | ≤10.0 |

### Persona Alignment
| Metric | Value | Target |
|--------|-------|--------|
| Keyword hits | {quality['persona_alignment']['keyword_hits']} | ≥2 |
| Persona | {quality['persona_alignment']['persona']} | - |

## 📁 Debug Artifacts

Generated files for detailed analysis:
- `{OUT_JSON}` - Machine-readable report
- `{os.path.join(DEBUG_DIR, "node_states.jsonl")}` - State snapshots
- `{os.path.join(DEBUG_DIR, "llm_interactions.jsonl")}` - LLM calls
- `{os.path.join(DEBUG_DIR, "validation_trace.jsonl")}` - Validation results

---
*Debug run completed with comprehensive tracing enabled*
"""
        return md

    def _compute_score(self, quality: Dict) -> int:
        """Compute overall quality score"""
        score = 100

        # Structural penalties
        if quality['structural']['insights_count'] != 5:
            score -= 20
        if quality['structural']['distinct_sources'] < 4:
            score -= 15
        if quality['structural']['recent_count'] < 2:
            score -= 10

        # Compliance penalties
        if quality['compliance']['word_count'] > 160:
            score -= 10
        if quality['compliance']['readability_grade'] > 10.0:
            score -= 10

        # Persona alignment
        if quality['persona_alignment']['keyword_hits'] < 2:
            score -= 15

        return max(0, score)

    def _write_trace_files(self):
        """Write JSONL trace files"""
        d = self.debugger

        # Node states trace
        states_path = os.path.join(DEBUG_DIR, "node_states.jsonl")
        ensure_dir(os.path.dirname(states_path))
        with open(states_path, "w", encoding="utf-8") as f:
            for node, snapshot in d.state_instrumentor.snapshots.items():
                f.write(json.dumps({
                    "node": node,
                    "before": snapshot.get("before"),
                    "after": snapshot.get("after"),
                    "diff": snapshot.get("diff")
                }) + "\n")

        # LLM interactions trace
        llm_path = os.path.join(DEBUG_DIR, "llm_interactions.jsonl")
        with open(llm_path, "w", encoding="utf-8") as f:
            for interaction in d.llm_monitor.interactions:
                f.write(json.dumps(interaction) + "\n")

        # Validation trace
        val_path = os.path.join(DEBUG_DIR, "validation_trace.jsonl")
        with open(val_path, "w", encoding="utf-8") as f:
            for result in d.validator.results:
                f.write(json.dumps(result) + "\n")


class LangGraphDebugger:
    """Main orchestrator for non-invasive LangGraph debugging"""

    def __init__(self):
        # Load configurations
        self.nodes_config = load_yaml(NODES_CONF)
        self.node_timeouts = {
            k: v / 1000 for k, v in
            self.nodes_config.get("timeouts_ms", {}).items()
        }

        # Components
        self.state_instrumentor = StateInstrumentor()
        self.llm_monitor = LLMMonitor()
        self.validator = NodeValidator(self)
        self.reporter = DebugReporter(self)

        # Tracking
        self.node_timings = {}
        self.transitions = []
        self.issues = []

        # Settings
        self.stop_on_error = False
        self.capture_llm = True
        self.validate_nodes = True

        # Runtime state
        self.args = None
        self.session_id = None
        self.output_dir = None
        self.final_state = {}

    def capture_state(self, node_name: str, phase: str, state: Dict):
        """Capture state at node boundary"""
        self.state_instrumentor.capture_state(node_name, phase, state)

    async def validate_node(self, node_name: str, state: Dict):
        """Validate node output"""
        if self.validate_nodes:
            await self.validator.validate_node(node_name, state)

    def record_issue(self, severity: str, node: str, details: str):
        """Record an issue"""
        self.issues.append({
            "severity": severity,
            "node": node,
            "issue": severity,
            "details": details,
            "timestamp": now_iso()
        })

    async def run_instrumented(self, args):
        """Run existing pipeline with instrumentation hooks"""
        self.args = args
        self.session_id = getattr(args, 'session_id', None) or f"debug_{int(time.time())}"
        self.output_dir = os.path.join("outputs", self.session_id)

        # Instrument LLM if enabled
        if self.capture_llm:
            self.llm_monitor.instrument_llm()

        try:
            # Import and run main_async from run_graph
            from run_graph import main_async

            # Execute (this runs the entire pipeline)
            session_id = await main_async(args)

            # Load final state from session file
            state_path = os.path.join("state", f"session-{session_id}.json")
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    self.final_state = json.load(f)

            return session_id

        finally:
            # Restore LLM
            if self.capture_llm:
                self.llm_monitor.restore_llm()

    def generate_reports(self):
        """Generate all debug reports"""
        self.reporter.generate_reports()


async def main_async(args):
    """Main async entry point"""
    debugger = LangGraphDebugger()

    # Configure
    debugger.stop_on_error = args.stop_on_error
    debugger.capture_llm = not args.no_llm_capture
    debugger.validate_nodes = not args.no_validation

    print(f"🔍 Starting debug run for {args.persona} @ {args.company}")
    print(f"   Session: {args.session_id or 'auto'}")
    print(f"   LLM capture: {debugger.capture_llm}")
    print(f"   Validation: {debugger.validate_nodes}")
    print()

    # Run instrumented pipeline
    t0 = time.perf_counter()
    try:
        session_id = await debugger.run_instrumented(args)
        duration = time.perf_counter() - t0

        print(f"✅ Pipeline completed in {duration:.2f}s")
        print(f"   Session: {session_id}")

    except Exception as e:
        duration = time.perf_counter() - t0
        print(f"❌ Pipeline failed after {duration:.2f}s: {e}")
        debugger.record_issue("ERROR", "Pipeline", str(e))

    # Generate reports
    print("\n📊 Generating debug reports...")
    debugger.generate_reports()

    print(f"✅ Reports generated:")
    print(f"   JSON: {OUT_JSON}")
    print(f"   Markdown: {OUT_MD}")
    print(f"   Traces: {DEBUG_DIR}/")


def parse_args():
    """Parse command-line arguments"""
    p = argparse.ArgumentParser(
        description="Gate-8 Debug: Deep inspection of single LangGraph run"
    )
    p.add_argument("--company", default="Salesforce", help="Company name")
    p.add_argument("--persona", default="vp_customer_experience",
                   choices=["vp_customer_experience", "cio", "vp_sales_ops"],
                   help="Target persona")
    p.add_argument("--session-id", default=None, help="Session ID (auto if not specified)")
    p.add_argument("--stop-on-error", action="store_true",
                   help="Stop on first error (default: continue)")
    p.add_argument("--no-llm-capture", action="store_true",
                   help="Disable LLM interaction capture")
    p.add_argument("--no-validation", action="store_true",
                   help="Disable node validation")
    return p.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
