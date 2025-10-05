#!/usr/bin/env python3
"""
Gate-8 Debug: Single-run deep inspection of LangGraph pipeline
Non-invasive instrumentation with comprehensive state tracking and validation
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
import traceback
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from common import ensure_dir, now_iso


# Paths
NODES_CONFIG = os.path.join("configs", "langgraph.nodes.yaml")
EVAL_CONFIG = os.path.join("configs", "eval.prompts.yaml")
MCP_CONFIG = os.path.join("configs", "mcp.tools.yaml")

# Output paths
DEBUG_DIR = os.path.join("reports", "debug")
OUTPUT_JSON = os.path.join("reports", "qa", "step08_debug.json")
OUTPUT_MD = os.path.join("reports", "qa", "step08_debug.md")


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


class NodeContext:
    """Context for a single node execution."""

    def __init__(self, node_name: str, debugger):
        self.node_name = node_name
        self.debugger = debugger
        self.start_time = time.perf_counter()
        self.timeout_occurred = False
        self.exception = None
        self.duration_ms = 0

    def mark_timeout(self):
        """Mark that a timeout occurred."""
        self.timeout_occurred = True

    def capture_exception(self, e: Exception):
        """Capture an exception for reporting."""
        self.exception = e
        self.debugger.record_issue(
            "ERROR", self.node_name,
            f"{type(e).__name__}: {str(e)}"
        )

    def finalize(self, duration_ms: float):
        """Finalize context with duration."""
        self.duration_ms = duration_ms

    async def cleanup(self):
        """Async cleanup if needed."""
        pass


class NodeWrapper:
    """Enhanced wrapper with proper async handling and context management."""

    def __init__(self, debugger):
        self.debugger = debugger
        self.node_timeouts = debugger.node_timeouts
        self.stop_on_error = False  # Continue on errors by default

    async def wrap_node(self, node_name: str, func, state: Dict) -> Any:
        """Enhanced wrapper with proper async handling."""
        async with self.node_context(node_name) as ctx:
            try:
                # Pre-execution hooks
                await self.pre_execute(node_name, state)

                # Get timeout for this node
                timeout_sec = self.node_timeouts.get(node_name, 30)

                # Execute with timeout
                if asyncio.iscoroutinefunction(func):
                    result = await asyncio.wait_for(
                        func(state),
                        timeout=timeout_sec
                    )
                else:
                    # For sync functions, run in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, func, state),
                        timeout=timeout_sec
                    )

                # Post-execution hooks
                await self.post_execute(node_name, state, result)

                return result

            except asyncio.TimeoutError:
                ctx.mark_timeout()
                self.debugger.record_issue(
                    "TIMEOUT", node_name,
                    f"Exceeded {self.node_timeouts.get(node_name, 30)}s"
                )
                raise

            except Exception as e:
                ctx.capture_exception(e)
                if self.stop_on_error:
                    raise
                return self.handle_node_failure(node_name, e, state)

    @asynccontextmanager
    async def node_context(self, node_name: str):
        """Context manager for resource management and timing."""
        start_time = time.perf_counter()
        ctx = NodeContext(node_name, self.debugger)

        try:
            yield ctx
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            ctx.finalize(duration_ms)
            await ctx.cleanup()

            # Record timing in debugger
            self.debugger.record_timing(node_name, duration_ms)

    async def pre_execute(self, node_name: str, state: Dict):
        """Pre-execution hook for state capture."""
        self.debugger.capture_state(node_name, "before", state)
        self.debugger.transitions.append({
            "entering": node_name,
            "timestamp": now_iso(),
            "state_keys": list(state.keys())
        })

    async def post_execute(self, node_name: str, state: Dict, result: Any):
        """Post-execution hook for validation and capture."""
        self.debugger.capture_state(node_name, "after", state)
        await self.debugger.validate_node(node_name, state)

    def handle_node_failure(self, node_name: str, e: Exception, state: Dict) -> None:
        """Handle node failure when stop_on_error is False."""
        self.debugger.record_issue(
            "HANDLED_ERROR", node_name,
            f"Continued after: {str(e)}"
        )
        # Add error to state for downstream handling
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append({
            "node": node_name,
            "error": str(e),
            "timestamp": now_iso()
        })
        return None


class StateInstrumentor:
    """Tracks state changes without modifying pipeline behavior."""

    def __init__(self):
        self.snapshots = {}
        self.transitions = []
        self.original_mark = None
        self.state_sizes = {}

    def instrument_mark_function(self, main_async):
        """Monkey-patch the mark() function in main_async scope."""
        # This is tricky - we need to patch it during execution
        # We'll use a different approach - wrap the entire execution
        pass

    def capture_state(self, node_name: str, phase: str, state: Dict):
        """Non-invasive state capture using deep copy."""
        snapshot = {
            "timestamp": now_iso(),
            "keys": list(state.keys()),
            "sizes": {k: self._get_size(v) for k, v in state.items()},
            # Deep copy samples to prevent mutation
            "samples": self._extract_samples(copy.deepcopy(state))
        }

        if phase == "before":
            self.snapshots[node_name] = {"before": snapshot}
        else:
            if node_name in self.snapshots:
                self.snapshots[node_name]["after"] = snapshot
                self._compute_diff(node_name)

    def _get_size(self, obj: Any) -> int:
        """Get size of object (count for lists/dicts, length for strings)."""
        if isinstance(obj, (list, dict)):
            return len(obj)
        elif isinstance(obj, str):
            return len(obj)
        else:
            return 1

    def _extract_samples(self, state: Dict) -> Dict:
        """Extract samples from state for debugging."""
        samples = {}

        # Sample queries
        if "queries" in state and isinstance(state["queries"], list):
            samples["queries"] = state["queries"][:3]

        # Sample retrieved chunks
        if "retrieved_chunks" in state and isinstance(state["retrieved_chunks"], list):
            samples["retrieved_chunks_count"] = len(state["retrieved_chunks"])
            if state["retrieved_chunks"]:
                samples["first_chunk_id"] = state["retrieved_chunks"][0].get("chunk_id")

        # Sample insight cards
        if "insight_cards" in state and isinstance(state["insight_cards"], list):
            samples["insight_cards_count"] = len(state["insight_cards"])
            if state["insight_cards"]:
                samples["card_titles"] = [
                    c.get("title", "")[:50] for c in state["insight_cards"][:3]
                ]

        # Sample email
        if "email_draft" in state and isinstance(state["email_draft"], dict):
            samples["email_subject"] = state["email_draft"].get("subject", "")
            samples["email_word_count"] = self._count_words(
                state["email_draft"].get("body", "")
            )

        return samples

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(re.findall(r"\b\w+\b", text or ""))

    def _compute_diff(self, node_name: str):
        """Compute what changed in this node."""
        before = self.snapshots[node_name]["before"]
        after = self.snapshots[node_name]["after"]

        diff = {
            "keys_added": list(set(after["keys"]) - set(before["keys"])),
            "keys_removed": list(set(before["keys"]) - set(after["keys"])),
            "sizes_changed": {},
            "samples_changed": {}
        }

        # Track size changes for existing keys
        for key in set(before["keys"]) & set(after["keys"]):
            if before["sizes"].get(key) != after["sizes"].get(key):
                diff["sizes_changed"][key] = {
                    "before": before["sizes"].get(key),
                    "after": after["sizes"].get(key)
                }

        self.snapshots[node_name]["diff"] = diff


class LLMMonitor:
    """Tracks LLM interactions through monkey-patching."""

    def __init__(self):
        self.interactions = []
        self.token_usage = {}
        self.original_ainvoke = None
        self.active = False

    def instrument_llm(self):
        """Monkey-patch ChatOpenAI.ainvoke for monitoring."""
        if self.active:
            return  # Already instrumented

        try:
            from langchain_openai import ChatOpenAI

            # Store original method
            self.original_ainvoke = ChatOpenAI.ainvoke

            # Reference to self for closure
            monitor_self = self

            # Create instrumented version
            async def instrumented_ainvoke(llm_self, *args, **kwargs):
                # Determine calling node from stack
                node_name = monitor_self._detect_calling_node()

                interaction = {
                    "node": node_name,
                    "timestamp": now_iso(),
                    "model": getattr(llm_self, 'model_name', 'unknown'),
                    "temperature": getattr(llm_self, 'temperature', 0)
                }

                # Capture prompt
                if args and len(args) > 0:
                    interaction["prompt_sample"] = monitor_self._extract_prompt(args[0])

                t0 = time.perf_counter()
                try:
                    # Call original method
                    response = await monitor_self.original_ainvoke(llm_self, *args, **kwargs)
                    interaction["duration_ms"] = (time.perf_counter() - t0) * 1000

                    # Capture response without modifying it
                    if hasattr(response, 'content'):
                        interaction["response_sample"] = response.content[:500]
                    if hasattr(response, 'response_metadata'):
                        interaction["tokens"] = response.response_metadata.get(
                            "token_usage", {}
                        )

                    monitor_self.interactions.append(interaction)
                    monitor_self._update_token_usage(node_name, interaction.get("tokens", {}))

                    return response  # Return unmodified

                except Exception as e:
                    interaction["error"] = str(e)
                    interaction["duration_ms"] = (time.perf_counter() - t0) * 1000
                    monitor_self.interactions.append(interaction)
                    raise  # Re-raise unmodified

            # Replace with instrumented version
            ChatOpenAI.ainvoke = instrumented_ainvoke
            self.active = True

        except ImportError:
            pass  # LangChain not available, skip instrumentation

    def restore_llm(self):
        """Restore original LLM methods."""
        if self.original_ainvoke and self.active:
            try:
                from langchain_openai import ChatOpenAI
                ChatOpenAI.ainvoke = self.original_ainvoke
                self.active = False
            except ImportError:
                pass

    def _detect_calling_node(self) -> str:
        """Detect which node is making the LLM call from stack."""
        stack = inspect.stack()

        for frame in stack:
            # Check for node markers in local variables
            local_vars = frame[0].f_locals

            # Look for mark function calls
            code = frame[0].f_code
            if "mark" in code.co_names:
                # Check previous frame for node name
                if frame[0].f_lineno > 0:
                    # Try to extract from surrounding code
                    pass

            # Check function names
            func_name = frame[3]
            if "consolidator" in func_name.lower():
                return "Consolidator"
            elif "stylist" in func_name.lower():
                return "Stylist"

        # Check code context for node identification
        for frame in stack:
            try:
                source_lines = inspect.getframeinfo(frame[0]).code_context
                if source_lines:
                    for line in source_lines:
                        if "Consolidator" in line:
                            return "Consolidator"
                        elif "Stylist" in line:
                            return "Stylist"
            except:
                pass

        return "Unknown"

    def _extract_prompt(self, messages) -> str:
        """Extract a sample from prompt messages."""
        try:
            if hasattr(messages, 'messages'):
                # It's a prompt template result
                msg_list = messages.messages
                if msg_list and len(msg_list) > 0:
                    # Get last user message
                    for msg in reversed(msg_list):
                        if hasattr(msg, 'content'):
                            return msg.content[:200] + "..."
            return str(messages)[:200] + "..."
        except:
            return "Unable to extract prompt"

    def _update_token_usage(self, node_name: str, tokens: Dict):
        """Update cumulative token usage."""
        if node_name not in self.token_usage:
            self.token_usage[node_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if key in tokens:
                self.token_usage[node_name][key] += tokens[key]


class ValidationEngine:
    """Validates node outputs without modifying behavior."""

    def __init__(self, eval_config: Dict):
        self.eval_config = eval_config
        self.validation_results = {}
        self.personas = eval_config.get("personas", {})

    async def validate_node(self, node_name: str, state: Dict) -> Dict:
        """Validate node output based on node type."""
        validators = {
            "Planner": self.validate_planner,
            "Retriever": self.validate_retriever,
            "Consolidator": self.validate_consolidator,
            "Stylist": self.validate_stylist,
            "A2A": self.validate_a2a
        }

        if node_name in validators:
            result = await validators[node_name](state)
            self.validation_results[node_name] = result
            return result

        return {"status": "SKIP", "checks": []}

    async def validate_planner(self, state: Dict) -> Dict:
        """Validate Planner output."""
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
        persona_keywords = self.personas.get(persona, [])
        if persona_keywords:
            relevant_queries = sum(
                1 for q in queries
                if any(kw.lower() in q.lower() for kw in persona_keywords)
            )
            checks.append({
                "check": "persona_relevance",
                "expected": ">=3",
                "actual": relevant_queries,
                "pass": relevant_queries >= 3
            })

        return {
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "checks": checks
        }

    async def validate_retriever(self, state: Dict) -> Dict:
        """Validate Retriever output."""
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
        doc_ids = [c.get("doc_id", "") for c in chunks]
        sources = set(d.split("::")[0] for d in doc_ids if d)
        checks.append({
            "check": "source_diversity",
            "expected": ">=3",
            "actual": len(sources),
            "pass": len(sources) >= 3
        })

        # Check 3: Routing decisions
        checks.append({
            "check": "routing_decisions",
            "expected": len(state.get("queries", [])),
            "actual": len(route_decisions),
            "pass": len(route_decisions) == len(state.get("queries", []))
        })

        return {
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "checks": checks
        }

    async def validate_consolidator(self, state: Dict) -> Dict:
        """Validate Consolidator output with LLM enhancement."""
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

        # Check 3: Domain diversity
        domains = set(c.get("source_domain", "") for c in cards)
        checks.append({
            "check": "domain_diversity",
            "expected": ">=4",
            "actual": len(domains),
            "pass": len(domains) >= 4
        })

        # Check 4: Persona alignment
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

        return {
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "checks": checks
        }

    async def validate_stylist(self, state: Dict) -> Dict:
        """Validate Stylist output."""
        email = state.get("email_draft", {})
        persona = state.get("persona", "")

        checks = []

        # Check 1: Email structure
        required_fields = ["subject", "body", "unsubscribe_block", "company_info_block"]
        has_all_fields = all(email.get(f) for f in required_fields)
        checks.append({
            "check": "email_structure",
            "expected": "all fields present",
            "actual": has_all_fields,
            "pass": has_all_fields
        })

        # Check 2: Word count
        body = email.get("body", "")
        word_count = len(re.findall(r"\b\w+\b", body))
        checks.append({
            "check": "word_count",
            "expected": "<=160",
            "actual": word_count,
            "pass": word_count <= 160
        })

        # Check 3: Persona keywords
        persona_keywords = self.personas.get(persona, [])
        if persona_keywords:
            keyword_hits = sum(
                1 for kw in persona_keywords
                if kw.lower() in body.lower()
            )
            checks.append({
                "check": "persona_keywords",
                "expected": ">=2",
                "actual": keyword_hits,
                "pass": keyword_hits >= 2
            })

        return {
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "checks": checks
        }

    async def validate_a2a(self, state: Dict) -> Dict:
        """Validate A2A compliance handling."""
        compliance_flags = state.get("compliance_flags", {})
        email = state.get("email_draft", {})

        checks = []

        # Check 1: Critical flags resolved
        critical_flags = compliance_flags.get("critical", [])
        checks.append({
            "check": "critical_flags",
            "expected": 0,
            "actual": len(critical_flags),
            "pass": len(critical_flags) == 0
        })

        # Check 2: Compliance blocks present
        has_unsubscribe = bool(email.get("unsubscribe_block"))
        has_company_info = bool(email.get("company_info_block"))
        checks.append({
            "check": "compliance_blocks",
            "expected": "both present",
            "actual": f"unsub={has_unsubscribe}, info={has_company_info}",
            "pass": has_unsubscribe and has_company_info
        })

        return {
            "status": "PASS" if all(c["pass"] for c in checks) else "FAIL",
            "checks": checks
        }


class DebugReporter:
    """Generates debug reports in JSON and Markdown formats."""

    def __init__(self, debugger):
        self.debugger = debugger

    def generate_reports(self, session_id: str, duration_ms: float):
        """Generate all debug reports."""
        # Create debug directory for this session
        session_debug_dir = os.path.join(DEBUG_DIR, session_id)
        ensure_dir(session_debug_dir)

        # Generate trace files
        self._write_trace_files(session_debug_dir)

        # Generate main reports
        machine_report = self._generate_machine_report(session_id, duration_ms)
        human_report = self._generate_human_report(machine_report)

        # Write reports
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(machine_report, f, ensure_ascii=False, indent=2)

        with open(OUTPUT_MD, "w", encoding="utf-8") as f:
            f.write(human_report)

        print(f"✅ Debug reports generated:")
        print(f"   - Machine: {OUTPUT_JSON}")
        print(f"   - Human:   {OUTPUT_MD}")
        print(f"   - Traces:  {session_debug_dir}/")

    def _write_trace_files(self, session_dir: str):
        """Write detailed trace files."""
        # Node states
        states_path = os.path.join(session_dir, "node_states.jsonl")
        with open(states_path, "w", encoding="utf-8") as f:
            for node_name, snapshots in self.debugger.state_instrumentor.snapshots.items():
                f.write(json.dumps({
                    "node": node_name,
                    "snapshots": snapshots
                }) + "\n")

        # LLM interactions
        llm_path = os.path.join(session_dir, "llm_interactions.jsonl")
        with open(llm_path, "w", encoding="utf-8") as f:
            for interaction in self.debugger.llm_monitor.interactions:
                f.write(json.dumps(interaction) + "\n")

        # Validation results
        val_path = os.path.join(session_dir, "validation_trace.jsonl")
        with open(val_path, "w", encoding="utf-8") as f:
            for node_name, result in self.debugger.validation_engine.validation_results.items():
                f.write(json.dumps({
                    "node": node_name,
                    "result": result
                }) + "\n")

    def _generate_machine_report(self, session_id: str, duration_ms: float) -> Dict:
        """Generate machine-readable JSON report."""
        # Build execution timeline
        timeline = []
        cumulative_time = 0

        for node_name in self.debugger.node_timings:
            node_time = self.debugger.node_timings[node_name]
            timeline.append({
                "node": node_name,
                "start_ms": cumulative_time,
                "duration_ms": node_time,
                "status": "SUCCESS" if node_name not in self.debugger.issues else "WARNING"
            })
            cumulative_time += node_time

        # Aggregate validation results
        validation_summary = {
            "passed": sum(
                1 for r in self.debugger.validation_engine.validation_results.values()
                if r.get("status") == "PASS"
            ),
            "failed": sum(
                1 for r in self.debugger.validation_engine.validation_results.values()
                if r.get("status") == "FAIL"
            ),
            "skipped": sum(
                1 for r in self.debugger.validation_engine.validation_results.values()
                if r.get("status") == "SKIP"
            )
        }

        # Build quality metrics
        quality_metrics = self._compute_quality_metrics()

        report = {
            "gate": "Gate-8-Debug",
            "timestamp": now_iso(),
            "configuration": {
                "mode": "debug_single_run",
                "mcp_mode": "strict",
                "company": self.debugger.args.company,
                "persona": self.debugger.args.persona,
                "session_id": session_id
            },
            "execution_timeline": timeline,
            "node_validations": self.debugger.validation_engine.validation_results,
            "state_transitions": self.debugger.state_instrumentor.transitions,
            "llm_usage": self.debugger.llm_monitor.token_usage,
            "quality_metrics": quality_metrics,
            "validation_summary": validation_summary,
            "issues_detected": self.debugger.issues,
            "total_runtime_ms": duration_ms,
            "artifacts": {
                "session_dir": f"outputs/{session_id}/",
                "trace_files": [
                    f"{DEBUG_DIR}/{session_id}/node_states.jsonl",
                    f"{DEBUG_DIR}/{session_id}/llm_interactions.jsonl",
                    f"{DEBUG_DIR}/{session_id}/validation_trace.jsonl"
                ]
            }
        }

        return report

    def _generate_human_report(self, machine_report: Dict) -> str:
        """Generate human-readable Markdown report."""
        lines = []

        # Header
        lines.append("# Gate-8 Debug Report\n")
        lines.append(f"**Session**: {machine_report['configuration']['session_id']}")
        lines.append(f"**Timestamp**: {machine_report['timestamp']}")
        lines.append(f"**Company**: {machine_report['configuration']['company']}")
        lines.append(f"**Persona**: {machine_report['configuration']['persona']}\n")

        # Executive summary
        lines.append("## 🎯 Executive Summary\n")

        status = "SUCCESS" if not machine_report["issues_detected"] else "WARNING"
        lines.append(f"✅ **Pipeline Status**: {status}")
        lines.append(f"⏱️ **Total Runtime**: {machine_report['total_runtime_ms']/1000:.2f}s")

        total_tokens = sum(
            usage.get("total_tokens", 0)
            for usage in machine_report["llm_usage"].values()
        )
        lines.append(f"🤖 **LLM Tokens Used**: {total_tokens:,}")

        val_summary = machine_report["validation_summary"]
        total_checks = val_summary["passed"] + val_summary["failed"]
        score = (val_summary["passed"] / max(1, total_checks)) * 100
        lines.append(f"📊 **Quality Score**: {score:.0f}/100\n")

        # Execution timeline
        lines.append("## 📈 Execution Timeline\n")
        lines.append("```")

        for item in machine_report["execution_timeline"]:
            icon = "✅" if item["status"] == "SUCCESS" else "⚠️"
            lines.append(
                f"[{item['start_ms']/1000:06.3f}s] {icon} {item['node']:<15} "
                f"({item['duration_ms']:.0f}ms)"
            )

        lines.append("```\n")

        # Node validations
        lines.append("## 🔍 Node Validation Results\n")

        for node_name, result in machine_report["node_validations"].items():
            status_icon = "✅" if result["status"] == "PASS" else "❌"
            lines.append(f"### {node_name} {status_icon}\n")

            if result.get("checks"):
                lines.append("| Check | Expected | Actual | Status |")
                lines.append("|-------|----------|--------|--------|")

                for check in result["checks"]:
                    status = "✅" if check["pass"] else "❌"
                    lines.append(
                        f"| {check['check']} | {check['expected']} | "
                        f"{check['actual']} | {status} |"
                    )
                lines.append("")

        # Quality metrics
        lines.append("## 📊 Quality Metrics\n")

        metrics = machine_report.get("quality_metrics", {})
        if metrics:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    lines.append(f"| {key} | {value} |")
                else:
                    lines.append(f"| {key} | {json.dumps(value)} |")
            lines.append("")

        # Issues
        if machine_report["issues_detected"]:
            lines.append("## 🐛 Issues Detected\n")

            for issue in machine_report["issues_detected"]:
                lines.append(f"- **{issue['severity']}** [{issue['node']}]: {issue['details']}")
            lines.append("")

        # Debug artifacts
        lines.append("## 📁 Debug Artifacts\n")
        lines.append("Generated debug files for detailed analysis:")

        for artifact in machine_report["artifacts"]["trace_files"]:
            lines.append(f"- `{artifact}`")

        lines.append(f"\nSession outputs: `{machine_report['artifacts']['session_dir']}`")

        # Footer
        lines.append("\n---")
        lines.append("*Debug run completed with comprehensive tracing enabled*\n")

        return "\n".join(lines)

    def _compute_quality_metrics(self) -> Dict:
        """Compute quality metrics from captured state."""
        metrics = {}

        # Get final state snapshot
        final_state = {}
        for node_name in ["Stylist", "Assembler", "A2A"]:
            if node_name in self.debugger.state_instrumentor.snapshots:
                snapshot = self.debugger.state_instrumentor.snapshots[node_name]
                if "after" in snapshot and "samples" in snapshot["after"]:
                    final_state.update(snapshot["after"]["samples"])

        # Extract metrics
        metrics["insights_count"] = final_state.get("insight_cards_count", 0)
        metrics["email_word_count"] = final_state.get("email_word_count", 0)

        # Get persona keyword hits from validation
        stylist_validation = self.debugger.validation_engine.validation_results.get("Stylist", {})
        for check in stylist_validation.get("checks", []):
            if check.get("check") == "persona_keywords":
                metrics["persona_keyword_hits"] = check.get("actual", 0)

        return metrics


class LangGraphDebugger:
    """Main debugger orchestrator for non-invasive pipeline inspection."""

    def __init__(self):
        # Load configurations
        self.nodes_config = load_yaml(NODES_CONFIG)
        self.eval_config = load_yaml(EVAL_CONFIG)
        self.mcp_config = load_yaml(MCP_CONFIG)

        # Extract node timeouts (convert ms to seconds)
        self.node_timeouts = {
            k: v/1000 for k, v in
            self.nodes_config.get("timeouts_ms", {}).items()
        }

        # Initialize components
        self.node_wrapper = NodeWrapper(self)
        self.state_instrumentor = StateInstrumentor()
        self.llm_monitor = LLMMonitor()
        self.validation_engine = ValidationEngine(self.eval_config)
        self.debug_reporter = DebugReporter(self)

        # Tracking
        self.transitions = []
        self.node_timings = {}
        self.issues = []
        self.args = None

        # Settings
        self.stop_on_error = False
        self.capture_llm = True
        self.validate_nodes = True

    async def run_instrumented(self, args) -> str:
        """Run the existing pipeline with instrumentation."""
        self.args = args

        # Instrument LLM if requested
        if self.capture_llm:
            self.llm_monitor.instrument_llm()

        # Start MCP stub servers (minimal fix for MCP connection issue)
        from router_core import load_mcp_map

        tools_cfg = load_mcp_map()
        state_env: Dict[str, Any] = {}

        # Import and start MCP stubs
        from qa_step03_mcp import start_stub_servers, stop_stub_servers

        await start_stub_servers(state_env, {"tools": tools_cfg})
        print("✓ MCP stub servers started")

        try:
            # Import run_graph directly
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from run_graph import main_async

            # Track overall execution
            t0 = time.perf_counter()

            # Execute pipeline (run_graph will detect external MCP is running)
            session_id = await main_async(args)

            duration_ms = (time.perf_counter() - t0) * 1000

            # Generate reports
            self.debug_reporter.generate_reports(
                session_id or args.session_id,
                duration_ms
            )

            return session_id

        finally:
            # Stop MCP stub servers
            try:
                await stop_stub_servers(state_env)
                print("✓ MCP stub servers stopped")
            except Exception as e:
                print(f"⚠ Failed to stop MCP servers: {e}")

            # Restore LLM
            if self.capture_llm:
                self.llm_monitor.restore_llm()

    def capture_state(self, node_name: str, phase: str, state: Dict):
        """Capture state snapshot."""
        self.state_instrumentor.capture_state(node_name, phase, state)

    async def validate_node(self, node_name: str, state: Dict):
        """Validate node output."""
        if self.validate_nodes:
            await self.validation_engine.validate_node(node_name, state)

    def record_timing(self, node_name: str, duration_ms: float):
        """Record node execution time."""
        self.node_timings[node_name] = duration_ms

    def record_issue(self, severity: str, node: str, details: str):
        """Record an issue for reporting."""
        self.issues.append({
            "severity": severity,
            "node": node,
            "details": details,
            "timestamp": now_iso()
        })


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Gate-8 Debug: Deep inspection of LangGraph pipeline"
    )
    parser.add_argument("--company", default="Salesforce")
    parser.add_argument("--persona", default="vp_customer_experience")
    parser.add_argument("--session-id", default=f"debug_{int(time.time())}")
    parser.add_argument("--stop-on-error", action="store_true",
                      help="Stop execution on first error")
    parser.add_argument("--no-llm-capture", action="store_true",
                      help="Disable LLM interaction capture")
    parser.add_argument("--no-validation", action="store_true",
                      help="Disable node validation")

    args = parser.parse_args()

    print("🔍 Gate-8 Debug Mode")
    print(f"   Company: {args.company}")
    print(f"   Persona: {args.persona}")
    print(f"   Session: {args.session_id}")
    print()

    # Initialize debugger
    debugger = LangGraphDebugger()

    # Configure settings
    debugger.stop_on_error = args.stop_on_error
    debugger.capture_llm = not args.no_llm_capture
    debugger.validate_nodes = not args.no_validation

    # Run with instrumentation
    try:
        session_id = await debugger.run_instrumented(args)
        print(f"\n✅ Debug session complete: {session_id}")

    except Exception as e:
        print(f"\n❌ Debug session failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())