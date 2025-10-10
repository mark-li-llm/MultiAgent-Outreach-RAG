#!/usr/bin/env python3
"""Generate LangGraph visualization."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_graph_langgraph import build_graph
from common import ensure_dir


def main():
    workflow = build_graph()
    app = workflow.compile()

    # Generate Mermaid diagram
    mermaid = app.get_graph().draw_mermaid()

    ensure_dir("reports/graphs")
    with open("reports/graphs/agent_workflow.mmd", "w") as f:
        f.write(mermaid)

    print("✓ Graph visualization saved to reports/graphs/agent_workflow.mmd")

    # Try to generate PNG (requires graphviz)
    try:
        png = app.get_graph().draw_mermaid_png()
        with open("reports/graphs/agent_workflow.png", "wb") as f:
            f.write(png)
        print("✓ PNG visualization saved to reports/graphs/agent_workflow.png")
    except Exception as e:
        print(f"⚠ PNG generation skipped (graphviz not installed): {e}")


if __name__ == "__main__":
    main()
