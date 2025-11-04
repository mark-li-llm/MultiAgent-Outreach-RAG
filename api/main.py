#!/usr/bin/env python3
"""FastAPI backend for RAG Email Generator.

This is a Phase 1 minimal backend that wraps the existing LangGraph workflow
with HTTP API endpoints. No authentication, database, or task queue.
"""

import sys
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory and scripts directory to path to import existing modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_dir = os.path.join(parent_dir, "scripts")
sys.path.insert(0, parent_dir)
sys.path.insert(0, scripts_dir)

from scripts.run_graph_langgraph import main_async


# Request/Response models
class GenerateEmailRequest(BaseModel):
    """Request model for email generation."""
    company: str = Field(..., description="Company name (e.g., 'Salesforce')")
    persona: str = Field(..., description="Persona type (e.g., 'vp_customer_experience')")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")


class GenerateEmailResponse(BaseModel):
    """Response model for email generation."""
    session_id: str = Field(..., description="Session ID for this generation")
    out_dir: str = Field(..., description="Output directory path")
    total_ms: float = Field(..., description="Total execution time in milliseconds")
    message: str = Field(..., description="Success message")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Email Generator API",
    description="Minimal backend for multi-agent RAG email generation system",
    version="1.0.0 (Phase 1)",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "RAG Email Generator",
        "version": "1.0.0 (Phase 1)",
        "endpoints": [
            {"path": "/", "method": "GET", "description": "Health check"},
            {"path": "/api/generate", "method": "POST", "description": "Generate email"},
            {"path": "/docs", "method": "GET", "description": "API documentation"},
        ]
    }


@app.post("/api/generate", response_model=GenerateEmailResponse)
async def generate_email(request: GenerateEmailRequest):
    """
    Generate a personalized email for a company and persona.

    This endpoint calls the existing LangGraph workflow to:
    1. Generate persona-specific search queries
    2. Retrieve relevant documents from vector indexes
    3. Synthesize insights
    4. Generate email draft
    5. Validate compliance
    6. Assemble final email with proof points

    Args:
        request: GenerateEmailRequest containing company, persona, and optional session_id

    Returns:
        GenerateEmailResponse with session_id, output directory, and execution time

    Raises:
        HTTPException: If workflow execution fails
    """
    try:
        # Create a simple args object to pass to main_async
        class Args:
            def __init__(self, company: str, persona: str, session_id: Optional[str]):
                self.company = company
                self.persona = persona
                self.session_id = session_id

        args = Args(
            company=request.company,
            persona=request.persona,
            session_id=request.session_id
        )

        # Call existing workflow function
        session_id = await main_async(args)

        # Read the timing.json to get total_ms
        import json
        out_dir = os.path.join("outputs", session_id)
        timing_path = os.path.join(out_dir, "timing.json")

        with open(timing_path, "r") as f:
            timing_data = json.load(f)

        total_ms = timing_data.get("total_runtime_ms", 0.0)

        return GenerateEmailResponse(
            session_id=session_id,
            out_dir=out_dir,
            total_ms=total_ms,
            message=f"Email generated successfully. Results available in {out_dir}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Email generation failed: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    # Check if required directories exist
    required_dirs = ["scripts", "configs", "data"]
    dirs_status = {d: os.path.isdir(d) for d in required_dirs}

    # Check if .env file exists
    env_exists = os.path.isfile(".env")

    return {
        "status": "healthy" if all(dirs_status.values()) else "degraded",
        "directories": dirs_status,
        "env_file": env_exists,
        "ready": all(dirs_status.values()) and env_exists
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start the FastAPI server.

    Args:
        host: Host to bind to (default: "0.0.0.0" for all interfaces)
        port: Port to bind to (default: 8000)
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FastAPI backend server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()

    print(f"Starting FastAPI server on {args.host}:{args.port}")
    print(f"API docs available at http://localhost:{args.port}/docs")
    start_server(host=args.host, port=args.port)
