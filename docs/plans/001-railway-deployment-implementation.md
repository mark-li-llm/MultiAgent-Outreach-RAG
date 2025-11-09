# Railway Deployment Implementation Plan

**Date**: 2025-11-04
**Status**: Planning
**Implements**: ADR-001 Railway Deployment Decision
**Target**: Production deployment of FastAPI backend on Railway.app

---

## Overview

This plan implements the Railway.app deployment strategy documented in ADR-001. The goal is to containerize the Phase 1 FastAPI backend (`api/main.py`) and deploy it to Railway with zero-cost operation using CPU-time billing optimization for our IO-bound workload.

## Current State Analysis

### ✅ What Exists

**Phase 1 FastAPI Backend** (`api/main.py` - 165 lines):
- HTTP endpoints: `GET /`, `GET /health`, `POST /api/generate`
- Successfully tested end-to-end (test-final-20251104)
- Execution time: 101.5s (target: optimize to 15-30s in future phases)
- Quality metrics: All passed (0 critical compliance flags, 85%+ recall)

**Two-Environment Architecture**:
- `age` (Python 3.13) - Runtime environment with LangGraph, OpenAI, FastAPI
- `ageFaiss` (Python 3.12) - FAISS index building only (offline use)

**Pre-Built Data Artifacts**:
- FAISS indexes in `data/vector/faiss/` (built with ageFaiss)
- Embeddings parquet in `data/vector/embeddings/`
- Chunked documents in `data/interim/chunks/`
- Document metadata in `data/interim/normalized/`
- Configuration files in `configs/`

**Deployment Analysis**:
- Detailed Railway cost analysis (ADR-001)
- Backend architecture design (docs/backend-architecture.md)
- Deployment strategy (docs/001-railway-deployment-analysis.md)

### ❌ What's Missing

**Docker Configuration**:
- [ ] Dockerfile for containerization
- [ ] .dockerignore for optimized builds

**Environment Configuration**:
- [ ] FastAPI dependencies not in `envs/age.yaml` (fastapi, uvicorn, starlette)
- [ ] Railway environment variable setup (OPENAI_API_KEY, PORT)

**Pre-Deployment Validation**:
- [ ] Verification script to ensure all required data files exist
- [ ] Local Docker build and test
- [ ] GitHub repository setup for Railway deployment

## Desired End State

After completing this implementation:

✅ **Dockerized Application**:
- Multi-stage Docker build with miniconda3 base
- Only `age` environment in runtime image
- Pre-built FAISS indexes included in image
- Image size: 1.5-2GB (acceptable for Railway)

✅ **Railway Deployment**:
- Automatic deployment on git push to main
- Environment variables configured in Railway dashboard
- Public HTTPS URL provided by Railway
- Zero cost operation within free tier ($5/month = 500 CPU-hours)

✅ **Validation**:
- Local Docker build succeeds
- Local Docker run passes health checks
- Railway deployment completes successfully
- API endpoint responds to test requests
- Full email generation workflow executes

### Success Criteria

#### Automated Verification:
- [ ] Docker build completes without errors: `docker build -t rag-api .`
- [ ] Health check passes: `curl http://localhost:8000/health` returns `{"status": "healthy"}`
- [ ] Local API test succeeds: `docker run -p 8000:8000 rag-api` and test generation endpoint
- [ ] Railway build completes: Check Railway dashboard build logs
- [ ] Railway health check passes: Railway marks service as "Deployed" (green)

#### Manual Verification:
- [ ] Railway public URL accessible in browser
- [ ] API documentation loads at `https://<railway-url>/docs`
- [ ] End-to-end email generation test via Railway URL completes successfully
- [ ] Generated email meets quality standards (≤160 words, 0 critical flags)
- [ ] Performance acceptable (≤120s for first request, ≤90s for subsequent)

## What We're NOT Doing

Explicitly out of scope for this deployment:

❌ **Phase 2 Features**:
- User authentication/authorization
- Database persistence (PostgreSQL/Redis)
- Background task queue (Celery)
- Rate limiting
- Advanced monitoring/alerting
- CORS configuration (frontend not yet implemented)

❌ **Performance Optimization**:
- Async task queue for long-running requests
- WebSocket for real-time updates
- Response caching
- Database connection pooling

❌ **FAISS Runtime Building**:
- Rebuilding FAISS indexes in Docker container
- Including `ageFaiss` environment in Docker image
- Dynamic index updates at runtime

❌ **Multi-Region Deployment**:
- Load balancing across regions
- CDN configuration
- Geographic routing

## Implementation Approach

### Strategy: Pre-Build + Single Runtime Environment

**Key Decision**: Use a **single-environment Docker image** with only `age` (Python 3.13) runtime.

**Rationale**:
1. FAISS index building happens offline (locally with `ageFaiss` before Docker build)
2. Runtime only needs to read pre-built FAISS indexes (no FAISS import needed)
3. Avoids dual-environment complexity in Docker (simpler, smaller, faster)
4. Prevents OpenMP conflicts entirely

**Docker Build Flow**:
```
┌─────────────────────────────────────────────────────┐
│ Local Pre-Build (Before Docker)                    │
├─────────────────────────────────────────────────────┤
│ conda run -n ageFaiss python scripts/qa_step02...  │
│ → Outputs: data/vector/faiss/index.faiss           │
│           data/vector/faiss/idmap.parquet           │
│           data/vector/faiss/manifest.json           │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Docker Build (Single-Stage)                        │
├─────────────────────────────────────────────────────┤
│ FROM continuumio/miniconda3:latest                 │
│ COPY envs/age.yaml .                               │
│ RUN conda env create -f age.yaml                   │
│ COPY data/ data/                                    │
│ COPY configs/ configs/                              │
│ COPY scripts/ scripts/                              │
│ COPY api/ api/                                      │
│ CMD conda run -n age python -m uvicorn ...         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ Railway Deployment                                  │
├─────────────────────────────────────────────────────┤
│ GitHub push → Railway auto-build → Deploy          │
│ Environment variables from Railway dashboard        │
│ Public URL: https://<project>.railway.app          │
└─────────────────────────────────────────────────────┘
```

## Phase 1: Update Environment Configuration

### Overview
Update `envs/age.yaml` to include FastAPI dependencies that are currently missing but needed for runtime.

### Changes Required

#### 1. Update `envs/age.yaml`
**File**: `envs/age.yaml`
**Changes**: Add FastAPI dependencies to pip section

Current pip dependencies (lines 14-24):
```yaml
  - pip:
    - openai>=1.0.0
    - python-dotenv>=1.0.0
    - tenacity>=8.2.0
    - langgraph>=0.2.20
    - langgraph-checkpoint-sqlite>=1.0.0
    - langchain-core>=0.3.0
    - langchain-openai>=0.2.0
    - langsmith>=0.1.0
    - aiosqlite>=0.19.0
```

**Add after `aiosqlite`**:
```yaml
    - fastapi>=0.121.0
    - uvicorn[standard]>=0.38.0
    - pydantic>=2.0.0
    - pydantic-settings>=2.0.0
```

**Reasoning**:
- `fastapi>=0.121.0` - HTTP framework (tested in Phase 1)
- `uvicorn[standard]>=0.38.0` - ASGI server with performance extras (tested in Phase 1)
- `pydantic>=2.0.0` - Request/response validation (FastAPI dependency)
- `pydantic-settings>=2.0.0` - Environment variable management (used in `api/main.py:14`)

### Success Criteria

#### Automated Verification:
- [ ] Conda environment recreates successfully: `conda env remove -n age && conda env create -f envs/age.yaml`
- [ ] FastAPI imports work: `conda run -n age python -c "import fastapi; import uvicorn; print('OK')"`
- [ ] Existing workflow still runs: `conda run -n age python scripts/run_graph_langgraph.py --help`

#### Manual Verification:
- [ ] FastAPI server starts: `conda run -n age python api/main.py --port 8001`
- [ ] Health endpoint responds: `curl http://localhost:8001/health`
- [ ] No import errors in startup logs

---

## Phase 2: Create Pre-Deployment Validation Script

### Overview
Create a verification script that checks all required files and directories exist before Docker build. This prevents Docker build failures due to missing data.

### Changes Required

#### 1. Create Validation Script
**File**: `scripts/pre_deploy_check.py`

```python
#!/usr/bin/env python3
"""Pre-deployment validation script.

Verifies all required files and directories exist before Docker build.
Run this locally before pushing to Railway.
"""

import os
import sys
from pathlib import Path

def check_exists(path: str, description: str) -> bool:
    """Check if a path exists."""
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"{status} {description}: {path}")
    return exists

def main():
    """Run all validation checks."""
    print("=== Pre-Deployment Validation ===\n")

    all_checks = []

    # Critical directories
    print("## Required Directories")
    all_checks.append(check_exists("api", "API directory"))
    all_checks.append(check_exists("scripts", "Scripts directory"))
    all_checks.append(check_exists("configs", "Configs directory"))
    all_checks.append(check_exists("data/vector/embeddings", "Embeddings directory"))
    all_checks.append(check_exists("data/interim/chunks", "Chunks directory"))
    all_checks.append(check_exists("data/interim/normalized", "Normalized directory"))

    # Critical files
    print("\n## Critical Data Files")
    all_checks.append(check_exists("data/vector/embeddings/embeddings.parquet", "Embeddings parquet"))
    all_checks.append(check_exists("data/interim/eval/salesforce_eval_seed.jsonl", "Eval seed"))

    # Check for at least one chunk file
    chunk_files = list(Path("data/interim/chunks").glob("*.chunks.jsonl"))
    has_chunks = len(chunk_files) > 0
    status = "✓" if has_chunks else "✗"
    print(f"{status} Chunk files: {len(chunk_files)} found")
    all_checks.append(has_chunks)

    # Check for at least one normalized file
    norm_files = list(Path("data/interim/normalized").glob("*.json"))
    has_norm = len(norm_files) > 0
    status = "✓" if has_norm else "✗"
    print(f"{status} Normalized files: {len(norm_files)} found")
    all_checks.append(has_norm)

    # Configuration files
    print("\n## Configuration Files")
    config_files = [
        "configs/router.heuristics.yaml",
        "configs/mcp.tools.yaml",
        "configs/eval.prompts.yaml",
        "configs/langgraph.nodes.yaml",
        "configs/vector.indexing.yaml",
        "configs/chunking.config.json",
    ]
    for cfg in config_files:
        all_checks.append(check_exists(cfg, Path(cfg).name))

    # Environment file (warning only)
    print("\n## Environment Configuration")
    has_env = check_exists(".env", "Environment file (.env)")
    if has_env:
        with open(".env") as f:
            content = f.read()
            has_key = "OPENAI_API_KEY" in content
            status = "✓" if has_key else "✗"
            print(f"{status} OPENAI_API_KEY present in .env")
            all_checks.append(has_key)
    else:
        print("⚠️  .env file missing (add OPENAI_API_KEY to Railway dashboard)")

    # Docker files
    print("\n## Docker Configuration")
    check_exists("Dockerfile", "Dockerfile")
    check_exists(".dockerignore", ".dockerignore")

    # Summary
    print("\n=== Summary ===")
    passed = sum(all_checks)
    total = len(all_checks)
    print(f"Passed: {passed}/{total} checks")

    if passed == total:
        print("\n✅ All checks passed! Ready for Docker build.")
        return 0
    else:
        print(f"\n❌ {total - passed} checks failed. Fix before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

#### 2. Make Script Executable
```bash
chmod +x scripts/pre_deploy_check.py
```

### Success Criteria

#### Automated Verification:
- [ ] Script runs without errors: `python scripts/pre_deploy_check.py`
- [ ] All checks pass: Script exits with code 0
- [ ] Missing files detected: Remove a test file, verify script reports failure

#### Manual Verification:
- [ ] Output is clear and readable
- [ ] Failed checks are easy to identify
- [ ] Script completes in <5 seconds

---

## Phase 3: Create Dockerfile

### Overview
Create a simple single-stage Dockerfile using continuumio/miniconda3 as the base image. This approach is more reliable and maintainable than multi-stage builds with conda-pack.

### Changes Required

#### 1. Create Dockerfile
**File**: `Dockerfile`

```dockerfile
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy conda environment file
COPY envs/age.yaml .

# Create conda environment and clean up in one layer
RUN conda env create -f age.yaml && \
    conda clean -afy

# Copy application code
COPY api/ api/
COPY scripts/ scripts/
COPY configs/ configs/

# Copy pre-built data files
COPY data/vector/ data/vector/
COPY data/interim/ data/interim/

# Create runtime directories
RUN mkdir -p outputs state logs data/cache/embeddings

# Expose port (Railway sets $PORT dynamically)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD conda run -n age python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with conda run (most reliable activation method)
# Use shell form to support environment variable substitution
CMD conda run --no-capture-output -n age python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

**Key Design Decisions**:
1. **Single-stage build**: Simple, reliable, standard conda+Docker approach
2. **continuumio/miniconda3 base**: Consistent conda environment throughout (~150MB base)
3. **conda run activation**: Most reliable method for activating conda environment in Docker
4. **Shell-form CMD**: Required for `${PORT:-8000}` environment variable substitution
5. **Selective COPY**: Only includes necessary data directories (vector/, interim/)
6. **Runtime directories**: Creates outputs/, state/, logs/ for generated files
7. **Health check**: Uses conda run to ensure environment is active

**Why Single-Stage (Not Multi-Stage)?**
- Both continuumio/miniconda3 and python:3.13-slim are ~130-150MB
- conda-pack adds complexity and potential binary compatibility issues
- Single-stage is simpler, more maintainable, and equally performant
- Standard approach used by most conda+Docker deployments

#### 2. Create .dockerignore
**File**: `.dockerignore`

```dockerfile
# Git
.git
.gitignore
.gitattributes

# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual environments
venv/
ENV/
env/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
*.cover
.hypothesis/

# Documentation (not needed in runtime)
docs/
docs.archive/
roadmap/
*.md
!CLAUDE.md

# Development outputs (will be regenerated)
outputs/
state/
logs/

# Large data files that should be generated, not copied
data/cache/

# Conda environments (built in Docker, not needed in image)
envs/

# Build artifacts
*.egg-info/
dist/
build/

# Railway-specific
.railway/
railway.json

# Environment files (secrets via Railway dashboard)
.env
.env.*

# Hack/test directories
hack/
icl/
thoughts/

# Temporary files
*.tmp
*.bak
*.log
```

### Success Criteria

#### Automated Verification:
- [ ] Docker build completes: `docker build -t rag-api .`
- [ ] Image size reasonable: `docker images rag-api` shows <2.5GB
- [ ] Container starts: `docker run -d -p 8000:8000 --env OPENAI_API_KEY=$OPENAI_API_KEY rag-api`
- [ ] Health check passes: `docker ps` shows "healthy" status after 40 seconds
- [ ] API responds: `curl http://localhost:8000/health` returns 200

#### Manual Verification:
- [ ] Build completes in <5 minutes
- [ ] No conda warnings about missing channels
- [ ] Container logs show uvicorn starting successfully
- [ ] API docs load: `http://localhost:8000/docs`
- [ ] Test generation endpoint with minimal request

---

## Phase 4: Test Local Docker Deployment

### Overview
Validate the Docker image locally before deploying to Railway. This catches issues early in the development cycle.

### Changes Required

#### 1. Create Local Test Script
**File**: `scripts/test_docker_local.sh`

```bash
#!/bin/bash
set -e

echo "=== Local Docker Deployment Test ==="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="rag-api"
CONTAINER_NAME="rag-api-test"
TEST_PORT=8001

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}✗ .env file not found. Create it with OPENAI_API_KEY.${NC}"
    exit 1
fi

# Load OPENAI_API_KEY from .env
export $(grep -v '^#' .env | xargs)

if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}✗ OPENAI_API_KEY not set in .env${NC}"
    exit 1
fi

echo -e "${GREEN}✓ OPENAI_API_KEY loaded from .env${NC}"

# Stop and remove existing container if running
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Build Docker image
echo ""
echo "## Building Docker image..."
docker build -t $IMAGE_NAME . || {
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
}
echo -e "${GREEN}✓ Docker build successful${NC}"

# Show image size
echo ""
echo "## Image size:"
docker images $IMAGE_NAME --format "{{.Repository}}:{{.Tag}} - {{.Size}}"

# Run container
echo ""
echo "## Starting container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $TEST_PORT:8000 \
    -e OPENAI_API_KEY=$OPENAI_API_KEY \
    -e PORT=8000 \
    $IMAGE_NAME || {
    echo -e "${RED}✗ Container failed to start${NC}"
    exit 1
}

echo -e "${GREEN}✓ Container started${NC}"
echo "Container ID: $(docker ps -q -f name=$CONTAINER_NAME)"

# Wait for health check
echo ""
echo "## Waiting for health check (up to 60s)..."
for i in {1..12}; do
    HEALTH=$(docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null || echo "starting")
    echo "  Attempt $i/12: $HEALTH"

    if [ "$HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✓ Health check passed${NC}"
        break
    fi

    if [ "$i" -eq 12 ]; then
        echo -e "${RED}✗ Health check failed or timed out${NC}"
        echo "Container logs:"
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        exit 1
    fi

    sleep 5
done

# Test API endpoints
echo ""
echo "## Testing API endpoints..."

# Test root endpoint
echo "  Testing GET /"
curl -s -f http://localhost:$TEST_PORT/ > /dev/null && \
    echo -e "${GREEN}  ✓ GET / - OK${NC}" || \
    echo -e "${RED}  ✗ GET / - FAILED${NC}"

# Test health endpoint
echo "  Testing GET /health"
HEALTH_RESPONSE=$(curl -s http://localhost:$TEST_PORT/health)
echo "  Response: $HEALTH_RESPONSE"

if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    echo -e "${GREEN}  ✓ GET /health - OK${NC}"
else
    echo -e "${RED}  ✗ GET /health - Status not healthy${NC}"
fi

# Test docs endpoint
echo "  Testing GET /docs"
curl -s -f http://localhost:$TEST_PORT/docs > /dev/null && \
    echo -e "${GREEN}  ✓ GET /docs - OK${NC}" || \
    echo -e "${RED}  ✗ GET /docs - FAILED${NC}"

# Summary
echo ""
echo "=== Test Summary ==="
echo "Image: $IMAGE_NAME"
echo "Container: $CONTAINER_NAME"
echo "URL: http://localhost:$TEST_PORT"
echo "Docs: http://localhost:$TEST_PORT/docs"
echo ""
echo "Container is running. Test the generate endpoint manually:"
echo ""
echo "  curl -X POST http://localhost:$TEST_PORT/api/generate \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"company\": \"Salesforce\", \"persona\": \"vp_customer_experience\"}'"
echo ""
echo "To stop the container:"
echo "  docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
```

#### 2. Make Script Executable
```bash
chmod +x scripts/test_docker_local.sh
```

### Usage

```bash
# Run pre-deployment validation
python scripts/pre_deploy_check.py

# Build and test Docker image locally
./scripts/test_docker_local.sh

# If successful, test the generate endpoint
curl -X POST http://localhost:8001/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"company": "Salesforce", "persona": "vp_customer_experience", "session_id": "test-docker"}'

# Stop container when done
docker stop rag-api-test && docker rm rag-api-test
```

### Success Criteria

#### Automated Verification:
- [ ] Test script completes without errors: `./scripts/test_docker_local.sh`
- [ ] All endpoint tests pass (GET /, GET /health, GET /docs)
- [ ] Health check reaches "healthy" status within 60 seconds

#### Manual Verification:
- [ ] API docs load and display all endpoints
- [ ] POST /api/generate completes successfully (may take 90-120s)
- [ ] Generated email.json exists in container: `docker exec rag-api-test ls outputs/test-docker/email.json`
- [ ] No errors in container logs: `docker logs rag-api-test`

---

## Phase 5: Railway Deployment Configuration

### Overview
Configure Railway project and deploy the Docker image from GitHub. Railway will auto-detect the Dockerfile and use its CMD for deployment - no additional configuration files needed.

### Changes Required

#### 1. Update .gitignore
**File**: `.gitignore` (add if not present)

```
# Environment variables (use Railway dashboard)
.env
.env.*

# Docker
.dockerignore

# Railway CLI
.railway/

# Runtime outputs
outputs/
state/
logs/
data/cache/

# Python
__pycache__/
*.py[cod]
```

### Railway Setup Steps

#### Step 1: Push Code to GitHub

```bash
# Verify all changes committed
git status

# Commit Docker files
git add Dockerfile .dockerignore scripts/pre_deploy_check.py scripts/test_docker_local.sh
git add envs/age.yaml  # Updated with FastAPI deps
git commit -m "feat: add Railway deployment configuration

- Add single-stage Dockerfile with conda run (simple, reliable)
- Add .dockerignore for optimized builds
- Add pre-deployment validation script
- Add local Docker test script
- Update age.yaml with FastAPI dependencies"

# Push to main branch
git push origin main
```

#### Step 2: Create Railway Project

1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Authenticate GitHub (if first time)
5. Select repository: `ag3` (or your repo name)
6. Select branch: `main`
7. Railway auto-detects Dockerfile and starts build

#### Step 3: Configure Environment Variables

In Railway dashboard:

1. Click on your service
2. Go to "Variables" tab
3. Click "New Variable"
4. Add variables:

   **Required**:
   - `OPENAI_API_KEY`: Your OpenAI API key

   **Optional** (Railway sets automatically):
   - `PORT`: Auto-set by Railway (don't manually configure)
   - `RAILWAY_ENVIRONMENT`: production/staging

5. Click "Deploy" to apply changes

#### Step 4: Monitor Deployment

1. Go to "Deployments" tab
2. Watch build logs (should take 3-5 minutes)
3. Wait for status: "Deployed" (green)
4. Note the public URL: `https://<project-name>.railway.app`

#### Step 5: Verify Deployment

```bash
# Test health endpoint
curl https://<project-name>.railway.app/health

# Expected response:
# {"status":"healthy","directories":{...},"env_file":false,"ready":true}

# Test docs
open https://<project-name>.railway.app/docs

# Test generation (may take 90-120s)
curl -X POST https://<project-name>.railway.app/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"company": "Salesforce", "persona": "vp_customer_experience"}'
```

### Success Criteria

#### Automated Verification:
- [ ] GitHub push succeeds: `git push origin main`
- [ ] Railway build completes: Check "Deployments" tab shows green "Deployed" status
- [ ] Health check passes: `curl https://<url>/health` returns `{"status":"healthy"}`
- [ ] API docs load: `curl https://<url>/docs` returns HTML

#### Manual Verification:
- [ ] Railway dashboard shows service running with 0 restarts
- [ ] Public URL accessible in browser
- [ ] API documentation page loads completely
- [ ] POST /api/generate endpoint completes successfully (test with Salesforce + vp_customer_experience)
- [ ] Response time acceptable (<120s for first request)
- [ ] Railway metrics show CPU time usage (not wall-clock time)

---

## Phase 6: Post-Deployment Validation

### Overview
Comprehensive validation of the deployed application on Railway, including performance monitoring and cost verification.

### Validation Steps

#### 1. Functional Testing

**Basic Health Checks**:
```bash
# Save Railway URL
RAILWAY_URL="https://<your-project>.railway.app"

# Test root endpoint
curl $RAILWAY_URL/

# Test health endpoint
curl $RAILWAY_URL/health | jq .

# Test API docs
curl -s $RAILWAY_URL/docs | grep -q "FastAPI" && echo "✓ Docs OK"
```

**End-to-End Workflow Test**:
```bash
# Test email generation for multiple personas
for persona in vp_customer_experience vp_sales_ops cio; do
    echo "Testing persona: $persona"

    curl -X POST $RAILWAY_URL/api/generate \
        -H 'Content-Type: application/json' \
        -d "{\"company\": \"Salesforce\", \"persona\": \"$persona\", \"session_id\": \"prod-test-$persona\"}" \
        -w "\nStatus: %{http_code}\nTime: %{time_total}s\n" \
        -o response_$persona.json

    # Check response
    if jq -e '.session_id' response_$persona.json > /dev/null; then
        echo "✓ Persona $persona succeeded"
    else
        echo "✗ Persona $persona failed"
    fi

    echo "---"
    sleep 5
done
```

#### 2. Performance Monitoring

**Railway Metrics Dashboard**:
1. Go to Railway project → "Metrics" tab
2. Monitor:
   - **CPU Usage**: Should spike during LLM calls, idle during OpenAI waits
   - **Memory Usage**: Should stay under 512MB-1GB
   - **Response Time**: 90-120s for first request, 60-90s for cached requests
   - **Restart Count**: Should be 0 (if >0, investigate logs)

**Expected CPU Time vs Wall-Clock Time**:
```
Request 1: Salesforce + vp_customer_experience
├── Wall-clock time: ~100s (user-perceived)
├── CPU time: ~5-8s (Railway billing)
└── Savings: ~92-95% cost reduction vs wall-clock billing
```

#### 3. Cost Verification

**Check Railway Usage**:
1. Go to Railway dashboard → "Usage"
2. Verify:
   - **CPU-hours used**: Should be <0.1 hours per request (~5s CPU time)
   - **Estimated monthly cost**: Should remain $0 for demo usage
   - **Free tier remaining**: Should have 499+ CPU-hours left

**Example Calculation**:
```
100 demo requests/month × 5s CPU time = 500s = 0.139 hours
Free tier: 500 CPU-hours/month
Usage: 0.139 / 500 = 0.028% of free tier
Cost: $0
```

### Troubleshooting Common Issues

#### Issue 1: "Service Unhealthy" in Railway

**Symptoms**: Railway shows red "Unhealthy" status

**Diagnosis**:
```bash
# Check Railway logs
railway logs

# Common causes:
# - Port binding incorrect (must use $PORT)
# - OPENAI_API_KEY missing
# - Data files missing from Docker image
```

**Fix**:
1. Verify `CMD` in Dockerfile uses `${PORT:-8000}`
2. Check environment variables in Railway dashboard
3. Re-run `scripts/pre_deploy_check.py` locally

#### Issue 2: Slow First Request (>180s)

**Symptoms**: First request times out or takes >3 minutes

**Diagnosis**:
- Cold start may take 30-40s (container initialization)
- OpenAI API rate limits or network issues
- Embedding cache not working

**Fix**:
1. Increase Railway health check timeout (Railway dashboard → Settings → Health Check)
2. Monitor OpenAI API status: https://status.openai.com
3. Check logs for embedding cache hits

#### Issue 3: High CPU Usage (Cost Concerns)

**Symptoms**: CPU-hours usage higher than expected

**Diagnosis**:
```bash
# Check Railway metrics
# CPU should be LOW except during active processing
# If constantly high, may indicate:
# - Infinite loop in code
# - Health check failures causing restarts
# - Inefficient vector operations
```

**Fix**:
1. Review Railway logs for errors
2. Check restart count (should be 0)
3. Verify FAISS indexes are pre-built (not being rebuilt at runtime)

### Success Criteria

#### Automated Verification:
- [ ] Health endpoint returns 200 for 10 consecutive checks (1-minute intervals)
- [ ] All three personas (vp_customer_experience, vp_sales_ops, cio) generate emails successfully
- [ ] Railway metrics show CPU time <10s per request
- [ ] Railway usage remains $0 (within free tier)

#### Manual Verification:
- [ ] First request completes in <120s
- [ ] Subsequent requests complete in <90s (caching benefits)
- [ ] Generated emails meet quality standards (✓ 0 critical flags, ✓ ≤160 words)
- [ ] Railway dashboard shows 0 restarts in past 24 hours
- [ ] Public URL accessible from external networks (test from different device/network)

---

## Testing Strategy

### Unit Tests
Not applicable for this deployment phase. Unit tests for application logic exist in the codebase but are out of scope for containerization.

### Integration Tests

**Docker Build Test**:
```bash
# Test Docker build completes
docker build -t rag-api-test .

# Test conda environment integrity
docker run --rm rag-api-test conda run -n age python -c "import fastapi, langgraph, openai; print('OK')"
```

**Local Container Test**:
```bash
# Full integration test with test script
./scripts/test_docker_local.sh

# Verify outputs directory created
docker exec rag-api-test ls -la outputs/

# Verify logs directory created
docker exec rag-api-test ls -la logs/
```

### Manual Testing Steps

**Step 1: Local Docker Test** (Before Railway)
1. Run `scripts/test_docker_local.sh`
2. Test each endpoint manually via curl or browser
3. Verify one full email generation workflow completes
4. Check container logs for any warnings/errors

**Step 2: Railway Staging Test** (After Deployment)
1. Test health endpoint immediately after deployment
2. Wait 2 minutes for full initialization
3. Test one email generation workflow (expect 90-120s)
4. Review Railway logs for any unexpected errors

**Step 3: Railway Production Test** (Final Validation)
1. Test all three personas (vp_customer_experience, vp_sales_ops, cio)
2. Verify response times acceptable (<120s first request, <90s subsequent)
3. Check generated emails for quality (0 critical flags, proper formatting)
4. Monitor Railway metrics for 24 hours
5. Verify CPU usage pattern (spikes during processing, low otherwise)

---

## Performance Considerations

### Expected Performance Metrics

**Local Docker Performance**:
- Build time: 3-5 minutes (first build), 30-60s (cached layers)
- Image size: 1.5-2GB (acceptable for Railway)
- Container start time: 20-30s
- Health check ready: 30-40s after start

**Railway Deployment Performance**:
- Build time: 4-6 minutes (Railway server builds from scratch)
- Deploy time: 1-2 minutes (container startup + health check)
- First request: 90-120s (includes LLM API calls)
- Subsequent requests: 60-90s (caching benefits)
- CPU time per request: 5-8s (only active processing time)

### Performance Optimization Opportunities (Future)

**Not included in this phase but documented for future reference**:

1. **Parallel LLM Calls**: Current implementation calls OpenAI sequentially (Consolidator → Stylist → A2A). Could parallelize non-dependent calls.

2. **Response Caching**: Cache generated emails for identical company+persona combinations (requires Redis in Phase 2).

3. **Async Task Queue**: Return immediately with job ID, process in background (requires Celery in Phase 2).

4. **Embedding Cache Pre-Population**: Pre-compute common query embeddings to reduce OpenAI API calls.

5. **Index Pre-Loading**: Load FAISS indexes at startup (currently lazy-loaded on first request).

---

## Migration Notes

### Pre-Deployment Checklist

Before deploying to Railway, ensure:

- [ ] **Local validation passed**: `python scripts/pre_deploy_check.py` returns 0
- [ ] **Docker build succeeded**: `docker build -t rag-api .` completes without errors
- [ ] **Local Docker test passed**: `./scripts/test_docker_local.sh` shows all green checks
- [ ] **Environment updated**: `envs/age.yaml` includes FastAPI dependencies
- [ ] **FAISS indexes built**: `data/vector/faiss/index.faiss` exists and is recent
- [ ] **Embeddings current**: `data/vector/embeddings/embeddings.parquet` includes all latest documents
- [ ] **GitHub pushed**: All changes committed and pushed to main branch
- [ ] **API key ready**: OPENAI_API_KEY available for Railway configuration

### Railway Environment Variables Setup

**Required Variables** (Set in Railway dashboard):

| Variable | Value | Source |
|----------|-------|--------|
| `OPENAI_API_KEY` | `sk-...` | From OpenAI dashboard (do NOT commit to git) |

**Auto-Set Variables** (Railway manages automatically):

| Variable | Value | Notes |
|----------|-------|-------|
| `PORT` | `8000-9999` | Dynamically assigned by Railway, do NOT manually set |
| `RAILWAY_ENVIRONMENT` | `production` | Auto-set by Railway |
| `RAILWAY_SERVICE_NAME` | `<service>` | Auto-set by Railway |

**Optional Variables** (For advanced configuration):

| Variable | Value | Use Case |
|----------|-------|----------|
| `AG7_LATENCY_MULTIPLIER` | `2.0` | Relax retrieval latency budgets for slower networks |
| `PYTHONUNBUFFERED` | `1` | Force Python stdout/stderr to be unbuffered (better logs) |
| `LOG_LEVEL` | `INFO` | Control uvicorn log verbosity |

### Rollback Plan

If deployment fails or critical issues discovered:

**Step 1: Immediate Rollback in Railway**
1. Go to Railway dashboard → "Deployments" tab
2. Find previous successful deployment
3. Click "Redeploy" on that version
4. Service reverts to previous state in <2 minutes

**Step 2: Fix Issues Locally**
1. Review Railway logs for error messages
2. Test fixes with local Docker: `./scripts/test_docker_local.sh`
3. Commit and push fixes
4. Railway auto-deploys new version

**Step 3: Disable Service (Emergency)**
If critical vulnerability discovered:
1. Railway dashboard → Service → "Settings"
2. Click "Pause Service"
3. Service stops immediately (no new requests)
4. Fix issue, unpause when ready

### Data Migration Notes

**No database migration needed** for this phase:
- Application is stateless (no persistent database)
- All data files are included in Docker image (read-only)
- Generated outputs (emails) are ephemeral (not persisted between deployments)

**Future Phase 2 Considerations**:
- When adding PostgreSQL, will need Railway database service
- When adding Redis, will need Railway Redis plugin
- User authentication will require database schema migrations

---

## References

### Internal Documentation
- **ADR-001**: Railway deployment decision (docs/adr/001-railway-deployment.md)
- **Backend Architecture**: Phase 1 implementation (docs/backend-architecture.md)
- **Deployment Analysis**: Cost and platform comparison (docs/001-railway-deployment-analysis.md)
- **System Overview**: Complete system architecture (roadmap/part1-overview.md)
- **Environment Setup**: Two-environment architecture (docs.archive/envs.md)

### External Resources
- **Railway Documentation**: https://docs.railway.com
  - [Deploy from Dockerfile](https://docs.railway.com/guides/dockerfiles)
  - [Using Variables](https://docs.railway.com/guides/variables)
  - [Deploy FastAPI](https://docs.railway.com/guides/fastapi)
- **Docker Best Practices**: https://docs.docker.com/develop/dev-best-practices/
- **FastAPI Documentation**: https://fastapi.tiangolo.com
  - [Settings and Environment Variables](https://fastapi.tiangolo.com/advanced/settings/)
- **Conda Docker Guide**: https://pythonspeed.com/articles/activate-conda-dockerfile/

---

## Next Steps After Deployment

Once Railway deployment is successful and validated:

### Immediate Next Steps
1. **Document public URL**: Add Railway URL to README.md
2. **Share with stakeholders**: Provide demo link with usage instructions
3. **Monitor for 7 days**: Check Railway metrics daily for anomalies
4. **Create runbook**: Document common operations (restart, logs, rollback)

### Phase 2 Planning (Future)
Evaluate whether these features are needed based on usage patterns:

- **User Authentication**: If multiple users need isolated access
- **Database Persistence**: If generated emails need to be stored/retrieved
- **Background Processing**: If 90-120s response time is unacceptable
- **CORS Configuration**: When Next.js frontend is implemented
- **Rate Limiting**: If abuse or excessive usage detected
- **Advanced Monitoring**: If Railway metrics insufficient (Sentry, Datadog)

### Continuous Improvement
- **Performance Monitoring**: Track Railway CPU-hours usage weekly
- **Cost Analysis**: Verify $0 cost maintained as usage scales
- **Quality Metrics**: Monitor generated email quality (0 critical flags target)
- **User Feedback**: Collect feedback on response times and email quality

---

**End of Railway Deployment Implementation Plan**

**Document Statistics**:
- **Phases**: 6 comprehensive implementation phases
- **Scripts**: 2 automation scripts (validation + Docker test)
- **Success Criteria**: 40+ automated and manual verification steps
- **Configuration Files**: 2 (Dockerfile, .dockerignore)
- **Estimated Implementation Time**: 4-6 hours (1 day)
- **Deployment Time**: 15-20 minutes (after testing)

**For Questions or Issues**:
- Review **Railway logs**: `railway logs` (if Railway CLI installed)
- Check **Phase 4 Troubleshooting**: Common issues and solutions
- Consult **ADR-001**: Original deployment decision rationale
- Review **Backend Architecture**: Phase 1 implementation details
