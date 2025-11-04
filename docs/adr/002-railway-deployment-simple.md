# Railway Deployment: Simplified Implementation

**Date**: 2025-11-04
**Status**: Ready for Implementation
**Implements**: ADR-001 Railway Deployment Decision
**Philosophy**: Simple, fast, good enough

---

## Overview

Deploy the Phase 1 FastAPI backend (`api/main.py`) to Railway.app with minimal complexity. Focus on getting it working, not perfection.

**Time to Deploy**: 1-2 hours (vs 6+ hours with complex approach)

---

## Current State

✅ **Working**:
- FastAPI backend in `api/main.py` (tested locally)
- Two conda environments: `age` (runtime), `ageFaiss` (FAISS build only)
- Pre-built FAISS indexes in `data/vector/faiss/`
- End-to-end test passed (test-final-20251104)

❌ **Missing**:
- FastAPI dependencies not in `envs/age.yaml`
- Dockerfile
- Railway configuration

---

## Implementation: 3 Simple Steps

### Step 1: Update age.yaml (2 minutes)

**File**: `envs/age.yaml`

Add after `aiosqlite>=0.19.0` (line 24):
```yaml
    - fastapi>=0.121.0
    - uvicorn[standard]>=0.38.0
    - pydantic>=2.0.0
```

**Verify**:
```bash
conda env remove -n age
conda env create -f envs/age.yaml
conda run -n age python -c "import fastapi; print('OK')"
```

---

### Step 2: Create Dockerfile (15 lines)

**File**: `Dockerfile`

```dockerfile
FROM continuumio/miniconda3:latest
WORKDIR /app

# Create conda environment
COPY envs/age.yaml .
RUN conda env create -f age.yaml && conda clean -afy

# Copy application code and data
COPY api/ api/
COPY scripts/ scripts/
COPY configs/ configs/
COPY data/vector/ data/vector/
COPY data/interim/ data/interim/

# Create runtime directories
RUN mkdir -p outputs state logs data/cache/embeddings

# Simple healthcheck (Railway expects this)
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD wget -q --spider http://localhost:${PORT:-8000}/health || exit 1

# Run with conda environment (Railway sets $PORT)
CMD ["sh", "-c", "conda run -n age python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

**File**: `.dockerignore`

```
# Exclude everything not needed for runtime
__pycache__/
*.pyc
.git/
.env
.venv/
outputs/
state/
logs/
docs/
roadmap/
hack/
*.md
!CLAUDE.md
envs/ageFaiss.yaml
```

**Test Locally** (optional but recommended):
```bash
docker build -t rag-api .
docker run -d --name test -p 8001:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY rag-api
sleep 40  # Wait for healthcheck
curl http://localhost:8001/health
docker stop test && docker rm test
```

---

### Step 3: Deploy to Railway (15 minutes)

#### A. Push to GitHub
```bash
git add Dockerfile .dockerignore envs/age.yaml
git commit -m "feat: add Railway deployment (simplified)"
git push origin main
```

#### B. Railway Dashboard Setup

1. **Create Project**: https://railway.app → "New Project" → "Deploy from GitHub repo"
2. **Select Repository**: Choose your repo and branch `main`
3. **Set Environment Variable**:
   - Go to "Variables" tab
   - Click "New Variable"
   - Name: `OPENAI_API_KEY`
   - Value: `sk-...` (your key)
4. **Wait for Build**: "Deployments" tab shows progress (5-7 minutes)
5. **Get Public URL**: Railway provides `https://<project>.railway.app`

#### C. Verify Deployment
```bash
# Health check
curl https://<your-project>.railway.app/health

# API docs
open https://<your-project>.railway.app/docs

# Test generation (takes 90-120s)
curl -X POST https://<your-project>.railway.app/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"company": "Salesforce", "persona": "vp_customer_experience"}'
```

---

## Success Criteria

**Must Pass**:
- [ ] Docker builds locally without errors
- [ ] Railway build completes (green "Deployed" status)
- [ ] Health endpoint returns `{"status":"healthy"}`
- [ ] API docs load at `/docs`
- [ ] One email generation completes successfully

**Good to Have** (but not blockers):
- Response time <120s
- Railway metrics show low CPU usage (5-10s per request)
- No service restarts in first 24 hours

---

## What We're NOT Doing

❌ Pre-deployment validation scripts
❌ Multi-stage Docker builds
❌ Fancy test automation
❌ Image size optimization (2GB is fine)
❌ conda-pack complexity

**Reasoning**: Railway handles these concerns, and they add minimal value for a demo app.

---

## Troubleshooting

### Issue: "Service Unhealthy" in Railway

**Check Railway logs**:
```bash
# Via Railway CLI (if installed)
railway logs

# Or via dashboard: Deployments → View logs
```

**Common causes**:
1. `OPENAI_API_KEY` not set → Add in Railway Variables
2. Data files missing → Re-run `docker build` locally to verify COPY commands worked
3. Port binding wrong → Verify Dockerfile CMD uses `${PORT:-8000}`

### Issue: Docker Build Fails Locally

**"File not found" errors**:
```bash
# Check files exist
ls data/vector/embeddings/embeddings.parquet
ls data/interim/chunks/*.chunks.jsonl

# If missing, rebuild locally:
conda run -n age python scripts/qa_step01_embeddings.py
conda run -n ageFaiss python scripts/qa_step02_indexes.py
```

**Conda environment creation fails**:
```bash
# Test environment file
conda env create -f envs/age.yaml --dry-run

# Check conda version
conda --version  # Should be >=23.x
```

### Issue: First Request Times Out (>180s)

**Expected behavior**: First request is slow (cold start + LLM calls)

**If timing out**:
1. Increase Railway timeout (default 300s, should be enough)
2. Check OpenAI API status: https://status.openai.com
3. Monitor Railway logs for OpenAI API errors (rate limits, quota)

---

## Cost Verification

After deployment, verify Railway usage:

1. Railway Dashboard → "Usage"
2. Check **CPU-hours used**
3. Expected: ~0.002 hours per request (5-8s CPU time)
4. Should remain $0 within free tier (500 CPU-hours/month)

**Example**:
- 100 requests/month × 5s = 500s = 0.14 hours
- 0.14 / 500 free tier = 0.028% usage
- Cost: **$0**

---

## What's Next

**Immediate**:
- [ ] Document Railway URL in README.md
- [ ] Test with all 3 personas (vp_customer_experience, vp_sales_ops, cio)
- [ ] Share demo link with stakeholders

**Future (Phase 2)**:
- Add CORS for frontend integration
- Add background task queue if 90-120s response time is unacceptable
- Add database if email persistence needed
- Add authentication if multi-user access needed

---

## Rollback Plan

**If deployment fails**:
1. Railway Dashboard → "Deployments"
2. Find last working deployment
3. Click "Redeploy" (reverts in <2 minutes)

**If critical bug discovered**:
1. Railway Dashboard → Service → "Settings"
2. Click "Pause Service" (stops immediately)
3. Fix locally, push to GitHub, unpause

---

## Files Created

This implementation creates **2 files**:

1. `Dockerfile` (15 lines)
2. `.dockerignore` (20 lines)

And modifies **1 file**:

1. `envs/age.yaml` (add 3 lines)

**Total Changes**: 38 lines of code

---

## Time Estimates

| Phase | Time |
|-------|------|
| Update age.yaml | 2 minutes |
| Create Dockerfile + .dockerignore | 10 minutes |
| Test locally (optional) | 15 minutes |
| Push to GitHub | 2 minutes |
| Railway setup | 5 minutes |
| Railway build + deploy | 7 minutes |
| Verify deployment | 10 minutes |
| **Total** | **~50 minutes** |

Add 30 minutes buffer for troubleshooting = **1.5 hours total**

---

## Key Differences from Complex Plan

| Aspect | Complex Plan | Simple Plan |
|--------|-------------|-------------|
| **Lines of Code** | 300+ | 38 |
| **Docker Build** | Multi-stage (2 stages) | Single-stage |
| **Image Size** | 1.5GB | 2GB |
| **Scripts Created** | 4 (validation, test, etc.) | 0 |
| **Implementation Time** | 4-6 hours | 1-2 hours |
| **Maintenance Burden** | High | Low |
| **Debugging Ease** | Complex | Simple |

**Trade-off**: 500MB larger image, less validation

**Worth it?** Yes. Railway handles large images well, and validation adds minimal value.

---

## References

- **ADR-001**: Original Railway decision (docs/adr/001-railway-deployment.md)
- **Backend Architecture**: Phase 1 implementation (docs/backend-architecture.md)
- **Railway Docs**: https://docs.railway.com/guides/dockerfiles
- **FastAPI Docs**: https://fastapi.tiangolo.com

---

**End of Simplified Deployment Plan**

Ready to implement in **~1 hour**. No over-engineering, no unnecessary complexity.
