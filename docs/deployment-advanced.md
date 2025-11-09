# Railway Deployment Guide

> Complete guide for deploying the Multi-Agent RAG Email Generator to Railway.app with production configuration, monitoring, and troubleshooting.

**Last Updated**: 2025-11-04
**Environment**: Railway Production
**Status**: ✅ Deployed and Verified

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Configuration](#deployment-configuration)
4. [Environment Variables](#environment-variables)
5. [Deployment Steps](#deployment-steps)
6. [Verification](#verification)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Performance](#performance)
10. [Next Steps](#next-steps)

---

## Overview

The Multi-Agent RAG system is deployed as a FastAPI backend service on Railway.app, providing HTTP API access to the email generation pipeline.

### Architecture

```
┌─────────────────────────────────────────────────┐
│            Railway.app Platform                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  Docker Container (Miniconda3 Base)       │ │
│  │                                           │ │
│  │  ├─ FastAPI Backend (api/main.py)        │ │
│  │  ├─ LangGraph Workflow (scripts/)        │ │
│  │  ├─ Vector Indexes (data/vector/)        │ │
│  │  ├─ Configs (configs/)                   │ │
│  │  └─ Conda Environment (age)              │ │
│  │                                           │ │
│  │  Port: $PORT (Railway-assigned)          │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  Public URL: multiagent-outreach-rag-          │
│              production.up.railway.app          │
└─────────────────────────────────────────────────┘
```

### Key Features

- **Stateless Design**: Each request is independent
- **Docker Containerization**: Reproducible builds
- **Conda Environment**: Python 3.13 with all dependencies
- **Auto-scaling**: Railway handles scaling (on paid plans)
- **HTTPS by Default**: Automatic SSL/TLS

---

## Prerequisites

### Required Accounts

1. **Railway.app Account** ([https://railway.app](https://railway.app))
   - Free tier available ($5 credit)
   - GitHub OAuth recommended

2. **OpenAI Account** ([https://platform.openai.com](https://platform.openai.com))
   - API key with GPT-3.5/GPT-4 access
   - Embedding API access (ada-002)

### Local Requirements

1. **Railway CLI** (optional but recommended)
   ```bash
   brew install railway
   railway login
   ```

2. **Git Repository**
   - Project must be in GitHub/GitLab
   - Connected to Railway for auto-deployment

3. **Docker** (for local testing)
   ```bash
   docker --version
   ```

---

## Deployment Configuration

### Dockerfile

Located at `/Dockerfile`:

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

**Key Points**:
- Uses `continuumio/miniconda3` base image
- Conda environment `age` (Python 3.13)
- Copies pre-built vector indexes (no build step)
- Health check endpoint required by Railway
- Dynamic port binding via `$PORT` environment variable

### Railway Configuration

Railway auto-detects the Dockerfile and builds automatically on every Git push.

**Build Settings** (auto-detected):
- Build Command: `docker build`
- Start Command: From Dockerfile `CMD`
- Port: Auto-assigned via `$PORT`

---

## Environment Variables

### Required Variables

Set in Railway Dashboard → Service → Variables:

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | `sk-proj-...` | ✅ Yes |

### Auto-Set by Railway

These are automatically provided by Railway:

| Variable | Description |
|----------|-------------|
| `PORT` | Port for the service to bind to |
| `RAILWAY_ENVIRONMENT` | Environment name (e.g., `production`) |
| `RAILWAY_PROJECT_NAME` | Project name |
| `RAILWAY_PUBLIC_DOMAIN` | Public URL of the service |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AG7_IGNORE_COVERAGE` | Skip coverage checks in Gate-7 | `0` |
| `AG7_LATENCY_MULTIPLIER` | Relax latency budgets | `1.0` |

---

## Deployment Steps

### Step 1: Prepare Repository

Ensure your repository has:
- ✅ `Dockerfile` in root
- ✅ `envs/age.yaml` with dependencies
- ✅ Pre-built vector indexes in `data/vector/`
- ✅ Application code in `api/` and `scripts/`

### Step 2: Create Railway Project

**Option A: Web Dashboard**

1. Go to [https://railway.app](https://railway.app)
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway auto-detects Dockerfile and starts building

**Option B: Railway CLI**

```bash
# In your project directory
railway init

# Link to existing project (if already created)
railway link

# Deploy
railway up
```

### Step 3: Configure Environment Variables

**Via Web Dashboard**:

1. Go to your project → Service
2. Click "Variables" tab
3. Click "New Variable"
4. Add `OPENAI_API_KEY` with your API key
5. Service auto-redeploys

**Via CLI**:

```bash
railway variables set OPENAI_API_KEY="sk-proj-your-key-here"
```

### Step 4: Generate Public Domain

**Via Web Dashboard**:

1. Service → Settings
2. Networking section
3. Click "Generate Domain"

**Via CLI**:

```bash
railway domain
```

Expected output:
```
✓ Generated domain: https://multiagent-outreach-rag-production.up.railway.app
```

### Step 5: Verify Deployment

See [Verification](#verification) section below.

---

## Verification

### Automated Test Script

Run the provided test script:

```bash
./test_railway_deployment.sh
```

Expected output:
```
======================================
Railway Deployment Test Suite
======================================
URL: https://multiagent-outreach-rag-production.up.railway.app

======================================
1. Basic Connectivity Tests
======================================
Test: Root endpoint (health check) ... PASS (HTTP 200)
Test: Detailed health check ... PASS (HTTP 200)
Test: API documentation ... PASS (HTTP 200)

======================================
2. API Functionality Tests
======================================
Test: Email generation API ... PASS (HTTP 200)

======================================
3. Error Handling Tests
======================================
Test: Invalid request handling ... PASS (HTTP 422)
Test: 404 handling ... PASS (HTTP 404)

======================================
Test Summary
======================================
Total Tests: 6
Passed: 6
Failed: 0

All tests passed!
```

### Manual Verification

#### 1. Health Check

```bash
curl https://multiagent-outreach-rag-production.up.railway.app/

# Expected:
{
  "status": "ok",
  "service": "RAG Email Generator",
  "version": "1.0.0 (Phase 1)",
  "endpoints": [...]
}
```

#### 2. Detailed Health Check

```bash
curl https://multiagent-outreach-rag-production.up.railway.app/health

# Expected:
{
  "status": "healthy",
  "directories": {
    "scripts": true,
    "configs": true,
    "data": true
  },
  "env_file": false,
  "ready": false  # .env not needed (uses Railway env vars)
}
```

#### 3. API Documentation

Open in browser:
```
https://multiagent-outreach-rag-production.up.railway.app/docs
```

Should see Swagger UI with interactive API documentation.

#### 4. Email Generation

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Salesforce",
    "persona": "vp_customer_experience",
    "session_id": "test-001"
  }'

# Expected (after 60-75 seconds):
{
  "session_id": "test-001",
  "out_dir": "outputs/test-001",
  "total_ms": 68832.02,
  "message": "Email generated successfully. Results available in outputs/test-001"
}
```

---

## Monitoring

### Railway Dashboard

**Metrics Available**:
- CPU usage
- Memory usage
- Network traffic
- Request count
- Response times

**Access Logs**:
```bash
# Via CLI
railway logs

# Follow logs in real-time
railway logs --follow
```

**In Web Dashboard**:
1. Go to Service
2. Click "Metrics" tab
3. View real-time graphs

### Health Check Endpoint

Railway automatically monitors `/health` endpoint:
- Interval: 30 seconds
- Timeout: 10 seconds
- Start period: 40 seconds

If health check fails 3 times consecutively, Railway restarts the service.

### Custom Monitoring

Add external monitoring services:
- **UptimeRobot**: Free uptime monitoring
- **Better Stack**: Log aggregation and alerts
- **Sentry**: Error tracking (integrate via Python SDK)

---

## Troubleshooting

### Common Issues

#### 1. `OPENAI_API_KEY` Not Found

**Symptom**:
```json
{
  "detail": "Email generation failed: The api_key client option must be set..."
}
```

**Solution**:
1. Check if variable is set:
   ```bash
   railway variables
   ```
2. If missing, add it:
   ```bash
   railway variables set OPENAI_API_KEY="sk-..."
   ```
3. Verify redeployment completed:
   ```bash
   railway status
   ```

#### 2. Build Failures

**Symptom**: Deployment stuck on "Building..."

**Solution**:
1. Check build logs:
   ```bash
   railway logs
   ```
2. Common causes:
   - Missing `Dockerfile`
   - Conda environment file errors
   - Out of memory (increase Railway plan)

#### 3. Service Crashed

**Symptom**: HTTP 503 errors

**Solution**:
1. Check logs for errors:
   ```bash
   railway logs
   ```
2. Common causes:
   - Port binding issue (use `$PORT`, not hardcoded)
   - Missing dependencies
   - Out of memory
3. Restart service:
   ```bash
   railway restart
   ```

#### 4. Slow Response Times

**Symptom**: Email generation takes >2 minutes

**Solution**:
- First request after idle: Cold start (expected)
- Railway free tier: Limited resources
- Upgrade to paid tier for better performance

#### 5. Data Files Missing

**Symptom**: Health check shows `"ready": false`

**Solution**:
1. Verify Dockerfile copies data:
   ```dockerfile
   COPY data/vector/ data/vector/
   COPY data/interim/ data/interim/
   ```
2. Rebuild and redeploy:
   ```bash
   railway up --detach
   ```

---

## Performance

### Baseline Metrics (Production)

**Test Date**: 2025-11-04

| Metric | Value | Notes |
|--------|-------|-------|
| **Health Check** | <1s | Immediate response |
| **Email Generation** | 68-75s | First request after idle may be slower |
| **Success Rate** | 100% | 6/6 tests passed |
| **Uptime** | 99.9%+ | Railway SLA |

### Performance Characteristics

**Cold Start** (first request after idle):
- Time: 10-15 seconds
- Cause: Container startup + conda env activation
- Mitigation: Keep-alive pings or paid tier with always-on

**Warm Requests**:
- Health checks: <500ms
- Email generation: 60-80 seconds (mostly LLM API calls)

**Concurrency**:
- Free tier: 1 concurrent request
- Paid tier: Multiple concurrent (limited by memory)

### Optimization Tips

1. **Reduce Cold Starts**:
   - Upgrade to Hobby plan ($5/month) for always-on
   - Implement periodic keep-alive pings

2. **Faster Email Generation**:
   - Pre-compute embeddings (done via Gate-1)
   - Cache LLM responses (future enhancement)
   - Use faster models (gpt-3.5-turbo vs gpt-4)

3. **Cost Optimization**:
   - Monitor OpenAI API usage
   - Implement request rate limiting
   - Add result caching

---

## Next Steps

### Immediate

- ✅ Deployment complete and verified
- ✅ Environment variables configured
- ✅ Public domain generated
- ✅ API documentation accessible

### Short-term Enhancements

1. **Authentication**
   - Add API key authentication
   - Implement rate limiting
   - Add user tracking

2. **Monitoring**
   - Set up external uptime monitoring
   - Add error tracking (Sentry)
   - Configure alerts

3. **Custom Domain**
   - Point your domain to Railway
   - Configure DNS (CNAME record)
   - Automatic SSL certificate

### Long-term Improvements

1. **Database Integration**
   - Store generated emails
   - Track usage analytics
   - Implement result caching

2. **Background Jobs**
   - Async email generation with task queue
   - Webhook notifications on completion
   - Job status polling endpoint

3. **Multi-environment Setup**
   - Staging environment for testing
   - Preview environments for PR branches
   - Production with blue-green deployments

---

## Resources

### Documentation

- [Railway Docs](https://docs.railway.app/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

### Related Project Documentation

- [API Usage Guide](./api.md)
- [Architecture Overview](../README.md)
- [Backend Architecture](./backend-architecture.md)
- [Test Results](../reports/deployment/railway_test_20251104.md)

### Support

- **Railway Support**: [https://railway.app/help](https://railway.app/help)
- **Railway Community**: Discord and GitHub Discussions
- **OpenAI Support**: [https://help.openai.com](https://help.openai.com)

---

## Deployment Checklist

Use this checklist for future deployments:

- [ ] Code pushed to GitHub
- [ ] `Dockerfile` tested locally
- [ ] Vector indexes built and committed
- [ ] Railway project created
- [ ] GitHub repo connected to Railway
- [ ] `OPENAI_API_KEY` environment variable set
- [ ] Public domain generated
- [ ] Health check endpoint returns 200
- [ ] API documentation accessible
- [ ] Email generation tested successfully
- [ ] Monitoring configured
- [ ] Documentation updated
- [ ] Team notified

---

**Deployment Contact**: Yunxiao Li (mark362852@gmail.com)
**Project**: MultiAgent-Outreach-RAG
**Environment**: production
**URL**: https://multiagent-outreach-rag-production.up.railway.app
