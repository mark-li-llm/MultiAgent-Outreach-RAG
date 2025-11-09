# Railway Deployment Quick Start

> Fast-track guide to deploy the Multi-Agent RAG Email Generator to Railway.app in under 10 minutes.

**Status**: ✅ Production Deployed | **URL**: https://multiagent-outreach-rag-production.up.railway.app

**Need more details?** See [Complete Deployment Guide](./deployment-advanced.md)

---

## ⚡ TL;DR - 2 Minute Deploy

```bash
# 1. Install and login to Railway CLI
brew install railway
railway login --browserless

# 2. Link to your project
cd /path/to/your/project
railway link

# 3. Set environment variable
railway variables set OPENAI_API_KEY="sk-proj-your-key-here"

# 4. Generate public domain
railway domain

# 5. Verify deployment
curl https://your-service.up.railway.app/health
```

**Done!** Your API is live at `https://your-service.up.railway.app`

---

## 📋 Prerequisites

**Required**:
- Railway account ([sign up free](https://railway.app))
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- GitHub repo connected to Railway

**Optional**:
- Railway CLI (`brew install railway`)
- Docker (for local testing)

---

## 🚀 5-Step Deployment

### Step 1: Create Railway Project

**Via Web Dashboard**:
1. Go to https://railway.app
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects `Dockerfile` and starts building

**Via CLI**:
```bash
railway init
# or link existing project:
railway link
```

---

### Step 2: Set Environment Variables

**Required variable**: `OPENAI_API_KEY`

**Via Web**:
1. Project → Service → Variables tab
2. Click "New Variable"
3. Name: `OPENAI_API_KEY`
4. Value: `sk-proj-...`
5. Service auto-redeploys

**Via CLI**:
```bash
railway variables set OPENAI_API_KEY="sk-proj-your-key"
```

---

### Step 3: Generate Public URL

**Via Web**:
1. Service → Settings → Networking
2. Click "Generate Domain"

**Via CLI**:
```bash
railway domain
```

Output: `https://your-service-production.up.railway.app`

---

### Step 4: Wait for Build

**Build time**: 3-5 minutes (first deploy)

**Check status**:
```bash
railway status
```

**View logs**:
```bash
railway logs
```

---

### Step 5: Verify Deployment

**Test health endpoint**:
```bash
curl https://your-service.up.railway.app/

# Expected response:
# {"status": "ok", "service": "RAG Email Generator", ...}
```

**Test email generation**:
```bash
curl -X POST https://your-service.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Salesforce",
    "persona": "vp_customer_experience"
  }'

# Expected: 200 OK after 60-75 seconds
```

**View API docs**:
```
https://your-service.up.railway.app/docs
```

---

## ✅ Verification Checklist

Run this quick test:

```bash
# 1. Health check
curl https://your-url.railway.app/health

# 2. API docs accessible
open https://your-url.railway.app/docs

# 3. Generate email (takes ~70 seconds)
curl -X POST https://your-url.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{"company": "Microsoft", "persona": "cto"}'
```

**All passed?** ✅ You're live!

---

## 🚨 Common Issues (Top 3)

### 1. "OPENAI_API_KEY not found"

**Symptom**: Email generation returns 500 error
**Fix**:
```bash
railway variables set OPENAI_API_KEY="sk-..."
# Wait for auto-redeploy (~2 min)
```

### 2. Build Failed

**Symptom**: Deployment stuck on "Building..."
**Fix**: Check logs for errors
```bash
railway logs
```
Common causes:
- Missing `Dockerfile` in root
- Out of memory (upgrade Railway plan)

### 3. Slow Response (>2 minutes)

**Symptom**: Email generation timeout
**Cause**: Cold start (first request after idle)
**Fix**:
- Upgrade to Hobby plan ($5/month) for always-on
- Or accept 10-15s cold start delay

---

## 📊 What Gets Deployed

**Docker Container** (continuumio/miniconda3):
```
├── Python 3.13 (conda environment: age)
├── FastAPI backend (api/main.py)
├── LangGraph workflow (scripts/)
├── Vector indexes (data/vector/)
├── Configs (configs/)
└── Pre-built embeddings (data/interim/)
```

**Port**: Auto-assigned by Railway via `$PORT`
**HTTPS**: Automatic SSL/TLS
**Uptime**: 99.9%+

---

## 🔍 Monitoring

### View Logs
```bash
# Real-time logs
railway logs --follow

# Recent logs
railway logs
```

### Check Metrics
**Via Dashboard**:
- Project → Service → Metrics tab
- CPU, memory, network graphs

### Health Check
Railway monitors `/health` endpoint:
- Interval: 30 seconds
- Auto-restart on 3 consecutive failures

---

## 📚 Next Steps

### Immediate
- ✅ Test all API endpoints
- ✅ Share API URL with team
- ✅ Monitor first few requests

### Short-term
- Add API authentication ([guide](./deployment-advanced.md#authentication))
- Set up external monitoring
- Configure custom domain

### Resources
- **[API Quick Start](./api.md)** - How to use the API
- **[Complete Deployment Guide](./deployment-advanced.md)** - Detailed reference
- **[API Reference](./api-reference.md)** - Full API documentation
- **[Test Reports](../reports/deployment/)** - Deployment verification

---

## 🆘 Need Help?

**Documentation**:
- [Complete Deployment Guide](./deployment-advanced.md) - Full troubleshooting
- [Railway Docs](https://docs.railway.app/)
- [OpenAI Platform](https://platform.openai.com/)

**Support**:
- Railway: https://railway.app/help
- Email: mark362852@gmail.com

---

## 📝 Deployment Info

**Deployed**: 2025-11-04
**Environment**: production
**Project**: mark
**Service**: MultiAgent-Outreach-RAG
**URL**: https://multiagent-outreach-rag-production.up.railway.app

**Test Results**: 6/6 tests passed ✅ | [View Report](../reports/deployment/railway_test_20251104.md)

---

**Quick Start Complete!** For advanced configuration, monitoring, and optimization, see the [Complete Deployment Guide](./deployment-advanced.md).
