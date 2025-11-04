# ADR-001: Use Railway.app for Backend Deployment

**Date**: 2025-11-04
**Status**: Accepted
**Technical Story**: Phase 1 backend deployment strategy

---

## Context

Need to deploy FastAPI backend for RAG email generation system:
- **Workload**: IO-bound (60-70% time waiting for OpenAI API)
- **Execution time**: 15-30 seconds wall-clock, 2-5 seconds CPU time
- **Usage**: Demo/showcase (~100 requests/month)
- **Budget**: $0

## Decision

Deploy backend on **Railway.app** using Docker container.

## Rationale

Railway bills by **CPU time**, not wall-clock time:
```
OpenAI API calls: 12-15s waiting → CPU idle → $0 cost
Railway only charges: 2-5s active CPU → within free tier
```

Cost comparison:
- Railway: **$0/month** (free tier: 500 CPU-hours)
- Vercel Hobby: $20/month (wall-clock billing)
- Render Free: $0 but sleeps after 15min idle

## Consequences

### Positive
- Zero cost for showcase usage (600K requests/month capacity)
- No timeout limits (unlike Vercel 60s)
- No cold-start issues (unlike Render sleep)
- Always-on, production-ready

### Negative
- Requires Docker knowledge (minimal learning curve)
- Potential vendor lock-in (mitigated by standard containerization)

## Alternatives Considered

| Platform | Cost | Issue |
|----------|------|-------|
| Vercel Serverless | $20/mo | 60s timeout risk, wall-clock billing |
| Render Free Tier | $0 | 15min idle → sleep (poor demo UX) |
| Local + Ngrok | $0 | Cannot keep machine always-on |

**Verdict**: Railway's CPU-time billing transforms IO-heavy workload from $20/month to free.

## Implementation

- [x] Phase 1 FastAPI backend (`api/main.py`)
- [ ] Dockerfile for containerization
- [ ] Railway deployment configuration
- [ ] Environment variables setup (`OPENAI_API_KEY`)

## References

- Detailed analysis: [001-railway-deployment-analysis.md](../001-railway-deployment-analysis.md)
- Railway pricing: https://railway.app/pricing
- Backend architecture: [backend-architecture.md](../backend-architecture.md)
