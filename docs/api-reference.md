# API Usage Documentation

> Complete API reference for the Multi-Agent RAG Email Generator HTTP API.

**Base URL (Production)**: `https://multiagent-outreach-rag-production.up.railway.app`
**Version**: 1.0.0 (Phase 1)
**Last Updated**: 2025-11-04

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Authentication](#authentication)
4. [Endpoints](#endpoints)
5. [Request/Response Models](#requestresponse-models)
6. [Error Handling](#error-handling)
7. [Rate Limits](#rate-limits)
8. [Examples](#examples)
9. [SDKs & Client Libraries](#sdks--client-libraries)
10. [Interactive Documentation](#interactive-documentation)

---

## Overview

The Multi-Agent RAG Email Generator provides a simple HTTP API for generating personalized outreach emails using a multi-agent RAG system.

### Key Features

- **RESTful API**: Standard HTTP methods and JSON payloads
- **Async Processing**: Concurrent document retrieval and LLM operations
- **Type Safety**: Pydantic models for request/response validation
- **Auto Documentation**: Interactive Swagger UI at `/docs`
- **Production Ready**: Deployed on Railway.app with HTTPS

### Architecture

```
Client Request
    ↓
FastAPI Backend (api/main.py)
    ↓
LangGraph Workflow (scripts/run_graph_langgraph.py)
    ↓
8 Agent Nodes:
    1. Intake      → Parse input
    2. Planner     → Generate 5 queries
    3. Retriever   → Multi-index search
    4. Synthesizer → Extract insights
    5. Consolidator→ Merge results
    6. Stylist     → Draft email
    7. A2A         → Compliance check
    8. Assembler   → Final output
    ↓
JSON Response
```

---

## Quick Start

### 1. Health Check

Verify the API is operational:

```bash
curl https://multiagent-outreach-rag-production.up.railway.app/
```

**Response**:
```json
{
  "status": "ok",
  "service": "RAG Email Generator",
  "version": "1.0.0 (Phase 1)",
  "endpoints": [...]
}
```

### 2. Generate Your First Email

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Salesforce",
    "persona": "vp_customer_experience"
  }'
```

**Response** (after ~60-75 seconds):
```json
{
  "session_id": "auto-generated-uuid",
  "out_dir": "outputs/auto-generated-uuid",
  "total_ms": 68832.02,
  "message": "Email generated successfully..."
}
```

### 3. Explore Interactive Docs

Visit the Swagger UI:
```
https://multiagent-outreach-rag-production.up.railway.app/docs
```

---

## Authentication

### Current Status: **No Authentication** (Phase 1)

The current deployment does not require authentication. This is suitable for:
- Internal testing
- Proof-of-concept deployments
- Trusted network environments

### Recommendations for Production

Before exposing to public internet, implement one of:

1. **API Key Authentication**
   ```bash
   curl -X POST https://api.example.com/api/generate \
     -H "X-API-Key: your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"company": "..."}'
   ```

2. **OAuth 2.0 / JWT**
   - Standard bearer token authentication
   - Supports user-level permissions

3. **Railway Private Networking**
   - Restrict access to specific IPs
   - Configure in Railway dashboard

---

## Endpoints

### Summary

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/` | GET | Basic health check | No |
| `/health` | GET | Detailed health check | No |
| `/docs` | GET | API documentation (Swagger UI) | No |
| `/api/generate` | POST | Generate personalized email | No |

---

### GET `/`

**Description**: Basic health check endpoint

**Parameters**: None

**Response**: `200 OK`

```json
{
  "status": "ok",
  "service": "RAG Email Generator",
  "version": "1.0.0 (Phase 1)",
  "endpoints": [
    {
      "path": "/",
      "method": "GET",
      "description": "Health check"
    },
    {
      "path": "/api/generate",
      "method": "POST",
      "description": "Generate email"
    },
    {
      "path": "/docs",
      "method": "GET",
      "description": "API documentation"
    }
  ]
}
```

**Example**:
```bash
curl https://multiagent-outreach-rag-production.up.railway.app/
```

---

### GET `/health`

**Description**: Detailed health check with infrastructure status

**Parameters**: None

**Response**: `200 OK`

```json
{
  "status": "healthy",
  "directories": {
    "scripts": true,
    "configs": true,
    "data": true
  },
  "env_file": false,
  "ready": false
}
```

**Status Values**:
- `healthy`: All systems operational
- `degraded`: Some non-critical issues detected

**Fields**:
- `directories`: Verification that required directories exist
- `env_file`: `.env` file presence (false on Railway - uses env vars)
- `ready`: Overall readiness status

**Example**:
```bash
curl https://multiagent-outreach-rag-production.up.railway.app/health
```

---

### GET `/docs`

**Description**: Interactive API documentation (Swagger UI)

**Parameters**: None

**Response**: HTML page with Swagger UI

**Example**:
Open in browser:
```
https://multiagent-outreach-rag-production.up.railway.app/docs
```

Features:
- Try endpoints directly in browser
- View request/response schemas
- See example payloads
- Copy curl commands

---

### POST `/api/generate`

**Description**: Generate a personalized email for a specific company and persona

**Request Body**: `application/json`

```json
{
  "company": "string (required)",
  "persona": "string (required)",
  "session_id": "string (optional)"
}
```

**Parameters**:

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `company` | string | ✅ Yes | Company name to target | `"Salesforce"` |
| `persona` | string | ✅ Yes | Target persona/role | `"vp_customer_experience"` |
| `session_id` | string | ❌ No | Custom session ID for tracking | `"my-campaign-001"` |

**Response**: `200 OK`

```json
{
  "session_id": "string",
  "out_dir": "string",
  "total_ms": "number",
  "message": "string"
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier (auto-generated if not provided) |
| `out_dir` | string | Output directory path (ephemeral on Railway) |
| `total_ms` | number | Total execution time in milliseconds |
| `message` | string | Success message |

**Execution Time**: Typically 60-80 seconds

**Example Request**:
```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Microsoft",
    "persona": "cto",
    "session_id": "campaign-ms-cto-001"
  }'
```

**Example Response**:
```json
{
  "session_id": "campaign-ms-cto-001",
  "out_dir": "outputs/campaign-ms-cto-001",
  "total_ms": 72450.33,
  "message": "Email generated successfully. Results available in outputs/campaign-ms-cto-001"
}
```

---

## Request/Response Models

### GenerateEmailRequest

Pydantic model for email generation requests:

```python
class GenerateEmailRequest(BaseModel):
    company: str = Field(..., description="Company name (e.g., 'Salesforce')")
    persona: str = Field(..., description="Persona type (e.g., 'vp_customer_experience')")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")
```

**JSON Schema**:
```json
{
  "type": "object",
  "properties": {
    "company": {
      "type": "string",
      "description": "Company name (e.g., 'Salesforce')"
    },
    "persona": {
      "type": "string",
      "description": "Persona type (e.g., 'vp_customer_experience')"
    },
    "session_id": {
      "type": "string",
      "description": "Optional session ID for tracking",
      "nullable": true
    }
  },
  "required": ["company", "persona"]
}
```

### GenerateEmailResponse

Pydantic model for email generation responses:

```python
class GenerateEmailResponse(BaseModel):
    session_id: str = Field(..., description="Session ID for this generation")
    out_dir: str = Field(..., description="Output directory path")
    total_ms: float = Field(..., description="Total execution time in milliseconds")
    message: str = Field(..., description="Success message")
```

**JSON Schema**:
```json
{
  "type": "object",
  "properties": {
    "session_id": {
      "type": "string",
      "description": "Session ID for this generation"
    },
    "out_dir": {
      "type": "string",
      "description": "Output directory path"
    },
    "total_ms": {
      "type": "number",
      "description": "Total execution time in milliseconds"
    },
    "message": {
      "type": "string",
      "description": "Success message"
    }
  },
  "required": ["session_id", "out_dir", "total_ms", "message"]
}
```

---

## Error Handling

### HTTP Status Codes

| Status Code | Meaning | When It Occurs |
|-------------|---------|----------------|
| `200` | OK | Request successful |
| `422` | Unprocessable Entity | Invalid request body (validation error) |
| `500` | Internal Server Error | Server-side error (e.g., OpenAI API failure) |

### Error Response Format

All errors return JSON with a `detail` field:

```json
{
  "detail": "Error description here"
}
```

### Common Errors

#### 422 Validation Error

**Cause**: Missing or invalid required fields

**Example**:
```bash
curl -X POST .../api/generate -H "Content-Type: application/json" -d '{}'
```

**Response**:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "company"],
      "msg": "Field required"
    },
    {
      "type": "missing",
      "loc": ["body", "persona"],
      "msg": "Field required"
    }
  ]
}
```

**Solution**: Include all required fields in request body

---

#### 500 Internal Server Error

**Cause**: OpenAI API key missing or invalid

**Example Response**:
```json
{
  "detail": "Email generation failed: The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
}
```

**Solution**: Ensure `OPENAI_API_KEY` environment variable is set in Railway

---

**Cause**: LangGraph workflow failure

**Example Response**:
```json
{
  "detail": "Email generation failed: [specific error message]"
}
```

**Solution**: Check Railway logs for detailed error trace:
```bash
railway logs
```

---

## Rate Limits

### Current Limits

**Phase 1 Deployment**: No explicit rate limits enforced

**Effective Limits**:
- **Railway Free Tier**: Limited by concurrent connections (1 request at a time)
- **OpenAI API**: Subject to your OpenAI account rate limits
- **Response Time**: ~70 seconds per request (natural throttling)

### Recommendations for Production

Implement rate limiting to prevent abuse:

1. **Per-IP Limits**: 10 requests/hour
2. **Per-API-Key Limits**: 100 requests/day
3. **Global Limits**: 1000 requests/day

Example using FastAPI middleware:
```python
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/generate")
@limiter.limit("10/hour")
async def generate_email(request: Request, ...):
    ...
```

---

## Examples

### Example 1: Basic Email Generation

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Apple",
    "persona": "vp_sales"
  }'
```

### Example 2: With Custom Session ID

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Google",
    "persona": "cto",
    "session_id": "google-cto-q1-2025"
  }'
```

### Example 3: Pretty Print Response

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Amazon",
    "persona": "vp_customer_experience"
  }' | jq .
```

### Example 4: Measure Response Time

```bash
time curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Netflix",
    "persona": "vp_product"
  }'
```

### Example 5: Using Python Requests

```python
import requests

url = "https://multiagent-outreach-rag-production.up.railway.app/api/generate"
payload = {
    "company": "Tesla",
    "persona": "vp_sales",
    "session_id": "tesla-vp-sales-001"
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Session ID: {result['session_id']}")
print(f"Execution time: {result['total_ms'] / 1000:.2f} seconds")
print(f"Message: {result['message']}")
```

### Example 6: Using JavaScript/Node.js

```javascript
const fetch = require('node-fetch');

const url = 'https://multiagent-outreach-rag-production.up.railway.app/api/generate';
const payload = {
  company: 'Spotify',
  persona: 'cto',
  session_id: 'spotify-cto-001'
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
  .then(res => res.json())
  .then(data => {
    console.log('Session ID:', data.session_id);
    console.log('Execution time:', (data.total_ms / 1000).toFixed(2), 'seconds');
    console.log('Message:', data.message);
  })
  .catch(err => console.error('Error:', err));
```

---

## SDKs & Client Libraries

### Official SDKs

**Phase 1**: No official SDKs available

**Planned**:
- Python SDK
- JavaScript/TypeScript SDK
- Go SDK

### Community Libraries

Contributions welcome! Submit PRs to the main repository.

### HTTP Client Recommendations

**Python**:
- `requests`: Simple and popular
- `httpx`: Async support
- `aiohttp`: High performance async

**JavaScript**:
- `axios`: Feature-rich
- `node-fetch`: Fetch API for Node.js
- `got`: Promise-based

**Command Line**:
- `curl`: Universal HTTP client
- `httpie`: User-friendly curl alternative
- `wget`: Download and testing

---

## Interactive Documentation

### Swagger UI

**URL**: https://multiagent-outreach-rag-production.up.railway.app/docs

**Features**:
- ✅ Interactive API explorer
- ✅ Try endpoints directly in browser
- ✅ View request/response schemas
- ✅ Copy curl commands
- ✅ No authentication required

**How to Use**:

1. **Open Swagger UI** in your browser
2. **Expand an endpoint** (e.g., POST /api/generate)
3. **Click "Try it out"**
4. **Fill in the request body**:
   ```json
   {
     "company": "Salesforce",
     "persona": "vp_customer_experience"
   }
   ```
5. **Click "Execute"**
6. **View the response** below

### ReDoc (Alternative)

**URL**: https://multiagent-outreach-rag-production.up.railway.app/redoc (if enabled)

**Features**:
- ✅ Clean, three-panel documentation
- ✅ Better for reading/reference
- ✅ No interactive testing

---

## Best Practices

### 1. Use Custom Session IDs

For tracking and debugging:
```bash
session_id="campaign-$(date +%Y%m%d-%H%M%S)"
curl -X POST .../api/generate -d "{\"session_id\": \"$session_id\", ...}"
```

### 2. Handle Long-Running Requests

Email generation takes 60-80 seconds:
```python
import requests

response = requests.post(
    url,
    json=payload,
    timeout=120  # 2 minutes
)
```

### 3. Implement Retries

For transient failures:
```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)

response = session.post(url, json=payload)
```

### 4. Monitor OpenAI API Usage

Track costs associated with API calls:
```bash
# Check OpenAI usage at:
https://platform.openai.com/usage
```

### 5. Validate Inputs

Ensure valid company and persona values before calling API.

---

## Performance Tips

### Expected Response Times

| Endpoint | Expected Time |
|----------|---------------|
| `/` | <500ms |
| `/health` | <500ms |
| `/docs` | <1s (page load) |
| `/api/generate` | 60-80s |

### Optimization Strategies

1. **Use Specific Session IDs**: Helps with debugging and tracking
2. **Batch Requests**: If generating multiple emails, space them out
3. **Cold Start**: First request after idle may take +10-15s
4. **Keep Alive**: Regular health checks can keep container warm (paid tier)

---

## Support & Resources

### Documentation

- **Deployment Guide**: [docs/deployment.md](./deployment.md)
- **Backend Architecture**: [docs/backend-architecture.md](./backend-architecture.md)
- **Test Reports**: [reports/deployment/](../reports/deployment/)

### Getting Help

- **GitHub Issues**: Report bugs and feature requests
- **Email**: mark362852@gmail.com
- **Railway Support**: For infrastructure issues

### External Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Railway Documentation](https://docs.railway.app/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Changelog

### Version 1.0.0 (Phase 1) - 2025-11-04

**Initial Release**

- ✅ Basic email generation endpoint
- ✅ Health check endpoints
- ✅ Interactive API documentation
- ✅ Railway deployment
- ✅ OpenAI integration

**Known Limitations**:
- No authentication
- No result persistence
- Single concurrent request (free tier)
- Output files ephemeral (container restarts)

**Coming in Future Versions**:
- API authentication
- Result caching
- Batch processing
- Webhook notifications
- Custom personas
- Template customization

---

**API Version**: 1.0.0 (Phase 1)
**Last Updated**: 2025-11-04
**Production URL**: https://multiagent-outreach-rag-production.up.railway.app
