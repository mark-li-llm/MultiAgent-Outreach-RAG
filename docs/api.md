# API Quick Start

> Fast-track guide to use the Multi-Agent RAG Email Generator HTTP API.

**Base URL**: `https://multiagent-outreach-rag-production.up.railway.app`
**Interactive Docs**: [Swagger UI](https://multiagent-outreach-rag-production.up.railway.app/docs)

**Need complete reference?** See [Full API Documentation](./api-reference.md)

---

## ⚡ TL;DR - 30 Second API Call

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Salesforce",
    "persona": "vp_customer_experience"
  }'
```

**Response** (60-75 seconds):
```json
{
  "session_id": "auto-generated-uuid",
  "out_dir": "outputs/auto-generated-uuid",
  "total_ms": 68832.02,
  "message": "Email generated successfully..."
}
```

---

## 📋 Available Endpoints

| Endpoint | Method | Description | Response Time |
|----------|--------|-------------|---------------|
| `/` | GET | Health check | <500ms |
| `/health` | GET | Detailed status | <500ms |
| `/api/generate` | POST | **Generate email** | 60-80s |
| `/docs` | GET | Interactive API docs | <1s |

---

## 🚀 Quick Start

### 1. Health Check

Verify the API is running:

```bash
curl https://multiagent-outreach-rag-production.up.railway.app/
```

**Response**:
```json
{
  "status": "ok",
  "service": "RAG Email Generator",
  "version": "1.0.0 (Phase 1)"
}
```

---

### 2. Generate Email (Main Endpoint)

**POST** `/api/generate`

**Request Body**:
```json
{
  "company": "string (required)",
  "persona": "string (required)",
  "session_id": "string (optional)"
}
```

**Example - Minimal**:
```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Microsoft",
    "persona": "cto"
  }'
```

**Example - With Session ID**:
```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "company": "Apple",
    "persona": "vp_sales",
    "session_id": "campaign-001"
  }'
```

**Response**:
```json
{
  "session_id": "campaign-001",
  "out_dir": "outputs/campaign-001",
  "total_ms": 72450.33,
  "message": "Email generated successfully. Results available in outputs/campaign-001"
}
```

---

### 3. Interactive Documentation

Visit the Swagger UI for hands-on testing:

**URL**: https://multiagent-outreach-rag-production.up.railway.app/docs

Features:
- ✅ Try endpoints directly in browser
- ✅ View request/response schemas
- ✅ Copy curl commands
- ✅ No authentication required

**How to use**:
1. Open the URL above
2. Expand `POST /api/generate`
3. Click "Try it out"
4. Fill in: `{"company": "Salesforce", "persona": "vp_customer_experience"}`
5. Click "Execute"
6. View response (takes ~70 seconds)

---

## 🐍 Python Example

```python
import requests

url = "https://multiagent-outreach-rag-production.up.railway.app/api/generate"

payload = {
    "company": "Google",
    "persona": "cto",
    "session_id": "my-test-001"
}

response = requests.post(url, json=payload, timeout=120)
result = response.json()

print(f"✅ Session: {result['session_id']}")
print(f"⏱️  Time: {result['total_ms'] / 1000:.1f}s")
print(f"📁 Output: {result['out_dir']}")
```

**Install requests**:
```bash
pip install requests
```

---

## 🟢 JavaScript/Node.js Example

```javascript
const fetch = require('node-fetch');

const url = 'https://multiagent-outreach-rag-production.up.railway.app/api/generate';

const payload = {
  company: 'Netflix',
  persona: 'vp_product'
};

fetch(url, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
})
  .then(res => res.json())
  .then(data => {
    console.log('✅ Session:', data.session_id);
    console.log('⏱️  Time:', (data.total_ms / 1000).toFixed(1) + 's');
  })
  .catch(err => console.error('Error:', err));
```

**Install node-fetch**:
```bash
npm install node-fetch
```

---

## 🚨 Error Handling

### Common Errors

#### 422 Validation Error

**Cause**: Missing required fields

**Example**:
```bash
curl -X POST .../api/generate -d '{}'
```

**Response**:
```json
{
  "detail": [
    {"loc": ["body", "company"], "msg": "Field required"},
    {"loc": ["body", "persona"], "msg": "Field required"}
  ]
}
```

**Fix**: Include `company` and `persona` fields

---

#### 500 Internal Error

**Cause 1**: OpenAI API key not configured

**Response**:
```json
{
  "detail": "Email generation failed: The api_key client option must be set..."
}
```

**Fix**: Environment variable issue - contact admin

---

**Cause 2**: Workflow failure

**Response**:
```json
{
  "detail": "Email generation failed: [specific error]"
}
```

**Fix**: Check error message, retry, or contact support

---

## 💡 Best Practices

### 1. Handle Timeouts

Email generation takes 60-80 seconds. Set appropriate timeout:

**Python**:
```python
response = requests.post(url, json=payload, timeout=120)  # 2 minutes
```

**JavaScript**:
```javascript
// Using node-fetch with abort controller
const controller = new AbortController();
setTimeout(() => controller.abort(), 120000);  // 2 minutes

fetch(url, { signal: controller.signal, ... })
```

---

### 2. Use Custom Session IDs

For tracking and debugging:

```bash
session_id="campaign-$(date +%Y%m%d-%H%M%S)"

curl -X POST .../api/generate \
  -d "{\"company\": \"Tesla\", \"persona\": \"vp_sales\", \"session_id\": \"$session_id\"}"
```

---

### 3. Retry on Transient Failures

```python
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('https://', adapter)

response = session.post(url, json=payload)
```

---

## 📊 Request/Response Reference

### GenerateEmailRequest

**Schema**:
```typescript
{
  company: string;      // Required: "Salesforce", "Microsoft", etc.
  persona: string;      // Required: "vp_customer_experience", "cto", etc.
  session_id?: string;  // Optional: Custom tracking ID
}
```

**Supported Personas**:
- `vp_customer_experience`
- `cto`
- `vp_sales`
- `vp_product`
- (others based on your configuration)

---

### GenerateEmailResponse

**Schema**:
```typescript
{
  session_id: string;   // Generated or provided session ID
  out_dir: string;      // Output directory path (ephemeral on Railway)
  total_ms: number;     // Execution time in milliseconds
  message: string;      // Success message
}
```

---

## ⏱️ Performance

**Expected Response Times**:

| Endpoint | Typical | Notes |
|----------|---------|-------|
| `/` | <500ms | Instant health check |
| `/health` | <500ms | Directory check included |
| `/api/generate` | 60-80s | LLM and retrieval time |

**First Request** (cold start): +10-15 seconds

---

## 🔐 Authentication

**Current Status**: No authentication (Phase 1)

**Suitable for**:
- Internal testing
- Proof-of-concept
- Trusted environments

**Not suitable for**:
- Public internet exposure
- Production with external users

**Future**: API key or OAuth 2.0 authentication

---

## 📚 More Resources

### Documentation
- **[Full API Reference](./api-reference.md)** - Complete endpoint details
- **[Deployment Guide](./deployment.md)** - How to deploy your own
- **[Test Reports](../reports/deployment/)** - Verification results

### Interactive Tools
- **[Swagger UI](https://multiagent-outreach-rag-production.up.railway.app/docs)** - Live API testing
- **[Postman Collection](./api-reference.md#postman)** - Import into Postman (see reference)

### Support
- **Email**: mark362852@gmail.com
- **Railway**: https://railway.app/help
- **OpenAI**: https://help.openai.com

---

## 🎯 Common Use Cases

### Use Case 1: Single Email Generation

```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{"company": "Stripe", "persona": "cto"}'
```

---

### Use Case 2: Batch Campaign (Sequential)

```bash
for company in "Salesforce" "Microsoft" "Apple"; do
  echo "Generating for $company..."
  curl -X POST .../api/generate \
    -H "Content-Type: application/json" \
    -d "{\"company\": \"$company\", \"persona\": \"vp_sales\"}"
  echo "\n---\n"
done
```

**Note**: Run sequentially to avoid overwhelming the service

---

### Use Case 3: Integration with CRM

```python
import requests

# Fetch contacts from your CRM
contacts = [
    {"company": "Salesforce", "persona": "vp_customer_experience"},
    {"company": "Microsoft", "persona": "cto"},
    # ... more contacts
]

url = "https://multiagent-outreach-rag-production.up.railway.app/api/generate"

for contact in contacts:
    print(f"Generating email for {contact['company']}...")
    response = requests.post(url, json=contact, timeout=120)

    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success: {result['session_id']}")
    else:
        print(f"❌ Failed: {response.status_code}")
```

---

## ✅ Quick Reference Card

**Generate Email**:
```bash
curl -X POST https://multiagent-outreach-rag-production.up.railway.app/api/generate \
  -H "Content-Type: application/json" \
  -d '{"company": "YourCompany", "persona": "your_persona"}'
```

**Test API**:
```bash
curl https://multiagent-outreach-rag-production.up.railway.app/health
```

**Interactive Docs**:
```
https://multiagent-outreach-rag-production.up.railway.app/docs
```

**Complete Reference**:
[api-reference.md](./api-reference.md)

---

**API Quick Start Complete!** For detailed schemas, error codes, and advanced features, see the [Full API Reference](./api-reference.md).
