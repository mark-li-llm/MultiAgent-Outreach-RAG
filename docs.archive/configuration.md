# Configuration Reference

Detailed documentation for all configuration files in the ag3 system.

## Configuration File Locations

All configuration files are located in the `configs/` directory:

```
configs/
├── vector.indexing.yaml      # Embedding and index settings
├── router.heuristics.yaml    # Query routing logic
├── mcp.tools.yaml            # MCP service endpoints
├── langgraph.nodes.yaml      # Agent graph orchestration
├── metadata.dictionary.yaml  # Metadata extraction rules
├── normalization.rules.yaml  # Text normalization rules
├── eval.prompts.yaml         # Evaluation prompt templates
├── agents.schema.yaml        # Agent schema definitions
├── compliance.template.yaml  # Compliance check templates
└── chunking.config.json      # Document chunking parameters
```

## vector.indexing.yaml

Defines embedding and index settings for all vector backends.

### Structure

```yaml
embedding:
  model: openai-ada-002        # OpenAI embedding model
  dim: 1536                    # Embedding dimensions
  batch_size: 20               # Batch size for API calls

faiss:
  index_type: HNSW             # Index algorithm
  M: 32                        # HNSW M parameter (connections per node)
  efConstruction: 200          # HNSW construction quality
  efSearch: 128                # HNSW search quality

weaviate:
  manifest:
    - class_name: Document
      vector_index_type: hnsw
      vector_index_config:
        ef: 128
        maxConnections: 32

pinecone:
  manifest:
    - index_name: ag3-documents
      dimension: 1536
      metric: cosine
      pod_type: p1.x1
```

### Key Settings

#### Embedding

- **model**: `openai-ada-002` - OpenAI's ada-002 embedding model (1536 dimensions)
- **dim**: `1536` - Must match model output dimensions
- **batch_size**: `20` - Reduced from default to avoid 8192 token limit per API call

#### FAISS (Hierarchical Navigable Small World)

- **index_type**: `HNSW` - Best balance of speed and quality for 1k-100k documents
- **M**: `32` - Number of connections per node (higher = better recall, slower build)
- **efConstruction**: `200` - Search effort during index build (higher = better quality)
- **efSearch**: `128` - Search effort during retrieval (higher = better recall)

**Tuning Guidelines**:
- For better recall: Increase `M` (32→64), `efSearch` (128→256)
- For faster search: Decrease `efSearch` (128→64)
- For faster build: Decrease `efConstruction` (200→100)

#### Weaviate

Weaviate settings are **simulated** for development (no network required).

- **class_name**: Schema class for documents
- **vector_index_type**: `hnsw` (same algorithm as FAISS)
- **ef**: Search effort (equivalent to FAISS efSearch)
- **maxConnections**: Equivalent to FAISS M

#### Pinecone

Pinecone settings are **simulated** for development (no network required).

- **index_name**: Pinecone index identifier
- **dimension**: Must be 1536 to match embeddings
- **metric**: `cosine` - Distance metric (cosine similarity)
- **pod_type**: Instance type for production deployment

### When to Modify

- **Never change `dim`** unless changing embedding model (will break existing indexes)
- Tune FAISS parameters if recall is too low or search is too slow
- Increase `batch_size` if you have larger OpenAI token limits
- Decrease `batch_size` if hitting token limit errors

## router.heuristics.yaml

Defines query routing logic to select between FAISS, Weaviate, and Pinecone backends.

### Structure

```yaml
weighting:
  similarity: 0.5              # Weight for similarity score
  recency: 0.3                 # Weight for document recency
  diversity: 0.2               # Weight for result diversity

routing_rules:
  - keywords: [press, release, announcement, news]
    backend: pinecone

  - keywords: [developer, api, code, sdk, documentation]
    backend: weaviate

  - keywords: [definition, what is, explain, glossary]
    backend: faiss

persona_bias:
  vp_customer_experience:
    preferred_backend: pinecone
    boost_keywords: [customer, satisfaction, feedback]

  chief_technology_officer:
    preferred_backend: weaviate
    boost_keywords: [technical, architecture, scalability]

fallback_order:
  - faiss
  - weaviate
  - pinecone
```

### Key Settings

#### Weighting

Controls how results are scored when no routing rule matches:

- **similarity**: Semantic similarity to query (0.5 = 50% weight)
- **recency**: Document age (0.3 = 30% weight, favors newer documents)
- **diversity**: Result diversity (0.2 = 20% weight, avoids duplicate sources)

Total must sum to 1.0.

#### Routing Rules

**First match wins** - rules are evaluated in order from top to bottom.

Each rule has:
- **keywords**: List of query keywords to match (case-insensitive)
- **backend**: Target backend (faiss, weaviate, or pinecone)

**Example**: Query "latest press release" matches first rule → routes to Pinecone

#### Persona Bias

Optional per-persona preferences:

- **preferred_backend**: Default backend for this persona if no rule matches
- **boost_keywords**: Keywords that get extra weight for this persona

**Example**: VP Customer Experience queries about "customer satisfaction" get boosted scoring

#### Fallback Order

If no routing rule matches and no persona bias applies, try backends in this order:
1. FAISS (general knowledge)
2. Weaviate (developer docs)
3. Pinecone (press/financial)

### When to Modify

- Add routing rules for new document types or query patterns
- Adjust weighting if recall is skewed (e.g., too many old documents)
- Add persona bias for new personas in `icl/persona/`
- Change fallback order based on index size/quality

## mcp.tools.yaml

Defines MCP (Model Context Protocol) service endpoints and configurations.

### Structure

```yaml
tools:
  kb.search:
    endpoint: http://localhost:7801/search
    timeout_ms: 2000
    retry_attempts: 3

  web.fetch:
    endpoint: http://localhost:7802/fetch
    timeout_ms: 5000
    retry_attempts: 2

  link.resolve:
    endpoint: http://localhost:7803/resolve
    timeout_ms: 1000
    retry_attempts: 2

  crm.lookup:
    endpoint: http://localhost:7804/lookup
    timeout_ms: 2000
    retry_attempts: 3

  safety.check:
    endpoint: http://localhost:7805/check
    timeout_ms: 3000
    retry_attempts: 1
```

### Key Settings

Each tool has:

- **endpoint**: HTTP endpoint URL (localhost for development stubs)
- **timeout_ms**: Maximum time to wait for response (milliseconds)
- **retry_attempts**: Number of retry attempts on failure

#### Tool Details

| Tool | Port | Purpose | Timeout | Retries |
|------|------|---------|---------|---------|
| `kb.search` | 7801 | Vector search across backends | 2000ms | 3 |
| `web.fetch` | 7802 | Web content fetching | 5000ms | 2 |
| `link.resolve` | 7803 | URL resolution | 1000ms | 2 |
| `crm.lookup` | 7804 | CRM data lookup | 2000ms | 3 |
| `safety.check` | 7805 | Compliance validation | 3000ms | 1 |

### When to Modify

- **Development**: Keep localhost endpoints for local stubs
- **Production**: Update endpoints to point to real services
- Increase timeouts if services are slow or network latency is high
- Increase retry attempts for flaky services
- Decrease retry attempts for safety-critical operations (e.g., safety.check)

### Production Example

```yaml
tools:
  kb.search:
    endpoint: https://api.example.com/kb/search
    timeout_ms: 5000
    retry_attempts: 3
    auth:
      type: bearer
      token_env: KB_API_TOKEN
```

## langgraph.nodes.yaml

Defines agent graph node configuration and timeout budgets.

### Structure

```yaml
nodes:
  - name: intake
    timeout_ms: 2000

  - name: planner
    timeout_ms: 5000

  - name: retriever
    timeout_ms: 10000

  - name: synthesizer
    timeout_ms: 5000

  - name: consolidator
    timeout_ms: 10000

  - name: stylist
    timeout_ms: 10000

  - name: a2a
    timeout_ms: 5000

  - name: assembler
    timeout_ms: 3000

edges:
  - from: intake
    to: planner

  - from: planner
    to: retriever

  - from: retriever
    to: synthesizer

  - from: synthesizer
    to: consolidator

  - from: consolidator
    to: stylist

  - from: stylist
    to: a2a

  - from: a2a
    to: assembler
    condition: no_critical_flags

  - from: a2a
    to: stylist
    condition: critical_flags_and_rounds_lt_2

max_a2a_rounds: 2
```

### Key Settings

#### Node Timeouts

Each node has a timeout budget (milliseconds):

- **intake**: 2s (fast validation)
- **planner**: 5s (query generation)
- **retriever**: 10s (longest - multi-backend search)
- **synthesizer**: 5s (chunk processing)
- **consolidator**: 10s (LLM call)
- **stylist**: 10s (LLM call)
- **a2a**: 5s (compliance check)
- **assembler**: 3s (final assembly)

**Total budget**: ~50-60s for full pipeline

#### Conditional Edges

- **no_critical_flags**: Proceed to assembler if no critical compliance violations
- **critical_flags_and_rounds_lt_2**: Loop back to stylist for revision if violations exist and rounds < 2

#### Max A2A Rounds

Maximum number of compliance negotiation rounds before giving up (default: 2).

### When to Modify

- Increase timeouts if nodes are timing out
- Decrease timeouts to catch performance regressions
- Adjust `max_a2a_rounds` for more/fewer compliance iterations
- Add nodes for new agent capabilities

## metadata.dictionary.yaml

Defines metadata extraction rules for structured fields.

### Structure

```yaml
fields:
  - name: document_type
    patterns:
      - regex: '10-[KQ]'
        value: sec_filing
      - regex: 'Press Release'
        value: press_release

  - name: date_published
    patterns:
      - regex: '\d{4}-\d{2}-\d{2}'
        extract: true

  - name: company_name
    patterns:
      - regex: 'Salesforce Inc\.'
        value: Salesforce

  - name: fiscal_period
    patterns:
      - regex: 'Q[1-4]\s+\d{4}'
        extract: true
```

### Key Settings

Each field has:

- **name**: Metadata field name
- **patterns**: List of extraction patterns

Each pattern has:
- **regex**: Regular expression to match
- **value**: Fixed value to assign (if no `extract`)
- **extract**: If true, extract matched text (default: false)

### When to Modify

- Add new fields for new metadata types
- Add patterns for new document formats
- Refine regex patterns if extraction is missing data

## normalization.rules.yaml

Defines text cleaning and normalization patterns.

### Structure

```yaml
rules:
  - name: remove_xbrl_metadata
    pattern: '<(?:ix:)?[^>]+>'
    replacement: ''

  - name: collapse_whitespace
    pattern: '\s+'
    replacement: ' '

  - name: remove_page_numbers
    pattern: 'Page \d+ of \d+'
    replacement: ''

  - name: normalize_quotes
    pattern: '[""''']
    replacement: '"'
```

### Key Settings

Each rule has:

- **name**: Rule identifier
- **pattern**: Regular expression to match
- **replacement**: Replacement text (empty string to remove)

Rules are applied in order from top to bottom.

### When to Modify

- Add rules for new noise patterns in documents
- Adjust patterns if normalization is too aggressive
- Reorder rules if execution order matters

## eval.prompts.yaml

Evaluation prompt templates for quality assessment.

### Structure

```yaml
prompts:
  persona_relevance:
    template: |
      Does the following insight align with the {persona} role?

      Insight: {insight_text}

      Respond with YES or NO.

  compliance_check:
    template: |
      Check the following email for compliance violations:

      Email: {email_text}

      Flag any critical issues.
```

### When to Modify

- Update templates for new evaluation criteria
- Add prompts for new quality checks
- Refine wording if LLM responses are inconsistent

## agents.schema.yaml

Agent schema definitions and validation rules.

### Structure

```yaml
agents:
  planner:
    input_schema:
      company: string
      persona: string
    output_schema:
      queries: list[string]

  retriever:
    input_schema:
      queries: list[string]
    output_schema:
      chunks: list[dict]
```

### When to Modify

- Add schemas for new agents
- Update schemas when agent contracts change

## compliance.template.yaml

Compliance check templates for generated content.

### Structure

```yaml
checks:
  - name: no_forward_looking_statements
    pattern: 'will|expect|anticipate|believe'
    severity: critical

  - name: proper_attribution
    pattern: 'Source:'
    severity: warning
    required: true
```

### When to Modify

- Add checks for new compliance requirements
- Adjust severity levels for business needs
- Add required patterns for audit trails

## chunking.config.json

Document chunking parameters (JSON format).

### Structure

```json
{
  "chunk_size": 512,
  "chunk_overlap": 64,
  "separators": ["\n\n", "\n", ". ", " "],
  "min_chunk_size": 100
}
```

### Key Settings

- **chunk_size**: Target chunk size in tokens (512 = ~2000 chars)
- **chunk_overlap**: Overlap between chunks in tokens (64 = ~250 chars)
- **separators**: Preferred split points (in priority order)
- **min_chunk_size**: Minimum viable chunk size

### When to Modify

- Increase `chunk_size` for more context per chunk (may hurt precision)
- Decrease `chunk_size` for finer granularity (may hurt context)
- Adjust `chunk_overlap` to balance context vs redundancy
- Add separators for new document formats

## Environment-Specific Configuration

Some configuration is environment-specific (not in config files):

### .env File

```bash
OPENAI_API_KEY=sk-...           # OpenAI API key (required)
AR_USER_AGENT=ag3-bot/1.0       # Custom user agent
AR_GLOBAL_RPS=5                 # Rate limit (requests per second)
```

Create with: `echo "OPENAI_API_KEY=your-key" > .env`

### Environment Variables

See [commands.md](commands.md#environment-variables-reference) for complete list.

## Configuration Validation

To validate configuration files:

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('configs/vector.indexing.yaml'))"

# Validate JSON syntax
python3 -c "import json; json.load(open('configs/chunking.config.json'))"
```

## Related Documentation

- **[architecture.md](architecture.md)** - How configuration affects system behavior
- **[commands.md](commands.md)** - Commands that use these configurations
- **[troubleshooting.md](troubleshooting.md)** - Debug configuration issues
- **[evaluation.md](evaluation.md)** - How configuration affects quality gates
