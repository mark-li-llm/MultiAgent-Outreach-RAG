Research the codebase to support building a minimal web demo for investor showcase at an
entrepreneur conference.

**System Context:**
- Pre-generated outputs are stored in: `/Users/liyunxiao/repo/ag3/worktrees/agent-faiss/outputs/`
- Directory naming pattern: `test-langgraph-{persona}` (e.g., `test-langgraph-vp-sales-2025`)
- Example path: `/Users/liyunxiao/repo/ag3/worktrees/agent-faiss/outputs/test-langgraph-vp-sales-2025`

**Demo Purpose:** 
Create a simple web interface where users can select a persona from a dropdown and instantly 
see a pre-generated personalized outreach email. No real-time processing - just serve cached 
results from the output directories. Currently we have results for only one company.

**Key Research Areas:**

1. **Pre-generated Email Outputs:**


2. **Email Data Structure:**
   - What's the JSON/file format for stored emails?
   - What fields are available (subject, body, metadata)?
   - How are files organized within each persona directory?
3。In the output content, what data are saved in different states?

3. **Existing Web/API Components:**
   - Any existing Flask/FastAPI endpoints?
   - Any frontend templates or UI components?
   - Any utility functions for loading/parsing results?

4. **Minimal Dependencies:**
   - What's needed for a read-only demo?
   - Can we avoid FAISS, embeddings, and heavy ML dependencies?
   - What's the bare minimum stack for serving static results?

**Goal:** 
Document what exists in the codebase and output directories. Identify relevant file contents, 
data structures, and components that can be leveraged for the demo. No implementation yet - 
just comprehensive documentation of available assets.