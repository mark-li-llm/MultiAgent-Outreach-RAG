Based on comprehensive analysis of
  your codebase, here's exactly what
  information you have for building the
   demo:

  ---
  1. Pre-Generated Email Inventory

  Total Available: 34 complete emails
  with compliance blocks and source
  attribution

  Persona Distribution:
  - vp_customer_experience: 16 emails
  - cio: 9 emails
  - vp_sales_ops: 9 emails

  Storage Location: /Users/liyunxiao/re
  po/ag3/worktrees/agent-faiss/outputs/
  <session-id>/email.json

  Session IDs (sample):
  - Named: step6demo
  - Hex IDs: 05b1a905f6f9,
  4ac2b75838ec, d0b6533b2b6a, etc.

  ---
  2. Email JSON Structure (What to 
  Display)

  Each email.json file contains 5
  fields:

  {
    "subject": "Ideas for improving CX 
  at Salesforce",
    "body": "Hi there,\n\nBased on 
  recent updates...",
    "unsubscribe_block": "You can 
  unsubscribe at any time...",
    "company_info_block": "Sent by ACME
   AI, 123 Market St...",
    "proof_points": [
      {"id":
  "crm::press::2025-05-27::...",
  "title": "Salesforce Signs..."}
    ]
  }

  For Your Demo Frontend, display:
  - subject - Email subject line
  (always ≤12 words)
  - body - Main content with bullets
  and URLs (100-160 words)
  - unsubscribe_block - Legal
  compliance footer
  - company_info_block - Sender
  identification
  - proof_points (optional) - Show
  "Sources" section with document
  titles

  Body Format: Plain text with \n
  newlines and -  bullets (not
  HTML/Markdown)

  ---
  3. Persona Definitions (For Dropdown)

  3 Core Personas Available:

  Option 1: VP Customer Experience

  - ID: vp_customer_experience
  - Label: "VP Customer Experience"
  - Keywords: NPS, CSAT, contact
  center, omnichannel, agent
  productivity
  - Description: "Customer experience
  executive focused on CX metrics and
  outcomes"

  Option 2: CIO

  - ID: cio
  - Label: "Chief Information Officer"
  - Keywords: Data integration,
  governance, security, TCO, platform,
  APIs
  - Description: "Technology leader
  focused on integration, security, and
   platform capabilities"

  Option 3: VP Sales Ops

  - ID: vp_sales_ops
  - Label: "VP Sales Operations"
  - Keywords: Pipeline, forecast
  accuracy, win rate, productivity,
  automation
  - Description: "Sales operations
  leader focused on metrics and
  business outcomes"

  Source:
  /Users/liyunxiao/repo/ag3/worktrees/a
  gent-faiss/configs/eval.prompts.yaml

  ---
  4. Company Data

  Single Company Available: Salesforce
  (NYSE: CRM)

  Document Coverage:
  - 100+ documents across 7 types
  - ~1,600 indexed chunks
  - SEC filings (10-K, 10-Q, 8-K)
  - Press releases
  - Product pages
  - Developer/help docs

  For Demo: Hardcode company as
  "Salesforce" (no dropdown needed)

  ---
  5. Sample Email Content

  Example from 05b1a905f6f9/email.json:

  Subject: "Ideas for improving CX at
  Salesforce"

  Body:
  Hi there,

  Based on recent updates, here are a
  few insights that may help your CX
  agenda:

  - Salesforce Signs Definitive
  Agreement to Acquire Informatica
  (2025-05-27) — https://www.salesforce
  .com/news/press-releases/2025/05/27/s
  alesforce-signs-definitive-agreement-
  to-acquire-informatica/?bc=OTH
  - crm-20250430 (2025-04-30) —
  https://www.sec.gov/Archives/edgar/da
  ta/1108524/000110852425000030/crm-202
  50430.htm
  - Salesforce Announces Fourth Quarter
   Fiscal 2025 Results (2025-01-31) —
  https://investor.salesforce.com/news/
  news-details/2025/Salesforce-Announce
  s-Fourth-Quarter-and-Fiscal-Year-2025
  -Results

  Would you be open to a quick chat to
  explore?

  ---
  6. Backend Data Access Pattern

  To serve emails via API:

  1. Scan directory:
  outputs/*/email.json → Get all
  session IDs
  2. Match persona: Read
  state/session-<id>.json → Extract
  persona field (line 3)
  3. Load email: Read
  outputs/<id>/email.json → Return JSON
   to frontend

  Python pseudocode:
  import json
  from pathlib import Path

  def get_emails_by_persona(persona):
      emails = []
      for state_file in
  Path("state").glob("session-*.json"):
          session_id =
  state_file.stem.replace("session-",
  "")
          # Read persona from state 
  (line 3)
          state =
  json.loads(state_file.read_text())
          if state.get("persona") ==
  persona:
              email_path = Path(f"outpu
  ts/{session_id}/email.json")
              if email_path.exists():
                  emails.append(json.lo
  ads(email_path.read_text()))
      return emails

  ---
  7. Metadata Files (Optional Context)

  Each session directory contains
  additional context files:

  - insights.json - Full insight cards
  with metadata
  - compliance_report.json - Compliance
   validation results
  - timing.json - Performance metrics
  - router_trace.jsonl - Query routing
  decisions
  - a2a_transcript.jsonl - Agent
  negotiation transcript (if
  applicable)

  For "How it works" section: You could
   show router trace or timing metrics

  ---
  8. Recommended Personas for 3-5 Demo 
  Options

  Minimum (3 personas): Use all 3 core
  personas
  - VP Customer Experience (16 emails
  available)
  - CIO (9 emails available)
  - VP Sales Operations (9 emails
  available)

  Extended (5 personas): Add 2 from
  evaluation set
  - CFO (from
  salesforce_eval_seed.jsonl)
  - VP Product (from
  salesforce_eval_seed.jsonl)
  - Note: These don't have
  pre-generated emails, so you'd need
  to run the generator

  ---
  9. Quality Examples to Feature

  Best email for demo (most complete
  formatting):
  - outputs/05b1a905f6f9/email.json -
  Clean, well-formatted, proper URLs

  Avoid:
  - outputs/step6demo/email.json - Body
   appears truncated

  ---
  10. What You DON'T Have (Gaps)

  - No real-time generation: All emails
   are pre-cached
  - Single company only: No Microsoft,
  Apple, etc.
  - No user personalization: Can't
  customize beyond persona selection
  - No email variations: Only 1 email
  per session (no A/B testing)
  - Subject line similarity: Many
  emails have same/similar subjects

  ---
  Summary: What Your Demo Can Show

  ✅ You Have:
  - 34 production-quality AI-generated
  emails
  - 3 distinct personas with different
  voices
  - Real Salesforce data (SEC filings,
  press, docs)
  - Full compliance (unsubscribe,
  company info)
  - Source attribution (proof points
  with URLs)

  🎯 Demo Flow:
  1. User scans QR code → Lands on web
  page
  2. Dropdown shows 3 personas
  3. User selects persona → See
  pre-generated email instantly
  4. Email displays with formatting,
  bullets, links
  5. Optional: "View Sources" expands
  proof_points

  📦 Data Bundle for Backend:
  - Copy outputs/*/email.json files (34
   files)
  - Copy state/session-*.json files (34
   files)
  - Create mapping: persona → email
  JSON
  - Serve via simple REST endpoint: GET

  /email?persona=vp_customer_experience

  ---

⏺ This is your complete demo inventory.
   You have everything you need:

  Core Assets:
  - 34 pre-generated emails ready to
  serve
  - 3 personas with distinct voices
  - Clean JSON structure (5 fields per
  email)
  - Real Salesforce data with
  compliance blocks

  For Your 10-Second Demo:
  - Frontend: Dropdown with 3 personas
  → Display email body + subject
  - Backend: Read outputs/*/email.json
  + match persona from
  state/session-*.json
  - Best example: Use 05b1a905f6f9 for
  clean formatting

  The data exists, is production-ready,
   and requires no real-time
  processing. You just need to wrap it
  in a simple web interface.

  Need anything else? More detail on
  file formats? Help picking which
  emails to feature? Backend code
  examples?

───────────────────────────────────────────
>  
───────────────────────────────────────────
  -- INSERT --




