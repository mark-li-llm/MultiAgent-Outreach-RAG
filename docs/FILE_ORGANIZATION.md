# File Organization Guide

This document records the principles, methods, and examples for organizing files in this project, based on actual file organization sessions.

## Philosophy

This project follows an **audit-ready, traceable documentation** approach:
- All development decisions should be documented
- Historical records should be preserved for learning
- Active documents should be easily discoverable
- Code and documentation should be cleanly separated

---

## Documentation Structure Overview

```
docs/
├── fixes/           # Bug fix archives (post-mortem)
├── features/        # Feature development archives (lifecycle)
├── tools/           # User-facing tool guides (active)
├── envs.md          # Environment setup
├── README.md        # General documentation
└── FILE_ORGANIZATION.md  # This guide

data/backup/
└── scripts/         # Code backups (not in main scripts/)
```

---

## Three-Category System

### 1. `docs/fixes/` - Bug Fix Archives

**Purpose**: Record completed bug fixes with full context

**When to use**:
- You fixed a bug or integration issue
- The fix involved code changes + verification
- You want to preserve the problem analysis and solution

**Structure**:
```
docs/fixes/{issue-name}-YYYY-MM-DD/
├── PLAN.md          # Problem analysis, root cause, approach
├── COMPLETED.md     # Solution summary, verification results
├── verify.py        # Optional: validation script
└── README.md        # Optional: additional context
```

**Example**: `assembler-fix-2025-10-05/`
- **Problem**: Assembler not receiving LLM-enhanced insights from Consolidator
- **Files**: PLAN.md (analysis), COMPLETED.md (solution), verify.py (validation)
- **Backup**: Code backup moved to `data/backup/scripts/run_graph.py.assembler_fix`

**Update**: Add entry to `docs/fixes/README.md`

---

### 2. `docs/features/` - Feature Development Archives

**Purpose**: Document complete feature development lifecycle

**When to use**:
- You built a new tool or major feature
- Development took multiple iterations
- You have design docs, implementation notes, and reviews
- The feature is complete and merged

**Structure**:
```
docs/features/{feature-name}-YYYY-MM-DD/
├── PLAN.md              # Architecture design, requirements
├── IMPLEMENTATION.md    # What was built, key decisions
├── CODE_REVIEW.md       # Optional: review findings
└── *.backup            # Optional: reference code
```

**Example**: `gate8-debug-tool-2025-10-05/`
- **Feature**: Deep-inspection debug tool for LangGraph pipeline
- **Files**: PLAN.md (862 lines design), IMPLEMENTATION.md (summary), CODE_REVIEW.md (issues)
- **Usage Guide**: Separated to `docs/tools/gate8-debug.md` (active document)

**Update**: Add entry to `docs/features/README.md`

---

### 3. `docs/tools/` - User-Facing Tool Guides

**Purpose**: Active documentation for users to discover and use tools

**When to use**:
- You have a tool that users need to run
- The tool will be used repeatedly (not one-off)
- Users need quick reference for commands and options

**Structure**:
```
docs/tools/
├── README.md              # Tool catalog with quick links
└── {tool-name}.md         # Individual tool usage guide
```

**Template for tool guide**:
```markdown
# {Tool Name}

## Quick Start
[Basic command with common options]

## Purpose
[What it does in 1-2 sentences]

## Key Features
- Feature 1
- Feature 2

## Options
| Option | Description | Default |

## Output Files
[Where to find results]

## Common Use Cases
[3-5 practical scenarios]

## Troubleshooting
[Common errors and fixes]
```

**Example**: `gate8-debug.md`
- Quick start command
- All options explained
- Output files location
- Example workflows

**Update**: Add entry to `docs/tools/README.md`

---

## Decision Framework

### Question 1: Is this a bug fix or new feature?

| Type | → Directory |
|------|-------------|
| Bug fix (correcting wrong behavior) | `docs/fixes/` |
| New feature (adding capability) | `docs/features/` |

### Question 2: Is this documentation still "active"?

| Active? | Example | → Directory |
|---------|---------|-------------|
| Yes - users need it regularly | Usage guide, API reference | `docs/tools/` |
| No - historical reference only | Design doc, code review | `docs/features/` or `docs/fixes/` |

### Question 3: Where should backups go?

| File Type | → Location |
|-----------|------------|
| Code backups (`.py`, `.js`, etc.) | `data/backup/scripts/{file}.{fix-name}` |
| Config backups (`.yaml`, `.json`) | `data/backup/configs/` |
| Large binary backups | `data/backup/` |

**Never put backups in**:
- `scripts/` (keeps active code clean)
- Project root (keeps root tidy)
- `docs/` (unless it's a small reference file)

---

## Case Study 1: Assembler Fix (Oct 5, 2025)

### Initial State
```
.
├── ASSEMBLER_FIX_PLAN.md
├── ASSEMBLER_FIX_COMPLETED.md
├── scripts/verify_assembler_fix.py
└── scripts/run_graph.py.backup_assembler_fix
```

### Analysis
- **Type**: Bug fix (Consolidator → Assembler integration)
- **Documents**: PLAN (analysis) + COMPLETED (solution)
- **Verification**: verify.py script
- **Backup**: Original run_graph.py

### Decision Process
1. Is this a fix or feature? → **Fix** (correcting broken integration)
2. Do users need ongoing access? → **No** (historical record)
3. Where should verify.py go? → With fix archive (not in scripts/)
4. Where should backup go? → `data/backup/scripts/` (not in scripts/)

### Final Organization
```
docs/fixes/assembler-fix-2025-10-05/
├── PLAN.md           # ← ASSEMBLER_FIX_PLAN.md
├── COMPLETED.md      # ← ASSEMBLER_FIX_COMPLETED.md
└── verify.py         # ← scripts/verify_assembler_fix.py

data/backup/scripts/
└── run_graph.py.assembler_fix  # ← scripts/run_graph.py.backup_assembler_fix
```

### Rationale
- ✅ All fix-related files in one place
- ✅ `scripts/` only contains active code
- ✅ Backup clearly labeled and separated
- ✅ Establishes pattern for future fixes

---

## Case Study 2: Gate-8 Debug Tool (Oct 5, 2025)

### Initial State
```
.
├── GATE8_DEBUG_PLAN.md (29K - architecture)
├── GATE8_DEBUG_IMPLEMENTATION.md (1.6K - summary)
├── CODE_REVIEW_GATE8_DEBUG.md (5.1K - review)
└── GATE8_DEBUG_USAGE.md (6.4K - user guide)
```

### Analysis
- **Type**: New feature (debug tool)
- **Documents**: Full development lifecycle (PLAN → IMPL → REVIEW → USAGE)
- **Active vs Archive**: USAGE is active, others are historical
- **User need**: Developers will regularly use this tool

### Decision Process
1. Is this a fix or feature? → **Feature** (new capability)
2. Which docs are "active"? → **USAGE** (users need it), others are historical
3. Should USAGE stay with PLAN? → **No** (different audiences)
4. Create new category? → **Yes** (`docs/tools/` for user guides)

### Final Organization
```
docs/features/gate8-debug-tool-2025-10-05/
├── PLAN.md              # ← GATE8_DEBUG_PLAN.md (development history)
├── IMPLEMENTATION.md    # ← GATE8_DEBUG_IMPLEMENTATION.md
└── CODE_REVIEW.md       # ← CODE_REVIEW_GATE8_DEBUG.md

docs/tools/
└── gate8-debug.md       # ← GATE8_DEBUG_USAGE.md (active guide)
```

### Rationale
- ✅ Separates "active docs" (tools/) from "archive docs" (features/)
- ✅ Users find usage guide easily without digging through development history
- ✅ Development history preserved for future reference
- ✅ Establishes two-tier documentation: user-facing + developer-facing

---

## Naming Conventions

### Directory Names
```
{topic}-{date}/           # For time-sensitive archives
{tool-name}/              # For active documentation
```

**Examples**:
- ✅ `assembler-fix-2025-10-05/` (fix with date)
- ✅ `gate8-debug-tool-2025-10-05/` (feature with date)
- ❌ `fix-oct-5/` (too vague)
- ❌ `my-fix/` (no context)

### File Names
```
PLAN.md           # Problem analysis, design approach
COMPLETED.md      # Solution summary (for fixes)
IMPLEMENTATION.md # Build summary (for features)
CODE_REVIEW.md    # Review findings
verify.py         # Validation script (short name, not verify_{fix}_fix.py)
```

**Backup naming**:
```
{original-name}.{context}
```

**Examples**:
- ✅ `run_graph.py.assembler_fix`
- ✅ `router.heuristics.yaml.v1`
- ❌ `run_graph.py.backup` (no context)
- ❌ `old_run_graph.py` (ambiguous)

---

## File Cleanup Checklist

When you have finished a fix or feature and need to organize files:

### Step 1: Identify File Types
- [ ] Which files are planning/analysis docs?
- [ ] Which files are implementation summaries?
- [ ] Which files are user-facing guides?
- [ ] Which files are code backups?
- [ ] Which files are validation/verification scripts?

### Step 2: Determine Category
- [ ] Is this a **bug fix** or **new feature**?
- [ ] Are there docs that users need long-term? (→ tools/)
- [ ] Is this work complete? (If not, maybe keep in root temporarily)

### Step 3: Create Target Structure
```bash
# For fixes:
mkdir -p docs/fixes/{name}-YYYY-MM-DD
mkdir -p data/backup/scripts  # if needed

# For features:
mkdir -p docs/features/{name}-YYYY-MM-DD
mkdir -p docs/tools  # if user docs needed
```

### Step 4: Move Files
- [ ] Move PLAN-type docs to features/ or fixes/
- [ ] Move COMPLETED/IMPLEMENTATION to features/ or fixes/
- [ ] Move USAGE/guide to tools/ (if active) or features/fixes/ (if archive)
- [ ] Move verify scripts to features/ or fixes/ (not scripts/)
- [ ] Move backups to data/backup/

### Step 5: Update Indexes
- [ ] Add entry to `docs/fixes/README.md` (if fix)
- [ ] Add entry to `docs/features/README.md` (if feature)
- [ ] Add entry to `docs/tools/README.md` (if user guide)

### Step 6: Verify Cleanup
- [ ] Project root is clean (no leftover .md files)
- [ ] `scripts/` only has active code (no `.backup` files)
- [ ] All related files are in one archive directory
- [ ] User can find active documentation in `docs/tools/`

---

## Red Flags: Don't Do This

### ❌ Don't: Leave files in project root
```
.
├── MY_FIX_PLAN.md        # ← Users see this, it's clutter
├── debug_notes.md        # ← Lost after a week
└── old_code_backup.py    # ← No context
```

**Why**: Root should only have essential project files (README, CLAUDE.md, etc.)

### ❌ Don't: Put backups in scripts/
```
scripts/
├── run_graph.py
├── run_graph.py.backup           # ← Confusing
├── run_graph.py.old              # ← Which is newer?
└── run_graph.py.before_fix       # ← Clutters directory
```

**Why**: Active code directory should only have active code

### ❌ Don't: Mix active and archive docs in same directory
```
docs/tools/
├── gate8-debug.md                # Active guide ✓
└── gate8-debug-dev-history.md    # Archive ✗ (should be in features/)
```

**Why**: Users looking for usage guides shouldn't wade through development history

### ❌ Don't: Use vague names
```
docs/fixes/
└── my-fix-yesterday/             # ← What fix? When?
    ├── notes.md                  # ← What kind of notes?
    └── script.py                 # ← What does it do?
```

**Why**: Future you (or teammates) won't remember context

---

## Templates

### Template: Fix Archive

Create `docs/fixes/{issue}-YYYY-MM-DD/PLAN.md`:

```markdown
# {Issue Name} Fix Plan

## Problem Statement
[Describe the bug/issue]

## Root Cause Analysis
[Why did this happen?]

## Proposed Solution
[How will we fix it?]

## Implementation Steps
1. Step 1
2. Step 2

## Verification Plan
[How to verify the fix works]

## Risks and Mitigations
[What could go wrong?]
```

Create `docs/fixes/{issue}-YYYY-MM-DD/COMPLETED.md`:

```markdown
# {Issue Name} Fix - Completed

## Summary
[1-2 sentence summary]

## Changes Made
- File 1: Change description
- File 2: Change description

## Verification Results
[Test results, before/after comparison]

## Artifacts
- Modified: `path/to/file.py`
- Backup: `data/backup/scripts/file.py.{context}`
- Tests: Link to test results

## Status
✅ Fixed and verified

## Related
- Original issue: [link or description]
- Related commits: [commit hashes]
```

---

### Template: Feature Archive

Create `docs/features/{feature}-YYYY-MM-DD/PLAN.md`:

```markdown
# {Feature Name} - Design Plan

## Executive Summary
[1-paragraph overview]

## Problem Statement
[What problem does this solve?]

## Proposed Architecture
[Design, components, flow]

## Key Components
- Component 1: Purpose
- Component 2: Purpose

## Implementation Plan
1. Phase 1
2. Phase 2

## Success Criteria
[How do we know it's done?]
```

Create `docs/features/{feature}-YYYY-MM-DD/IMPLEMENTATION.md`:

```markdown
# {Feature Name} - Implementation Summary

## Summary
[What was built]

## Files Created/Modified
- `path/to/file.py` (NNN lines) - Purpose

## Key Features Implemented
1. Feature 1
2. Feature 2

## Usage
[Basic example]

## Status
✅ Implemented and working

## Known Limitations
[If any]

## Next Steps
[Future enhancements]
```

---

### Template: Tool Guide

Create `docs/tools/{tool-name}.md`:

```markdown
# {Tool Name}

## Quick Start

\`\`\`bash
conda run -n age python scripts/{tool}.py [options]
\`\`\`

## Purpose
[What this tool does]

## Key Features
- Feature 1
- Feature 2
- Feature 3

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--option1` | Description | value |

## Output Files

- `path/to/output.json` - Description
- `path/to/output.md` - Description

## Common Use Cases

### Use Case 1: {Title}
\`\`\`bash
[command]
\`\`\`
[Explanation]

## Troubleshooting

### Error: "message"
**Cause**: [Why this happens]
**Fix**: [How to resolve]

## Related Documentation
- Development history: `docs/features/{tool}-YYYY-MM-DD/`
- Main README: [link]
```

---

## Maintenance

### Weekly Check
- [ ] Are there loose `.md` files in project root?
- [ ] Are there `.backup` files in `scripts/`?
- [ ] Do new tools have documentation in `docs/tools/`?

### After Completing Work
- [ ] Run through cleanup checklist above
- [ ] Update relevant README.md indexes
- [ ] Verify all related files are archived together

### When Adding New Categories
If you find yourself creating many similar files that don't fit `fixes/`, `features/`, or `tools/`:

1. **Propose a new category** (e.g., `docs/experiments/`, `docs/benchmarks/`)
2. **Update this guide** with the new category's purpose and structure
3. **Create a README.md** for the new category with index and template

---

## Examples from This Project

### Well-Organized
✅ `docs/fixes/assembler-fix-2025-10-05/` - Clear fix archive
✅ `docs/features/gate8-debug-tool-2025-10-05/` - Complete feature lifecycle
✅ `docs/tools/gate8-debug.md` - Active user guide
✅ `data/backup/scripts/run_graph.py.assembler_fix` - Labeled backup

### Previously Problematic (now fixed)
❌ `ASSEMBLER_FIX_PLAN.md` in root → ✅ Moved to `docs/fixes/`
❌ `scripts/verify_assembler_fix.py` → ✅ Moved to `docs/fixes/` (verification, not active tool)
❌ `scripts/run_graph.py.backup_assembler_fix` → ✅ Moved to `data/backup/scripts/`
❌ `GATE8_DEBUG_USAGE.md` in root → ✅ Moved to `docs/tools/`

---

## Summary: The Three Rules

1. **Separate active from archive**: Users need `docs/tools/`, developers need `docs/features/` and `docs/fixes/`

2. **Keep code directories clean**: `scripts/` = active code only, backups go to `data/backup/`

3. **Archive together**: All files for one fix/feature go in one dated directory

---

**Last Updated**: 2025-10-05
**Established By**: File organization sessions for assembler-fix and gate8-debug-tool
