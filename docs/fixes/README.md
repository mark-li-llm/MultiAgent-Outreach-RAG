# Fix Archive

This directory contains historical bug fixes, feature integrations, and debugging sessions with complete documentation for traceability.

## Index

### 2025-10-05: Assembler-Consolidator Integration Fix

**Directory**: `assembler-fix-2025-10-05/`

**Problem**: Assembler node was not receiving LLM-enhanced insight cards from Consolidator, resulting in low-quality email generation.

**Solution**:
- Modified Consolidator to use ChatOpenAI for persona-aware enhancement
- Added LLM-based fields: `persona_relevance`, `metric_impact`, `action_suggestion`
- Verified end-to-end flow from retrieval → consolidation → email generation

**Files**:
- `PLAN.md` - Initial analysis and fix plan
- `COMPLETED.md` - Implementation summary and verification results
- `verify.py` - Validation script for the fix
- `data/backup/scripts/run_graph.py.assembler_fix` - Backup of modified file

**Status**: ✅ Completed and verified

**Related Commits**: (Add git commit hash if applicable)

---

## Adding New Fixes

When documenting a new fix, create a new directory with format:

```
fixes/
└── {component}-{issue}-YYYY-MM-DD/
    ├── PLAN.md          # Problem analysis and approach
    ├── COMPLETED.md     # Results and verification
    ├── verify.py        # Optional: validation script
    └── *.backup         # Optional: backups (or put in data/backup/)
```

Then update this README.md with a new entry in the Index section.
