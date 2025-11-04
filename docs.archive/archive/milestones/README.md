# Project Milestones

This directory records major project milestones, sign-offs, and validation checkpoints that demonstrate system readiness at specific points in time.

## Philosophy

Unlike `docs/features/` (which tracks individual feature development) or `docs/fixes/` (which tracks bug corrections), **milestones** capture complete system state validation across multiple components.

**When to create a milestone record**:
- ✅ You validated the entire pipeline end-to-end
- ✅ Multiple quality gates passed simultaneously
- ✅ System achieved a notable state of completeness
- ✅ You need to prove system readiness at a specific date
- ✅ External sign-off or approval was obtained

---

## Milestone Index

### 2025-09-08: Day-1 Sign-off (Gates G01-G08)

**File**: [`day1-signoff-2025-09-08.md`](day1-signoff-2025-09-08.md)

**Status**: ✅ All gates PASS

**Summary**: Complete pipeline validation from data collection through day-1 sign-off, covering all 8 quality gates.

**Key Results**:
- **G01-G08**: All gates passed
- **Coverage**: 97 documents with publish dates (target: ≥80)
- **Duplicate ratio**: 0.0583 (target: ≤0.15)
- **Link health**: 100% (target: 100%)
- **Required fields**: 98.97% presence (target: ≥98%)
- **PR recency**: 73 docs since 2024, 67 in last 12 months

**Artifacts**:
- Inventory CSV: `data/final/inventory/salesforce_inventory.csv` (98 docs)
- Eval seed: `data/interim/eval/salesforce_eval_seed.jsonl` (26.9 KB)
- Gate reports: `reports/qa/gate0{1-8}_*.json`
- Sign-off log: `logs/signoff/20250908_004557.log`

**Reproducibility**:
```bash
python3 scripts/build_inventory_csv.py
python3 scripts/verify_day1_milestones.py
python3 scripts/link_health_check.py && python3 scripts/qa_verify_link_health.py
python3 scripts/qa_verify_day1_signoff.py
```

**Related Work**:
- Initial data pipeline established
- Quality gates G01-G08 implemented
- Evaluation seed generated for downstream testing

---

## Future Milestones (Template)

### YYYY-MM-DD: {Milestone Name}

**Status**: 🔄 In Progress / ✅ Complete / ⚠️ Partial

**Summary**: [1-2 sentence description]

**Key Results**:
- Metric 1: value (target)
- Metric 2: value (target)

**Artifacts**:
- File 1: path
- File 2: path

**Reproducibility**:
```bash
command 1
command 2
```

---

## Milestone vs Feature vs Fix

| Aspect | Milestone | Feature | Fix |
|--------|-----------|---------|-----|
| **Scope** | System-wide validation | Single new capability | Bug correction |
| **Gates** | Multiple gates at once | May trigger 1-2 gates | Usually 0-1 gates |
| **Audience** | Stakeholders, PMs | Developers, users | Developers, maintainers |
| **Frequency** | Monthly or quarterly | Weekly | As needed |
| **Example** | "Day-1 Sign-off" | "Gate-8 Debug Tool" | "Assembler fix" |

---

## Adding New Milestones

When documenting a new milestone:

1. **Create milestone document**:
   ```
   docs/milestones/{name}-YYYY-MM-DD.md
   ```

2. **Required sections**:
   - Status (emoji + text)
   - Summary (executive overview)
   - Key Results (metrics with targets)
   - Artifacts (outputs, reports, logs)
   - Reproducibility (commands to re-verify)
   - Related Work (links to features/fixes involved)

3. **Update this README.md** with:
   - Date and milestone name
   - Status badge
   - Key results summary
   - Link to full document

4. **Cross-reference**:
   - Link from related features: "Part of milestone: X"
   - Link from related fixes: "Enabled milestone: X"

---

## Milestone Naming Convention

```
{descriptive-name}-YYYY-MM-DD.md
```

**Examples**:
- ✅ `day1-signoff-2025-09-08.md` (clear purpose + date)
- ✅ `production-readiness-2025-11-01.md`
- ✅ `v1.0-release-2025-12-15.md`
- ❌ `milestone1.md` (no context)
- ❌ `sept-validation.md` (ambiguous date)

---

## Suggested Future Milestones

### Production Readiness
- All gates passing with production thresholds
- Gate-8 quality score ≥90/100 consistently
- LLM token costs within budget
- Response latency p95 < 5s

### V1.0 Release
- Complete API documentation
- User guides for all tools
- CI/CD pipeline configured
- Monitoring and alerting enabled

### Scale Validation
- 1000+ documents indexed
- Multi-company support verified
- Performance benchmarks documented

---

**Last Updated**: 2025-10-05
**Total Milestones**: 1
