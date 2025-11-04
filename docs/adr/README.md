# Architecture Decision Records (ADR)

This directory contains architecture decisions made for this project.

## Format

Each ADR consists of two parts:
- **`adr/NNN-title.md`** - Concise decision record (1 page)
- **`../NNN-title-analysis.md`** - Detailed technical analysis (in docs/ root)

## Index

| ADR | Title | Date | Status |
|-----|-------|------|--------|
| [001](./001-railway-deployment.md) | Use Railway.app for Backend Deployment | 2025-11-04 | Accepted |

## ADR Template

Use this structure for new ADRs:

```markdown
# ADR-NNN: [Title]

**Date**: YYYY-MM-DD
**Status**: [Proposed | Accepted | Deprecated | Superseded]
**Technical Story**: [Brief context]

## Context
[What is the issue we're facing?]

## Decision
[What did we decide?]

## Rationale
[Why did we choose this?]

## Consequences
### Positive
### Negative

## Alternatives Considered
[What other options did we evaluate?]

## References
[Links to detailed analysis and related documents]
```

## Status Definitions

- **Proposed**: Decision under discussion
- **Accepted**: Decision approved and implemented
- **Deprecated**: No longer relevant
- **Superseded**: Replaced by another ADR

## Naming Convention

- ADR files: `adr/NNN-short-title.md`
- Analysis files: `../NNN-short-title-analysis.md`
- NNN: Zero-padded 3-digit number (001, 002, 003...)
