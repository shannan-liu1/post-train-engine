# ADR-0005: Local Campaign Control Plane First

Status: accepted

## Context

One researcher is the primary operator, but future work may involve many cooperating agents. JSONL registries do not provide leases, uniqueness, atomic Incumbent updates, or efficient campaign queries.

## Decision

Immutable evidence remains in portable run bundles. A local SQLite campaign index will store hypotheses, proposals, comparison groups, budgets, work leases, Run state, and the current Incumbent reference.

Only an atomic promotion transaction may update the Incumbent. Proposal hashes prevent accidental duplicate experiments. Work leases coordinate agents. External databases and object stores remain future adapters.

## Consequences

- The first implementation remains operable by one researcher.
- Multi-agent collaboration gains atomic claims without requiring distributed infrastructure.
- Remote scale can replace adapters without changing campaign semantics.
