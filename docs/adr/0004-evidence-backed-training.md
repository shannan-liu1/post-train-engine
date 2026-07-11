# ADR-0004: Evidence-Backed TrainingViews

Status: accepted

## Context

The local smoke path creates traces and TrainingViews, while real GRPO training consumes raw task rows and records incomplete policy provenance.

## Decision

Every method adapter accepts a validated TrainingView. Raw task records cannot cross the training seam. Every TrainingView records source Trace identifiers, allowed SplitRoles, selection policy, method compatibility, and privileged visibility.

Every generated Trace records policy checkpoint, policy version or optimizer step, rollout group, sampling contract, verifier contract, reward contract, token evidence, and cost evidence when available.

Replay is a query over the evidence substrate, not a competing raw storage format.

## Consequences

- The real GRPO path will consume the same evidence contract as local tests.
- Stale or ungrouped policy-gradient evidence can fail closed.
- Learning lineage can trace a promoted Candidate to its source evidence.
