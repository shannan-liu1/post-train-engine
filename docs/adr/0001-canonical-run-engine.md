# ADR-0001: One Canonical RunEngine

Status: accepted

## Context

The repository currently contains API-first, local-smoke, generic climb, and manual RunPod orchestration implementations. They duplicate configuration, evaluation, promotion, reporting, and manifest behavior.

## Decision

The repository will expose one mutable execution entry point: `pte run --config <path>`. A canonical RunEngine will own stage order, state transitions, artifact finalization, evaluation, promotion, lifecycle, and campaign updates.

Task, method, provider, and compute variation will enter through adapters. Adapters may execute work and return typed evidence. They may not decide promotion, mutate the Incumbent, or write an alternative final manifest.

Compatibility commands may remain temporarily, but they must compile their input into the canonical RunPlan and delegate to RunEngine.

## Consequences

- The implementation will delete duplicate orchestration after compatibility migration.
- Every execution mode will receive the same validation and promotion law.
- Adapter contract tests will replace path-specific orchestration tests.
