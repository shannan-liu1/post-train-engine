# ADR-0002: Sealed Evaluation and Strict Promotion

Status: accepted

## Context

The manual RunPod path currently selects checkpoints and promotes the selected checkpoint on the same examples. Multiple promotion implementations enforce different gates.

## Decision

Selection and promotion must use disjoint examples. EvalSuites assign explicit roles: `selection`, `diagnostic`, `promotion`, `canary`, and `unseen`.

The strict paired PromotionDecision is the only acceptance authority. It must evaluate comparability, primary improvement, uncertainty, protected regressions, parser behavior, token and cost budgets, verifier integrity, leakage, and artifact integrity. Missing required evidence rejects the Candidate.

Promotion disclosure defaults to aggregate results. Diagnostic suites may expose per-example failures. Campaign state records promotion-suite exposure and requires rotation with a bridge evaluation when the exposure budget expires.

## Consequences

- Checkpoint selection can no longer reuse promotion rows.
- API and RunPod promotion helpers will be removed.
- Weighted objectives remain diagnostic.
