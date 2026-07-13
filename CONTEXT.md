# Post-Train Engine Domain Context

This file defines the canonical language for the post-training research flywheel.

## Core concepts

**Campaign**
: A bounded research program that starts from one incumbent and accumulates hypotheses, experiments, evidence, costs, and promotion decisions.

**Hypothesis**
: A falsifiable claim about a data, method, evaluation, or runtime change and the evidence that would support or reject it.

**ExperimentProposal**
: An immutable request to test one hypothesis against a named parent candidate under a fixed comparison contract and budget.

**ComparisonGroup**
: A set of experiments that share every contract field except the declared ablation axes.

**RunPlan**
: The fully resolved, secret-redacted, immutable execution contract produced by the RunEngine `resolve` phase from an experiment specification.

**Run**
: One attempt to execute a RunPlan. A Run ends in `failed`, `rejected`, or `promoted`; interrupted work remains `running` until resumed or explicitly failed.

**Candidate**
: A checkpoint or model reference produced by a Run and eligible for evaluation. A Candidate does not become an Incumbent without a PromotionDecision.

**Incumbent**
: The only promoted Candidate at the head of a Campaign lineage.

**DatasetSnapshot**
: An immutable, revision-pinned set of normalized examples with stable identifiers, content hashes, and assigned SplitRoles.

**SplitRole**
: One of `train`, `selection`, `diagnostic`, `promotion`, `canary`, or `unseen`. Training may consume only roles explicitly allowed by the method contract. Selection may never consume promotion or canary examples.

**Trace**
: Append-only evidence for one model interaction, including policy identity, prompt, completion or action spans, parser output, verifier output, reward evidence, token usage, cost, and source artifacts.

**RolloutGroup**
: A view over sibling Trace identifiers sampled for one example under one policy and sampling contract.

**TrainingView**
: An immutable, method-specific projection over eligible Trace evidence. Every training update must name its TrainingView.

**EvalSuite**
: A versioned collection of examples, metrics, slices, protected functions, verifier contracts, and disclosure rules for one evaluation role.

**PromotionDecision**
: The deterministic, fail-closed decision that compares one Candidate with the Incumbent under a sealed promotion contract.

**RunManifest**
: The canonical provenance envelope for a Run, containing resolved identities, artifact hashes, state, lineage, evaluation evidence, costs, and PromotionDecision references.

**Lifecycle**
: The ordered process that persists evidence, validates artifacts, uploads required data, promotes or rejects the Candidate, and safely prunes checkpoint bytes.

**CostRecord**
: Measured resource use and attributed cost for one Run phase.

## Invariants

1. `pte run --config <path>` is the only mutable execution entry point.
2. RunEngine owns consequential input resolution before it freezes the RunPlan.
3. Compatibility commands delegate to the same RunEngine implementation.
4. Compute, task, and method adapters never decide promotion.
5. Promotion examples never enter training, selection, prompts, teacher context, or replay.
6. Checkpoint selection and promotion use disjoint examples.
7. Every TrainingView preserves source Trace identifiers and SplitRoles.
8. Every Trace records the policy version that generated it.
9. Weighted objectives diagnose and rank. The strict PromotionDecision owns acceptance.
10. Missing required evidence fails closed.
11. Every paid Run records measured cost or an explicit non-certifying missing-cost state.
12. Only an atomic promotion transaction may replace the Campaign Incumbent.
13. Rejected, failed, and negative results remain durable campaign evidence.
