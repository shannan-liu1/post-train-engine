# Canonical Research Flywheel Specification

Status: foundation implemented and locally tested; adversarial review reopened production certification gates

Owner: repository user

Tracking authority: this document

Last updated: 2026-07-12

## Problem Statement

The repository contains many strong post-training primitives, but its real execution surfaces do not yet form one canonical research flywheel. Separate paths define different configuration, data, evaluation, promotion, artifact, and lifecycle behavior. This duplicity weakens scientific validity, traceability, cost accounting, and extensibility.

The engine must continuously select sufficiently learnable evidence, choose an appropriate post-training mechanism, train Candidates, evaluate them without leakage, reject important regressions, promote only proven improvements, preserve reasoning and learning lineage, measure costs, and use every result to choose the next experiment.

The first operator is one researcher. The design must remain safe and legible when many independent agents propose, execute, ablate, compare, and combine research directions.

## Solution

Build one evidence-driven RunEngine and one campaign control plane. The RunEngine compiles every experiment into a canonical RunPlan, executes a fixed sequence of typed stages, and finalizes one portable RunManifest. Task, method, provider, and compute behavior vary through adapters. Evaluation and promotion remain centralized and fail closed.

The campaign control plane records hypotheses, proposals, comparison groups, budgets, leases, outcomes, and the Incumbent. It treats rejected and failed Runs as evidence. A deterministic planner recommends the next experiment from observed failure modes, uncertainty, data value, regression risk, runtime, and cost.

## User Stories

1. As a researcher, I want one execution command, so that every experiment follows the same scientific law.
2. As a researcher, I want every config compiled into one RunPlan, so that backends cannot silently change semantics.
3. As a researcher, I want checkpoint selection separated from promotion, so that model selection does not bias promotion evidence.
4. As a researcher, I want sealed promotion and canary suites, so that important capabilities cannot regress unnoticed.
5. As a researcher, I want promotion-suite exposure tracked, so that repeated adaptive evaluation cannot masquerade as fresh evidence.
6. As a researcher, I want every training example linked to source Traces, so that I can reconstruct why a Candidate learned.
7. As a researcher, I want every Trace linked to the policy version that generated it, so that on-policy and stale evidence remain distinguishable.
8. As a researcher, I want sufficiently learnable data selected by evidence, so that training avoids already-solved and zero-signal examples.
9. As a researcher, I want protected replay mixed into training, so that new learning does not erase important capabilities.
10. As a researcher, I want method choice driven by observed signal, so that the engine chooses SFT, preference training, RLVR, distillation, evaluation repair, or runtime work for a reason.
11. As a researcher, I want verifier quality recorded and tested, so that reward hacking cannot appear as capability gain.
12. As a researcher, I want portable Run bundles, so that results remain inspectable after moving machines or storage locations.
13. As a researcher, I want exact code, data, model, verifier, and environment identity, so that a Run can be reproduced honestly.
14. As a researcher, I want phase-level runtime and cost records, so that I can optimize quality, speed, and cost together.
15. As a researcher, I want rejected and failed Runs retained, so that the research loop learns from negative evidence.
16. As a researcher, I want atomic promotion, so that no concurrent worker can replace the Incumbent incorrectly.
17. As a researcher, I want resumable stages, so that infrastructure failure does not destroy paid work or duplicate side effects.
18. As a researcher, I want one report and diagnostics interface for every backend, so that operations do not depend on manifest variants.
19. As an agent, I want immutable experiment proposals and claim leases, so that many agents can collaborate without duplicate execution.
20. As an agent, I want comparison-group hashes, so that ablations change only declared axes.
21. As an agent, I want aggregate sealed-promotion feedback, so that I cannot overfit to hidden examples.
22. As an agent, I want explicit budgets and stop rules, so that exploration remains bounded and falsifiable.
23. As an agent, I want orthogonal successful mechanisms combined through a new experiment, so that informal merging cannot bypass evaluation.
24. As a maintainer, I want adapter contract tests, so that new tasks, methods, and compute frameworks cannot create alternate engine behavior.
25. As a maintainer, I want schema migrations, so that durable evidence remains readable as the engine evolves.

## Outcome Hypotheses

1. If every execution mode delegates to one RunEngine, then semantic drift should decrease because orchestration, evaluation, promotion, and finalization have one implementation. We will evaluate this with contract tests that run local fake, API, and RunPod adapters through the same public interface and compare manifest and decision schemas.
2. If checkpoint selection and promotion use disjoint suites, then promotion estimates should become more trustworthy because selection no longer optimizes the reported holdout. We will evaluate this with tests that reject overlapping roles and with campaign exposure records.
3. If method adapters require TrainingViews, then learning lineage should become complete because raw datasets cannot bypass evidence selection. We will evaluate this by tracing every trained Candidate back to source Trace identifiers and SplitRoles.
4. If Run bundles use relative content-addressed references and honest source identity, then reproducibility should improve because bundles survive relocation and identify dirty code. We will evaluate this by relocating bundles and reproducing validation results.
5. If campaign state records negative results and comparison contracts, then next-experiment quality should improve because planners can avoid repeated failures and invalid comparisons. We will evaluate this with deterministic recommendation tests and duplicate-proposal rejection.
6. If phase-level costs and useful-evidence yield are measured, then runtime and cost efficiency should improve because optimization targets measured bottlenecks. We will evaluate this with before-and-after equivalent-output benchmarks.
7. If agent work uses leases and atomic promotion, then parallel research throughput should improve without corrupting the Incumbent. We will evaluate this with concurrent claim and promotion tests.

## Implementation Decisions

- One RunEngine owns the mutable execution path.
- One versioned RunManifest describes every backend.
- One strict paired PromotionDecision owns acceptance.
- Selection, diagnostic, promotion, canary, and unseen evaluation roles remain explicit and disjoint where required.
- Method adapters accept TrainingViews, not raw task datasets.
- Evidence records policy, verifier, reward, sampling, token, cost, and parent identities.
- Run bundles are portable and content-addressed.
- Local SQLite indexes campaign state; immutable artifacts remain files.
- Synchronous correctness precedes asynchronous or multi-node optimization.
- Cost begins as a hard budget and Pareto diagnostic. Cost cannot compensate for protected regressions.
- Local evidence remains authoritative. External experiment trackers may mirror it.
- A second executable-verifier task will prove task extensibility before broad plugin work.

## Working And Epistemic Principles

These rules constrain implementation and experiment selection. They are decision
rules, not slogans.

1. Name the owner and evidence for every requirement. Delete requirements that do
   not protect truth, money, safety, or recoverability.
2. Delete before simplifying. Simplify before accelerating. Accelerate before
   automating. A deleted stage is the cheapest and most reliable stage.
3. Start from provider and machine constraints, not inherited repository shapes.
   Never preserve an interface solely because it already exists.
4. Every ExperimentProposal states a problem, a causal explanation, the details
   that make that explanation hard to vary, a risky prediction, and a criticism
   that would reject it.
5. Evidence criticizes competing explanations. A passing metric does not prove a
   mechanism; promotion requires the declared prediction and all regression gates.
6. Negative results update the explanation and planner. They do not trigger ad hoc
   metric, split, or hypothesis changes.
7. Prefer one decisive experiment over many weak experiments. Optimize expected
   information gain per dollar only after validity constraints pass.
8. Automate only a demonstrated, stable loop. Human or agent review remains at
   boundaries where the provider cannot supply idempotency, authoritative cost, or
   sealed-evaluation isolation.

## Testing Decisions

- Tests exercise public interfaces and complete vertical slices.
- Every behavior change follows one RED to GREEN cycle.
- Adapter contract tests verify identical engine semantics across compute modes.
- Evaluation tests cover split leakage, adaptive exposure, uncertainty, protected regressions, verifier integrity, and missing evidence.
- Artifact tests cover relocation, hash mismatch, schema migration, interrupted finalization, and idempotent resume.
- Evidence tests cover policy version, rollout groups, TrainingView lineage, privileged visibility, and illegal SplitRoles.
- Campaign tests cover duplicate proposals, budgets, leases, atomic promotion, negative results, and deterministic recommendations.
- Runtime changes require output-equivalence and promotion-equivalence tests before performance claims.
- Full repository tests, Ruff, and diff checks gate each milestone.

## Acceptance Criteria

### Foundation: findings 1 through 4

- [x] F1. One RunManifest and RunBundle reader cover local, API, and RunPod Runs.
- [x] F2. Run bundles validate after relocation.
- [ ] F3. Source identity distinguishes every clean, tracked-dirty, untracked, staged, and mixed source state.
- [ ] F4. Dataset and model revisions reflect resolved inputs across every compiler and backend.
- [x] F5. Selection and promotion examples are disjoint and role-validated.
- [x] F6. One strict PromotionDecision handles every execution mode.
- [ ] F7. RunPod promotion requires an immutable evaluation contract and all configured strict evidence.
- [x] F8. Every execution mode validates its final bundle before returning success.
- [ ] F9. One RunEngine owns stage order, state transitions, and all consequential adapter work.
- [x] F10. Compatibility commands delegate to RunEngine.
- [x] F11. Method adapters cannot mutate the Incumbent or decide promotion.
- [x] F12. Real GRPO consumes a TrainingView and records policy-versioned Trace lineage.
- [ ] F13. Promotion examples cannot enter any TrainingView, replay query, or teacher context.
- [ ] F14. Interrupted stages and finalization resume idempotently without duplicate provider or training side effects.
- [x] F15. Every production TrainingView uses measured learnability or an explicit non-training outcome; configured partition membership alone is insufficient.

### Campaign and next-experiment loop

- [x] C1. Campaign state stores hypotheses, proposals, comparison groups, budgets, Runs, and Incumbent lineage.
- [x] C2. Proposal hashes detect accidental duplicates.
- [x] C3. Renewable fenced work leases prevent duplicate concurrent execution and stale-worker commits.
- [x] C4. Atomic promotion prevents conflicting Incumbent updates.
- [x] C5. Failed and rejected Runs remain queryable evidence.
- [ ] C6. Production compilers use Run evidence to select and configure the next experiment mechanism.
- [ ] C7. Promotion-suite exposure and rotation remain durable and fail closed across every production Run.

### Cost, runtime, and scale

- [ ] R1. Paid Run phases record provider-derived resource use and cost or an explicit non-certifying missing state.
- [x] R2. Reports show cost per Candidate, useful Trace, evaluation, and promoted metric gain.
- [ ] R3. Production baseline evaluation caching uses exact model, suite, generation, and verifier contract hashes.
- [ ] R4. Batching and model reuse improve measured runtime without output drift.
- [x] R5. Distributed plans distinguish modeled topology from executed topology.
- [ ] R6. Production asynchronous execution records and enforces policy-staleness limits before use.

### Extensibility and collaboration

- [x] E1. A second executable-verifier task runs through the same RunEngine.
- [ ] E2. Executable SFT, DPO, and GRPO paths use the same production method seam and evidence law.
- [ ] E3. Executable OPD or OPSD paths enter only through explicit evidence requirements and OOD guards.
- [ ] E4. Multiple agents use renewable fenced leases to claim distinct proposals and submit comparable results safely.
- [ ] E5. Production composition turns successful mechanisms into a new controlled experiment without bypassing evaluation.

## Unknown-Unknown Audit Protocol

Before each implementation slice, inspect:

1. Semantic leakage beyond exact example overlap.
2. Adaptive holdout exposure and disclosure.
3. Synthetic-data ancestry and correlated errors.
4. Protected capabilities not represented by current slices.
5. Privileged information visible to policies, teachers, verifiers, or agents.
6. Policy staleness, rollout grouping, and distributed ordering.
7. Partial failure, retry, and duplicate side effects.
8. Concurrent claims and Incumbent races.
9. Cost attribution gaps and performance changes that alter statistics.
10. Data rights, secret handling, model licensing, and remote retention.

Record each discovered risk in the register below. Fix it in the current slice when it threatens correctness. Otherwise add a falsifiable follow-up with an owner phase.

## Unknown-Unknown Register

| ID | Discovery | Impact | Decision or test | Status |
| --- | --- | --- | --- | --- |
| U-001 | Detailed promotion failures can leak sealed-suite content without exact row access. | Adaptive overfitting | Expose aggregate promotion results; keep detailed failures in diagnostic suites. | accepted |
| U-002 | Synthetic evidence can amplify correlated generator errors. | False learning signal | No synthetic generator adapter exists. Require generator lineage, verifier contract, diversity, and filtering before adding one. | deferred until generator adapter |
| U-003 | Runtime changes can alter the sampled training distribution. | Invalid performance comparison | Require output and promotion equivalence before accepting runtime optimizations; `scripts/benchmark_runpod_eval.py` fails on any output drift. | implemented locally; remote timing pending |
| U-004 | Independent successful weights or adapters may not compose. | Regression after merge | Treat every composition method as a new Candidate and controlled experiment. | accepted |
| U-005 | A Git SHA can appear valid while the executed source is almost entirely untracked. | False reproducibility | Add dirty and untracked source digests to source identity. | foundation |
| U-006 | Policy-changing GRPO traces currently record only the base checkpoint. | Invalid on-policy lineage | Add policy version and optimizer-step evidence to Trace records. | foundation |
| U-007 | Exact ID overlap misses sub-document and paraphrase contamination. | Inflated promotion evidence | Require a content-separation certificate with configurable n-gram similarity bounds. | implemented |
| U-008 | Counting promotion-suite exposures does not itself preserve adaptive validity. | Holdout overfitting | Enforce a maximum exposure limit and require suite rotation before another Candidate. | implemented |
| U-009 | Direct preference methods can over-optimize their implicit proxy despite avoiding a separate reward model. | False quality improvement | Require an independent promotion verifier for learned or heuristic training rewards. | implemented |
| U-010 | Test temp links and files can race with source identity capture. | Run failure or dishonest dirty-state evidence | Hash untracked files, links, directories, and disappeared paths with distinct type markers. | implemented |
| U-011 | A RunPod TrainingView that selects only a configured train partition provides no model-derived learnability evidence. | Wasteful or zero-signal training | Probe the parent policy, select a measured frontier, and emit a non-training result when no learnable rows exist. | implemented locally |
| U-012 | Output-equivalent batching is locally proven but its wall-clock benefit depends on GPU memory, padding waste, and generation-length variance. | False runtime optimization claim | Keep R4 open until the same RunPod topology shows lower measured phase time with identical outputs. | runtime |
| U-013 | A stage can finish append-only side effects and crash before its receipt becomes durable. | Duplicate Traces or control records after resume | Require stage outputs to overwrite deterministic artifacts or use an atomic commit protocol; fault-inject missing receipts and verify stable cardinality. | partial; deterministic local upserts implemented, provider transaction open in U-032 |
| U-014 | The legacy API `overwrite` option can destroy a finalized Run under the same Run identifier. | Loss of durable positive or negative evidence | Finalized Runs are immutable and idempotent even when a compatibility config says overwrite; a new attempt requires a new Run identifier. | implemented |
| U-015 | API prompt adaptation consumed raw rollout rows outside the TrainingView seam. | Incomplete learning lineage and partition-only training | Persist policy-versioned Traces and RolloutGroups, select a measured success-rate frontier, pass a validated TrainingView, or record an explicit non-training outcome. | implemented |
| U-016 | Non-main RunPod ranks returned after checkpoint selection, before the selected checkpoint's distributed promotion evaluation. | Collective deadlock or incomplete promotion evidence | Add an engine coordinator seam; every rank executes all non-promotion stages while rank zero alone persists intents, receipts, decisions, and the manifest. | partial; stage coordination implemented, post-stage persistence fault open in U-036 |
| U-017 | A remote lifecycle upload can succeed immediately before the local completion receipt is persisted. | Duplicate remote commits after process loss | Persist a started transaction journal before upload, complete it after the wrapper receipt, and fail closed on an ambiguous rerun until provider reconciliation. | implemented; provider exactly-once remains external |
| U-018 | Retired `pte train`, `ClimbOrchestrator`, and `HillClimber` surfaces could create Candidates or Incumbents under shadow laws. | Semantic drift and duplicate acceptance paths | Remove the command, modules, exports, and legacy tests after local, API, and RunPod compatibility migration. | implemented |
| U-019 | A configured cost budget treated an unknown phase cost as zero. | Unbounded paid execution under a false certificate | Make missing phase cost fail the Run when a hard budget is configured; record cost-certifying state and missing phases otherwise. | implemented |
| U-020 | One distributed rank could fail before a barrier while peers continued. | Deadlock and incomplete evidence | Exchange rank-local errors before every receipt barrier and propagate one distributed stage failure to every rank. | implemented |
| U-021 | A terminal stage failure left no readable report, and replay tried to open a nonexistent final report. | Negative evidence loss and misleading command success | Persist a valid failed Run bundle, expose failure reports and diagnostics, and make command replay fail with the recorded stage and evidence path. | implemented |
| U-022 | Legacy promotion policies, a standalone promotion command, and `ExperimentEvidenceRecord` exposed shadow acceptance and provenance authorities. | Multiple scientific laws despite a canonical engine | Remove the unused authorities and enforce their absence with architecture contract tests. | implemented |
| U-023 | Campaign outcome, Incumbent mutation, and budget reservation used independent public methods. | Promoted outcomes without CAS, CAS without Run evidence, and lease-budget races | Route outcome, exposure, and Incumbent mutation through `finalize_run`; enforce quality and cost budgets atomically when a proposal lease is claimed. | implemented |
| U-024 | TrainingViews embedded absolute data paths. | Learning lineage stopped resolving after Run relocation | Write new TrainingView data references relative to the Run root and keep one reader for legacy absolute references. | implemented |
| U-025 | Campaign outcomes and proposal parents were not cross-checked against status, decision, and expected Incumbent lineage. | Inconsistent campaign history or promotion against the wrong parent | Validate outcome pairs, RunPlan parent binding, and proposal parent binding before campaign finalization. | implemented |
| U-026 | A CUDA 12.8 image was allocated on a CUDA 12.4 host because the create request omitted the CUDA filter. | Pod startup failure, wasted spend, and false SSH-timeout diagnosis | Derive `execution.cuda_version` from each pinned image tag and pass it as `allowedCudaVersions` on every create request; reject images whose requirement cannot be parsed. | implemented |
| U-027 | RunPod catalog stock and single-GPU UI availability did not guarantee that two GPUs existed on one host. | Repeated failed allocations and misleading capacity assumptions | Query and create with the exact GPU count, trust the create response rate, and preserve the two-GPU evidence requirement instead of weakening it. | accepted provider constraint |
| U-028 | The local execution sandbox prohibited uploading the audited private workspace bundle even after explicit user approval. | A healthy paid Pod cannot execute the repository benchmark from this environment. | Check transfer capability before allocation; use exact Git commit mode only after a reviewed push, or use a user-supplied prebuilt image or checkout. Never bypass the policy. | delivery decision deferred to later user instruction |
| U-029 | Promotion compared candidate-produced artifacts without binding them to one immutable suite, protected-row content, prompt, verifier, generation, and RunPlan contract. | A candidate or adapter could change the evaluation law or substitute rows and still satisfy numeric gates. | Add a content-addressed EvalContract to RunPlan; bind protected IDs and content; require both paired artifacts, row membership, and primary metric to match it before `decide_promotion`. | implemented locally |
| U-030 | Final Run bundles exposed per-example promotion rows and failure annotations through normal artifact surfaces. | Researchers or agents could adapt to sealed holdout content despite split separation. | Mark promotion artifacts sealed, require explicit sealed access for internal certification, redact their status paths, and omit them from normal reports. | implemented locally |
| U-031 | Content and verifier separation checks existed as isolated helpers and tests, while replay records lacked a canonical role-aware query path. | Training could bypass leakage checks through a non-TrainingView evidence path. | Require validated content and verifier separation certificates in every RunPlan and persisted manifest; expose only train, probe, and replay traces through the canonical training query. | implemented locally |
| U-032 | Stage receipts referenced mutable paths rather than immutable output hashes, and provider side effects can complete before receipt durability. | Resume could accept changed outputs or duplicate paid work. | Stage receipts bind output bytes; provider operations persist intent, reconcile replay-safe jobs, and fail ambiguous non-replayable jobs. | implemented locally |
| U-033 | Campaign, budget, exposure, and next-experiment fields were optional primitives that production compilers did not consistently populate. | Runs could bypass the research flywheel while appearing canonical. | Require explicit non-certifying smoke mode or a complete fenced campaign binding; make smoke decisions reject promotion. | implemented locally |
| U-034 | Work leases had no renewal heartbeat or fencing token, and finalization did not recheck measured cost against the reserved budget. | A stale worker could commit after lease expiry or overspend a campaign reservation. | Add renewable monotonic fencing, actual-cost rechecks, and two-phase provider billing settlement. | implemented locally |
| U-035 | RunPod cost evidence used configured heuristics rather than provider receipts, and provider rates may already represent the total Pod rate. | Cost reports could double-count GPUs or certify an incorrect spend. | Use the create-response total Pod rate for deadlines; keep campaign promotion pending until a hashed billing receipt settles actual total cost. | control plane implemented locally; remote receipt pending |
| U-036 | A distributed rank can fail while writing or exchanging a receipt after the last coordinated error check. | Peers can block indefinitely or finalize incomplete evidence. | Add one plan timeout to preparation and stage barriers; retain fault injection at post-stage persistence boundaries. | partial; bounded barriers implemented locally |
| U-037 | The project had no lockfile, dependencies floated, and Muon resolved from a mutable Git branch. | A commit could not reproduce its Python or optimizer environment. | Commit a cross-platform lock and pin Git dependencies to reviewed immutable commits before production certification. | implemented locally |
| U-038 | API and RunPod compilers perform dataset loading and other consequential work before RunEngine begins. | The canonical stage machine does not own all mutable or costly execution. | Move consequential preparation behind typed RunEngine stages and keep compilers pure. | open, canonical path |
| U-039 | Trace, report, diagnostic, and upload-plan readers did not all share a containment and byte-integrity guard. | A crafted artifact could escape the Run root or consume changed external data. | Require stage artifacts to remain inside the Run root before entering prior state; verify receipt hashes on every resume; route manifest consumers through `RunBundle.verified_artifact_path`. | implemented locally |
| U-040 | The repository lacked CI and a declared license. | Future changes could merge without the local evidence gate, and reuse terms remained ambiguous. | Enforce locked tests, Ruff, build, and secret scanning in CI; declare Apache-2.0. | implemented locally |
| U-041 | A valid ownership marker did not prevent overwrite or retention code from traversing a Windows junction. | Cleanup could remove files outside the intended managed tree. | Reject symlinks and junctions at ArtifactStore overwrite, lifecycle pruning, and checkpoint-retention boundaries; cover simulated junctions in tests. | implemented locally |
| U-042 | The two-A40 smoke config requested a persistent volume despite a bounded ephemeral benchmark attempt. | Retained storage can outlive compute teardown and weaken the spend and data-retention boundary. | Set `volume_gb: 0` in the smoke config and assert that the canonical RunPod plan derives zero persistent volume. | implemented locally |
| U-043 | Local private-repository access did not prove that the execution boundary permitted sending a PAT to a third-party Pod. | A paid Pod could pass hardware gates but remain unable to receive its exact source commit. | Require anonymous access to the exact public commit before allocation, or use an independently proven credential transport for a private repository. | resolved by verified anonymous public access |
| U-044 | A REST-created Pod did not expose `RUNPOD_POD_ID`, and a Windows read/write-only private-key ACL prevented deletion. | The deletion watchdog can target an empty ID, and teardown can retain sensitive key material. | Bind the literal create-response Pod ID, verify Pod-side API authentication, grant local key deletion permission, and assert key absence during teardown. | operational procedure implemented |
| U-045 | RunPod and synchronous chat-completion APIs do not document a universal idempotency key or durable result lookup. | A crash after provider completion but before local receipt durability can duplicate paid work. | Persist operation intent before submission; reconcile RunPod by deterministic Pod name and replay-safe providers by durable handle; fail ambiguity instead of resubmitting. | implemented locally |
| U-046 | RunPod billing history is time-bucketed and can settle after Pod teardown. | Immediate local estimates can be mistaken for authoritative final spend. | Record `pending_settlement`, reserve campaign budget, and block Incumbent mutation until a hashed provider billing receipt atomically settles cost and the bundle. | implemented locally; remote receipt pending |
| U-047 | Agents sharing one operating-system identity can bypass RunBundle visibility and read local sealed files directly. | Interface-level sealing cannot prevent adaptive holdout leakage in an adversarial agent population. | Keep protected suites outside worker-readable Run roots and evaluate through a separate authority that returns aggregate signed evidence. | accepted threat model, implementation pending |
| U-048 | SQLite coordinates one local authority but is not a multi-host consensus system. | Treating it as distributed consensus can permit split-brain claims across machines or network filesystems. | Certify SQLite only for one host and one database authority; preserve an explicit control-plane adapter seam for later multi-host deployment. | accepted scope boundary |
| U-049 | The lock selected Pydantic 1 even though the code requires Pydantic 2, and an unused vLLM extra forced an obsolete CUDA build on Windows. | A locked clean environment failed during test collection and could not reproduce local evidence. | Require `pydantic>=2,<3`; delete speculative vLLM extras until an output-equivalent benchmark justifies them; refresh the lock. | implemented locally |
| U-050 | Checked-in 300-step RunPod configs could spend materially without campaign authority. | A researcher or agent could run an expensive non-certifying experiment by mistake. | Delete executable 300-step defaults; retain one one-step, two-A40, ephemeral smoke. Generate full configs only from claimed proposals. | implemented locally |
| U-051 | A single scalar-first runtime trial gave the optimized path warmed caches and certified any speedup above one. | Order bias and noise could create a false R4 claim. | Warm shared execution, synchronize CUDA, run paired ABBA trials, and certify only `min(baseline) / max(optimized) >= 1.05` with exact output parity. | implemented locally; remote timing pending |
| U-052 | An attempt-local RunPod deadline ignored settled spend from prior attempts under the same user authorization. | Repeated attempts could exceed the campaign's total dollar cap. | Bind settled campaign spend into `RunPodBudget` and subtract it plus reserve before calculating the hard deadline. | implemented locally |
| U-053 | Paid preflight applied a full timeout to every command, continued after failure, and lost its report on `TimeoutExpired`. | A failed gate could consume most of the budget without durable evidence. | Remove duplicated local checks, enforce one aggregate deadline, stop at the first failure, catch timeouts, and always persist the receipt. | implemented locally |
| U-054 | Unconstrained editable installation could replace the image's compatible Torch/CUDA stack. | Dependency resolution could break CUDA after allocation or make the run irreproducible. | Install a `uv.lock`-bound non-Torch requirements file, install the project with `--no-deps`, and verify image Torch and CUDA remain unchanged. | implemented locally |
| U-055 | A successful Pod-create response without an ID raised before deterministic-name reconciliation. | A live Pod could continue billing after the client abandoned it. | Persist ambiguous state and reconcile by deterministic name before returning or failing. | implemented locally |
| U-056 | SSH instructions treated every account key as temporary although the operator chose a dedicated persistent service identity. | Teardown could delete the canonical key and force repeated manual setup. | Record the identity lifecycle explicitly; retain persistent service keys and delete only task-scoped keys. | implemented operationally |
| U-057 | Pod creation could succeed before the durable create receipt failed to write. | The process could exit while an unjournaled Pod continued billing. | Delete the known Pod on any post-create receipt persistence failure and preserve the original failure. | implemented locally |
| U-058 | A missing remote executable raised before the paid preflight recorded the failed gate. | The durable report lacked the actionable cause of a remote setup failure. | Convert operating-system launch errors into typed failed-gate evidence and stop immediately. | implemented locally |

## Progress Ledger

| Phase | State | Acceptance | Evidence | Next action |
| --- | --- | --- | --- | --- |
| 0. Governing context and decisions | completed | Docs and ADRs exist | This spec, CONTEXT.md, ADR-0001 through ADR-0005 | Keep decisions current |
| 1. Portable RunManifest and RunBundle | partial | F1-F4, F8, F14 | Canonical writer, relocation and source tests, immutable Hub revision resolver, atomic manifest finalization, ArtifactStore ownership marker, lifecycle deletion containment, TrainingView consumption hashes, stage receipt SHA-256 verification on every resume path, and one containment plus hash-verification reader for manifest consumers | Complete source-state coverage |
| 2. Sealed evaluation and strict promotion | partial | F5-F8 | EvaluationRoles, one PromotionDecision, EvalContract bindings for protected IDs and content across all compilers, runtime row-membership checks, sealed promotion artifacts and evaluate receipts excluded from normal surfaces, and mandatory content and verifier separation certificates persisted with every plan | Complete suite-rotation bridge evidence |
| 3. Canonical RunEngine | partial | F9-F11, F14 | One config dispatcher, shared stage order, central manifest and promotion, and removal of shadow orchestration | Move consequential compiler and adapter preparation behind engine stages |
| 4. Evidence-backed training | partial | F12-F13, F15 | MethodTrainingRequest, consumption-time TrainingView hash checks, policy-versioned GRPO traces, a role-aware training trace query that excludes sealed evaluation roles, measured frontier selection, and semantic RunBundle GRPO validation | Bind replay sampling decisions to explicit policy lineage and measured outcomes |
| 5. Campaign control plane | partial | C1-C7 | Explanation-bound proposals, renewable fenced leases, actual-cost rechecks, two-phase provider settlement, atomic Incumbent CAS, exposure accounting, and explicit compiler certification modes | Add automatic proposal-to-config materialization after pure compiler work |
| 6. Cost and runtime | in progress | R1-R4 | Cumulative $1.50 control-plane policy, exact Secure 2xA40 ephemeral request, create-rate deadline, malformed-response reconciliation, delayed billing receipts, Torch-preserving frozen RunPod requirements, fail-fast aggregate preflight, conservative paired runtime evidence, output-equivalence tests, and dynamic CUDA validation | After an explicit paid-run instruction, run R4 and only the one-step smoke if R4 passes and settled spend leaves budget |
| 7. Extensibility | partial | E1-E3 | Exact-math engine task and typed SFT, DPO, GRPO, OPD, and OPSD contracts | Prove executable production paths rather than isolated contract surfaces |
| 8. Agent and distributed scale | partial | R5-R6, E4-E5 | Renewable fenced leases, one-host authority scope, topology and staleness contracts, composition proposals, and bounded stage barriers | Finish post-receipt fault injection; add multi-host consensus only when required |
| 9. Final compliance and fresh-eyes review | in progress | All criteria | Apache-2.0, 303 locked tests, Ruff, build, secret scanning, two post-change fresh-eyes passes, and the adversarial risk register | Preserve honest external blockers and collect remote certification evidence only under explicit authorization |

## Research Decisions

- Adaptive reuse of a holdout can overfit even without direct row access. CampaignStore therefore enforces suite exposure limits and rotation rather than reporting exposure only. Primary source: Dwork et al., `Generalization in Adaptive Data Analysis and Holdout Reuse`, https://arxiv.org/abs/1506.02629.
- Content contamination can occur below document granularity. The evidence law now includes a content-separation certificate in addition to exact IDs. Primary sources: `PALOMA`, https://arxiv.org/abs/2312.10523, and Hidayat et al., `Simulating Training Data Leakage in Multiple-Choice Benchmarks for LLM Evaluation`, https://arxiv.org/abs/2505.24263.
- DPO simplifies optimization but does not eliminate proxy over-optimization. Learned and heuristic training verifiers therefore require an independent promotion verifier. Primary sources: Rafailov et al., `Direct Preference Optimization`, https://arxiv.org/abs/2305.18290, and `Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms`, https://arxiv.org/abs/2406.02900.
- Asynchronous rollout throughput creates policy-staleness risk. The architecture keeps synchronous execution canonical until an asynchronous executor records and enforces policy lag. Primary source: Fu et al., `AReaL`, https://arxiv.org/abs/2505.24298.

## Out of Scope Until Their Preconditions Hold

- Asynchronous rollout before synchronous evidence and policy-version contracts pass.
- Multi-node automation before one-node executed topology is measured and recorded.
- A plugin marketplace before a second task proves the task seam.
- A production OPD, OPSD, reward-model, or ECHO trainer without method-required evidence and promotion guards.
- A dashboard before RunBundle and campaign query contracts stabilize.
- External state as the source of truth.

## Completion Definition

The initiative is complete only when every acceptance criterion is checked with current evidence, every open unknown unknown has a decision or explicit external blocker, all mutable execution delegates to one RunEngine, the full verification suite passes, and a fresh-eyes review finds no unresolved consequential issue.
