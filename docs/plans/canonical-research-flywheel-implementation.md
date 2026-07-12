# Canonical Research Flywheel Implementation Plan

## Goal

Implement the governing flywheel specification through verified vertical slices, beginning with findings 1 through 4 and continuing through campaign, cost, extensibility, and collaboration capabilities.

## Source Of Truth

- `docs/specs/canonical-research-flywheel.md`
- `CONTEXT.md`
- `docs/adr/0001-canonical-run-engine.md`
- `docs/adr/0002-evaluation-and-promotion-law.md`
- `docs/adr/0003-portable-run-bundles.md`
- `docs/adr/0004-evidence-backed-training.md`
- `docs/adr/0005-local-campaign-control-plane.md`

## Behavior Contract

- `pte run --config <path>` is the only mutable execution interface.
- Every adapter produces the same RunManifest and validates the same RunBundle contract.
- Selection, diagnostic, promotion, canary, and unseen roles remain explicit.
- The strict paired PromotionDecision owns acceptance.
- Every method consumes a TrainingView.
- Every Run preserves reasoning, learning, source, artifact, and cost lineage.
- Campaign state updates the Incumbent atomically and preserves negative results.
- Missing required evidence fails closed with an actionable error.

## Execution Order: Delete Before Automating

The remaining work proceeds as vertical slices in this order:

1. Provider operation intent and reconciliation. A crash after external completion
   never causes an unexamined retry.
2. Pure compilation plus RunEngine-owned resolution. Delete consequential work
   outside the stage machine.
3. Certifying campaign binding. Delete implicit standalone production Runs.
4. Renewable fenced leases and measured-cost settlement. Delete clock-only worker
   authority and estimate-as-actual accounting.
5. Authoritative RunPod create and billing receipts with a $1.50 target. Delete
   inferred topology pricing and immediate-settlement claims.
6. Bounded distributed receipt commits with fault injection. Delete unbounded
   collectives.
7. Explanation-driven experiment routing. Automate only after the evidence loop is
   stable and every proposal exposes a risky prediction and rejection criterion.

Each slice uses one public RED test, the minimum GREEN implementation, focused
regression tests, then the full repository gate. No remote spend occurs before
slices 1 through 6 pass locally, the reviewed tree is committed and pushed, and
anonymous checkout of that exact commit succeeds.

## Foundation Slice 1: Portable RunBundle

Files and interfaces:

- Create `src/post_train_engine/run_bundle.py` with `RunManifest`, `ArtifactRef`, `SourceIdentity`, `RunBundle`, and finalization helpers.
- Modify `src/post_train_engine/artifacts.py`, `src/post_train_engine/reports.py`, `src/post_train_engine/diagnostics.py`, and lifecycle planning to use RunBundle.
- Add schema-version compatibility readers for current local and API fixtures.

First RED test:

- Copy a valid run directory to a different parent and validate it from a different working directory.
- Command: `python -m pytest tests/test_run_bundle.py -q`
- Expected RED: current artifact paths resolve against the original process working directory or no RunBundle interface exists.
- Expected GREEN: copied bundle validates with identical artifact hashes.

Further cycles:

1. Reject path traversal outside the run root for local relative artifacts.
2. Record source clean state, tracked diff hash, and untracked source hash.
3. Reject requested dataset revisions that do not match resolved revisions.
4. Read existing manifest schemas into one normalized view.
5. Finalize atomically and resume an interrupted finalization without duplicating side effects.

## Foundation Slice 2: Sealed Evaluation And Promotion

Files and interfaces:

- Extend task split schemas with canonical SplitRoles.
- Make all evaluation paths emit canonical EvalArtifacts.
- Route API and RunPod decisions through `decide_promotion()`.
- Persist PromotionSuiteState in the RunBundle and campaign index.

First RED test:

- Configure checkpoint selection and promotion with the same example identifiers.
- Command: `python -m pytest tests/test_evaluation_roles.py -q`
- Expected RED: the current RunPod path accepts the overlap.
- Expected GREEN: RunPlan validation rejects the overlap before model loading.

Further cycles:

1. RunPod cannot promote without configured strict evidence.
2. Missing overlap certification rejects promotion.
3. Promotion disclosure omits per-example sealed outputs.
4. Suite exposure increments once per evaluated Candidate and rotation fails closed when required.
5. API and RunPod promotion artifacts share one schema.

## Foundation Slice 3: Canonical RunEngine

Files and interfaces:

- Create `src/post_train_engine/engine.py` with RunEngine and typed stage outcomes.
- Create adapter modules for task, method, provider, and compute variation.
- Change CLI registration so `pte run` accepts canonical configs.
- Route `pte hillclimb` through a compatibility compiler before eventual removal.
- Split orchestration out of `api_hillclimb.py` and `runpod_grpo.py`.

First RED test:

- Execute local fake and RunPod-fake adapters through the public RunEngine interface.
- Command: `python -m pytest tests/test_run_engine.py -q`
- Expected RED: no shared public interface exists and manifests differ.
- Expected GREEN: both Runs traverse the same stage sequence and emit the same manifest and PromotionDecision schemas.

Further cycles:

1. Adapter code cannot import or invoke promotion mutation.
2. Stage receipts support idempotent resume.
3. Provider errors produce failed RunManifests rather than partial success bundles.
4. Compatibility commands delegate to RunEngine.
5. Remove duplicate promotion and report writers.

## Foundation Slice 4: Evidence-Backed Training

Files and interfaces:

- Extend TraceRecord with policy, rollout-group, sampling, verifier, reward, and TrainingView identities.
- Make TrainingView validation own SplitRole eligibility.
- Change method adapters to accept TrainingViews.
- Change real GRPO row construction and trace capture to preserve lineage.
- Replace independent replay records with evidence queries or compatibility views.

First RED test:

- Invoke a method adapter with a raw dataset path and no TrainingView.
- Command: `python -m pytest tests/test_method_training_views.py -q`
- Expected RED: current runners accept raw data paths.
- Expected GREEN: every method rejects missing or incompatible TrainingViews before loading the GPU stack.

Further cycles:

1. GRPO traces record policy version and rollout group.
2. Promotion, canary, and unseen roles cannot enter a TrainingView.
3. Privileged teacher visibility is explicit and method-valid.
4. Candidate lineage resolves to source Trace identifiers.
5. Real and fake method adapters satisfy the same contract tests.

## Later Vertical Slices

1. Campaign SQLite schema, migrations, repositories, and atomic Incumbent transaction.
2. Hypothesis and ExperimentProposal hashing with one-axis comparison validation.
3. Work leases and concurrent claim tests.
4. Deterministic next-experiment recommendations from failure and evidence categories.
5. Phase-level cost instrumentation and exact-contract baseline caches.
6. Batched evaluation and model reuse with equivalence tests.
7. Second executable-verifier task through the canonical TaskAdapter.
8. SFT, DPO, and GRPO adapter parity.
9. Agent collaboration and mechanism-combination proposals.
10. Executed topology and policy-staleness enforcement before asynchronous scale.

## Broader Verification Commands

- Focused RED/GREEN command named in each slice.
- `python -m pytest -q`
- `ruff check .`
- `git diff --check`
- Deterministic dry run through `pte run --config configs/experiments/gsm8k_smoke.yaml`
- Artifact validation after relocating the produced run directory.

## Spec Compliance Checklist

- [ ] Foundation acceptance F1 through F15 has current evidence.
- [ ] Campaign acceptance C1 through C7 has current evidence.
- [ ] Runtime acceptance R1 through R6 has current evidence.
- [ ] Extensibility acceptance E1 through E5 has current evidence.
- [ ] Unknown-unknown register contains no unowned correctness risk.
- [x] Full tests and Ruff pass after the final local code change: 307 tests pass.
- [x] Two fresh-eyes passes found and fixed allocation, billing, identity, dependency, runtime-certification, preflight, and provider-handle defects; the final targeted and full gates pass.

The adversarial review reopened production certification at the promotion-contract,
sealed-evidence, replay-lineage, receipt-idempotency, campaign-wiring, lease-fencing,
provider-cost, dependency-locking, and distributed-failure boundaries. U-029 through
U-065 in the governing spec own those findings. Local tests certify implemented
mechanics only. They do not certify a production-safe research flywheel.

R4 remains open after a Secure two-A40 attempt measured a 1.973 conservative speedup
but failed exact output parity. GRPO did not run. A later remote attempt must diagnose
the scalar-versus-batched BF16 generation drift, receive explicit user authorization,
select exactly one source-delivery mode from `docs/runpod_deployment.md`, and bind the
CUDA allocation filter to the chosen image. No local review or commit authorizes a
second RunPod action.

## Risks And Escape Hatches

- Risk: compatibility migrations may expand scope. Escape hatch: normalize legacy manifests at read time while writing only the canonical schema.
- Risk: strict promotion requirements may block existing smoke fixtures. Escape hatch: use explicit non-certifying smoke states that can never promote.
- Risk: SQLite work may precede stable domain contracts. Escape hatch: complete portable RunBundle and RunEngine first.
- Risk: RunPod integration cannot be fully exercised locally. Escape hatch: require adapter contract tests with injected fakes and preserve one explicit remote verification gate.
- Risk: a new repository can make broad local tests look like production evidence. Escape hatch: keep unsupported criteria open and treat the first reviewed commit as a foundation checkpoint.
