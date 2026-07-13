# Post-Train Engine

Post-Train Engine is an early evidence-driven research flywheel for model post-training. The repository implements a canonical local stage machine, portable Run bundles, typed training evidence, paired promotion gates, and campaign primitives. The governing spec distinguishes locally verified mechanics from open production certification work.

The repository optimizes first for one researcher. Its immutable proposals, renewable fenced SQLite leases, canonical Run bundles, and coordinated stage receipts provide a foundation for multi-agent work. External sealed-evaluation authority, provider-derived production cost evidence, multi-host consensus, and end-to-end production wiring remain open before that foundation can safely support many concurrent agents.

## Governing contracts

- [`CONTEXT.md`](CONTEXT.md) defines domain terms and invariants.
- [`docs/specs/canonical-research-flywheel.md`](docs/specs/canonical-research-flywheel.md) owns acceptance and progress.
- [`docs/plans/canonical-research-flywheel-implementation.md`](docs/plans/canonical-research-flywheel-implementation.md) owns implementation order and verification.
- [`docs/adr/`](docs/adr/) records accepted architecture decisions.

The target core invariants are below. The spec progress ledger records which ones
have complete production evidence.

1. `pte run --config <path>` is the only mutable experiment command.
2. `pte hillclimb` is a compatibility alias that delegates to the same config dispatcher and `RunEngine`.
3. Every Run traverses `prepare`, `data`, `evidence`, `train`, `select`, `evaluate`, `promote`, and `finalize`.
4. `RunEngine` alone writes the final `RunManifest` and invokes strict promotion.
5. Method adapters consume validated `TrainingView` evidence. They cannot decide promotion or mutate the Incumbent.
6. Selection and promotion examples remain disjoint.
7. Missing required evidence rejects the Candidate.

## Install and verify

```powershell
uv sync --frozen --extra dev
uv run --frozen pytest -q
uv run --frozen ruff check .
```

Codex on Windows can use the repository-local wrapper when the global `uv` cache is sandboxed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/codex_uv_run.ps1 `
  --with pytest --with pydantic --with pyyaml --with numpy --with torch `
  --with safetensors python -m pytest -q
```

## Run locally

The deterministic local Run exercises the complete bundle, evidence, promotion, lifecycle, and report contracts without model inference:

```powershell
uv run pte run --config configs/experiments/gsm8k_smoke.yaml
uv run pte artifacts validate --run runs/gsm8k-local-smoke
uv run pte report --run runs/gsm8k-local-smoke
uv run pte diagnose --run runs/gsm8k-local-smoke
```

## Run API prompt adaptation

The fake provider config exercises provider requests, policy-linked Traces, measured frontier selection, prompt-adaptation TrainingViews, and central promotion:

```powershell
uv run pte run --config configs/gsm8k_tiny_dryrun.yaml
```

Remote OpenAI-compatible configs resolve secrets from environment variables or an explicit dotenv file. Resolved configs and provider logs redact secret values.

```powershell
uv run pte run --config configs/gsm8k_api_smoke.yaml --env .env
```

## Run GRPO on RunPod

Follow the fail-closed [RunPod deployment contract](docs/runpod_deployment.md) before creating a Pod. In particular, the allocation request must pin an `allowedCudaVersions` value compatible with the configured image.

Run the preflight inside the target image before spending a training budget:

```bash
python scripts/runpod_preflight.py --out runs/runpod-preflight/preflight_report.json
```

Launch the canonical command with the configured topology:

```bash
accelerate launch --num_processes 2 -m post_train_engine.cli run \
  --config configs/gsm8k_runpod_smoke.yaml
```

The RunPod path currently combines pre-engine compilation with `RunEngine` stages
for these operations. U-038 tracks moving every consequential preparation step
behind the engine boundary.

1. Resolve immutable Hugging Face model and dataset revisions.
2. Assign disjoint train, selection, and promotion roles.
3. Probe the parent policy with grouped stochastic rollouts.
4. Select examples whose measured success rate lies strictly between zero and one.
5. Emit a non-training outcome when no learnable frontier exists.
6. Train GRPO only from the validated TrainingView.
7. Select checkpoints on selection examples.
8. Re-evaluate the selected checkpoint on independent promotion examples.
9. Apply the current central promotion, lifecycle, cost, and campaign mechanics.

This path is not production-certified. The governing spec keeps immutable
evaluation binding, sealed-row disclosure, provider-derived costs, and distributed
receipt-failure handling open.

Multi-rank workers execute the same non-promotion stages. Rank zero alone writes intents, receipts, promotion evidence, and the final manifest.

## Runtime benchmark

R4 requires measured GPU evidence. The current script is a fail-closed,
non-promoting verification utility, not a canonical Run. U-071 blocks another paid
certification attempt until this measurement enters typed RunEngine stages. The
historical development invocation below does not grant standing authorization:

```bash
accelerate launch --num_processes 2 scripts/benchmark_runpod_eval.py \
  --config configs/gsm8k_runpod_smoke.yaml \
  --out runs/runtime-benchmarks/gsm8k-runpod-smoke.json
```

The benchmark fails unless one-load-per-shard model reuse with scalar tensor shapes exactly matches one-load-per-example outputs and improves max-rank wall time. Its JSON artifact records resolved revisions, topology, environment, timings, load counts, output hash, and certifying status. Batching remains disabled until a separate exact-contract GPU experiment proves equivalence.

## Run bundle

Every finalized Run writes `manifest.json` with schema `post_train_run_v1`. Required artifacts use paths relative to the Run root and carry SHA-256 hashes. The manifest records:

- clean, tracked-dirty, and untracked source identity;
- requested and resolved model, dataset, verifier, teacher, and environment inputs;
- Candidate and parent lineage;
- policy-versioned Traces, RolloutGroups, and TrainingViews;
- canonical evaluation and PromotionDecision evidence;
- stage receipts, costs, topology, lifecycle, and next-experiment recommendation.

Moving a Run directory does not invalidate the bundle. Hash, path, role, lineage, promotion, and report inconsistencies fail validation.

## Campaign control plane

`CampaignStore` indexes immutable bundle evidence in local SQLite. It stores hypotheses, proposals, comparison groups, budgets, leases, negative outcomes, suite exposure, and the current Incumbent.

A non-certifying smoke carries no campaign binding and can never promote. A certifying `RunPlan.campaign` binding names the campaign database, fenced proposal lease, expected Incumbent, promotion suite, and cost-settlement mode. Stage-measured Runs finalize atomically. Provider-billed Runs remain `pending_settlement` until a hashed billing receipt settles actual cost and compare-and-swaps the Incumbent. Reconciliation is idempotent across either write order.

## Extension seams

- Task adapters produce executable verifier evidence.
- Method adapters accept `MethodTrainingRequest` with a validated TrainingView.
- Provider and compute adapters execute work but cannot decide promotion.
- `RunEngine` owns orchestration and finalization.
- `CampaignStore` owns multi-agent coordination and Incumbent mutation.

The exact-math task proves a second executable-verifier task through the same engine. SFT, DPO, GRPO, OPD, and OPSD retain method-specific evidence contracts instead of collapsing into one generic trainer.

## Repository map

```text
src/post_train_engine/
  engine.py                 canonical RunPlan and eight-stage RunEngine
  run_bundle.py             portable RunManifest and RunBundle validation
  campaign.py               SQLite proposals, leases, outcomes, and Incumbent
  api_hillclimb.py           API provider adapter
  runpod_grpo.py             distributed RunPod GRPO adapter
  evaluation_roles.py       split-role law
  evals/promotion.py        strict paired PromotionDecision
  traces/                   policy and verifier interaction evidence
  training_views/           immutable method-specific projections
  training/                 method adapters and checkpoint mechanics
  task_adapters/            executable-verifier task adapters
tests/                       public behavior and contract tests
```

Do not add another orchestration loop, final manifest writer, or promotion helper. Extend an adapter or the canonical engine contract instead.
