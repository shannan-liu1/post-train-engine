# Lean Flywheel Plan

Status: historical. `docs/plans/canonical-research-flywheel-implementation.md` supersedes this plan.

This repo should become a small, agent-operable post-training flywheel, not a
general RL research framework. The minimum durable loop is:

```text
config -> examples -> remote samples -> grading -> metrics -> promotion -> artifacts
```

Local code orchestrates and verifies. Remote/API providers do compute.

## Phase Review

### Phase A: API-first GSM8K dry-run

Purpose: prove the loop without secrets, GPUs, network calls, or local model
serving.

Minimum slice:

- load a small deterministic GSM8K fixture
- split train/eval deterministically
- register a baseline candidate
- generate fake deterministic rollouts
- create one fake deterministic prompt-adapted candidate
- evaluate baseline and candidate
- compute paired accuracy statistics
- apply fail-closed promotion gates
- write raw prompts, raw outputs, evals, decision, report, and manifest

Required modules:

- `src/post_train_engine/api_schemas.py`
- `src/post_train_engine/api_hillclimb.py`
- `src/post_train_engine/artifact_store.py`
- `src/post_train_engine/artifacts.py`
- `src/post_train_engine/providers/base.py`
- `src/post_train_engine/providers/fake.py`
- `src/post_train_engine/tasks/gsm8k.py`
- `src/post_train_engine/evals/statistics.py`
- `src/post_train_engine/cli/hillclimb.py`
- `configs/gsm8k_tiny_dryrun.yaml`

Required tests:

- dry-run writes the complete minimal bundle
- dry-run can be rerun with `overwrite: true` without appending stale logs
- missing or malformed metrics fail closed
- artifact hashes validate
- promotion decision, manifest status, and final report agree
- no secret-bearing fields are emitted

Acceptance criteria:

- `python -m post_train_engine.cli hillclimb --config configs/gsm8k_tiny_dryrun.yaml`
  writes a deterministic run bundle.
- `pte artifacts validate --run <run_dir>` passes for that bundle.
- `promotion_decision.json` is the single promotion decision artifact.

Cut candidates:

- per-run `promotion_decisions.jsonl`; useful only once there is a multi-candidate
  loop in a single run directory
- event logs that duplicate final status and decision
- local GPU runners, training views, lifecycle upload, and trace groups on this
  path

Failure modes:

- fake provider accidentally treated as real training evidence
- config silently resolves missing secrets
- provider output missing eval coverage
- manifest hash index omitted or stale
- promotion report disagrees with promotion decision

Do not build yet:

- GRPO controller
- RunPod worker image
- OPD/OPSD/reward-model trainers
- async queues
- dashboard or plugin marketplace

### Phase B: API smoke with provider-neutral chat-completions inference

Purpose: prove remote inference/adaptation calls while keeping improvement honest
as prompt adaptation, not weight training.

Minimum slice:

- resolve `.env` without mutating process state
- submit provider-neutral chat-completion jobs through an OpenAI-compatible HTTP
  wire protocol
- preserve request/response metadata with secrets redacted
- evaluate provider outputs through the same local GSM8K grader
- fail closed on missing secrets, malformed responses, timeouts, and incomplete
  eval coverage

Required modules:

- `src/post_train_engine/env.py`
- `src/post_train_engine/providers/openai_compatible.py`
- `configs/gsm8k_api_smoke.yaml`
- `.env.example`

Required tests:

- missing required secret fails before network call
- malformed response fails closed
- token fields are not redacted as secrets
- raw provider response is preserved in redacted artifacts

Acceptance criteria:

- filling `.env` is the only manual setup for the API smoke path
- no local GPU assumptions enter this path
- final report calls the improvement `prompt_adaptation`

Cut candidates:

- any code that implies a prompt-adapted candidate is a trained checkpoint
- provider-specific assumptions in core promotion logic

Failure modes:

- API provider returns non-JSON adaptation payload
- cost metadata unavailable
- sync calls are slow at scale

Do not build yet:

- provider capability discovery
- generic job queues
- provider retry orchestration beyond fail-loud request errors

### Phase C: Real remote training adapter

Purpose: replace prompt adaptation with one true remote candidate-improvement
mechanism.

Minimum slice:

- one `TrainingProvider` adapter with explicit job submission, polling, result
  fetch, timeout, cancellation, and redacted request/response artifacts
- one fake deterministic training provider for tests
- one real provider adapter, likely RunPod serverless first
- training result returns a candidate reference plus durable artifact refs

Required modules:

- `src/post_train_engine/providers/training.py`
- `src/post_train_engine/providers/runpod_serverless.py`
- `src/post_train_engine/providers/fake_training.py`
- `tests/test_remote_training_provider.py`

Required tests:

- fake training provider exercises the same orchestration contract as real
  providers
- RunPod adapter submits, polls, fetches, and cancels using injected transport
- timeout and malformed remote payloads fail closed
- no API keys appear in artifacts or exceptions

Acceptance criteria:

- API hillclimb can select prompt adaptation or true remote training by config
- candidate artifact records whether the improvement was prompt-only or weights
- provider-specific details do not leak into promotion code

Cut candidates:

- local pod SSH
- multi-node training
- HF upload
- GRPO internals

Failure modes:

- remote worker returns checkpoint path without evidence bundle
- job completes but lacks eval-ready candidate reference
- provider state is hidden outside artifacts

Do not build yet:

- async RL sample queues
- distributed orchestration
- trainer-specific objective variants

### Phase D: GRPO as first real RL method

Purpose: add the first canonical hill-climbing RL technique only after remote
training has a working job boundary.

Minimum slice:

- GSM8K rollout groups with reward variance
- GRPO training view consumed by the remote worker
- correctness reward plus optional correctness-gated length penalty
- early group pruning for all-wrong or all-correct groups
- policy-version metadata to reject stale samples

Required modules:

- `src/post_train_engine/traces/schema.py`
- `src/post_train_engine/traces/rollouts.py`
- `src/post_train_engine/training_views/schema.py`
- `src/post_train_engine/methods/grpo/`
- `tests/test_rollout_groups.py`
- `tests/test_training_view_schema.py`

Required tests:

- all-wrong groups are pruned or diagnostic-only
- missing rewards fail closed
- degenerate reward groups are recorded
- stale policy samples are rejected according to config
- length penalty never rewards wrong short answers

Acceptance criteria:

- GRPO artifacts contain group rewards, variance, source policy version, verifier
  version, and split role
- promotion still depends only on eval artifacts, not training loss

Cut candidates:

- OPD/OPSD/reward-model schemas unless GRPO consumes the same evidence
- modified-GRPO knobs not used by the first remote worker

Failure modes:

- off-policy/stale samples treated as current
- low-variance groups waste training budget
- length penalty optimizes brevity over correctness

Do not build yet:

- ECHO traces
- terminal-agent environments
- complex RL environment marketplace

## Codebase Audit

### `src/post_train_engine/api_hillclimb.py`

Problem: owns config loading, dataset splitting, provider orchestration, grading,
promotion, reporting, and manifest writing in one file.

Why it matters: it is readable today but will become the dumping ground if real
remote training is added here.

Minimal fix: keep the file for the GSM8K API smoke path, but split only when the
second task uses the same code. Candidate future splits are `api_datasets.py`,
`api_eval.py`, and `api_reports.py`; do not split preemptively.

Action: keep, harden.

Tests: `tests/test_api_hillclimb.py`.

### `src/post_train_engine/api_schemas.py`

Problem: duplicates names already present in `search.py`, `climb.py`, and
`config.py`.

Why it matters: duplicate `Candidate` and config concepts confuse agents.

Minimal fix: keep API schemas as the source of truth for the API-first path.
Do not import old local GPU config into this path.

Action: keep; document boundary.

Tests: schema validation through API hillclimb tests.

### `src/post_train_engine/artifact_store.py`

Problem: previously pre-created directories and supported event logs even when
the API path did not need them.

Why it matters: empty directories and redundant logs look like contract surface.

Minimal fix: create directories lazily and emit hash refs through one helper.

Action: simplify.

Tests: dry-run bundle and artifact validation.

### `src/post_train_engine/artifacts.py`

Problem: artifact validation assumed lifecycle-backed local flywheel manifests.

Why it matters: API-first manifests should validate without pretending they have
lifecycle upload evidence.

Minimal fix: branch semantic promotion consistency by explicit manifest schema.

Action: harden, do not generalize further.

Tests: local smoke artifact tests plus API dry-run validation.

### `src/post_train_engine/providers/base.py`

Problem: `RemoteProvider` is intentionally tiny, but it is insufficient for real
remote training cancellation and retry.

Why it matters: training adapter work will need cancel/timeout semantics.

Minimal fix: do not expand yet. Add a separate training provider contract only
when Phase C starts.

Action: keep.

Tests: provider lifecycle tests.

### `src/post_train_engine/providers/fake.py`

Problem: fake provider knows GSM8K answers.

Why it matters: that is correct for orchestration tests, but it must never be
confused with model evidence.

Minimal fix: keep provider type and metadata explicitly fake.

Action: keep.

Tests: dry-run smoke tests.

### `src/post_train_engine/providers/openai_compatible.py`

Problem: synchronous calls are fine for smoke and bad for scale. The historical
module name describes the HTTP wire protocol, not the public provider surface.

Why it matters: an async queue would be premature until one real training adapter
exists.

Minimal fix: keep synchronous provider; add async only after remote training
exists.

Action: keep.

Tests: injected transport tests.

### `src/post_train_engine/cli/main.py`

Problem: exposes many commands from different eras.

Why it matters: agents may choose `run`, `train`, or `runpod` when the intended
API-first path is `hillclimb`.

Minimal fix: README should make `hillclimb` the default command. Do not remove
older commands until their tests and users are migrated.

Action: keep, later deprecate old local smoke commands.

Tests: CLI dispatch tests.

### `src/post_train_engine/cli/run.py`

Problem: a large local GSM8K smoke runner with its own manifest shape.

Why it matters: useful for local evidence contracts, but not the API-first path.

Minimal fix: quarantine as legacy/local smoke. Do not add new remote compute
logic here.

Action: keep temporarily; future merge/delete after API hillclimb replaces it.

Tests: GSM8K flywheel bundle tests.

### `src/post_train_engine/climb.py`, `search.py`, `config.py`

Problem: older local hill-climb abstractions overlap with API hillclimb concepts.

Why it matters: wrapping them would create abstract soup.

Minimal fix: do not use them in API hillclimb. Retain for existing tests until a
deliberate deletion pass.

Action: quarantine.

Tests: existing climb/search/scaffold tests.

### `src/post_train_engine/flywheel.py`

Problem: rich manifest schema includes optimizer, topology, shard plans, and
pricing.

Why it matters: correct for older local/distributed experiments, excessive for
API smoke.

Minimal fix: API hillclimb uses a lean manifest instead of `FlywheelRunManifest`.

Action: keep for local smoke, do not import into API path.

Tests: flywheel manifest tests.

### `src/post_train_engine/traces/*`, `training_views/*`

Problem: useful evidence models but not consumed by API hillclimb yet.

Why it matters: adding them to the API path before GRPO would create ceremony.

Minimal fix: keep dormant until Phase D.

Action: defer.

Tests: existing trace and training-view schema tests.

### `src/post_train_engine/opsd/*`, `methods/*`, `training/*`

Problem: method internals exist before the API-first remote training boundary.

Why it matters: easy for agents to start optimizing local trainer internals while
the real gap is remote execution and evidence preservation.

Minimal fix: do not touch for API MVP. Treat as method library code, not engine
core.

Action: defer.

Tests: existing method/training tests.

### `src/post_train_engine/runpod.py`

Problem: writes plans but does not execute remote jobs.

Why it matters: the name can imply capability that is not implemented.

Minimal fix: keep fail-closed plan-only behavior until Phase C adds a real
serverless adapter.

Action: keep, then supersede with `providers/runpod_serverless.py`.

Tests: RunPod dry-run tests.

## Bare Target Architecture

### `api_schemas.py`

Responsibility: immutable config, candidate, provider-job, eval, and promotion
schemas for the API-first loop.

Inputs: YAML-loaded mappings and provider payloads.

Outputs: Pydantic objects and JSON-safe dictionaries.

Invariants: no loose extra fields; finite metrics; explicit provider types;
missing secrets resolved outside the schema.

Must not own: dataset loading, provider HTTP, promotion math.

Tests: config rejection, finite metric rejection, redaction.

### `api_hillclimb.py`

Responsibility: one concrete GSM8K API hillclimb orchestration.

Inputs: config path and optional env path.

Outputs: deterministic run directory and final report dictionary.

Invariants: same eval IDs for baseline and candidate; complete sample-index-0
coverage; missing provider output fails closed; promotion is deterministic.

Must not own: training internals, provider-specific HTTP details, GRPO internals.

Tests: dry-run, missing secret, underpowered eval, coverage mismatch.

### `providers/*`

Responsibility: turn provider-specific request/response formats into `JobResult`.

Inputs: `JobRequest`.

Outputs: `JobHandle`, `JobStatus`, `JobResult`.

Invariants: provider ID matches; malformed remote payloads raise; secrets are
not returned through result artifacts.

Must not own: metrics, promotion, dataset splits.

Tests: fake provider dry-run; chat-completions injected transport.

### `tasks/gsm8k.py`

Responsibility: GSM8K loading, prompting, parsing, and numeric verification.

Inputs: examples or provider completions.

Outputs: normalized examples and parser/verifier results.

Invariants: NaN/inf fail closed; official/promotion splits do not leak into
training.

Must not own: remote provider execution or promotion policy.

Tests: parser, verifier, deterministic splits.

### `evals/statistics.py`

Responsibility: paired statistical summaries.

Inputs: baseline/candidate correctness vectors.

Outputs: deltas, bootstrap CI, McNemar/sign evidence.

Invariants: paired vectors same length and deterministic seed.

Must not own: candidate selection.

Tests: paired stats and reproducible CI.

### `artifact_store.py` and `artifacts.py`

Responsibility: write artifacts and validate manifest-indexed bundles.

Inputs: JSON-safe bodies and run directory.

Outputs: files plus `artifact_status.json`.

Invariants: required artifact hashes match; manifest/promotion/report agree;
semantic checks are schema-specific.

Must not own: model behavior or promotion math.

Tests: hash mismatch, malformed refs, API promotion consistency.

### CLI

Responsibility: expose the few commands agents should run.

Required now:

- `pte hillclimb --config ...`
- `pte artifacts validate --run ...`

Useful soon:

- `pte report --run ...`
- `pte diagnose --run ...`

Must not own: hidden defaults beyond config parsing.

Tests: no traceback on config/provider errors.

## Artifact Contract

API hillclimb writes only:

- `config.raw.yaml`
- `config.resolved.json`
- `env.redacted.json`
- `manifest.json`
- `provider_requests.jsonl`
- `provider_responses.jsonl`
- `candidates/baseline.json`
- `candidates/candidate.json`
- `datasets/splits.json`
- `datasets/train.jsonl`
- `datasets/eval.jsonl`
- `rollouts/baseline_train_rollouts.jsonl`
- `evals/baseline.json`
- `evals/candidate.json`
- `promotion_decision.json`
- `final_report.json`
- `final_report.md`

Deliberately not emitted:

- one-row `promotion_decisions.jsonl`
- event logs duplicating the final decision
- lifecycle upload evidence
- training-view artifacts
- GRPO rollout groups
- checkpoint registry rows

Those artifacts become necessary only when the corresponding capability exists.

## Next Implementation Steps

1. Keep API hillclimb as the only documented default loop.
2. Add a true remote training provider contract and one fake provider.
3. Add one real RunPod serverless adapter with transport-stub tests.
4. Add a config option that chooses `prompt_adaptation` or real remote training.
5. Only then introduce GRPO rollout-group evidence.

## Explicit Deferred Scope

- GRPO internals
- OPD/OPSD
- reward-model training
- ECHO-style environment spans
- async RL queues
- multi-node orchestration
- dashboard UI
- HF publishing
- local GPU training as a default path
