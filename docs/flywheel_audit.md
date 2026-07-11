# Flywheel Audit and Minimal Architecture Plan

Status: historical. `docs/specs/canonical-research-flywheel.md` supersedes execution and architecture recommendations in this audit.

Date: 2026-06-16

## Scope Gate

Precise question: What is the smallest truthful path from the current
`post-train-engine` scaffold to an agent-operable, eval-driven post-training
flywheel that preserves evidence, builds method-specific training views, runs
deterministic promotion gates, records reproducible outcomes, and only then
adds OPD, OPSD, reward-model, or ECHO-style extensions?

Complete answer must contain:

1. A repository audit grounded in current files, not generic architecture.
2. A literature-grounded failure-mode summary for GRPO/RLVR, OPD, OPSD, reward
   modeling, and ECHO-style agentic rewards.
3. A target architecture and gap analysis against the live scaffold.
4. A phased implementation plan with files, tests, acceptance criteria, and
   risks.
5. Concrete file-level changes, test plan, cut candidates, MVP scope, and
   deferred scope.

Research type: practical and exploratory. The papers are useful for interface
design, artifact fields, diagnostics, and gates; they are not enough evidence to
import complex new algorithms into an MVP.

Research failure modes to watch:

1. Treating a new algorithm paper as production validation.
2. Conflating training data, traces, difficulty maps, and training methods.
3. Overfitting architecture to GSM8K while claiming a task-agnostic flywheel.

## Executive Verdict

This repository is already more than a pile of scripts. It has a credible
spine: normalized task examples, verifier-backed probing, difficulty buckets,
GSM8K curriculum generation, strict eval artifacts, paired promotion gates,
split-leakage checks, lifecycle evidence bundles, checkpoint registry rows,
replay records, and GPU runner adapters. The missing flywheel is not another
training method. The missing flywheel is a durable evidence substrate:
`TraceStore` as the append-only raw evidence layer, `RolloutStore` as a
rollout-specific API/view over trace groups, typed `TrainingView` artifacts, a
`TrainingViewBuilder`, generic run/report/diagnose CLI commands, and
method-specific artifact diagnostics.

The implementation priority is not to add more methods. It is to make every
method consume typed evidence and produce promotion-grade artifacts.

The strongest counterargument is that the repo should first prove one complete
GSM8K local run before adding any general abstractions. That is correct as a
sequencing constraint. It is not a reason to keep saying "dataset" where the
system needs "training view"; the naming boundary is already load-bearing.

## Literature-Grounded Failure Modes

Primary sources inspected:

- [DeepSeekMath / GRPO, arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- [Self-distillation degradation, arXiv:2603.24472](https://arxiv.org/abs/2603.24472)
- [Revisiting OPD, arXiv:2603.25562](https://arxiv.org/abs/2603.25562)
- [OGLS-SD / OPSD, arXiv:2605.12400](https://arxiv.org/abs/2605.12400)
- [Many Faces of OPD, arXiv:2605.11182](https://arxiv.org/abs/2605.11182)
- [Trust Region OPD, arXiv:2606.01249](https://arxiv.org/abs/2606.01249)
- [Teacher-Guided Policy Optimization, arXiv:2605.13230](https://arxiv.org/abs/2605.13230)
- [Gradient Starvation in Binary-Reward GRPO, arXiv:2605.07689](https://arxiv.org/abs/2605.07689)
- [EP-GRPO, arXiv:2605.04960](https://arxiv.org/abs/2605.04960)
- [LongRLVR, arXiv:2603.02146](https://arxiv.org/abs/2603.02146)
- [LLMs Gaming Verifiers, arXiv:2604.15149](https://arxiv.org/abs/2604.15149)
- [Noisy Data is Destructive to RLVR, arXiv:2603.16140](https://arxiv.org/abs/2603.16140)
- [Reasoning Boundary Paradox, arXiv:2510.02230](https://arxiv.org/abs/2510.02230)
- [ECHO, arXiv:2605.24517](https://arxiv.org/abs/2605.24517)
- [Deep RL from Human Preferences, arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- [InstructGPT / RLHF, arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

### GRPO / RLVR

Works when the verifier is trustworthy, the base policy has nonzero success,
rollout groups have reward variance, and promotion is held out. DeepSeekMath
introduced GRPO as a memory-efficient PPO variant for math reasoning. Recent
failure papers point to the real engineering contract: binary-reward GRPO can
starve when groups are all correct or all wrong; token-level updates can assign
credit to uninformative or wrong tokens; outcome-only long-context rewards are
too sparse for grounding; imperfect verifiers can be hacked; noisy labels are
destructive; and on-policy RL can shrink pass@k by reinforcing high-likelihood
solutions while suppressing low-likelihood ones.

Required artifact fields: group size, rollout id, per-sample reward, reward
component breakdown, group reward mean and variance, degenerate group count,
advantage statistics, parse status, verifier id/version, completion tokens,
pass@k, accuracy@1, prompt hash, split role, and canary/perturbation outcomes.

Promotion gates: fail closed on missing reward variance evidence when the
candidate was trained or selected using group-relative RLVR/GRPO evidence.
More generally, missing method-required evidence fails closed. Also fail closed
on missing token evidence, train/promotion overlap, verifier uncertainty, parse
regression, token/cost explosion, pass@k collapse, easy-slice regression, OOD
regression, and canary or perturbation failures.

Out of MVP: EP-GRPO, sign advantages, trust-region variants, learned process
reward shaping, and dense process rewards without verifiable process signals.

### OPD

Works when teacher feedback is reliable on student-generated prefixes, the
teacher and student distributions are not too far apart, tokenizers and special
tokens match, and teacher traces are verifier-filtered. The OPD papers identify
sampled-token supervision imbalance, unreliable teacher guidance on off-support
prefixes, tokenizer/special-token mismatch, large teacher-student divergence,
and uninformative negative feedback under large policy divergence.

Required artifact fields: teacher id, teacher checkpoint, student checkpoint,
teacher visibility of gold/verifier information, teacher reliability score,
tokenizer ids, special-token masks, KL or compatibility diagnostics, local
support/top-k metadata, verifier result for teacher trace, source trace ids,
and reason for inclusion.

Promotion gates: OPD-trained candidates must pass the same held-out promotion
contract as GRPO candidates, plus OOD/slice checks and tokenizer compatibility
checks before training starts.

Out of MVP: TGPO, TrOPD, top-k reverse-KL implementations, and trust-region
teacher-prefix rollouts. The MVP should only preserve the fields and block
unsafe OPD configs.

### OPSD

Works only in narrower settings than the name suggests. Self-distillation can
shorten traces while degrading math reasoning by suppressing uncertainty
expression; OPSD can fail when privileged information is instance-specific and
unavailable at test time; self-reflected teacher responses can impose biased
templates and miscalibrated token-level supervision.

Required artifact fields: mode (`opsd_verified_self_trace` versus
`opsd_privileged_context`), source policy, target policy, reference source,
whether gold/verifier information was visible during generation, whether it is
visible during training, whether it is metadata-only, verifier pass/fail,
response length, self-correction markers, uncertainty markers, parse status,
and OOD/slice outcomes.

Promotion gates: reject in-domain-only gains without OOD and easy-slice guards;
flag response-length collapse and parse/format regressions; never treat
answer-only final targets as reasoning supervision.

Out of MVP: outcome-guided logit steering, privileged-context distillation
trainer, and minimum-reasoning-length rewards.

### SFT, DPO, Rejection Sampling, And Reward Models

SFT is useful for bootstrapping zero-reward regions, parser repair, action
protocols, and high-quality demonstrations. DPO/preference training is useful
when pairwise comparisons are more reliable than scalar rewards. Rejection
sampling is only a selection operator: it can emit SFT, preference, OPD, OPSD,
GRPO replay, or diagnostics records. Reward modeling is a fallible learned
scorer, not a verified reward function; classic preference-reward work and
InstructGPT-style RLHF both rely on learned reward models from comparisons, so
calibration and held-out checks are part of the contract.

Required artifact fields: comparison basis, chosen/rejected reward, strict or
weak preference status, parser/verifier output, source checkpoint, privileged
visibility, reward-model id/version, calibration eval, slice coverage, and
known invalid domains.

Promotion gates: learned reward improvements are diagnostic only unless backed
by held-out task evals and deterministic promotion evidence.

Out of MVP: training a reward model. Keep schema support and diagnostics hooks.

### ECHO-Style Agentic Rewards

The relevant ECHO paper is `ECHO: Terminal Agents Learn World Models for Free`,
arXiv `2605.24517`. The useful design pattern is not "add an ECHO trainer now."
It is that multi-turn agent rollouts contain action tokens and environment
observation tokens; terminal outputs are dense supervision already present in
failed and successful traces; and an auxiliary observation-prediction loss can
be combined with policy-gradient loss on action tokens.

Required artifact fields: message/span id, role, action span token range,
environment-observation span token range, command/tool metadata, stdout/stderr
or environment state hash, final verifier result, and loss masks for action
versus observation tokens.

Promotion gates: final task success remains the promotion signal; environment
prediction is diagnostic or auxiliary training signal.

Out of MVP: full ECHO objective, terminal-agent benchmark runner, and
verifier-free self-improvement.

## Repository Audit

| Area | What exists | Role in flywheel | Status | Main gap |
| --- | --- | --- | --- | --- |
| `README.md` | Goal, scaffold map, lifecycle contract, next slice | Direction-setting | Core, updated here | Needs future pruning once the live CLI path proves the intended architecture |
| `pyproject.toml` | Package metadata, CLI entry points, deps | Packaging | Core | Heavy dependency set at import/test time |
| `configs/` | GSM8K task, eval, probe, and train smoke configs | Example contracts | Core seed | No generic experiment config examples for full run loop |
| `config.py` | Strict Pydantic experiment, optimizer, checkpoint, lifecycle configs | ExperimentSpec seed | Core | No first-class eval suite, rollout, trace store, or training-view config |
| `flywheel.py` | Run manifest, artifact refs, cost records, topology, shard plans | RunManifest seed | Core | Not yet written by a generic `pte run` path |
| `tasks/schema.py` | Normalized `Example` and duplicate-id guard | TaskExamples | Core | No split role or prompt hash on `Example`; handled elsewhere |
| `tasks/gsm8k.py` | GSM8K loader, prompt, splits, parser, verifier | Task plugin | Core seed | Task-specific, not general plugin registry beyond current pattern |
| `rewards/gsm8k.py` | Exact numeric reward, parse bonus, length penalty | Verifier/reward seed | Core seed | No general reward component schema |
| `generation.py` | vLLM default and HF fallback with seeded rollouts | Rollout backend | Core | Per-rollout interface, not batched RolloutRunner |
| `probe.py` | Probe traces, early-exit config, JSONL rows | ProbeRunner and raw evidence | Core | Probe rows are task-shaped, not a general TraceStore schema |
| `difficulty.py` | Difficulty bands and GSM8K-like buckets | DifficultyMap | Core | Checkpoint-local semantics are implicit in artifact metadata |
| `data_builders/gsm8k_curriculum.py` | GRPO frontier, easy regression, OPSD-hard, quarantine outputs | TrainingViewBuilder seed | Core seed | Outputs are still method-specific dicts, not typed TrainingView artifacts |
| `data_builders/opsd.py` | Thin OPSD-hard row filter | Helper | Optional | Too thin to justify long-term if TrainingViewBuilder absorbs it |
| `data/overlap.py` | Train/promotion overlap certification by id and prompt hash | Leakage guard | Core | Needs artifact-level promotion leakage guard beyond examples |
| `data/sft_dataset.py` | Tokenized SFT examples and packing collator | SFT view consumption | Core | SFT records do not yet carry full provenance schema |
| `data/preference.py` | Preference pair loading and validation | DPO view consumption | Core | Pair provenance and privileged flags are not typed |
| `evals/harness.py` | EvalSpec, metrics, slices, report writing | EvalHarness | Core | Paired promotion statistics exist, but generic EvalReport should surface uncertainty, underpowered decisions, and confidence metadata more explicitly |
| `evals/promotion.py` | Paired promotion gate with strict metrics, parse, token, severity, canary checks | PromotionPolicy | Core | Only correctness-style primary metrics are supported, deliberately |
| `evals/statistics.py` | Paired stats, bootstrap, McNemar, hierarchical bootstrap | Statistical evidence | Core | Needs more reporting integration, not new math first |
| `evals/suites.py` | Promotion suite accounting and rotation recommendation | Eval suite memory | Core | Needs CLI/report integration |
| `evals/hf_cli.py`, `run_eval.py`, `source.py`, `grades.py` | Exact-match eval CLI and source/grade contracts | Eval execution | Core seed | Narrow exact-match path |
| `opsd/context.py`, `opsd/loss.py` | OPSD privileged contexts and forward-KL loss | OPSD seed | Experimental | Missing privilege visibility metadata in artifact contract |
| `methods/*/loss.py` | SFT, DPO, GRPO loss primitives | Method primitives | Core primitives | Losses are not the flywheel; runners and artifacts are the gap |
| `training/gpu_runners.py` | TRL/Transformers SFT, DPO, GRPO adapters | MethodRunner seed | Core adapter | Real CUDA runs not locally proven; modified-GRPO knobs fail closed |
| `training/checkpoint.py` | Atomic save/load and retention | ArtifactStore seed | Core | Not tied to generic run manifest emission |
| `training/grpo_config.py`, `optimizers.py`, `runner.py` | GRPO knob detection, Muon/AdamW, runner protocol | Method config | Core | No OPD/reward-model runner contracts |
| `lifecycle.py` | Evidence bundle, HF upload, safe discard/pruning | ArtifactStore and checkpoint lifecycle | Core | Good local contract; generic `push-hf` CLI still absent |
| `registry/checkpoints.py` | JSONL checkpoint registry with remote/local state | CheckpointRegistry | Core | No query CLI/report writer |
| `registry/replay.py` | JSONL replay trace buffer | ReplayBuffer seed | Core seed | Too shallow for raw trace store and span-level agent traces |
| `search.py` | Trainer-agnostic hill climber over opaque candidates | Search contract | Optional/core | Overlaps with `climb.py`; keep until generic loop is settled |
| `climb.py` | Full orchestrator: train, eval/probe, promote, record lifecycle/replay | Flywheel loop seed | Core | Needs generic CLI and artifact bundle smoke path |
| `cli/gsm8k.py` | GSM8K split/probe/curriculum/eval commands | Agent-operable task CLI | Core seed | Task-specific command shape, no generic `pte run` |
| `cli/train.py` | GRPO config/data validation and dry-run manifest | Training CLI seed | Core seed | Real dispatch intentionally not wired for modified GRPO |
| `cli/promote.py` | Paired promotion compare CLI | Promotion CLI | Core | No `pte report` or suite diagnostics command |
| `tests/` | Focused unit and contract tests across main boundaries | Regression shield | Core | Missing end-to-end local run bundle smoke test |

## Target Architecture

The target architecture should preserve the current spine and deepen the
interfaces in this order:

```text
TaskExamples
-> ProbeRunner / RolloutRunner
-> TraceStore
-> DifficultyMap
-> CurriculumSpec
-> TrainingViewBuilder
-> MethodRunner
-> EvalHarness
-> PromotionPolicy
-> CheckpointLifecycle / Registry
-> ReplayBuffer
-> Diagnostics
-> ReportWriter
```

Required component contracts:

- `ExperimentSpec`: task, model, checkpoint, method, eval suite, seeds, budget,
  output paths, rollout config, and artifact-store policy.
- `RunManifest`: immutable run record with config hash, git commit, model and
  dataset versions, artifact hashes, hardware, start/end time, costs, and
  parent checkpoint lineage.
- `TraceStore`: append-only raw probe, rollout, eval, parser, verifier, reward,
  token, and environment evidence with split roles and prompt hashes.
- `RolloutStore`: rollout-specific API/view over trace groups; it must not
  become a second raw-evidence storage system.
- `DifficultyMap`: checkpoint-local index of easy, frontier, hard-solved,
  unsolved, parser-issue, verifier-suspect, and quarantine buckets.
- `CurriculumSpec`: sampling policy over difficulty bands, slices, replay, and
  budget.
- `TrainingView`: immutable method-specific projection over trace evidence:
  SFT, preference, RLVR/GRPO rollout groups, OPD, OPSD, reward-model, or
  ECHO-compatible views.
- `TrainingViewBuilder`: converts traces plus difficulty map plus split policy
  plus method spec into a typed training view with full provenance.
- `EvalSpec`: eval datasets, metrics, slices, thresholds, sample size, and
  confidence/error-bar requirements.
- `PromotionPolicy`: deterministic fail-closed promotion decision over paired
  candidate/incumbent eval evidence.
- `ArtifactStore`: local and Hugging Face artifact management with safe upload
  and discard semantics.
- `Verifier` / `RewardFunction`: task-specific exact-answer, parser, unit-test,
  tool-use, environment, or judge-assisted grading interface.
- `RewardModel`: learned scorer with calibration and held-out validation,
  never treated as ground truth.
- `Trainer` / `MethodRunner`: SFT, DPO, GRPO/RLVR, OPD, OPSD, reward-model, and
  future methods as pluggable strategies.
- `Diagnostics`: failure clustering, slice regressions, parse failures, reward
  hacking, length drift, pass@k collapse, and cost regressions.
- `ReportWriter`: human Markdown plus machine JSON for every run.
- `CLI`: `pte run`, `pte eval`, `pte promote`, `pte rollout`,
  `pte build-data`, `pte diagnose`, `pte push-hf`, and `pte report`.

## Gap Analysis

| Target | Current support | Gap | Priority |
| --- | --- | --- | --- |
| `ExperimentSpec` | `ExperimentConfig` | Missing rollout/eval-suite/training-view/store config | P1 |
| `RunManifest` | `FlywheelRunManifest` | Not emitted by generic run path | P1 |
| `TraceStore` | `ProbeTrace`, `ProbeArtifactRow`, `ReplayTrace` | No typed append-only raw evidence store | P1 |
| `RolloutStore` | GSM8K probe JSONL rows | No generic rollout-group API/view over trace records or env transcript support | P1 |
| `DifficultyMap` | `DifficultyMap`, bucket records | Needs checkpoint-local manifest linkage | P1 |
| `TrainingViewBuilder` | GSM8K curriculum builder | Needs typed method views and provenance contract | P1 |
| `EvalSpec` | `EvalSpec`, `EvalArtifact` | Paired promotion statistics exist, but generic EvalReport should expose uncertainty and underpowered decisions uniformly | P2 |
| `PromotionPolicy` | Strong paired gate | Good for correctness metrics; not universal by design | P1 |
| `ArtifactStore` | Lifecycle manager | Good; needs CLI exposure and report links | P2 |
| `MethodRunner` | SFT/DPO/GRPO GPU runners | No OPD/OPSD/reward-model runner contracts | P3 |
| `Diagnostics` | Promotion reasons and difficulty buckets | No report writer or failure clustering command | P2 |
| `ECHO-compatible traces` | None beyond text completions | Need span schema before trainer | P4 |
| `CLI` | GSM8K, promote compare, GRPO dry run | Generic agent-operable commands missing | P1 |

## Minimal Implementation Plan

### Phase 0: Freeze terminology and method boundaries

Objective: Make the conceptual model unambiguous before touching runner logic.

Files to modify: `README.md`, `docs/flywheel_audit.md`, and later `config.py`
only if public config names change.

Expected behavior: docs distinguish traces, rollout groups, difficulty maps,
training views, methods, and promotion gates. "Training data builder" remains
only as backwards-compatible language; the conceptual interface is
`TrainingViewBuilder`.

Tests: no code tests required unless config enums change.

Acceptance criteria: docs define `TraceStore`, `RolloutStore` as a view over
trace groups, `TrainingView`, and `TrainingViewBuilder`; docs state GRPO/RLVR
is default only for verifiable nonzero-success tasks; OPD/OPSD are optional
distillation views or auxiliaries; answer-only distillation is not reasoning
OPSD; rejection sampling is a selection operator.

Risks: renaming modules before the run path is proven would break imports for
cosmetic gain.

### Phase 1: Prove one coherent local run bundle

Objective: Make the existing GSM8K scaffold emit a complete local artifact
bundle without real GPU training.

Files to modify: `src/post_train_engine/cli/main.py`,
`src/post_train_engine/cli/gsm8k.py`, `src/post_train_engine/flywheel.py`,
`tests/test_gsm8k_cli_contracts.py`, new `scripts/smoke_gsm8k_flywheel.py` or
`tests/test_local_flywheel_smoke.py`.

Expected behavior: one command or test path creates split, probe, curriculum,
dry-run checkpoint manifest, eval artifact, promotion decision, lifecycle
evidence, ledger entries, and a run manifest under one run directory.

Tests: end-to-end filesystem smoke using injected deterministic generation.

Acceptance criteria: every recorded artifact path exists, hashes are stable,
rejected candidates cannot become incumbents, and promotion artifacts are
paired.

Risks: test runtime and accidental coupling to GSM8K-specific command names.

### Phase 2: Introduce typed TraceStore and rollout group views

Objective: Replace loose task-shaped JSONL rows with a shared raw evidence
contract while keeping GSM8K adapters.

Files to create/modify: new `src/post_train_engine/traces/schema.py`,
`src/post_train_engine/traces/store.py`,
`src/post_train_engine/traces/rollouts.py`, `src/post_train_engine/probe.py`,
`src/post_train_engine/registry/replay.py`, tests under `tests/test_traces.py`
and `tests/test_rollout_groups.py`.

Expected behavior: traces record split role, prompt hash, source checkpoint,
target checkpoint, parser/verifier/reward evidence, reward components, tokens,
and privileged visibility metadata.

Tests: reject missing example ids, prompt hashes, split roles, non-finite
rewards, malformed booleans, and train/promotion leakage.

Acceptance criteria: GSM8K probe rows can be converted losslessly to generic
traces; rollout groups reference trace IDs and record group size, reward mean,
reward variance, degenerate-group status, parser success count, verifier
identity, verifier version, and sampling config hash.

Risks: schema churn before the smoke path proves artifact needs.

### Phase 3: Rename and type TrainingViewBuilder

Objective: Make training views explicit and method-specific.

Files to create/modify: new `src/post_train_engine/training_views.py`,
`src/post_train_engine/data_builders/gsm8k_curriculum.py`, optionally move
`data_builders/` names after compatibility wrappers exist.

Expected behavior: GRPO frontier prompts, easy-regression rehearsal, OPSD-hard
privileged contexts, SFT records, and preference pairs are typed views with
source trace ids and inclusion reasons.

Tests: each view rejects missing provenance, missing privilege flags, and
promotion split contamination.

Acceptance criteria: curriculum files declare view type and method
compatibility, not just arbitrary JSONL shape.

Risks: premature generalization; mitigate by deriving fields from current
GSM8K rows.

### Phase 4: Add generic run/report/diagnose CLI

Objective: Make the flywheel agent-operable without task-specific choreography.

Files to modify/create: `src/post_train_engine/cli/main.py`, new
`cli/run.py`, `cli/report.py`, `cli/diagnose.py`, `cli/push_hf.py`,
`src/post_train_engine/reports.py`.

Expected behavior: an agent can call `pte run --config ...`, inspect
`pte report --run ...`, and get machine JSON plus human Markdown.

Tests: CLI smoke tests with temporary run directory and injected methods.

Acceptance criteria: no hidden notebook/manual step is required for the local
dry-run bundle.

Risks: CLI grows before internal contracts settle.

### Phase 5: Harden promotion/eval diagnostics

Objective: Make promotion reports explain uncertainty and regression sources.

Files to modify: `evals/harness.py`, `evals/promotion.py`,
`evals/statistics.py`, `reports.py`.

Expected behavior: reports include primary delta, CI, McNemar/sign evidence,
sample counts, power/underpowered flag, parse/token/cost/slice/canary gates,
and next experiment recommendation.

Tests: underpowered evals are flagged; missing error-bar evidence fails closed
where required.

Acceptance criteria: every promotion decision answers "compared to what, by how
much, with what uncertainty, and what regressed?"

Risks: overblocking small local smoke tests; keep strictness configurable.

### Phase 6: Add OPD/OPSD/reward-model config guards, not trainers first

Objective: Preserve method semantics and prevent silent unsafe distillation.
The first implementation pass should not add OPD, OPSD, reward-model training,
ECHO loss, or new GRPO variants; it should add schema hooks and fail-closed
config validation so those methods cannot silently misuse evidence.

Files to modify/create: `config.py`, `training_views.py`, `opsd/context.py`,
`opsd/loss.py`, tests for config rejection.

Expected behavior: OPD/OPSD configs require teacher identity, tokenizer
compatibility evidence, privileged visibility flags, verifier filtering policy,
and OOD/slice gate declarations.

Tests: malformed privilege metadata, missing teacher id, or unverifier-filtered
teacher traces fail closed.

Acceptance criteria: the repo can store and audit these views before training
on them.

Risks: too much schema before enough examples; keep optional until Phase 3 is
complete.

### Phase 7: Add artifact lifecycle/report CLI exposure

Objective: Let agents inspect, dry-run push, and report artifacts without
touching internals.

Files to create/modify: `cli/push_hf.py`, `cli/artifacts.py`, `cli/runpod.py`,
`push_hf.py`, `runpod.py`, `reports.py`, `lifecycle.py`,
`registry/checkpoints.py`.

Expected behavior: `pte push-hf --run runs/<run_id> --dry-run` shows exact HF
target paths, `pte runpod plan --run runs/<run_id> --dry-run` writes an
auditable remote execution plan, and `pte artifacts --run runs/<run_id>` lists
local evidence, promotion, trace, training-view, and lifecycle artifacts.

Tests: dry-run HF push refuses missing evidence bundles, refuses promoted
checkpoints without promotion JSON, and never requires credentials. Dry-run
RunPod planning refuses malformed run bundles, refuses raw secret values,
records secret names only, records image/GPU/setup/command/topology, and never
submits a job or mutates promotion/lifecycle state.

Acceptance criteria: rejected checkpoints keep registry rows even when local
bytes are discarded, every report links lifecycle artifacts, and remote
execution remains a launcher boundary rather than a trainer or lifecycle
backend.

Risks: HF and RunPod integration tests must stay local/dry-run by default.
Actual RunPod execution should only be added behind explicit credentials,
budget limits, and artifact-sync checks.

### Phase 8: Add ECHO-compatible trace spans

Objective: Support terminal/tool/browser/code-agent traces without importing a
full ECHO trainer.

Files to create/modify: `traces/spans.py`, `training_views.py`, possibly
`core/masking.py`.

Expected behavior: action spans and environment-observation spans can be
stored, masked, and converted into optional auxiliary training views.

Tests: span ranges are valid, non-overlapping where required, and final
verifier success remains separate from observation-prediction diagnostics.

Acceptance criteria: ECHO-compatible data can be represented and reported.

Risks: terminal-agent scope creep.

## Concrete File-Level Changes In This Patch

- Updated `README.md` to define `TraceStore`, `RolloutStore`,
  `TrainingViewBuilder`, method boundaries, and method-required-evidence
  fail-closed behavior.
- Added this audit and implementation plan at `docs/flywheel_audit.md`.
- No runtime code behavior changed in this patch.

## Test Plan

Immediate documentation patch checks:

- `git diff --check`
- focused import or contract tests are not required for docs-only changes, but
  running core promotion/curriculum tests is a useful regression guard before
  merging into active code work.

Phase test matrix:

- Trace/schema tests: malformed booleans, non-finite rewards, missing hashes,
  duplicate ids, invalid split roles, missing privilege flags.
- Training-view tests: GRPO groups require per-sample rewards and group
  variance; OPD requires teacher id and tokenizer compatibility; OPSD requires
  privileged visibility metadata; SFT/preference views require provenance.
- Eval/promotion tests: missing metrics, metric mismatch, underpowered sample
  count, unsupported primary metric, pass@k collapse, parse regression, token
  explosion, slice regression, canary failure, train/promotion overlap.
- CLI tests: `pte run`, `pte report`, `pte diagnose`, `pte push-hf` can run
  against deterministic fakes without GPU or network.
- Lifecycle tests: evidence sidecars exist before discard; upload and discard
  reject non-checkpoint directories.
- Smoke test: one local GSM8K dry-run bundle records all expected artifact
  paths and rejects candidate promotion under deterministic fake evidence.
- Method-required-evidence tests: GRPO views require rollout reward evidence;
  SFT, DPO, OPD, OPSD, and reward-model views require their own declared
  evidence instead of inheriting GRPO-specific gates.

## Proposed Cuts

* `Cut candidate:` `README.md` FinPost Extraction Plan section
* `Rationale:` It describes historical extraction intent rather than the current target flywheel contract.
* `Risk of cutting:` It may remove useful provenance for why current modules have their present names.
* `Keep if:` More code is still expected to be extracted from `finpost`.

* `Cut candidate:` `src/post_train_engine/data_builders/opsd.py`
* `Rationale:` It is currently a thin bucket filter that belongs inside a typed TrainingViewBuilder once training views exist.
* `Risk of cutting:` Existing imports may rely on it as a stable helper.
* `Keep if:` OPSD builder logic grows before the general TrainingViewBuilder lands.

* `Cut candidate:` Generic `src/post_train_engine/search.py` hill climber
* `Rationale:` It overlaps with `ClimbOrchestrator` and can become a second acceptance path if not wired through strict promotion.
* `Risk of cutting:` It is a clean trainer-agnostic search abstraction and may be useful for non-checkpoint candidate searches.
* `Keep if:` It remains diagnostic-only and cannot promote checkpoints.

* `Cut candidate:` Any modified-GRPO config knobs without a runner adapter
* `Rationale:` Unsupported knobs create false belief that the experiment was actually run under the requested algorithm.
* `Risk of cutting:` They document intended future experiments.
* `Keep if:` They remain fail-closed and are marked as future adapter contracts.

* `Cut candidate:` Production OPSD trainer work before TraceStore and TrainingView are typed
* `Rationale:` OPSD is high-risk without explicit privileged-information and OOD-regression evidence.
* `Risk of cutting:` It delays experiments that may be useful for format repair or verified self-trace distillation.
* `Keep if:` The trainer only accepts verifier-filtered views with explicit privilege metadata and cannot promote on in-domain gains alone.

* `Cut candidate:` Any separate `RolloutStore` implementation that writes its own independent storage format
* `Rationale:` Rollouts should be grouped trace records, not a second raw-evidence substrate.
* `Risk of cutting:` GRPO code may temporarily need an adapter for efficient group loading.
* `Keep if:` It is only an API/view over `TraceStore` records.

## Minimal Viable Flywheel Scope

MVP:

1. One deterministic local GSM8K dry-run bundle under a run directory.
2. Typed raw trace and rollout evidence with prompt hashes, split roles,
   verifier outputs, reward components, parse status, tokens, and source
   checkpoint ids.
3. Typed method-specific training views for GRPO frontier, easy-regression
   rehearsal, OPSD-hard diagnostics, SFT repair, and preference pairs.
4. Paired promotion decision with machine JSON and human Markdown report.
5. Lifecycle bundle with eval performance, costs, promotion gate, lineage, and
   artifact hashes.
6. Generic `pte run`, `pte report`, and `pte diagnose` for the dry-run path.
7. Dry-run artifact lifecycle/report commands for inspecting local evidence.
8. Tests proving rejected candidates cannot become incumbents and promotion
   examples cannot enter training.

## Do Later Scope

Do later:

1. Real CUDA training sweeps and distributed launchers.
2. Full OPD, OPSD, reward-model, and ECHO trainers.
3. Alternative GRPO estimators or trust-region variants.
4. Verifier-free self-improvement.
5. Multi-task plugin marketplace.
6. UI/dashboard work.
7. Aggressive cuts or module renames.

## Final Recommendation

The next engineering slice should not be OPD, OPSD, or ECHO. It should be the
boring local proof: one agent-callable run directory that contains split,
probe, traces, difficulty, training view, dry-run checkpoint manifest, eval,
promotion decision, lifecycle bundle, report, ledger, and manifest. After that
exists, adding OPD/OPSD/ECHO-compatible data becomes a schema extension rather
than a new pile of scripts.
