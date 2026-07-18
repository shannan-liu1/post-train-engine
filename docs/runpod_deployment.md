# RunPod deployment contract

Read this file before creating any RunPod Pod. Treat every check as fail closed.

## Invariants

1. Obtain an explicit total spend cap before deployment.
2. Query the starting account balance and active Pods. Create at most one Pod at a time.
3. Use the GPU count required by the evidence claim. A one-GPU run cannot certify a two-GPU claim.
4. Pin the container image, GPU type, and GPU count. Derive the CUDA allocation filter from the image tag. Never hardcode one global CUDA version.
5. Verify one dedicated account SSH identity before Pod creation. Retain a persistent service identity; remove only an explicitly task-scoped temporary identity during teardown.
6. Use the canonical exact-public-Git-commit source path. The attempt compiler binds the full commit and rejects private repository URLs.
7. Persist create intent, arm the provider-authoritative local deletion watchdog, and set the provider `terminateAfter` deadline before submitting the create mutation.
8. Delete the Pod on any failed gate. Verify zero active Pods and zero current hourly spend at teardown.
9. Before loading the RunPod credential, require a clean local worktree whose `HEAD` exactly matches public remote `main` and the prepared attempt SHA.
10. Never upload a repository bundle, Git credential, provider key, or private SSH key to the Pod. The executor may pass the private-key path to local OpenSSH but never reads or serializes its contents.
11. Use `RunPodAllocationPolicy` and `RunPodBudget`. Supply settled campaign spend before creation. The checked-in allocation policy permits Secure Cloud, exactly two A40s, 40 GB ephemeral container disk, zero persistent volume, and SSH only. It does not grant standing authorization. Obtain a new explicit total spend cap for every paid campaign.
12. Probe SSH with `BatchMode=yes`, `PasswordAuthentication=no`, one connection attempt, and a short timeout. Never allow an unattended password prompt to consume paid time.
13. Apply an independent attempt runtime ceiling. The canonical default is 20 minutes. Use the earlier of the runtime ceiling and the dollar-derived deadline.
14. Treat the dollar budget as a fail-closed local target, not a provider-enforced absolute cap. The provider-scheduled runtime deadline is the independent backstop because the authoritative Pod rate arrives only after creation.

## Image and CUDA compatibility

The image tag and the host CUDA capability are separate allocation inputs. Derive the RunPod filter from the pinned image tag for every deployment. For example,
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853` yields `12.8`.
RunPod can otherwise allocate an incompatible machine, reject the image before startup, and never expose SSH.

`ManualRunPodExecution.cuda_version` is computed from `container_image`; it is not a second configurable value. Config validation rejects an image without a parseable `cudaMAJOR.MINOR` tag. Pass that computed value as the sole `allowedCudaVersions` entry.

For the example image above, the create request includes:

```json
{
  "allowedCudaVersions": ["12.8"],
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853"
}
```

Before upload, verify all of the following over SSH:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count())"
```

Require the configured GPU count, configured GPU name, visible CUDA devices, and a Torch CUDA build compatible with the image. Delete the Pod on any mismatch.

## Capacity and price

RunPod catalog prices and stock can race with allocation. Multi-GPU stock also differs from single-GPU stock because all requested GPUs must exist on one host.

- Query availability with the exact `gpuCount`.
- Treat the create response `adjustedCostPerHr` or `costPerHr` as the authoritative Pod rate.
- Do not infer the final topology price by multiplying or dividing catalog fields.
- Delete the Pod immediately when its authoritative rate exceeds the attempt budget.
- A create response without an assigned machine or usable SSH mapping is not a healthy allocation.
- Fail immediately when RunPod reports a terminal Pod status. The CUDA allocation filter and digest-pinned image provide the pre-allocation image compatibility contract.

## Create request

Use the official image entrypoint. Do not replace it with a watchdog or setup command. A custom entrypoint can cause restart loops.

The minimum request shape below is an example. Substitute the selected image and its derived `allowedCudaVersions` value together:

```json
{
  "allowedCudaVersions": ["12.8"],
  "cloudType": "SECURE",
  "computeType": "GPU",
  "containerDiskInGb": 40,
  "gpuCount": 2,
  "gpuTypeIds": ["NVIDIA A40"],
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853",
  "interruptible": false,
  "ports": ["22/tcp"],
  "supportPublicIp": true,
  "volumeInGb": 0
}
```

Choose `SECURE` or `COMMUNITY` deliberately. Repository upload approval must cover the selected trust boundary.

## SSH lifecycle

RunPod injects account SSH keys at Pod creation. Prefer one dedicated persistent service identity for repeated Codex operations. A task-scoped identity is optional and temporary. Register either identity before creating the Pod.

1. Generate or select one dedicated private key outside the workspace. Record whether its lifecycle is `persistent_service` or `task_scoped`. Prefer operator-generated keys when the private half must remain outside the agent-readable workspace. Automation may consume the key path through SSH, but it must never read, print, copy, or serialize the private-key contents.
2. Restrict its Windows ACL so OpenSSH accepts it.
3. Add its public key to the existing RunPod account keys without replacing unrelated keys.
   In the RunPod console, expand the SSH public keys accordion before editing.
   Require the `Public key updated` confirmation, reload Settings, expand the
   accordion again, and verify that the new key remains present. A value visible
   before reload is not persistence evidence.
4. Create the Pod.
5. Pin the host key in a task-specific `known_hosts` file.
6. Confirm the local provider-deletion watchdog remains active. Do not create a Pod-side watchdog or send provider authority to the Pod.
7. During teardown, retain a `persistent_service` key. Remove and delete only a `task_scoped` key, then verify all unrelated account keys remain.

On Windows, grant the task owner deletion permission on a task-scoped private key. A read/write-only ACL can make teardown unable to delete it. Verify that both task-scoped key files are absent after teardown. Do not apply this deletion step to the persistent service identity.

The canonical attempt runner launches the local watchdog from the trusted
workstation after durable intent and before provider create. The standalone
command exists for recovery and inspection:

```powershell
pte runpod watchdog `
  --journal artifacts/runpod/<attempt>/runpod_operation.json `
  --receipt artifacts/runpod/<attempt>/watchdog.json `
  --log artifacts/runpod/<attempt>/watchdog.log
```

The command reads `PTE_REMOTE_RUNPOD_ALL` from the process environment or local
`.env`, reads the deterministic Pod name or literal Pod ID and absolute deletion
deadline from the create journal, and starts a detached local worker. It never places the API key in the
child command or an artifact, and it excludes unrelated environment secrets from
the child. If the absolute deadline is already too close, the launcher deletes
and verifies the Pod synchronously instead of racing an `armed` receipt against
terminal evidence. At the deadline, the worker retries deletion and provider
absence verification three times. Exhaustion leaves the operation journal in
`delete_unverified`, never a false terminal state. Require `state: armed`, verify
the recorded PID is alive, and keep the trusted workstation awake until provider
deletion is verified. If the operation journal becomes unreadable during failure
handling, the independent watchdog receipt still records the failure and journal
error type. Do not hand-write another task-local watchdog.

## Source delivery contract

Use only an exact public Git commit. A local commit without a matching public
`main` ref is not deliverable through this path.

1. Require a clean worktree after the reviewed commit.
2. Record the local commit with `git rev-parse HEAD`.
3. Push that exact commit through the user-approved remote and branch workflow before allocating a Pod.
4. Clone the public repository without credentials, then run `git checkout --detach <expected-sha>`.
5. Require `git rev-parse HEAD` to equal the expected full SHA and require a clean checkout before installation or execution.

Treat branch names and tags as discovery aids, not execution identity. Only the
detached full commit SHA certifies the source.

## Remote execution order

The repository ships one executable paid training config: `configs/gsm8k_runpod_smoke.yaml`.
It is a one-step non-certifying smoke. The repository intentionally ships no
300-step paid default. Generate any full certifying config only after claiming a
fenced campaign proposal and binding provider billing settlement.

1. Verify exact local and public source identity before loading the provider credential.
2. Persist create intent, arm the name-based local watchdog, and submit one create request with provider-scheduled termination.
3. Enter a new work directory that contains only the verified source.
4. Run `python src/post_train_engine/runpod_preflight.py --constraints-only` directly so this stdlib-only gate works before project dependencies exist. Before installing anything, require CUDA, exactly two GPUs, and `A40` in both device names. Record the image-provided Torch version and CUDA build. Install the frozen non-Torch environment with `python -m pip install --require-hashes -r requirements/runpod.txt`, then install the repository without dependency resolution using `python -m pip install --no-deps -e ".[rlvr]"`. Require the Torch version and CUDA build to remain unchanged.
5. Run `python scripts/check_cuda_stack.py --config <config-path>`.
6. Load every RunPod config with `load_runpod_grpo_config`.
7. Confirm the distributed topology with `accelerate env` and a two-rank CUDA probe.
8. Launch R4 only through `accelerate launch --num_processes 2 -m post_train_engine.cli run --config configs/gsm8k_runpod_r4.yaml --no-env`. Confirm its canonical RunManifest reports `runtime_certified: true` before any GRPO smoke.
9. Require exit code zero, exact output parity, paired ABBA trials, conservative `speedup >= 1.05`, and `certifying: true` under benchmark schema v3. The current candidate isolates model reuse with scalar tensor shapes. Do not re-enable batching until a separate exact-contract experiment proves output equivalence.
10. Download the JSON artifact and logs before teardown.

Prepare the immutable attempt only after the reviewed commit is pushed:

```powershell
pte runpod plan `
  --config configs/gsm8k_runpod_smoke.yaml `
  --out artifacts/runpod/plans/<attempt>.json `
  --command "accelerate launch --num_processes 2 -m post_train_engine.cli run --config configs/gsm8k_runpod_r4.yaml --no-env" `
  --dry-run

pte runpod attempt prepare `
  --plan artifacts/runpod/plans/<attempt>.json `
  --attempt-dir artifacts/runpod/<attempt> `
  --repo-url https://github.com/shannan-liu1/post-train-engine.git `
  --commit-sha <full-reviewed-sha> `
  --target-spend-usd 1.5 `
  --settled-spend-usd 0.5037198807985988 `
  --max-runtime-seconds 1200
```

`prepare` is local-only. It copies the hashed plan into the otherwise empty attempt
directory. Inspect `plan.json` and `attempt.json`. Paid execution requires a second,
exact spend confirmation and the persistent service private key:

```powershell
pte runpod attempt execute `
  --attempt artifacts/runpod/<attempt>/attempt.json `
  --ssh-private-key <absolute-service-key-path> `
  --confirm-spend-cap-usd 1.5
```

The attempt runner rejects any active Pod before creation. It starts the local
watchdog before provider create, rehashes the frozen plan and referenced config,
verifies a detached public Git SHA, preserves image Torch,
runs the remote preflight and two-rank CUDA probe, runs R4 first, runs GRPO only
after certification with at least nine minutes remaining, downloads bounded
evidence, and deletes in `finally`. A lost delete response triggers provider
absence reconciliation instead of a second blind mutation. Teardown requires two
consecutive provider-absence observations and rejects any unexpected active Pod.
It also rejects conflicting `PTE_REMOTE_RUNPOD_ALL` values between the process
environment and the selected dotenv file, so an explicit `.env` cannot silently
target a different account.

The runner refuses bootstrap unless the full bootstrap, R4, evidence, and teardown
reserve remains. It rechecks the R4 reserve after bootstrap, caps evidence at
256 MiB before SCP, pins the first SSH endpoint and host-key bytes across resume,
and holds one atomic execution lock. If a process dies and leaves that ignored lock,
remove it only after provider inventory proves zero active Pods or the provider
termination deadline has elapsed.

Run the full locked test suite, Ruff, architecture constraints check, and diff check locally before allocation. The paid preflight intentionally runs only remote-specific config, CUDA, and TRL compatibility gates. It stops after the first failure, uses one aggregate deadline, and always writes its JSON receipt. Do not repeat local tests or Ruff on paid compute.

When dependencies change, run the export command recorded at the top of `requirements/runpod.txt`, then restore `# uv-lock-sha256: <sha256 of normalized uv.lock>` within its first three lines. Keep package hashes enabled. `runpod_preflight.py --constraints-only` rejects a stale, hashless, or missing binding.

## Teardown and evidence

Teardown runs after success, failure, timeout, or interruption:

1. Download any available logs and evidence.
2. Delete the Pod through the REST API and require the canonical operation journal to transition from `created` to `deleted` with the exact Pod ID and deletion time.
3. Verify the target Pod is absent and no unintended Pod remains.
4. Remove only a task-scoped account SSH key and verify the original key set. Retain the dedicated persistent service identity.
5. Delete only task-scoped private keys and the local staging bundle.
6. Query ending balance and current hourly spend.
7. Record balance delta, authoritative Pod rate, elapsed time, topology, image, CUDA filter, and artifact hash.

Billing settlement requires two matching provider observations. Record the first
nonempty amount as provisional. After Pod absence is verified, query again with an
explicit end time. Settle only when the second amount matches the durable first
observation. A first nonempty billing response is never final evidence.

Use the canonical settlement command for both observations:

```powershell
pte runpod attempt settle --journal <operation.json> --pod-id <id> `
  --start-time <iso-time> --out <billing-receipt.json>

pte runpod attempt settle --journal <operation.json> --pod-id <id> `
  --start-time <iso-time> --end-time <iso-time> --final `
  --out <billing-receipt.json>
```

## Failures already observed

| Failure | Cause | Permanent prevention |
| --- | --- | --- |
| Image not found | Stale `runpod/pytorch:2.4.0-py3.11-cuda12.1` tag | Use a current official image and verify it before paid allocation. |
| Minimum CUDA version not met | The host did not satisfy the selected image's CUDA requirement | Derive `allowedCudaVersions` from each image tag and reject images whose requirement cannot be parsed. |
| Pod never exposed SSH | Image compatibility failure was mistaken for slow startup | Inspect provider status and CUDA compatibility before polling SSH. |
| No two-GPU capacity | Single-GPU or catalog stock was treated as multi-GPU availability | Query and create with the exact GPU count; accept capacity failure without weakening evidence. |
| SSH key unavailable | Key was registered after Pod creation | Register the temporary key before creation and restore account state afterward. |
| Restart loop | Watchdog replaced the official image entrypoint | Keep the default entrypoint and start the watchdog after SSH. |
| Bundle contained caches | Allowlisted directories still contained `__pycache__` | Exclude caches and audit every archive member before upload. |
| Local sandbox denied source upload | The execution environment prohibited private workspace transfer even after explicit user approval | Confirm transfer capability before renting; otherwise use the exact Git commit mode, a user-supplied prebuilt image, or a user-controlled checkout already present on the Pod. Never bypass the policy. |
| PowerShell state update deleted a healthy Pod | A new property was assigned to a fixed `PSCustomObject`, which threw inside the fail-closed handler | Rebuild operational state through an ordered map, validate it locally, then persist it. |
| Private GitHub PAT could not be sent to the Pod | The execution security boundary rejected credential transmission to a third-party machine despite explicit workflow approval | Use a repository-scoped read-only deploy key through SSH agent forwarding, or require a user-controlled checkout. Never bypass the credential boundary. |
| Watchdog had an empty Pod ID | A REST-created Pod did not export `RUNPOD_POD_ID` | Bind the create-response Pod ID literally and verify Pod-side API authentication without exposing it. |
| Temporary private key survived teardown | Its Windows ACL granted read/write but not delete permission | Grant the task owner deletion permission and verify both key files are absent after teardown. |
| Pod-specific `SSH_PUBLIC_KEY` did not authorize REST-created Pods | Two Secure 2xA40 Pods exposed healthy SSH endpoints but rejected the injected ed25519 key; the first non-batch probe waited on authentication | Do not rely on create-request environment injection for SSH. Verify an account-level key with `runpodctl ssh list-keys` or the RunPod console before allocation, and use batch-mode SSH so rejection fails in seconds. |
| RunPod console key edit appeared saved but reverted after reload | The SSH key form remained mounted while its accordion was collapsed, so automation could inspect hidden state without a reliable interactive submission | Expand the SSH public keys accordion, submit through its visible controls, require the success confirmation, then reload and verify persistence before allocation. |
| Pre-install constraint check could not import Pydantic | Importing the package-scoped verifier executed the package root before frozen dependencies existed | Execute `src/post_train_engine/runpod_preflight.py --constraints-only` directly before installation; it uses only the standard library. |
| R4 batching changed generated text | The benchmark combined model reuse with BF16 tensor-shape batching, and the combined candidate produced different greedy completions | Keep the failure closed. Benchmark model reuse alone with scalar tensor shapes. Treat numerical batch-shape sensitivity as a separate conjecture that requires its own exact-parity experiment. |
| Security boundary rejected account API-key transfer to the Pod | A Pod-side provider-deletion watchdog would require disclosing the account credential to the rented machine | Start a local provider-authoritative deletion watchdog immediately after create; add Pod-side deletion only when authority already exists without credential transfer. |
| Deleted Pod left the operation journal in `created` state | Teardown evidence was split across an ad hoc sidecar while the canonical control-plane record remained stale | Update the operation journal atomically after a successful provider delete and retain the Pod ID and deletion time. |
| Failed R4 artifact collapsed all trial drift into one boolean | The artifact could not distinguish baseline instability from optimized-path drift | Record warmup-relative parity separately for both baseline and both optimized ABBA trials without exposing completions. |
| Local watchdog started from an ignored task script | The safety-critical process had no canonical launcher, absolute deadline, or secret-minimized child environment | Use `pte runpod watchdog`, require its armed receipt, and keep the trusted host awake until teardown is verified. |
| Watchdog used one absence check, could overwrite terminal evidence when launched after its deadline, and lost its failure receipt if the journal update failed | Eventual consistency, late launch, or journal corruption could leave a billable Pod with misleading or missing local evidence | Delete expired targets synchronously, retry provider deletion and absence verification three times, retain `delete_unverified` on exhaustion, and write the independent watchdog receipt even when journal repair fails. |

## Primary references

- [RunPod create Pod API](https://docs.runpod.io/api-reference/pods/POST/pods)
- [RunPod GraphQL Pod management](https://docs.runpod.io/sdks/graphql/manage-pods)
- [Official runpodctl GraphQL client](https://github.com/runpod/runpodctl/blob/main/internal/api/graphql.go)
- [Pinned RunPod PyTorch image digest](https://hub.docker.com/layers/runpod/pytorch/2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04/images/sha256-cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853)
- [RunPod SSH configuration](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod billing](https://docs.runpod.io/accounts-billing/billing)
