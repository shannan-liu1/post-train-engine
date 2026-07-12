# RunPod deployment contract

Read this file before creating any RunPod Pod. Treat every check as fail closed.

## Invariants

1. Obtain an explicit total spend cap before deployment.
2. Query the starting account balance and active Pods. Create at most one Pod at a time.
3. Use the GPU count required by the evidence claim. A one-GPU run cannot certify a two-GPU claim.
4. Pin the container image, GPU type, and GPU count. Derive the CUDA allocation filter from the image tag. Never hardcode one global CUDA version.
5. Verify one dedicated account SSH identity before Pod creation. Retain a persistent service identity; remove only an explicitly task-scoped temporary identity during teardown.
6. Use exactly one audited source-delivery mode: an allowlist bundle or an exact Git commit. Never mix source modes within one attempt.
7. Start a provider-authoritative local deletion watchdog immediately after creation. Add a Pod-side watchdog after SSH only when the Pod already has provider deletion authority; never transmit an account API key into the Pod to create that authority.
8. Delete the Pod on any failed gate. Verify zero active Pods and zero current hourly spend at teardown.
9. Verify that the current execution environment permits the approved bundle transfer before renting a Pod. An approved user intent does not guarantee that the local sandbox permits external source upload.
10. Prove the exact private-repository authentication mechanism before renting. A local authenticated `git ls-remote` proves repository access, but it does not prove that the execution environment permits transmitting that credential to a Pod.
11. Use `RunPodAllocationPolicy` and `RunPodBudget`. Supply settled campaign spend before creation. The current authorization is Secure Cloud, exactly two A40s, 40 GB ephemeral container disk, zero persistent volume, SSH only, and at most $1.50 total spend across attempts.
12. Probe SSH with `BatchMode=yes`, `PasswordAuthentication=no`, one connection attempt, and a short timeout. Never allow an unattended password prompt to consume paid time.

## Image and CUDA compatibility

The image tag and the host CUDA capability are separate allocation inputs. Derive the RunPod filter from the pinned image tag for every deployment. For example,
`runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` yields `12.8`.
RunPod can otherwise allocate an incompatible machine, reject the image before startup, and never expose SSH.

`ManualRunPodExecution.cuda_version` is computed from `container_image`; it is not a second configurable value. Config validation rejects an image without a parseable `cudaMAJOR.MINOR` tag. Pass that computed value as the sole `allowedCudaVersions` entry.

For the example image above, the create request includes:

```json
{
  "allowedCudaVersions": ["12.8"],
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
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
- Inspect provider status and image compatibility before waiting on SSH.

## Create request

Use the official image entrypoint. Do not replace it with a watchdog or setup command. A custom entrypoint can cause restart loops.

The minimum request shape below is an example. Substitute the selected image and its derived `allowedCudaVersions` value together:

```json
{
  "allowedCudaVersions": ["12.8"],
  "cloudType": "SECURE",
  "computeType": "GPU",
  "containerDiskInGb": 40,
  "globalNetworking": true,
  "gpuCount": 2,
  "gpuTypeIds": ["NVIDIA A40"],
  "gpuTypePriority": "availability",
  "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
  "interruptible": false,
  "ports": ["22/tcp"],
  "supportPublicIp": true,
  "volumeInGb": 0
}
```

Choose `SECURE` or `COMMUNITY` deliberately. Repository upload approval must cover the selected trust boundary.

## SSH lifecycle

RunPod injects account SSH keys at Pod creation. Prefer one dedicated persistent service identity for repeated Codex operations. A task-scoped identity is optional and temporary. Register either identity before creating the Pod.

1. Generate or select one dedicated private key outside the workspace. Record whether its lifecycle is `persistent_service` or `task_scoped`.
2. Restrict its Windows ACL so OpenSSH accepts it.
3. Add its public key to the existing RunPod account keys without replacing unrelated keys.
   In the RunPod console, expand the SSH public keys accordion before editing.
   Require the `Public key updated` confirmation, reload Settings, expand the
   accordion again, and verify that the new key remains present. A value visible
   before reload is not persistence evidence.
4. Create the Pod.
5. Pin the host key in a task-specific `known_hosts` file.
6. Confirm the local provider-deletion watchdog remains active. Start a Pod-side watchdog as the first remote command only when the Pod already has provider deletion authority.
7. During teardown, retain a `persistent_service` key. Remove and delete only a `task_scoped` key, then verify all unrelated account keys remain.

On Windows, grant the task owner deletion permission on a task-scoped private key. A read/write-only ACL can make teardown unable to delete it. Verify that both task-scoped key files are absent after teardown. Do not apply this deletion step to the persistent service identity.

Example watchdog for a 20-minute attempt:

```bash
nohup bash -lc 'sleep 1200; runpodctl pod delete "$RUNPOD_POD_ID" || kill 1' \
  >/workspace/pte-watchdog.log 2>&1 &
```

Do not assume that a REST-created Pod exports `RUNPOD_POD_ID`. Bind every watchdog to the literal Pod ID returned by the create response. Keep provider credentials on the trusted local machine. A local watchdog must call the provider delete API after its short deadline and record whether it deleted an active Pod or found it absent. If the Pod already exposes authenticated provider deletion, verify that authority without printing it and add a Pod-side watchdog. If Pod-side authenticated deletion is unavailable, do not transmit an account API key into the Pod and do not treat `kill 1` as sufficient because the provider may restart a Pod whose desired state remains `RUNNING`.

The repository does not yet ship a canonical detached-process launcher for the
local watchdog. This is a blocker for another paid attempt. Add and test that
launcher before creating a Pod; do not hand-write another task-local watchdog.

## Source delivery contract

Choose one source-delivery mode before allocation and record it in the attempt plan.

### Allowlist bundle mode

The benchmark bundle contains only:

- `src/`
- `post_train_engine/`
- `scripts/`
- `configs/`
- `requirements/`
- `pyproject.toml`
- `uv.lock`
- `README.md`

Exclude `__pycache__`, `*.pyc`, `.pytest_cache`, `.env*`, `.git`, `tests`, `docs`, `runs`, and local artifacts. List every archive member and reject the bundle if any forbidden path appears. Record its byte count, member count, and SHA-256 before upload. Verify the remote SHA-256 after upload.

### Exact Git commit mode

Use Git only when the reviewed source exists in a reachable remote repository. A local commit without a configured and verified remote is not deliverable through this mode.

1. Require a clean worktree after the reviewed commit.
2. Record the local commit with `git rev-parse HEAD`.
3. Push that exact commit through the user-approved remote and branch workflow before allocating a Pod.
4. On the Pod, authenticate with a user-supplied ephemeral credential. Never embed a token in a clone URL, command history, config file, or Run artifact.
5. Clone or fetch the repository, then run `git checkout --detach <expected-sha>`.
6. Require `git rev-parse HEAD` to equal the expected full SHA before installation or execution.
7. Remove the ephemeral credential during teardown. Do not persist it in the image or volume.

The execution environment must permit the chosen credential transport. Do not infer this permission from the user's approval or from a successful local Git check. If direct PAT transmission is prohibited, use a repository-scoped, read-only GitHub deploy key with SSH agent forwarding:

1. Generate the deploy key outside the repository.
2. Ask the repository owner to register only its public key with write access disabled.
3. Verify the exact remote commit locally with that key.
4. Keep the private key on the trusted local machine and load it into a task-local SSH agent.
5. Connect to the Pod with agent forwarding and clone through the forwarded agent. Never copy the deploy private key to the Pod.
6. Remove the deploy key from GitHub and delete the local private key during teardown.

Prove that agent forwarding works before allocation when the environment permits a non-billable test. Otherwise treat the first paid connection as a fail-fast gate and delete immediately on failure.

Treat branch names and tags as discovery aids, not execution identity. Only the detached full commit SHA certifies the source. If the remote is private, the user must explicitly approve that repository and trust boundary.

## Remote execution order

The repository ships one executable paid config: `configs/gsm8k_runpod_smoke.yaml`.
It is a one-step non-certifying smoke. The repository intentionally ships no
300-step paid default. Generate any full certifying config only after claiming a
fenced campaign proposal and binding provider billing settlement.

1. Start the watchdog.
2. Deliver source through the selected mode. Verify the bundle hash or exact detached Git SHA.
3. Enter a new work directory that contains only the verified source.
4. Run `python src/post_train_engine/runpod_preflight.py --constraints-only` directly so this stdlib-only gate works before project dependencies exist. Record the image-provided Torch version and CUDA build. Install the frozen non-Torch environment with `python -m pip install -r requirements/runpod.txt`, then install the repository without dependency resolution using `python -m pip install --no-deps -e ".[rlvr]"`. Require the Torch version and CUDA build to remain unchanged.
5. Run `python scripts/check_cuda_stack.py --config <config-path>`.
6. Load every RunPod config with `load_runpod_grpo_config`.
7. Confirm the distributed topology with `accelerate env` and a two-rank CUDA probe.
8. Run the benchmark with `accelerate launch --num_processes 2`.
9. Require exit code zero, exact output parity, paired ABBA trials, conservative `speedup >= 1.05`, and `certifying: true` under benchmark schema v2.
10. Download the JSON artifact and logs before teardown.

Run the full locked test suite, Ruff, architecture constraints check, and diff check locally before allocation. The paid preflight intentionally runs only remote-specific config, CUDA, and TRL compatibility gates. It stops after the first failure, uses one aggregate deadline, and always writes its JSON receipt. Do not repeat local tests or Ruff on paid compute.

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
| R4 batching changed generated text | BF16 GPU generation produced different greedy completions for scalar and batched tensor shapes despite deterministic sampling settings | Keep R4 failed closed. Preserve exact-output parity as the certification law and diagnose numerical batch-shape sensitivity before another paid attempt. |
| Security boundary rejected account API-key transfer to the Pod | A Pod-side provider-deletion watchdog would require disclosing the account credential to the rented machine | Start a local provider-authoritative deletion watchdog immediately after create; add Pod-side deletion only when authority already exists without credential transfer. |
| Deleted Pod left the operation journal in `created` state | Teardown evidence was split across an ad hoc sidecar while the canonical control-plane record remained stale | Update the operation journal atomically after a successful provider delete and retain the Pod ID and deletion time. |
| Failed R4 artifact collapsed all trial drift into one boolean | The artifact could not distinguish baseline instability from optimized-path drift | Record warmup-relative parity separately for both baseline and both optimized ABBA trials without exposing completions. |

## Primary references

- [RunPod create Pod API](https://docs.runpod.io/api-reference/pods/POST/pods)
- [RunPod SSH configuration](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod billing](https://docs.runpod.io/accounts-billing/billing)
