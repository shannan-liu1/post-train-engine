# RunPod deployment contract

Read this file before creating any RunPod Pod. Treat every check as fail closed.

## Invariants

1. Obtain an explicit total spend cap before deployment.
2. Query the starting account balance and active Pods. Create at most one Pod at a time.
3. Use the GPU count required by the evidence claim. A one-GPU run cannot certify a two-GPU claim.
4. Pin the container image, GPU type, and GPU count. Derive the CUDA allocation filter from the image tag. Never hardcode one global CUDA version.
5. Add temporary SSH access before Pod creation. Remove only that key during teardown.
6. Use exactly one audited source-delivery mode: an allowlist bundle or an exact Git commit. Never mix source modes within one attempt.
7. Start a Pod-side deletion watchdog immediately after SSH becomes available.
8. Delete the Pod on any failed gate. Verify zero active Pods and zero current hourly spend at teardown.
9. Verify that the current execution environment permits the approved bundle transfer before renting a Pod. An approved user intent does not guarantee that the local sandbox permits external source upload.

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
  "containerDiskInGb": 50,
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

RunPod injects account SSH keys at Pod creation. Register the temporary public key before creating the Pod.

1. Generate or copy one task-specific private key outside the workspace.
2. Restrict its Windows ACL so OpenSSH accepts it.
3. Add its public key to the existing RunPod account keys without replacing unrelated keys.
4. Create the Pod.
5. Pin the host key in a task-specific `known_hosts` file.
6. Start the watchdog as the first remote command.
7. During teardown, remove only the temporary public key, delete the private key, and verify the original account keys remain.

Example watchdog for a 20-minute attempt:

```bash
nohup bash -lc 'sleep 1200; runpodctl pod delete "$RUNPOD_POD_ID" || kill 1' \
  >/workspace/pte-watchdog.log 2>&1 &
```

## Source delivery contract

Choose one source-delivery mode before allocation and record it in the attempt plan.

### Allowlist bundle mode

The benchmark bundle contains only:

- `src/`
- `post_train_engine/`
- `scripts/`
- `configs/`
- `pyproject.toml`
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

Treat branch names and tags as discovery aids, not execution identity. Only the detached full commit SHA certifies the source. If the remote is private, the user must explicitly approve that repository and trust boundary.

## Remote execution order

1. Start the watchdog.
2. Deliver source through the selected mode. Verify the bundle hash or exact detached Git SHA.
3. Enter a new work directory that contains only the verified source.
4. Install the required extras: `python -m pip install -e ".[dev,rlvr]"`.
5. Run `python scripts/check_cuda_stack.py --config <config-path>`.
6. Load every RunPod config with `load_runpod_grpo_config`.
7. Confirm the distributed topology with `accelerate env` and a two-rank CUDA probe.
8. Run the benchmark with `accelerate launch --num_processes 2`.
9. Require exit code zero, exact output parity, `speedup > 1`, and `certifying: true`.
10. Download the JSON artifact and logs before teardown.

The repository preflight script includes tests and Ruff. Run it only when the uploaded bundle includes those surfaces. The reduced benchmark bundle instead relies on the full verified local suite plus the remote CUDA, config, dependency, and distributed probes above.

## Teardown and evidence

Teardown runs after success, failure, timeout, or interruption:

1. Download any available logs and evidence.
2. Delete the Pod through the REST API.
3. Verify the target Pod is absent and no unintended Pod remains.
4. Remove the temporary account SSH key and verify the original key set.
5. Delete the outside private key and local staging bundle.
6. Query ending balance and current hourly spend.
7. Record balance delta, authoritative Pod rate, elapsed time, topology, image, CUDA filter, and artifact hash.

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

## Primary references

- [RunPod create Pod API](https://docs.runpod.io/api-reference/pods/POST/pods)
- [RunPod SSH configuration](https://docs.runpod.io/pods/configuration/use-ssh)
- [RunPod billing](https://docs.runpod.io/accounts-billing/billing)
