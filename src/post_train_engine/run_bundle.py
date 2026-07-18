"""Portable run-bundle contracts and validation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from post_train_engine.traces.schema import TraceRecord

CANONICAL_RUN_SCHEMA_VERSION = "post_train_run_v1"
_FROZEN_FORBID = ConfigDict(frozen=True, extra="forbid")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourceIdentity(BaseModel):
    """Identity of the source tree used to execute a Run."""

    model_config = _FROZEN_FORBID

    commit_sha: str | None = None
    state: Literal["clean", "tracked_dirty", "untracked_dirty", "mixed", "unknown"]
    tracked_diff_sha256: str | None = None
    untracked_sha256: str | None = None

    @field_validator("commit_sha", "tracked_diff_sha256", "untracked_sha256")
    @classmethod
    def _optional_identity_must_be_nonempty(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("source identity values must be non-empty when provided")
        return value


class ArtifactRef(BaseModel):
    """Portable reference to an artifact stored inside a Run bundle."""

    model_config = _FROZEN_FORBID

    path: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)
    sha256: str
    required: StrictBool = True
    visibility: Literal["standard", "sealed"] = "standard"

    @field_validator("path")
    @classmethod
    def _path_must_be_portable_relative(cls, value: str) -> str:
        if "\\" in value:
            raise ValueError("artifact path must use forward slashes")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must remain inside the run bundle")
        if any(part in {"", "."} for part in path.parts):
            raise ValueError("artifact path must be normalized")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha256_must_be_canonical(cls, value: str) -> str:
        if not _SHA256_RE.fullmatch(value):
            raise ValueError("artifact sha256 must use sha256:<64 lowercase hex chars>")
        return value


class ResolvedInput(BaseModel):
    """Requested and observed identity for one external Run input."""

    model_config = _FROZEN_FORBID

    kind: Literal["model", "dataset", "verifier", "teacher", "environment"]
    requested_id: str = Field(..., min_length=1)
    resolved_id: str | None = None
    requested_revision: str | None = None
    resolved_revision: str | None = None
    fingerprint: str | None = None
    resolution_state: Literal["exact", "provider_managed", "unresolved"]
    non_certifying_reason: str | None = None

    @field_validator(
        "resolved_id",
        "requested_revision",
        "resolved_revision",
        "non_certifying_reason",
    )
    @classmethod
    def _optional_text_must_be_nonempty(cls, value: str | None) -> str | None:
        if value == "":
            raise ValueError("input identity values must be non-empty when provided")
        return value

    @field_validator("fingerprint")
    @classmethod
    def _fingerprint_must_be_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256_RE.fullmatch(value):
            raise ValueError(
                "input fingerprint must use sha256:<64 lowercase hex chars>"
            )
        return value

    @model_validator(mode="after")
    def _resolution_evidence_must_be_honest(self) -> ResolvedInput:
        if self.resolution_state == "exact":
            if not self.resolved_id:
                raise ValueError("exact input identity requires resolved_id")
            if not self.resolved_revision and not self.fingerprint:
                raise ValueError(
                    "exact input identity requires resolved_revision or fingerprint"
                )
        elif not self.non_certifying_reason:
            raise ValueError("non-exact input identity requires non_certifying_reason")
        return self


class RunManifest(BaseModel):
    """Canonical provenance envelope for every Run execution mode."""

    model_config = _FROZEN_FORBID

    schema_version: Literal["post_train_run_v1"] = CANONICAL_RUN_SCHEMA_VERSION
    run_id: str = Field(..., min_length=1)
    candidate_id: str = Field(..., min_length=1)
    parent_candidate_id: str | None = None
    task_name: str = Field(..., min_length=1)
    model_id: str = Field(..., min_length=1)
    status: Literal[
        "planned", "running", "pending_settlement", "failed", "rejected", "promoted"
    ]
    source: SourceIdentity
    inputs: dict[str, ResolvedInput]
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _finalized_runs_require_resolved_core_inputs(self) -> RunManifest:
        missing = {"model", "dataset"}.difference(self.inputs)
        if missing:
            raise ValueError(
                "Run inputs missing required identities: " + ", ".join(sorted(missing))
            )
        if self.status in {"pending_settlement", "promoted", "rejected"}:
            unresolved = sorted(
                name
                for name, identity in self.inputs.items()
                if identity.resolution_state == "unresolved"
            )
            if unresolved:
                raise ValueError(
                    "finalized Run inputs must be resolved: " + ", ".join(unresolved)
                )
        return self


class RunBundle:
    """Read and validate one portable Run directory."""

    def __init__(self, root: str | Path, manifest: RunManifest) -> None:
        self.root = Path(root).resolve()
        self.manifest = manifest

    @classmethod
    def load(cls, root: str | Path) -> RunBundle:
        root = Path(root)
        manifest_path = root / "manifest.json"
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"manifest does not exist: {manifest_path}"
            ) from exc
        if not isinstance(raw, dict):
            raise ValueError(f"manifest JSON root must be an object: {manifest_path}")
        return cls(root, RunManifest.model_validate(raw))

    def artifact_path(self, name: str, *, allow_sealed: bool = False) -> Path:
        try:
            ref = self.manifest.artifacts[name]
        except KeyError as exc:
            raise ValueError(f"manifest missing artifact {name!r}") from exc
        if ref.visibility == "sealed" and not allow_sealed:
            raise ValueError(
                f"sealed artifact {name!r} requires explicit sealed evidence access"
            )
        candidate = (self.root / PurePosixPath(ref.path)).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"artifact path escapes run bundle: {ref.path}") from exc
        return candidate

    def verified_artifact_path(
        self,
        name: str,
        *,
        allow_sealed: bool = False,
    ) -> Path:
        """Return one contained artifact only when its recorded bytes still match."""

        path = self.artifact_path(name, allow_sealed=allow_sealed)
        if not path.is_file():
            raise ValueError(f"artifact is not a file: {name!r}")
        actual = _sha256(path)
        expected = self.manifest.artifacts[name].sha256
        if actual != expected:
            raise ValueError(
                f"artifact hash mismatch for {name!r}: expected {expected}, got {actual}"
            )
        return path

    def validate(self) -> dict[str, Any]:
        manifest_path = self.root / "manifest.json"
        manifest_digest = _sha256(manifest_path)
        statuses: list[dict[str, Any]] = [
            {
                "name": "manifest",
                "kind": "manifest",
                "path": "manifest.json",
                "required": True,
                "status": "ok",
                "exists": True,
                "expected_sha256": manifest_digest,
                "actual_sha256": manifest_digest,
            }
        ]
        for name, ref in sorted(self.manifest.artifacts.items()):
            path = self.artifact_path(name, allow_sealed=True)
            actual = _sha256(path) if path.is_file() else None
            if not path.is_file():
                status = "missing"
            elif actual != ref.sha256:
                status = "hash_mismatch"
            else:
                status = "ok"
            statuses.append(
                {
                    "name": name,
                    "kind": ref.kind,
                    "path": "[SEALED]" if ref.visibility == "sealed" else ref.path,
                    "visibility": ref.visibility,
                    "required": ref.required,
                    "status": status,
                    "exists": path.is_file(),
                    "expected_sha256": ref.sha256,
                    "actual_sha256": actual,
                }
            )
        physical_failures = any(
            status["required"] and status["status"] != "ok" for status in statuses
        )
        if not physical_failures:
            statuses.append(self._promotion_consistency_status())
            statuses.append(self._training_view_leakage_status())
            statuses.append(self._grpo_reward_evidence_status())
            statuses.append(self._policy_training_trace_status())
        failures = [
            status
            for status in statuses
            if status["required"] and status["status"] != "ok"
        ]
        return {
            "run_id": self.manifest.run_id,
            "candidate_id": self.manifest.candidate_id,
            "status": "ok" if not failures else "failed",
            "required_count": sum(status["required"] for status in statuses),
            "ok_count": sum(
                status["required"] and status["status"] == "ok" for status in statuses
            ),
            "failure_count": len(failures),
            "failures": failures,
            "artifacts": statuses,
        }

    def _promotion_consistency_status(self) -> dict[str, Any]:
        try:
            self._validate_promotion_consistency()
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            return _semantic_status(
                "promotion_consistency",
                status="semantic_failed",
                message=str(exc),
            )
        return _semantic_status("promotion_consistency", status="ok")

    def _validate_promotion_consistency(self) -> None:
        if self.manifest.status not in {"promoted", "rejected"}:
            return

        expected_decision = (
            "promote" if self.manifest.status == "promoted" else "reject"
        )
        decision = self._read_artifact_object("promotion_decision")
        if decision.get("decision") != expected_decision:
            raise ValueError(
                "promotion decision does not match manifest status: "
                f"expected {expected_decision!r}"
            )

        if "final_report_json" in self.manifest.artifacts:
            report = self._read_artifact_object("final_report_json")
            promotion = report.get("promotion")
            if not isinstance(promotion, dict):
                raise ValueError("final report promotion must be an object")
            if promotion.get("decision") != expected_decision:
                raise ValueError(
                    "final report promotion does not match manifest status: "
                    f"expected {expected_decision!r}"
                )

        if "lifecycle" in self.manifest.artifacts:
            lifecycle = self._read_artifact_object("lifecycle")
            promoted = lifecycle.get("promoted")
            if not isinstance(promoted, bool):
                raise ValueError("lifecycle promoted must be a boolean")
            if promoted != (expected_decision == "promote"):
                raise ValueError("lifecycle promotion does not match manifest status")

    def _read_artifact_object(self, name: str) -> dict[str, Any]:
        if name not in self.manifest.artifacts:
            raise ValueError(f"manifest missing artifact {name!r}")
        path = self.verified_artifact_path(name, allow_sealed=True)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"artifact {name!r} JSON root must be an object")
        return value

    def _training_view_leakage_status(self) -> dict[str, Any]:
        try:
            trace_artifact = (
                "input_traces"
                if "input_traces" in self.manifest.artifacts
                else "traces"
            )
            traces_by_id: dict[str, dict[str, Any]] | None = None
            for name, ref in self.manifest.artifacts.items():
                if not (name.endswith("_view") or ref.kind.endswith("_view")):
                    continue
                view = self._read_artifact_object(name)
                trace_ids = view.get("source_trace_ids")
                roles = view.get("source_split_roles")
                methods = view.get("method_compatibility")
                if view.get("run_id") != self.manifest.run_id:
                    raise ValueError(f"{name} belongs to a different Run")
                if view.get("task_id") != self.manifest.task_name:
                    raise ValueError(f"{name} belongs to a different task")
                if not isinstance(trace_ids, list) or not trace_ids:
                    raise ValueError(f"{name} missing source_trace_ids")
                if not isinstance(roles, list) or not roles:
                    raise ValueError(f"{name} missing source_split_roles")
                protected = {
                    "selection",
                    "diagnostic",
                    "promotion",
                    "canary",
                    "unseen",
                } & set(roles)
                if protected:
                    raise ValueError(
                        f"{name} uses protected split sources: "
                        + ", ".join(sorted(protected))
                    )
                if not isinstance(methods, list) or not methods:
                    raise ValueError(f"{name} missing method_compatibility")
                if traces_by_id is None:
                    traces_by_id = self._traces_by_id(trace_artifact)
                missing = sorted(set(trace_ids).difference(traces_by_id))
                if missing:
                    raise ValueError(f"{name} references missing trace: {missing[0]}")
                observed_roles = {
                    str(traces_by_id[trace_id].get("split_role"))
                    for trace_id in trace_ids
                }
                if observed_roles != set(roles):
                    raise ValueError(
                        f"{name} split roles do not match referenced trace evidence"
                    )
                row_trace_ids, row_roles = self._validate_training_view_rows(
                    name, view, traces_by_id
                )
                if row_trace_ids != set(trace_ids):
                    raise ValueError(
                        f"{name} row trace IDs do not match its lineage envelope"
                    )
                if row_roles != set(roles):
                    raise ValueError(
                        f"{name} row roles do not match its lineage envelope"
                    )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            return _semantic_status(
                "training_view_leakage",
                status="semantic_failed",
                message=str(exc),
            )
        return _semantic_status("training_view_leakage", status="ok")

    def _grpo_reward_evidence_status(self) -> dict[str, Any]:
        try:
            grpo_views: list[tuple[str, dict[str, Any]]] = []
            for name, ref in self.manifest.artifacts.items():
                if not (name.endswith("_view") or ref.kind.endswith("_view")):
                    continue
                view = self._read_artifact_object(name)
                if view.get("view_type") == "grpo_rollout":
                    grpo_views.append((name, view))
            if not grpo_views:
                return _semantic_status("grpo_reward_evidence", status="ok")
            trace_artifact = (
                "input_traces"
                if "input_traces" in self.manifest.artifacts
                else "traces"
            )
            traces_by_id = self._traces_by_id(trace_artifact)
            for view_name, view in grpo_views:
                source_trace_ids = set(view.get("source_trace_ids", ()))
                missing = sorted(source_trace_ids.difference(traces_by_id))
                if missing:
                    raise ValueError(
                        f"{view_name} references missing trace: {missing[0]}"
                    )
                for trace_id in sorted(source_trace_ids):
                    trace = traces_by_id[trace_id]
                    for field in (
                        "policy_version",
                        "policy_step",
                        "policy_step_evidence",
                        "rollout_group_id",
                        "sampling_config",
                        "verifier_id",
                    ):
                        value = trace.get(field)
                        if value is None or value == "":
                            raise ValueError(f"trace {trace_id} missing {field}")
                    if (
                        not isinstance(trace["sampling_config"], dict)
                        or not trace["sampling_config"]
                    ):
                        raise ValueError(f"trace {trace_id} missing sampling_config")
            groups = self._read_jsonl_artifact("rollout_groups")
            if not groups:
                raise ValueError("missing rollout groups")
            if not any(
                isinstance(group.get("rewards"), list)
                and group["rewards"]
                and isinstance(group.get("reward_variance"), int | float)
                for group in groups
            ):
                raise ValueError("GRPO rollout view requires reward evidence")
            if not any(group.get("degenerate_group") is False for group in groups):
                raise ValueError(
                    "GRPO rollout view requires at least one non-degenerate reward group"
                )
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            return _semantic_status(
                "grpo_reward_evidence",
                status="semantic_failed",
                message=str(exc),
            )
        return _semantic_status("grpo_reward_evidence", status="ok")

    def _policy_training_trace_status(self) -> dict[str, Any]:
        names = sorted(
            name
            for name in self.manifest.artifacts
            if name.startswith("policy_trace_rank")
        )
        try:
            is_runpod_grpo = self.manifest.metadata.get("execution_mode") == "runpod_grpo"
            if not names:
                if is_runpod_grpo:
                    train_result = self._read_artifact_object("train_result")
                    if train_result.get("status") == "trained":
                        raise ValueError(
                            "trained RunPod GRPO run omitted policy training traces"
                        )
                return _semantic_status("policy_training_traces", status="ok")
            train_result = self._read_artifact_object("train_result")
            distributed = train_result.get("distributed")
            world_size = (
                distributed.get("world_size") if isinstance(distributed, dict) else None
            )
            expected = (
                [f"policy_trace_rank{rank}" for rank in range(world_size)]
                if (type(world_size) is int and world_size > 0)
                else []
            )
            if set(names) != set(expected):
                raise ValueError(
                    "policy training traces do not cover every distributed rank"
                )
            training_view = self._read_artifact_object("method_training_view")
            data_ref = training_view.get("data_artifact")
            if not isinstance(data_ref, dict) or not isinstance(
                data_ref.get("path"), str
            ):
                raise ValueError("policy training traces require a TrainingView data path")
            data_path = (self.root / PurePosixPath(data_ref["path"])).resolve()
            try:
                data_path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("TrainingView data path escapes the Run") from exc
            if not data_path.is_file() or _sha256(data_path) != data_ref.get("sha256"):
                raise ValueError("TrainingView data bytes do not match policy trace evidence")
            eligible_example_ids = {
                str(row.get("example_id"))
                for row in self._read_jsonl_path(data_path, label="TrainingView data")
                if row.get("example_id") is not None
            }
            if not eligible_example_ids:
                raise ValueError("TrainingView data contains no eligible example IDs")
            trace_ids: set[str] = set()
            for name in names:
                rows = self._read_jsonl_artifact(name)
                if not rows:
                    raise ValueError(f"{name} contains no policy-changing traces")
                for row in rows:
                    trace = TraceRecord.model_validate(row)
                    if trace.split_role != "train":
                        raise ValueError(f"{name} contains a non-training trace")
                    if trace.run_id != self.manifest.run_id:
                        raise ValueError(f"{name} belongs to a different Run")
                    if trace.task_id != self.manifest.task_name:
                        raise ValueError(f"{name} belongs to a different task")
                    if trace.example_id not in eligible_example_ids:
                        raise ValueError(
                            f"{name} trace example was not eligible in the TrainingView"
                        )
                    if trace.trace_id in trace_ids:
                        raise ValueError(
                            f"duplicate policy training trace id: {trace.trace_id}"
                        )
                    trace_ids.add(trace.trace_id)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
            return _semantic_status(
                "policy_training_traces",
                status="semantic_failed",
                message=str(exc),
            )
        return _semantic_status("policy_training_traces", status="ok")

    def _traces_by_id(self, artifact_name: str) -> dict[str, dict[str, Any]]:
        traces: dict[str, dict[str, Any]] = {}
        for trace in self._read_jsonl_artifact(artifact_name):
            trace_id = trace.get("trace_id")
            if not isinstance(trace_id, str) or not trace_id:
                raise ValueError(
                    f"artifact {artifact_name!r} contains an invalid trace id"
                )
            if trace_id in traces:
                raise ValueError(f"duplicate trace id in {artifact_name!r}: {trace_id}")
            traces[trace_id] = trace
        return traces

    def _validate_training_view_rows(
        self,
        view_name: str,
        view: dict[str, Any],
        traces_by_id: dict[str, dict[str, Any]],
    ) -> tuple[set[str], set[str]]:
        data_ref = view.get("data_artifact")
        if not isinstance(data_ref, dict):
            raise ValueError(f"{view_name} missing data_artifact")
        raw_path = data_ref.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{view_name} data_artifact has no path")
        path = (self.root / PurePosixPath(raw_path)).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"{view_name} data_artifact escapes the Run") from exc
        if not path.is_file() or _sha256(path) != data_ref.get("sha256"):
            raise ValueError(f"{view_name} data_artifact bytes do not match")
        row_trace_ids: set[str] = set()
        row_roles: set[str] = set()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{view_name} training row {line_number} must be an object"
                )
            trace_ids = row.get("source_trace_ids")
            roles = row.get("source_split_roles")
            if not isinstance(trace_ids, list) or not trace_ids:
                raise ValueError(
                    f"{view_name} training row {line_number} missing source_trace_ids"
                )
            if not isinstance(roles, list) or not roles:
                raise ValueError(
                    f"{view_name} training row {line_number} missing source_split_roles"
                )
            missing = sorted(set(trace_ids).difference(traces_by_id))
            if missing:
                raise ValueError(
                    f"{view_name} training row {line_number} references missing trace"
                )
            observed = {
                str(traces_by_id[trace_id].get("split_role"))
                for trace_id in trace_ids
            }
            if observed != set(roles):
                raise ValueError(
                    f"{view_name} training row {line_number} roles do not match traces"
                )
            row_trace_ids.update(trace_ids)
            row_roles.update(roles)
        return row_trace_ids, row_roles

    def _read_jsonl_artifact(self, name: str) -> list[dict[str, Any]]:
        path = self.verified_artifact_path(name, allow_sealed=True)
        return self._read_jsonl_path(path, label=f"artifact {name!r}")

    @staticmethod
    def _read_jsonl_path(path: Path, *, label: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{label} line {line_number} must be an object")
            rows.append(value)
        return rows


def capture_source_identity(repo_root: str | Path) -> SourceIdentity:
    """Capture commit and dirty-source digests without hiding untracked code."""

    repo_root = Path(repo_root).resolve()
    try:
        commit_sha = _git(repo_root, "rev-parse", "HEAD").decode("ascii").strip()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return SourceIdentity(commit_sha=None, state="unknown")

    tracked_diff = b"\n".join(
        (
            _git(repo_root, "diff", "--binary", "HEAD", "--", "."),
            _git(repo_root, "diff", "--binary", "--cached", "HEAD", "--", "."),
        ),
    ).strip()
    raw_untracked = _git(
        repo_root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    untracked_paths = tuple(
        sorted(
            path.decode("utf-8", errors="surrogateescape")
            for path in raw_untracked.split(b"\0")
            if path
        ),
    )
    has_tracked = bool(tracked_diff)
    has_untracked = bool(untracked_paths)
    if has_tracked and has_untracked:
        state = "mixed"
    elif has_tracked:
        state = "tracked_dirty"
    elif has_untracked:
        state = "untracked_dirty"
    else:
        state = "clean"
    return SourceIdentity(
        commit_sha=commit_sha or None,
        state=state,
        tracked_diff_sha256=_sha256_bytes(tracked_diff) if has_tracked else None,
        untracked_sha256=(
            _untracked_source_sha256(repo_root, untracked_paths)
            if has_untracked
            else None
        ),
    )


def make_artifact_ref(
    run_root: str | Path,
    artifact_path: str | Path,
    *,
    kind: str,
    required: bool = True,
    visibility: Literal["standard", "sealed"] = "standard",
) -> ArtifactRef:
    """Build one portable content-addressed reference inside a Run bundle."""

    root = Path(run_root).resolve()
    path = Path(artifact_path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact must remain inside the run bundle: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {path}")
    return ArtifactRef(
        path=relative.as_posix(),
        kind=kind,
        sha256=_sha256(path),
        required=required,
        visibility=visibility,
    )


def write_manifest_atomic(run_root: str | Path, manifest: RunManifest) -> Path:
    """Finalize a manifest atomically and tolerate an identical resume."""

    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "manifest.json"
    body = json.dumps(
        manifest.model_dump(mode="json", exclude_none=True),
        indent=2,
        sort_keys=True,
    )
    if target.is_file():
        current = target.read_text(encoding="utf-8")
        if current == body:
            return target
    temporary = root / ".manifest.json.tmp"
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(target)
    return target


def _git(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _untracked_source_sha256(repo_root: Path, paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for relative in paths:
        portable = PurePosixPath(relative.replace("\\", "/"))
        if portable.is_absolute() or ".." in portable.parts:
            raise ValueError(f"untracked source path escapes repository: {relative}")
        path = repo_root.joinpath(*portable.parts)
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            digest.update(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"file\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        elif path.is_dir():
            digest.update(b"directory\0")
        else:
            digest.update(b"missing-during-capture\0")
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _semantic_status(
    name: str,
    *,
    status: Literal["ok", "semantic_failed"],
    message: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "kind": "semantic",
        "path": f"semantic://{name}",
        "required": True,
        "status": status,
        "exists": True,
        "expected_sha256": None,
        "actual_sha256": None,
    }
    if message is not None:
        result["message"] = message
    return result


__all__ = [
    "ArtifactRef",
    "CANONICAL_RUN_SCHEMA_VERSION",
    "RunBundle",
    "RunManifest",
    "ResolvedInput",
    "SourceIdentity",
    "capture_source_identity",
    "make_artifact_ref",
    "write_manifest_atomic",
]
