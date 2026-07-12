"""Deterministic fake providers for dry-run and CI."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from post_train_engine.api_schemas import Candidate, JobHandle, JobRequest, JobResult, JobStatus


class _ImmediateProvider:
    provider_id: str
    provider_type: str
    recovery_policy = "replay_safe"

    def __init__(self, provider_id: str, provider_type: str) -> None:
        self.provider_id = provider_id
        self.provider_type = provider_type
        self._results: dict[str, JobResult] = {}

    def poll_job(self, handle: JobHandle) -> JobStatus:
        if handle.job_id not in self._results:
            return JobStatus(state="failed", message="unknown provider job id")
        return self._results[handle.job_id].status

    def fetch_result(self, handle: JobHandle) -> JobResult:
        try:
            return self._results[handle.job_id]
        except KeyError as exc:
            raise RuntimeError(f"unknown provider job id: {handle.provider_job_id}") from exc

    def reconcile_job(
        self,
        request: JobRequest,
        _handle: JobHandle | None,
    ) -> JobResult | None:
        return self._results.get(request.job_id)


class FakeInferenceProvider(_ImmediateProvider):
    """Deterministic GSM8K-like inference backend.

    This provider is not a model simulator. It exists to exercise orchestration,
    persistence, parsing, metrics, and promotion without secrets or network I/O.
    """

    def __init__(self, provider_id: str = "fake-inference") -> None:
        super().__init__(provider_id, "fake")

    def submit_job(self, request: JobRequest) -> JobHandle:
        if request.provider_id != self.provider_id:
            raise ValueError("request provider_id does not match fake inference provider")
        if request.job_type not in {"rollout_generation", "evaluation"}:
            raise ValueError(f"fake inference cannot handle job type: {request.job_type}")
        candidate = Candidate.model_validate(request.payload["candidate"])
        generations = [
            self._generate(candidate, prompt, request.job_type)
            for prompt in list(request.payload.get("prompts", ()))
        ]
        handle = JobHandle(
            job_id=request.job_id,
            job_type=request.job_type,
            provider_id=self.provider_id,
            provider_job_id=f"fake://{request.job_id}",
            metadata={"provider_type": self.provider_type},
        )
        self._results[request.job_id] = JobResult(
            handle=handle,
            status=JobStatus(state="succeeded"),
            payload={"generations": generations},
            metadata={"deterministic_fake": True},
        )
        return handle

    def _generate(
        self,
        candidate: Candidate,
        prompt_row: dict[str, Any],
        job_type: str,
    ) -> dict[str, Any]:
        prompt = str(prompt_row["prompt"])
        example_id = str(prompt_row["example_id"])
        sample_index = int(prompt_row.get("sample_index", 0))
        gold = _gold_answer_for_prompt(prompt)
        correct = _fake_correctness(
            candidate=candidate,
            example_id=example_id,
            sample_index=sample_index,
            job_type=job_type,
        )
        answer = gold if correct else "999"
        completion = f"fake deterministic reasoning\n<answer>{answer}</answer>"
        return {
            "example_id": example_id,
            "sample_index": sample_index,
            "split_role": str(prompt_row.get("split_role", "eval")),
            "completion": completion,
            "completion_tokens": len(completion.split()),
            "finish_reason": "stop",
            "provider_job_id": f"fake://{example_id}/{sample_index}",
            "raw_response": {
                "provider": "fake",
                "gold_known_to_fake": True,
                "correct": correct,
            },
        }


class FakePromptAdapterProvider(_ImmediateProvider):
    """Deterministic fake prompt-adaptation backend for dry-run tests."""

    def __init__(self, provider_id: str = "fake-trainer") -> None:
        super().__init__(provider_id, "fake_prompt_adapter")

    def submit_job(self, request: JobRequest) -> JobHandle:
        if request.provider_id != self.provider_id:
            raise ValueError("request provider_id does not match fake adapter provider")
        if request.job_type != "candidate_adaptation":
            raise ValueError(f"fake adapter cannot handle job type: {request.job_type}")
        baseline = Candidate.model_validate(request.payload["baseline_candidate"])
        candidate_id = str(request.payload.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate adaptation requires a planned candidate_id")
        candidate = Candidate(
            candidate_id=candidate_id,
            model_id=baseline.model_id,
            parent_id=baseline.candidate_id,
            system_prompt=(
                baseline.system_prompt
                + "\nSolve arithmetic carefully and always emit <answer>number</answer>."
            ).strip(),
            prompt_prefix=baseline.prompt_prefix,
            prompt_suffix=baseline.prompt_suffix,
            adapter_kind="fake_prompt_adapter",
            metadata={
                "provider": self.provider_id,
                "adaptation": "deterministic_fake_prompt_adapter",
            },
        )
        handle = JobHandle(
            job_id=request.job_id,
            job_type=request.job_type,
            provider_id=self.provider_id,
            provider_job_id=f"fake://{request.job_id}",
            metadata={"provider_type": self.provider_type},
        )
        self._results[request.job_id] = JobResult(
            handle=handle,
            status=JobStatus(state="succeeded"),
            payload={"candidate": candidate.to_json()},
            metadata={"deterministic_fake": True},
        )
        return handle


def _gold_answer_for_prompt(prompt: str) -> str:
    normalized = prompt.lower()
    answers = {
        "ana has 2 apples": "5",
        "4 plus 4": "8",
        "5 plus 6": "11",
        "7 minus 1": "6",
        "3 times 4": "12",
        "12 divided by 3": "4",
        "9 plus 10": "19",
        "8 minus 5": "3",
    }
    for fragment, answer in answers.items():
        if fragment in normalized:
            return answer
    return "0"


def _fake_correctness(
    *,
    candidate: Candidate,
    example_id: str,
    sample_index: int,
    job_type: str,
) -> bool:
    if candidate.adapter_kind == "fake_prompt_adapter":
        return True
    payload = json.dumps(
        {
            "candidate": candidate.candidate_id,
            "example": example_id,
            "sample": sample_index,
            "job_type": job_type,
        },
        sort_keys=True,
    )
    digest = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)
    if job_type == "evaluation":
        return example_id.endswith("000003")
    return digest % 3 != 0


__all__ = ["FakeInferenceProvider", "FakePromptAdapterProvider"]
