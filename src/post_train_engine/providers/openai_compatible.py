"""Chat-completions provider for OpenAI-compatible HTTP APIs.

The public config surface calls this provider ``chat_completions``. The module
keeps the historical filename because the wire protocol is commonly described
as OpenAI-compatible by hosting providers.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from post_train_engine.api_schemas import (
    Candidate,
    JobHandle,
    JobRequest,
    JobResult,
    JobStatus,
    redact_secret_text,
)
from post_train_engine.http_transport import open_no_redirect, read_bounded_response

HttpTransport = Callable[
    [str, dict[str, str], dict[str, Any], float],
    dict[str, Any],
]
class OpenAICompatibleProvider:
    """Synchronous provider over an OpenAI-compatible chat-completions endpoint."""

    recovery_policy = "non_replayable"

    def __init__(
        self,
        *,
        provider_id: str,
        provider_type: str = "chat_completions",
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 60.0,
        max_tokens_field: str = "max_tokens",
        transport: HttpTransport | None = None,
    ) -> None:
        if not provider_id:
            raise ValueError("provider_id is required")
        if not base_url:
            raise ValueError("base_url is required")
        parsed_base_url = urlsplit(base_url)
        if parsed_base_url.scheme != "https" or not parsed_base_url.hostname:
            raise ValueError("base_url must use HTTPS")
        if parsed_base_url.username or parsed_base_url.password:
            raise ValueError("base_url must not contain credentials")
        if parsed_base_url.query or parsed_base_url.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        if not api_key:
            raise ValueError("api_key is required")
        if not model:
            raise ValueError("model is required")
        if max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("max_tokens_field must be max_tokens or max_completion_tokens")
        self.provider_id = provider_id
        self.provider_type = provider_type
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens_field = max_tokens_field
        self.transport = transport or _default_transport
        self._results: dict[str, JobResult] = {}

    def submit_job(self, request: JobRequest) -> JobHandle:
        if request.provider_id != self.provider_id:
            raise ValueError("request provider_id does not match provider")
        if request.job_type in {"rollout_generation", "evaluation"}:
            result = self._run_generation_job(request)
        elif request.job_type == "candidate_adaptation":
            result = self._run_adaptation_job(request)
        else:
            raise ValueError(f"unsupported job type: {request.job_type}")
        self._results[request.job_id] = result
        return result.handle

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

    def _run_generation_job(self, request: JobRequest) -> JobResult:
        candidate = Candidate.model_validate(request.payload["candidate"])
        generation_config = dict(request.payload.get("generation", {}))
        generations: list[dict[str, Any]] = []
        for prompt_row in list(request.payload.get("prompts", ())):
            response = self._chat_completion(
                messages=_messages_for_candidate(candidate, str(prompt_row["prompt"])),
                generation_config=generation_config,
            )
            choice = _first_choice(response)
            content = _choice_content(choice)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            generations.append(
                {
                    "example_id": str(prompt_row["example_id"]),
                    "sample_index": int(prompt_row.get("sample_index", 0)),
                    "split_role": str(prompt_row.get("split_role", "eval")),
                    "completion": content,
                    "completion_tokens": _usage_int(usage, "completion_tokens", content),
                    "finish_reason": choice.get("finish_reason"),
                    "provider_job_id": str(response.get("id", request.job_id)),
                    "raw_response": response,
                }
            )
        handle = _handle(request, self.provider_id, response_id=request.job_id)
        return JobResult(
            handle=handle,
            status=JobStatus(state="succeeded"),
            payload={"generations": generations},
            metadata={"provider_type": self.provider_type, "model": self.model},
        )

    def _run_adaptation_job(self, request: JobRequest) -> JobResult:
        baseline = Candidate.model_validate(request.payload["baseline_candidate"])
        candidate_id = str(request.payload.get("candidate_id") or "")
        if not candidate_id:
            raise ValueError("candidate adaptation requires a planned candidate_id")
        response = self._chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You improve GSM8K answer-format prompts. Return strict JSON only "
                        "with a string field system_prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "baseline_candidate": baseline.to_json(),
                            "training_view": request.payload.get("training_view"),
                            "training_rows": request.payload.get("training_rows", []),
                            "instruction": (
                                "Propose a concise system prompt that improves arithmetic "
                                "accuracy while preserving <answer>...</answer> output."
                            ),
                        },
                        sort_keys=True,
                    ),
                },
            ],
            generation_config={"temperature": 0.0, "max_output_tokens": 256},
        )
        content = _choice_content(_first_choice(response))
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("adaptation provider returned non-JSON content") from exc
        system_prompt = parsed.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise RuntimeError("adaptation provider response missing system_prompt")
        candidate = Candidate(
            candidate_id=candidate_id,
            model_id=baseline.model_id,
            parent_id=baseline.candidate_id,
            system_prompt=system_prompt.strip(),
            prompt_prefix=baseline.prompt_prefix,
            prompt_suffix=baseline.prompt_suffix,
            adapter_kind=_adapter_kind(self.provider_type),
            metadata={
                "provider": self.provider_id,
                "provider_response_id": str(response.get("id", "")),
            },
        )
        handle = _handle(request, self.provider_id, response_id=str(response.get("id", request.job_id)))
        return JobResult(
            handle=handle,
            status=JobStatus(state="succeeded"),
            payload={"candidate": candidate.to_json(), "raw_response": response},
            metadata={"provider_type": _adapter_kind(self.provider_type), "model": self.model},
        )

    def _chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        generation_config: dict[str, Any],
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(generation_config.get("temperature", 0.0)),
            "top_p": float(generation_config.get("top_p", 1.0)),
            self.max_tokens_field: int(generation_config.get("max_output_tokens", 256)),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            return self.transport(
                url=f"{self.base_url}/chat/completions",
                headers=headers,
                body=body,
                timeout_seconds=self.timeout_seconds,
            )
        except Exception as exc:
            raise RuntimeError(f"provider {self.provider_id} request failed: {_safe_error(exc)}") from exc


def _default_transport(
    *,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with open_no_redirect(request, timeout=timeout_seconds) as response:
            raw = read_bounded_response(response).decode("utf-8")
    except urllib.error.HTTPError as exc:
        body_text = exc.read(501).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body_text[:500]}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("provider response root must be an object")
    return parsed


def _messages_for_candidate(candidate: Candidate, prompt: str) -> list[dict[str, str]]:
    user_prompt = f"{candidate.prompt_prefix}{prompt}{candidate.prompt_suffix}"
    messages: list[dict[str, str]] = []
    if candidate.system_prompt:
        messages.append({"role": "system", "content": candidate.system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _first_choice(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("malformed chat completion response: missing choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise RuntimeError("malformed chat completion response: choice is not an object")
    return first


def _choice_content(choice: dict[str, Any]) -> str:
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("malformed chat completion response: missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("malformed chat completion response: missing message content")
    return content


def _usage_int(usage: dict[str, Any], key: str, content: str) -> int:
    value = usage.get(key)
    if type(value) is bool or not isinstance(value, int | float):
        return len(content.split())
    return int(value)


def _handle(request: JobRequest, provider_id: str, *, response_id: str) -> JobHandle:
    return JobHandle(
        job_id=request.job_id,
        job_type=request.job_type,
        provider_id=provider_id,
        provider_job_id=response_id,
        metadata={},
    )


def _adapter_kind(provider_type: str) -> str:
    return "chat_completions_prompt_adapter"


def _safe_error(exc: Exception) -> str:
    return redact_secret_text(str(exc))


__all__ = ["OpenAICompatibleProvider"]
