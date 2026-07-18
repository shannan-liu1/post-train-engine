"""Bounded, no-redirect HTTP primitives for secret-bearing provider calls."""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any

MAX_PROVIDER_RESPONSE_BYTES = 1024 * 1024


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    """Prevent urllib from forwarding authorization headers across redirects."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        del msg, newurl
        raise urllib.error.HTTPError(
            req.full_url,
            code,
            "provider redirect rejected",
            headers,
            fp,
        )


_NO_REDIRECT_OPENER = urllib.request.build_opener(RejectRedirects())


def open_no_redirect(
    request: urllib.request.Request,
    *,
    timeout: float,
) -> Any:
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def read_bounded_response(response: Any) -> bytes:
    payload = response.read(MAX_PROVIDER_RESPONSE_BYTES + 1)
    if len(payload) > MAX_PROVIDER_RESPONSE_BYTES:
        raise RuntimeError("provider response exceeded 1 MiB")
    return payload
