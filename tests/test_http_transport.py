from __future__ import annotations

import urllib.error
import urllib.request
from io import BytesIO

import pytest

from post_train_engine.http_transport import RejectRedirects


def test_secret_bearing_http_transport_rejects_redirects() -> None:
    request = urllib.request.Request(
        "https://api.example.test/v1/jobs",
        headers={"Authorization": "Bearer secret"},
    )

    with pytest.raises(urllib.error.HTTPError, match="redirect rejected"):
        RejectRedirects().redirect_request(
            request,
            BytesIO(b"redirect"),
            302,
            "Found",
            {"Location": "https://attacker.example.test/collect"},
            "https://attacker.example.test/collect",
        )
