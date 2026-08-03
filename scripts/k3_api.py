"""Shared standard-library HTTP helpers for Kimi K3 clients."""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from typing import Any


def post_json(
    url: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": {"message": raw}}
        return exc.code, body
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        ConnectionError,
    ) as exc:
        return 0, {"error": {"message": f"{type(exc).__name__}: {exc}"}}


def error_message(body: dict[str, Any]) -> str:
    error = body.get("error", body)
    if isinstance(error, dict):
        return str(error.get("message", error))
    return str(error)
