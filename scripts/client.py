#!/usr/bin/env python3
"""Minimal OpenAI-compatible client for a local Kimi K3 server."""

from __future__ import annotations

import argparse
import json
import os
import sys

from k3_api import error_message, post_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "http://127.0.0.1:30000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("KIMI_MODEL", "moonshotai/Kimi-K3"),
    )
    parser.add_argument("--prompt", default="Write a one-sentence readiness check.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    status, result = post_json(
        f"{args.base_url.rstrip('/')}/chat/completions", payload, args.timeout
    )
    if status != 200:
        sys.stderr.write(f"request failed (HTTP {status}): {error_message(result)}\n")
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
