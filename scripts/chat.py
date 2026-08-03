#!/usr/bin/env python3
"""Streaming multi-turn terminal chat client for the local Kimi K3 server."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator

from k3_api import error_message


METRIC_FIELDS = [
    "timestamp",
    "job_id",
    "turn_index",
    "model",
    "request_started_at",
    "response_headers_at",
    "first_token_at",
    "first_reasoning_token_at",
    "first_content_token_at",
    "request_finished_at",
    "time_to_headers_seconds",
    "ttft_seconds",
    "time_to_first_reasoning_seconds",
    "time_to_first_content_seconds",
    "total_latency_seconds",
    "http_status",
    "finish_reason",
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "total_tokens",
    "max_output_tokens",
    "reasoning_effort",
    "error",
    "usage_json",
]


def now() -> dt.datetime:
    return dt.datetime.now().astimezone()


def iso_milliseconds(value: dt.datetime | None) -> str:
    return value.isoformat(timespec="milliseconds") if value else ""


def elapsed(value: float | None, start: float) -> str:
    return f"{value - start:.3f}" if value is not None else ""


def sse_payloads(response: BinaryIO) -> Iterator[str]:
    """Yield complete data payloads from an SSE response body."""

    data_lines: list[str] = []
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines.clear()
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield "\n".join(data_lines)


def merge_tool_call(
    tool_calls: list[dict[str, Any]], delta: dict[str, Any]
) -> None:
    index = int(delta.get("index", len(tool_calls)))
    while len(tool_calls) <= index:
        tool_calls.append(
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
    target = tool_calls[index]
    for key in ("id", "type"):
        value = delta.get(key)
        if isinstance(value, str):
            if key == "id":
                target[key] = str(target.get(key, "")) + value
            elif value:
                target[key] = value
    function = delta.get("function")
    if isinstance(function, dict):
        target_function = target.setdefault("function", {})
        for key in ("name", "arguments"):
            value = function.get(key)
            if isinstance(value, str):
                target_function[key] = str(target_function.get(key, "")) + value


@dataclass
class StreamResult:
    status: int = 0
    assistant: dict[str, Any] = field(default_factory=lambda: {"role": "assistant"})
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""
    error: str = ""
    headers_perf: float | None = None
    headers_at: dt.datetime | None = None
    first_token_perf: float | None = None
    first_token_at: dt.datetime | None = None
    first_reasoning_perf: float | None = None
    first_reasoning_at: dt.datetime | None = None
    first_content_perf: float | None = None
    first_content_at: dt.datetime | None = None


def stream_chat_completion(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
) -> StreamResult:
    """Stream one completion, display it, and retain precise client timings."""

    result = StreamResult()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    printed_reasoning = False
    printed_content = False

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result.status = response.status
            result.headers_perf = time.perf_counter()
            result.headers_at = now()
            for raw_payload in sse_payloads(response):
                if raw_payload == "[DONE]":
                    break
                try:
                    event = json.loads(raw_payload)
                except json.JSONDecodeError as exc:
                    result.error = f"invalid SSE JSON: {exc}"
                    break

                usage = event.get("usage")
                if isinstance(usage, dict):
                    result.usage = usage
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason:
                    result.finish_reason = str(finish_reason)
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                role = delta.get("role")
                if isinstance(role, str) and role:
                    result.assistant["role"] = role

                reasoning = delta.get("reasoning_content")
                if isinstance(reasoning, str) and reasoning:
                    received_perf = time.perf_counter()
                    received_at = now()
                    if result.first_token_perf is None:
                        result.first_token_perf = received_perf
                        result.first_token_at = received_at
                    if result.first_reasoning_perf is None:
                        result.first_reasoning_perf = received_perf
                        result.first_reasoning_at = received_at
                    reasoning_parts.append(reasoning)
                    if not printed_reasoning:
                        print("thinking> ", end="", flush=True)
                        printed_reasoning = True
                    print(reasoning, end="", flush=True)

                content = delta.get("content")
                if isinstance(content, str) and content:
                    received_perf = time.perf_counter()
                    received_at = now()
                    if result.first_token_perf is None:
                        result.first_token_perf = received_perf
                        result.first_token_at = received_at
                    if result.first_content_perf is None:
                        result.first_content_perf = received_perf
                        result.first_content_at = received_at
                    content_parts.append(content)
                    if not printed_content:
                        if printed_reasoning:
                            print()
                        print("kimi> ", end="", flush=True)
                        printed_content = True
                    print(content, end="", flush=True)

                for tool_delta in delta.get("tool_calls") or []:
                    if isinstance(tool_delta, dict):
                        merge_tool_call(tool_calls, tool_delta)
    except urllib.error.HTTPError as exc:
        result.status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"error": {"message": raw}}
        result.error = error_message(body)
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        ConnectionError,
    ) as exc:
        result.error = f"{type(exc).__name__}: {exc}"

    if printed_reasoning or printed_content:
        print()
    if reasoning_parts:
        result.assistant["reasoning_content"] = "".join(reasoning_parts)
    if content_parts:
        result.assistant["content"] = "".join(content_parts)
    else:
        result.assistant["content"] = ""
    if tool_calls:
        result.assistant["tool_calls"] = tool_calls
    return result


def usage_value(usage: dict[str, Any], key: str) -> int | str:
    value = usage.get(key, "")
    return value if isinstance(value, int) else ""


def nested_usage_value(usage: dict[str, Any], *keys: str) -> int | str:
    value: Any = usage
    for key in keys:
        if not isinstance(value, dict):
            return ""
        value = value.get(key)
    return value if isinstance(value, int) else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        default=os.environ.get("KIMI_TARGET_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        help="Serving Slurm job ID; metrics go to logs/JOB_ID/k3-chat.csv.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "http://127.0.0.1:30000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("KIMI_MODEL", "moonshotai/Kimi-K3"),
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--reasoning-effort", choices=("low", "high", "max"), default="low")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    if not args.job_id or not args.job_id.isdigit():
        parser.error("--job-id must be the numeric serving Slurm job ID")
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")

    root_dir = Path(
        os.environ.get("KIMI_K3_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    output_path = root_dir / "logs" / args.job_id / "k3-chat.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    if not write_header:
        with output_path.open(newline="", encoding="utf-8") as existing:
            if next(csv.reader(existing), []) != METRIC_FIELDS:
                parser.error(
                    f"output CSV has an incompatible schema: {output_path}; "
                    "move the old file aside before rerunning this job"
                )

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    messages: list[dict[str, Any]] = []
    print("Commands: /clear resets the conversation; /exit quits.")
    print(f"Metrics: {output_path}")

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
            handle.flush()

        turn_index = 0
        while True:
            try:
                prompt = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not prompt:
                continue
            if prompt == "/exit":
                return 0
            if prompt == "/clear":
                messages.clear()
                print("Conversation cleared.")
                continue

            turn_index += 1
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": args.model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "reasoning_effort": args.reasoning_effort,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            started_at = now()
            start_perf = time.perf_counter()
            result = stream_chat_completion(endpoint, payload, args.timeout)
            end_perf = time.perf_counter()
            finished_at = now()

            usage = result.usage
            reasoning_tokens = usage_value(usage, "reasoning_tokens")
            if reasoning_tokens == "":
                reasoning_tokens = nested_usage_value(
                    usage, "completion_tokens_details", "reasoning_tokens"
                )
            cached_tokens = nested_usage_value(
                usage, "prompt_tokens_details", "cached_tokens"
            )
            row = {
                "timestamp": finished_at.isoformat(timespec="seconds"),
                "job_id": args.job_id,
                "turn_index": turn_index,
                "model": args.model,
                "request_started_at": iso_milliseconds(started_at),
                "response_headers_at": iso_milliseconds(result.headers_at),
                "first_token_at": iso_milliseconds(result.first_token_at),
                "first_reasoning_token_at": iso_milliseconds(result.first_reasoning_at),
                "first_content_token_at": iso_milliseconds(result.first_content_at),
                "request_finished_at": iso_milliseconds(finished_at),
                "time_to_headers_seconds": elapsed(result.headers_perf, start_perf),
                "ttft_seconds": elapsed(result.first_token_perf, start_perf),
                "time_to_first_reasoning_seconds": elapsed(
                    result.first_reasoning_perf, start_perf
                ),
                "time_to_first_content_seconds": elapsed(
                    result.first_content_perf, start_perf
                ),
                "total_latency_seconds": f"{end_perf - start_perf:.3f}",
                "http_status": result.status,
                "finish_reason": result.finish_reason,
                "prompt_tokens": usage_value(usage, "prompt_tokens"),
                "cached_prompt_tokens": cached_tokens,
                "completion_tokens": usage_value(usage, "completion_tokens"),
                "reasoning_tokens": reasoning_tokens,
                "total_tokens": usage_value(usage, "total_tokens"),
                "max_output_tokens": args.max_tokens,
                "reasoning_effort": args.reasoning_effort,
                "error": result.error,
                "usage_json": json.dumps(
                    usage, ensure_ascii=False, separators=(",", ":")
                ),
            }
            writer.writerow(row)
            handle.flush()

            ttft = row["ttft_seconds"] or "n/a"
            ttfc = row["time_to_first_content_seconds"] or "n/a"
            print(
                f"metrics> TTFT={ttft}s first_content={ttfc}s "
                f"total={row['total_latency_seconds']}s"
            )

            if result.status != 200 or result.error:
                sys.stderr.write(
                    f"request failed (HTTP {result.status}): {result.error}\n"
                )
                messages.pop()
                continue
            messages.append(result.assistant)


if __name__ == "__main__":
    raise SystemExit(main())
