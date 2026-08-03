#!/usr/bin/env python3
"""Probe Kimi K3 prompt length and long-range recall through its OpenAI API."""

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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from k3_api import error_message, post_json


def flush_cache(base_url: str, timeout: float) -> tuple[int, str, float]:
    """Flush SGLang's radix/KV/Mamba caches and return status, text, latency."""
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    query = urllib.parse.urlencode({"timeout": f"{timeout:g}"})
    request = urllib.request.Request(
        f"{api_root}/flush_cache?{query}",
        data=b"",
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout + 5) as response:
            text = response.read().decode("utf-8", errors="replace").strip()
            return response.status, text, time.perf_counter() - start
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace").strip()
        return exc.code, text, time.perf_counter() - start
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        TimeoutError,
        ConnectionError,
    ) as exc:
        text = f"{type(exc).__name__}: {exc}"
        return 0, text, time.perf_counter() - start


def make_prompt(marker: str, filler_tokens: int) -> str:
    return (
        "This is a long-context retrieval test. Memorize the secret identifier "
        f"{marker}. Do not change it.\nBEGIN FILLER\n"
        + (" x" * filler_tokens)
        + "\nEND FILLER\nWhat was the secret identifier? Reply with the exact "
        "identifier and no other text."
    )


def make_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning_effort": "low",
        "temperature": 0.0,
    }


def response_text(body: dict[str, Any]) -> tuple[str, str]:
    try:
        choice = body["choices"][0]
        message = choice["message"]
        text = "\n".join(
            value
            for value in (message.get("reasoning_content"), message.get("content"))
            if value
        )
        return text, str(choice.get("finish_reason", ""))
    except (KeyError, IndexError, TypeError):
        return "", ""


def parse_targets(value: str) -> list[int]:
    try:
        targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("targets must be comma-separated integers") from exc
    if not targets or any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError("targets must contain positive integers")
    return targets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        default=os.environ.get("KIMI_TARGET_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
        help="Serving Slurm job ID; results go to logs/JOB_ID/k3-context.csv.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KIMI_BASE_URL", "http://127.0.0.1:30000/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("KIMI_MODEL", "moonshotai/Kimi-K3"),
    )
    parser.add_argument(
        "--targets",
        type=parse_targets,
        default=parse_targets("4096,8192,16384,24576,30000,32000,32637"),
        help="Comma-separated target prompt-token counts.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument(
        "--context-overhead-tokens",
        type=int,
        default=3,
        help=(
            "Tokens counted by SGLang's context guard but omitted from OpenAI "
            "prompt usage for this K3 chat template."
        ),
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument(
        "--flush-timeout",
        type=float,
        default=120,
        help="Seconds SGLang may wait for an idle server before flushing caches.",
    )
    parser.add_argument(
        "--no-flush-cache",
        action="store_true",
        help="Retain radix/KV/Mamba caches between targets instead of isolating them.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue to larger targets after an HTTP or recall failure.",
    )
    args = parser.parse_args()

    if not args.job_id or not args.job_id.isdigit():
        parser.error("--job-id must be a numeric serving Slurm job ID")
    if (
        args.max_tokens <= 0
        or args.timeout <= 0
        or args.flush_timeout <= 0
        or args.context_limit <= 0
        or args.context_overhead_tokens < 0
    ):
        parser.error(
            "max-tokens, timeout, flush-timeout, and context-limit must be positive; "
            "context-overhead-tokens must be nonnegative"
        )
    for target in args.targets:
        reserved_tokens = target + args.max_tokens + args.context_overhead_tokens
        if reserved_tokens > args.context_limit:
            parser.error(
                f"target {target} + max-tokens {args.max_tokens} + context overhead "
                f"{args.context_overhead_tokens} = {reserved_tokens}, which exceeds "
                f"context-limit {args.context_limit}"
            )

    root_dir = Path(
        os.environ.get("KIMI_K3_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    output_path = root_dir / "logs" / args.job_id / "k3-context.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "job_id",
        "request_started_at",
        "request_finished_at",
        "request_start_epoch",
        "request_end_epoch",
        "phase",
        "cache_mode",
        "cache_flush_http_status",
        "cache_flush_latency_seconds",
        "cache_flush_error",
        "target_prompt_tokens",
        "actual_prompt_tokens",
        "max_output_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_seconds",
        "http_status",
        "marker_found",
        "finish_reason",
        "filler_tokens",
        "error",
        "usage_json",
        "response_excerpt",
    ]
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    if not write_header:
        with output_path.open(newline="", encoding="utf-8") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != fieldnames:
            parser.error(
                f"output CSV has an incompatible schema: {output_path}; "
                "move the old file aside before rerunning this job"
            )

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            handle.flush()

        for target in args.targets:
            marker = f"K3_NEEDLE_{target:06d}_COBALT"
            cache_mode = "retained" if args.no_flush_cache else "isolated"

            def write_failure(
                phase: str,
                status: int,
                message: str,
                *,
                flush_status: int | str = "",
                flush_latency: float | str = "",
                flush_error: str = "",
            ) -> None:
                now = dt.datetime.now().astimezone()
                row = {
                    "timestamp": now.isoformat(timespec="seconds"),
                    "job_id": args.job_id,
                    "request_started_at": "",
                    "request_finished_at": "",
                    "request_start_epoch": "",
                    "request_end_epoch": "",
                    "phase": phase,
                    "cache_mode": cache_mode,
                    "cache_flush_http_status": flush_status,
                    "cache_flush_latency_seconds": (
                        f"{flush_latency:.3f}" if isinstance(flush_latency, float) else flush_latency
                    ),
                    "cache_flush_error": flush_error,
                    "target_prompt_tokens": target,
                    "actual_prompt_tokens": "",
                    "max_output_tokens": args.max_tokens,
                    "completion_tokens": "",
                    "total_tokens": "",
                    "latency_seconds": "",
                    "http_status": status,
                    "marker_found": False,
                    "finish_reason": "",
                    "filler_tokens": "",
                    "error": message,
                    "usage_json": "",
                    "response_excerpt": "",
                }
                writer.writerow(row)
                handle.flush()
                print(json.dumps(row, ensure_ascii=False))

            if not args.no_flush_cache:
                flush_status, flush_text, flush_elapsed = flush_cache(
                    args.base_url, args.flush_timeout
                )
                if flush_status != 200:
                    write_failure(
                        "pre_calibration_flush",
                        flush_status,
                        f"cache flush failed: {flush_text}",
                        flush_status=flush_status,
                        flush_latency=flush_elapsed,
                        flush_error=flush_text,
                    )
                    if not args.continue_on_failure:
                        return 1
                    continue

            # Calibrate the fixed prompt and chat-template overhead for this
            # exact marker. The pinned Kimi tokenizer maps every repeated
            # filler unit " x" to exactly one token.
            status, calibration = post_json(
                endpoint,
                make_payload(args.model, make_prompt(marker, 0), 1),
                args.timeout,
            )
            if status != 200 or "usage" not in calibration:
                message = error_message(calibration)
                write_failure("zero_filler_calibration", status, message)
                print(f"target={target} calibration_failed status={status}: {message}")
                if not args.continue_on_failure:
                    return 1
                continue

            base_tokens = int(calibration["usage"]["prompt_tokens"])

            # Inserting the first " x" can also split a token at the adjacent
            # newline boundary. Measure that transition separately; every
            # additional repeated " x" is exactly one tokenizer token.
            status, one_filler_calibration = post_json(
                endpoint,
                make_payload(args.model, make_prompt(marker, 1), 1),
                args.timeout,
            )
            if status != 200 or "usage" not in one_filler_calibration:
                message = error_message(one_filler_calibration)
                write_failure("one_filler_calibration", status, message)
                print(f"target={target} filler_calibration_failed status={status}: {message}")
                if not args.continue_on_failure:
                    return 1
                continue

            one_filler_tokens = int(one_filler_calibration["usage"]["prompt_tokens"])
            if target == base_tokens:
                filler_tokens = 0
            else:
                filler_tokens = target - one_filler_tokens + 1
            if filler_tokens < 0:
                message = (
                    f"target is below calibrated prompt sizes "
                    f"{base_tokens}/{one_filler_tokens}"
                )
                write_failure("token_calibration", 0, message)
                print(f"target={target} {message}")
                if not args.continue_on_failure:
                    return 1
                continue

            flush_status: int | str = ""
            flush_elapsed: float | str = ""
            flush_error = ""
            if not args.no_flush_cache:
                # Calibration requests populate the radix cache.  Flush again
                # so the measured request starts from a comparable cache and
                # allocator state at every target length.
                flush_status, flush_text, flush_elapsed = flush_cache(
                    args.base_url, args.flush_timeout
                )
                if flush_status != 200:
                    flush_error = flush_text
                    write_failure(
                        "pre_measurement_flush",
                        flush_status,
                        f"cache flush failed: {flush_text}",
                        flush_status=flush_status,
                        flush_latency=flush_elapsed,
                        flush_error=flush_error,
                    )
                    if not args.continue_on_failure:
                        return 1
                    continue

            prompt = make_prompt(marker, filler_tokens)
            started_at = dt.datetime.now().astimezone()
            start_epoch = time.time()
            start = time.perf_counter()
            status, body = post_json(
                endpoint,
                make_payload(args.model, prompt, args.max_tokens),
                args.timeout,
            )
            elapsed = time.perf_counter() - start
            end_epoch = time.time()
            finished_at = dt.datetime.now().astimezone()
            text, finish_reason = response_text(body)
            usage = body.get("usage", {}) if status == 200 else {}
            actual_prompt_tokens = usage.get("prompt_tokens", "")
            marker_found = marker in text
            message = "" if status == 200 else error_message(body)
            excerpt = text.replace("\r", " ").replace("\n", " ")[:300]

            row = {
                "timestamp": finished_at.isoformat(timespec="seconds"),
                "job_id": args.job_id,
                "request_started_at": started_at.isoformat(timespec="milliseconds"),
                "request_finished_at": finished_at.isoformat(timespec="milliseconds"),
                "request_start_epoch": f"{start_epoch:.3f}",
                "request_end_epoch": f"{end_epoch:.3f}",
                "phase": "needle_recall",
                "cache_mode": cache_mode,
                "cache_flush_http_status": flush_status,
                "cache_flush_latency_seconds": (
                    f"{flush_elapsed:.3f}" if isinstance(flush_elapsed, float) else ""
                ),
                "cache_flush_error": flush_error,
                "target_prompt_tokens": target,
                "actual_prompt_tokens": actual_prompt_tokens,
                "max_output_tokens": args.max_tokens,
                "completion_tokens": usage.get("completion_tokens", ""),
                "total_tokens": usage.get("total_tokens", ""),
                "latency_seconds": f"{elapsed:.3f}",
                "http_status": status,
                "marker_found": marker_found,
                "finish_reason": finish_reason,
                "filler_tokens": filler_tokens,
                "error": message,
                "usage_json": json.dumps(usage, ensure_ascii=False, separators=(",", ":")),
                "response_excerpt": excerpt,
            }
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, ensure_ascii=False))

            passed = status == 200 and marker_found and actual_prompt_tokens == target
            if not passed and not args.continue_on_failure:
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
