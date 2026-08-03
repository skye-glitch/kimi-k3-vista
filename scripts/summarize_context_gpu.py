#!/usr/bin/env python3
"""Join K3 context-probe rows with GPU samples from the same Slurm job."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def as_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job-id",
        default=os.environ.get("KIMI_TARGET_JOB_ID") or os.environ.get("SLURM_JOB_ID"),
    )
    args = parser.parse_args()
    if not args.job_id or not args.job_id.isdigit():
        parser.error("--job-id must be numeric")

    root_dir = Path(
        os.environ.get("KIMI_K3_ROOT", Path(__file__).resolve().parents[1])
    ).resolve()
    job_dir = root_dir / "logs" / args.job_id
    context_path = job_dir / "k3-context.csv"
    gpu_path = job_dir / "k3-gpu.csv"
    output_path = job_dir / "k3-context-gpu-summary.csv"
    for path in (context_path, gpu_path):
        if not path.is_file():
            parser.error(f"missing input: {path}")

    with context_path.open(newline="", encoding="utf-8") as handle:
        context_rows = list(csv.DictReader(handle))

    gpu_by_epoch: dict[int, list[dict[str, str]]] = defaultdict(list)
    with gpu_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["sample_status"] != "ok":
                continue
            try:
                epoch = int(row["epoch"])
                float(row["memory_used_mib"])
                float(row["memory_total_mib"])
                float(row["gpu_util_pct"])
            except (KeyError, ValueError):
                continue
            gpu_by_epoch[epoch].append(row)

    fieldnames = [
        "job_id",
        "request_started_at",
        "request_finished_at",
        "target_prompt_tokens",
        "actual_prompt_tokens",
        "max_output_tokens",
        "completion_tokens",
        "api_prompt_plus_max_tokens",
        "latency_seconds",
        "http_status",
        "marker_found",
        "passed",
        "cache_mode",
        "gpu_sample_epochs",
        "gpu_rows_ok",
        "gpu_memory_baseline_avg_mib",
        "gpu_memory_baseline_max_mib",
        "gpu_memory_request_avg_mib",
        "gpu_memory_peak_epoch_avg_mib",
        "gpu_memory_peak_mib",
        "gpu_min_free_mib",
        "gpu_util_request_avg_pct",
        "gpu_util_peak_pct",
        "error",
    ]

    summaries: list[dict[str, Any]] = []
    all_epochs = sorted(gpu_by_epoch)
    for context in context_rows:
        start = as_float(context.get("request_start_epoch", ""))
        end = as_float(context.get("request_end_epoch", ""))
        request_epochs = (
            [epoch for epoch in all_epochs if start <= epoch <= end]
            if start is not None and end is not None
            else []
        )
        request_gpu = [row for epoch in request_epochs for row in gpu_by_epoch[epoch]]

        baseline_rows: list[dict[str, str]] = []
        if start is not None:
            baseline_candidates = [epoch for epoch in all_epochs if epoch < start]
            if baseline_candidates:
                baseline_rows = gpu_by_epoch[baseline_candidates[-1]]

        memory = [float(row["memory_used_mib"]) for row in request_gpu]
        total = [float(row["memory_total_mib"]) for row in request_gpu]
        utilization = [float(row["gpu_util_pct"]) for row in request_gpu]
        baseline_memory = [float(row["memory_used_mib"]) for row in baseline_rows]
        epoch_memory_means = [
            fmean(float(row["memory_used_mib"]) for row in gpu_by_epoch[epoch])
            for epoch in request_epochs
        ]

        actual = context.get("actual_prompt_tokens", "")
        target = context.get("target_prompt_tokens", "")
        max_output = context.get("max_output_tokens", "")
        try:
            api_reserved: int | str = int(actual) + int(max_output)
        except ValueError:
            api_reserved = ""
        passed = (
            context.get("http_status") == "200"
            and context.get("marker_found") == "True"
            and actual == target
        )

        summaries.append(
            {
                "job_id": args.job_id,
                "request_started_at": context.get("request_started_at", ""),
                "request_finished_at": context.get("request_finished_at", ""),
                "target_prompt_tokens": target,
                "actual_prompt_tokens": actual,
                "max_output_tokens": max_output,
                "completion_tokens": context.get("completion_tokens", ""),
                "api_prompt_plus_max_tokens": api_reserved,
                "latency_seconds": context.get("latency_seconds", ""),
                "http_status": context.get("http_status", ""),
                "marker_found": context.get("marker_found", ""),
                "passed": passed,
                "cache_mode": context.get("cache_mode", ""),
                "gpu_sample_epochs": len(request_epochs),
                "gpu_rows_ok": len(request_gpu),
                "gpu_memory_baseline_avg_mib": (
                    f"{fmean(baseline_memory):.1f}" if baseline_memory else ""
                ),
                "gpu_memory_baseline_max_mib": (
                    f"{max(baseline_memory):.0f}" if baseline_memory else ""
                ),
                "gpu_memory_request_avg_mib": f"{fmean(memory):.1f}" if memory else "",
                "gpu_memory_peak_epoch_avg_mib": (
                    f"{max(epoch_memory_means):.1f}" if epoch_memory_means else ""
                ),
                "gpu_memory_peak_mib": f"{max(memory):.0f}" if memory else "",
                "gpu_min_free_mib": (
                    f"{min(limit - used for limit, used in zip(total, memory)):.0f}"
                    if memory
                    else ""
                ),
                "gpu_util_request_avg_pct": (
                    f"{fmean(utilization):.1f}" if utilization else ""
                ),
                "gpu_util_peak_pct": f"{max(utilization):.0f}" if utilization else "",
                "error": context.get("error", ""),
            }
        )

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(summaries)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
