#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 JOB_ID [requests] [concurrency]" >&2
    exit 2
fi

job_id=$1
requests=${2:-100}
concurrency=${3:-32}

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

rank0_host=$(squeue -h -j "$job_id" -t R -o '%B' 2>/dev/null || true)
rank0_host=$(awk 'NF && $1 != "(null)" && tolower($1) != "n/a" {print $1; exit}' <<<"$rank0_host")

if [[ -z "$rank0_host" ]]; then
    echo "Could not resolve rank-0 host for Job $job_id." >&2
    exit 1
fi

# We don't need any modules loaded for this pure Python script
python3 "$project_root/scripts/batch_eval.py" \
    --base-url "http://${rank0_host}:30000/v1" \
    --requests "$requests" \
    --concurrency "$concurrency"