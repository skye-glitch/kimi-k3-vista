#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage: $0 JOB_ID [chat.py options...]" >&2
    exit 2
fi

job_id=$1
shift

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "JOB_ID must be numeric: $job_id" >&2
    exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
rank0_host=""

# Slurm's %B field is the executing batch host, which is node zero for this
# batch job and therefore the node running the rank-0 API process.
if command -v squeue >/dev/null; then
    rank0_host=$(squeue -h -j "$job_id" -t R -o '%B' 2>/dev/null || true)
    rank0_host=$(
        awk 'NF && $1 != "(null)" && tolower($1) != "n/a" {print $1; exit}' \
            <<<"$rank0_host"
    )
fi

if [[ -z "$rank0_host" ]]; then
    echo "Could not resolve rank-0 host for Job $job_id." >&2
    echo "The job must be running and visible to squeue." >&2
    exit 1
fi

echo "Resolved Job $job_id rank-0 host: $rank0_host" >&2


exec python3 "$project_root/scripts/chat.py" \
    --job-id "$job_id" \
    --base-url "http://${rank0_host}:30000/v1" \
    "$@"
