#!/usr/bin/env bash

set -eo pipefail

module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

set -u

job_id=${1:?Usage: monitor_k3_gpu.sh JOB_ID NODELIST [INTERVAL_SECONDS]}
nodelist=${2:?Usage: monitor_k3_gpu.sh JOB_ID NODELIST [INTERVAL_SECONDS]}
interval_seconds=${3:-15}
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=${KIMI_K3_ROOT:-$(cd "$script_dir/.." && pwd)}
job_log_dir="$root_dir/logs/$job_id"
output_csv="$job_log_dir/k3-gpu.csv"
max_all_failed_samples=${KIMI_GPU_MONITOR_MAX_ALL_FAILED_SAMPLES:-3}

[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "JOB_ID must be numeric." >&2; exit 2; }
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || {
    echo "INTERVAL_SECONDS must be a positive integer." >&2
    exit 2
}
[[ "$max_all_failed_samples" =~ ^[1-9][0-9]*$ ]] || {
    echo "KIMI_GPU_MONITOR_MAX_ALL_FAILED_SAMPLES must be a positive integer." >&2
    exit 2
}

mapfile -t hosts < <(scontrol show hostnames "$nodelist")
[[ ${#hosts[@]} -gt 0 ]] || { echo "NODELIST resolved to zero hosts." >&2; exit 2; }
sample_script="$root_dir/scripts/sample_k3_gpu.sh"
[[ -r "$sample_script" ]] || { echo "Missing GPU sample helper: $sample_script" >&2; exit 2; }

mkdir -p "$job_log_dir"
if [[ ! -s "$output_csv" ]]; then
    printf '%s\n' \
        'timestamp,epoch,job_id,host,gpu_index,gpu_uuid,memory_used_mib,memory_total_mib,gpu_util_pct,memory_util_pct,power_draw_w,power_limit_w,temperature_c,sm_clock_mhz,memory_clock_mhz,pstate,sample_status' \
        >"$output_csv"
fi

echo "Sampling ${#hosts[@]} GPUs every ${interval_seconds}s via overlapping Slurm steps into $output_csv"

consecutive_all_failed=0
while true; do
    sample_started_epoch=$(date +%s)
    export sample_timestamp
    export sample_epoch
    sample_timestamp=$(date --iso-8601=seconds)
    sample_epoch=$(date +%s)

    raw_rows=''
    sample_status=ok
    if ! raw_rows=$(timeout --signal=TERM --kill-after=10 60 \
        srun \
            --jobid="$job_id" \
            --overlap \
            --nodes="${#hosts[@]}" \
            --ntasks="${#hosts[@]}" \
            --ntasks-per-node=1 \
            --cpus-per-task=1 \
            --kill-on-bad-exit=1 \
            bash "$sample_script"); then
        sample_status=srun_error
    fi

    declare -A rows_by_host=()
    if [[ "$sample_status" == ok ]]; then
        while IFS= read -r raw_row; do
            row_host=${raw_row%%,*}
            if [[ -n "$row_host" ]]; then
                rows_by_host["$row_host"]="$raw_row"
            fi
        done <<<"$raw_rows"
    fi

    rows=''
    for host in "${hosts[@]}"; do
        if [[ -n ${rows_by_host[$host]:-} ]]; then
            host_row=${rows_by_host[$host]}
        else
            missing_status=$sample_status
            if [[ "$missing_status" == ok ]]; then
                missing_status=missing_rank
            fi
            host_row="$host,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,$missing_status"
        fi
        rows+="${sample_timestamp},${sample_epoch},${job_id},${host_row}"$'\n'
    done
    rows=${rows%$'\n'}
    printf '%s\n' "$rows" >>"$output_csv"

    successful_gpus=$(awk -F',' '$17 == "ok" {count++} END {print count + 0}' <<<"$rows")
    awk -F',' -v timestamp="$sample_timestamp" '
        $17 == "ok" {
            count++
            memory += $7
            total += $8
            gpu += $9
            if ($9 > max_gpu) max_gpu = $9
            if ($7 > max_memory) max_memory = $7
        }
        END {
            printf "sample=%s successful_gpus=%d memory_avg_mib=%.1f memory_max_mib=%.0f memory_pct_avg=%.1f gpu_util_avg=%.1f gpu_util_max=%.0f\n",
                timestamp, count, count ? memory / count : 0, max_memory,
                total ? 100 * memory / total : 0, count ? gpu / count : 0, max_gpu
        }
    ' <<<"$rows"

    if [[ "$successful_gpus" -eq 0 ]]; then
        consecutive_all_failed=$((consecutive_all_failed + 1))
        if [[ "$consecutive_all_failed" -ge "$max_all_failed_samples" ]]; then
            echo "All ${#hosts[@]} GPU samples failed for ${consecutive_all_failed} consecutive rounds; stopping."
            exit 0
        fi
    else
        consecutive_all_failed=0
    fi

    sample_elapsed=$(( $(date +%s) - sample_started_epoch ))
    sleep_seconds=$(( interval_seconds - sample_elapsed ))
    if (( sleep_seconds > 0 )); then
        sleep "$sleep_seconds"
    fi
done
