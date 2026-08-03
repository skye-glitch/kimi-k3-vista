#!/usr/bin/env bash
# Emit one host's GPU metrics for monitor_k3_gpu.sh. The parent monitor loads
# the required TACC modules before launching this script.

set -uo pipefail

host=$(hostname -s)
status=ok
if ! metrics=$(nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.current.sm,clocks.current.memory,pstate \
    --format=csv,noheader,nounits 2>/dev/null | tr -d ' '); then
    metrics='NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA'
    status=nvidia_smi_error
elif ! awk -F',' 'NR == 1 { fields = NF } END { exit !(NR == 1 && fields == 12) }' \
    <<<"$metrics"; then
    metrics='NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA'
    status=parse_error
fi

printf '%s,%s,%s\n' "$host" "$metrics" "$status"
