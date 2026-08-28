#!/usr/bin/env bash
#SBATCH --job-name=k3-download
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH -A ASC26075

module load gcc
module load cuda
module load tacc-apptainer

root_dir=${KIMI_K3_ROOT:-${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required}}
job_id=${SLURM_JOB_ID:?SLURM_JOB_ID is required}
export KIMI_K3_ROOT="$root_dir"
export KIMI_JOB_LOG_DIR="$root_dir/logs/$job_id"

mkdir -p "$KIMI_JOB_LOG_DIR"
exec >"$KIMI_JOB_LOG_DIR/k3-download.out" 2>&1
cd "$KIMI_JOB_LOG_DIR"



set -euo pipefail

image_path="$root_dir/images/sglang-kimi-k3.sif"
model_path=${KIMI_MODEL_PATH:-$root_dir/models/Kimi-K3}
host_hf_home=${HF_HOME:-$root_dir/.cache/huggingface}

[[ -r "$image_path" ]] || { echo "Missing image: $image_path" >&2; exit 2; }
mkdir -p "$model_path" "$host_hf_home"

bind_args=(--bind "$root_dir:$root_dir")
if [[ "$host_hf_home" != "$root_dir" && "$host_hf_home" != "$root_dir/"* ]]; then
    bind_args+=(--bind "$host_hf_home:$host_hf_home")
fi

export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_HF_HOME="$host_hf_home"
export APPTAINERENV_HF_XET_HIGH_PERFORMANCE=1
if [[ -n ${HF_TOKEN:-} ]]; then
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
else
    echo "HF_TOKEN is unset; the public download may be rate-limited."
fi

echo "Hugging Face cache: $host_hf_home"

apptainer exec \
    --cleanenv \
    "${bind_args[@]}" \
    "$image_path" \
    hf download moonshotai/Kimi-K3 --local-dir "$model_path"

bash "$root_dir/scripts/preflight.sh"
