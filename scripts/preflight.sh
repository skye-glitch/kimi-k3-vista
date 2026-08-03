#!/usr/bin/env bash

module load gcc/13.2.0
module load cuda/12.6
module load python3/3.11.8

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root_dir=${KIMI_K3_ROOT:-$(cd "$script_dir/.." && pwd)}
image_path="$root_dir/images/sglang-kimi-k3-cu12-74968e5653-arm64.sif"
model_path=${KIMI_MODEL_PATH:-$root_dir/models/Kimi-K3}

[[ -r "$image_path" ]] || { echo "FAIL: missing SGLang image: $image_path" >&2; exit 1; }
[[ -r "$model_path/model.safetensors.index.json" ]] || {
    echo "FAIL: missing model index: $model_path/model.safetensors.index.json" >&2
    exit 1
}

shopt -s nullglob
shards=("$model_path"/model-*-of-000096.safetensors)
if [[ ${#shards[@]} -ne 96 ]]; then
    echo "FAIL: expected 96 production weight shards, found ${#shards[@]}." >&2
    exit 1
fi

checkpoint_bytes=$(stat --format='%s' "${shards[@]}" | awk '{total += $1} END {printf "%.0f", total}')
if [[ "$checkpoint_bytes" -lt 1500000000000 ]]; then
    echo "FAIL: 96 shards exist but total only $checkpoint_bytes bytes; expected over 1.5 TB." >&2
    exit 1
fi

python3 - "$model_path" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
index_path = model_path / "model.safetensors.index.json"

try:
    with index_path.open(encoding="utf-8") as handle:
        index = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"FAIL: cannot parse model index {index_path}: {exc}")

weight_map = index.get("weight_map")
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit("FAIL: model index has no non-empty weight_map object.")

referenced = set(weight_map.values())
if not all(isinstance(name, str) and Path(name).name == name for name in referenced):
    raise SystemExit("FAIL: model index contains an invalid or non-local shard path.")

expected = {f"model-{part:05d}-of-000096.safetensors" for part in range(1, 97)}
actual = {path.name for path in model_path.glob("model-*-of-000096.safetensors")}

missing_references = sorted(referenced - actual)
unreferenced_files = sorted(actual - referenced)
unexpected_references = sorted(referenced - expected)
if missing_references or unreferenced_files or unexpected_references or referenced != expected:
    raise SystemExit(
        "FAIL: model index/shard mismatch: "
        f"missing={missing_references[:3]} "
        f"unreferenced={unreferenced_files[:3]} "
        f"unexpected={unexpected_references[:3]} "
        f"referenced_count={len(referenced)}"
    )

zero_size = sorted(name for name in referenced if (model_path / name).stat().st_size == 0)
if zero_size:
    raise SystemExit(f"FAIL: zero-size checkpoint shards: {zero_size[:3]}")

print(f"PASS: index_tensors={len(weight_map)} referenced_shards={len(referenced)}")
PY

echo "PASS: image=$image_path"
echo "PASS: model=$model_path"
echo "PASS: shards=${#shards[@]} bytes=$checkpoint_bytes"
echo "PASS: production topology target=32 GH200 nodes (TP32/EP32)"
