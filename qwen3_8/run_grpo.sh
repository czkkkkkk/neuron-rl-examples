#!/usr/bin/env bash
# Train inside the pinned image. CONFIG_PATH is independent of the caller directory.
set -euo pipefail
example=$(cd "$(dirname "$0")" && pwd)
: "${MODEL_DIR:?Set MODEL_DIR to a Qwen3.8-27B checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to the prepared GSM8K dataset}"
: "${PLUGIN_DIR:?Set PLUGIN_DIR to the installed Neuron verl plugin}"
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/model.safetensors.index.json"
test -f "$DATA_DIR/train.parquet"
test -f "$DATA_DIR/test.parquet"
export VERL_LOGGING_LEVEL=INFO RAY_DEDUP_LOGS=0 RAY_USAGE_STATS_ENABLED=0
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
export TORCH_CACHING_PRECOMPILE=0 VLLM_NEURON_DISABLE_PARALLEL_TRACE=1
python "$example/check_sdk.py" --hardware
export CONFIG_PATH="$example/config" CONFIG_NAME=qwen38_grpo
exec bash "$PLUGIN_DIR/scripts/accelerators/neuron/run_neuron.sh" "$@"
