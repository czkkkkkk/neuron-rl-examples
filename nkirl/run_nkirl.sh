#!/usr/bin/env bash
# Run the NKIRL recipe inside the head node's container, on a Ray cluster that is already up.
#
#   RAY_ADDRESS=<head ip>:<port> RUN_DIR=<shared run directory> MODEL_DIR=... DATA_DIR=... \
#       bash /workspace/nkirl/run_nkirl.sh [hydra overrides...]
#
# Start one container per node and join them into one Ray cluster first, with your cluster's launcher,
# then call this script in the head container. RUN_DIR must be on storage that every node can read.
# It does what the plugin's scripts/accelerators/neuron/run_neuron.sh does, except that it joins the
# existing multi-node cluster (run_neuron.sh refuses one, because it is written for one host), and it
# forwards the NKIRL tool and reward settings to every Ray worker.
set -euo pipefail
: "${RAY_ADDRESS:?set RAY_ADDRESS to the running cluster}"
: "${RUN_DIR:?set RUN_DIR to a shared run directory}"
: "${MODEL_DIR:?set MODEL_DIR}"
: "${DATA_DIR:?set DATA_DIR (train_hermes.parquet, val_hermes.parquet)}"
# Not CONFIG_PATH / CONFIG_NAME: the image sets those to the GSM8K example for every container.
NKIRL_CONFIG_PATH=${NKIRL_CONFIG_PATH:-/workspace/nkirl/config}
NKIRL_CONFIG_NAME=${NKIRL_CONFIG_NAME:-neuron_nkirl_qwen3_32b}
export RAY_ADDRESS RUN_DIR MODEL_DIR DATA_DIR
# The paths are passed as explicit overrides below. The config's ${oc.env:...} defaults resolve in the
# TaskRunner Ray actor, which inherits the container environment: the image sets MODEL_DIR and DATA_DIR
# to the GSM8K example there.

# The same controller settings as run_neuron.sh.
export VERL_PLATFORM=aws TORCH_DEVICE_BACKEND_AUTOLOAD=0 VLLM_PLUGINS="" NEURON_RT_MAP_HBM=1 \
    NEURON_RT_ENABLE_P2P=0 FI_EFA_ENABLE_SHM_TRANSFER=0 TOKENIZERS_PARALLELISM=false \
    VERL_NEURON_PROCESS_ROLE=controller
export NEURON_LOGICAL_NC_CONFIG=${NEURON_LOGICAL_NC_CONFIG:-2} OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
: "${VERL_NEURON_PYTHON:?the image sets VERL_NEURON_PYTHON}"
export NKI_COMPILE_CACHE_URL=${NKI_COMPILE_CACHE_URL:-/cache/nki}
# Required by the actor's TP=8 x FSDP=8 layout (see the plugin's runtime.py).
export VERL_NEURON_ACTOR_CORE_ORDER=${VERL_NEURON_ACTOR_CORE_ORDER-tp8_fsdp8}
# Keep compiled graphs across containers; the default is /tmp.
export TORCH_NEURONX_NEFF_CACHE_DIR=${TORCH_NEURONX_NEFF_CACHE_DIR:-/cache/neff}
export TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR=${TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR:-/cache/neff_local}
# Ranks waiting on a long compile break the default 1,200 s lock and compile the same graph again.
export TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT=${TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT:-7200}
export LD_LIBRARY_PATH="$(dirname "$VERL_NEURON_PYTHON")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Recipe outputs, all under RUN_DIR.
export VERL_FILE_LOGGER_PATH=${VERL_FILE_LOGGER_PATH:-$RUN_DIR/metrics.jsonl}
export NKIRL_PROFILER_TRACE_DIR=${NKIRL_PROFILER_TRACE_DIR:-$RUN_DIR/profiler_trace}
export KERNEL_ARTIFACTS_DIR=${KERNEL_ARTIFACTS_DIR:-$RUN_DIR/kernels}
export NKIRL_RESPONSE_DUMP_DIR=${NKIRL_RESPONSE_DUMP_DIR:-$RUN_DIR/generations}
export NKIRL_DIR=${NKIRL_DIR:-$RUN_DIR}
# The NKIRL recipe defaults, the same as the earlier profiled run.
export KERNEL_EXTRACT_STRATEGY=${KERNEL_EXTRACT_STRATEGY:-best_speedup} TEST_CONFIG_MODE=${TEST_CONFIG_MODE:-first}
export NKIRL_REWARD_MODE=${NKIRL_REWARD_MODE:-speedup} NKIRL_SPEEDUP_MAX=${NKIRL_SPEEDUP_MAX:-2.0}
export NKIRL_USE_JUDGE=${NKIRL_USE_JUDGE:-0} COPY_SIMILARITY_THRESHOLD=${COPY_SIMILARITY_THRESHOLD:-0.95}
export NKIRL_DROP_INFRA_FAILURES=${NKIRL_DROP_INFRA_FAILURES:-1} NKIRL_INFRA_ABORT_THRESHOLD=${NKIRL_INFRA_ABORT_THRESHOLD:-0.3}
export NKIRL_TOOL_FORMAT=${NKIRL_TOOL_FORMAT:-hermes}
export PROFILER_TIMEOUT_SECONDS=${PROFILER_TIMEOUT_SECONDS:-600} PROFILER_MAX_PARALLEL_CHUNKS=${PROFILER_MAX_PARALLEL_CHUNKS:-4}
mkdir -p "$RUN_DIR" "$NKIRL_PROFILER_TRACE_DIR" "$KERNEL_ARTIFACTS_DIR" "$NKIRL_RESPONSE_DUMP_DIR"

# A deployed profiler (https) needs SigV4 and refreshable credentials (see README.md).
: "${PROFILER_REMOTE_URL:?set PROFILER_REMOTE_URL to the profiler service}"
if [ "${PROFILER_REMOTE_AUTH_MODE:-}" = sigv4 ]; then
    : "${PROFILER_REMOTE_AUTH_REGION:?set PROFILER_REMOTE_AUTH_REGION for SigV4}"
    : "${AWS_CONFIG_FILE:?set AWS_CONFIG_FILE to a profile whose credential_process is nkirl/aws_credential_process.py}"
    version=$("$VERL_NEURON_PYTHON" /workspace/nkirl/sigv4_get.py "${PROFILER_REMOTE_URL%/}/version" "$PROFILER_REMOTE_AUTH_REGION") \
        || { echo "${PROFILER_REMOTE_URL}/version is not reachable with SigV4 (credentials, region?)" >&2; exit 1; }
else
    version=$(curl -fsS -m 20 "${PROFILER_REMOTE_URL%/}/version") || { echo "${PROFILER_REMOTE_URL}/version is not reachable" >&2; exit 1; }
fi
echo "profiler_version $(date +%s) ${version}"

# Every Ray worker: the plugin's runtime env, plus PYTHONPATH (the tool is imported by dotted path) and
# the recipe settings above.
mapfile -t overrides < <("$VERL_NEURON_PYTHON" - <<'PY'
import json, os, sys
from contextlib import redirect_stdout
with redirect_stdout(sys.stderr):
    from verl_hardware_plugin.accelerators.neuron.runtime import public_runtime_env
    runtime = public_runtime_env()
env = dict(runtime["env_vars"])
env["PYTHONPATH"] = os.pathsep.join(p for p in ("/workspace", os.environ.get("PYTHONPATH", "")) if p)
for name in (
    "VERL_FILE_LOGGER_PATH", "NKIRL_PROFILER_TRACE_DIR", "KERNEL_ARTIFACTS_DIR", "NKIRL_RESPONSE_DUMP_DIR", "NKIRL_DIR",
    "KERNEL_EXTRACT_STRATEGY", "TEST_CONFIG_MODE", "NKIRL_REWARD_MODE", "NKIRL_SPEEDUP_MAX", "NKIRL_USE_JUDGE",
    "COPY_SIMILARITY_THRESHOLD", "NKIRL_DROP_INFRA_FAILURES", "NKIRL_INFRA_ABORT_THRESHOLD", "NKIRL_TOOL_FORMAT",
    "PROFILER_REMOTE_URL", "PROFILER_TIMEOUT_SECONDS", "PROFILER_MAX_PARALLEL_CHUNKS", "PROFILER_REMOTE_AUTH_MODE",
    "PROFILER_REMOTE_AUTH_REGION", "AWS_CONFIG_FILE", "NKI_COMPILE_CACHE_URL", "TOKENIZERS_PARALLELISM",
    "TORCH_NEURONX_NEFF_CACHE_DIR", "TORCH_NEURONX_NEFF_LOCAL_CACHE_DIR", "VERL_NEURON_MEMORY_SNAPSHOT_DIR",
    "VERL_NEURON_ACTOR_CORE_ORDER", "TORCH_NEURONX_NEFF_CACHE_LOCK_TIMEOUT",
):
    if os.environ.get(name):
        env[name] = os.environ[name]
print("ray_kwargs.ray_init.runtime_env.py_executable=" + json.dumps(runtime["py_executable"]))
for name, value in env.items():
    print("++ray_kwargs.ray_init.runtime_env.env_vars." + name + "=" + json.dumps(value))
PY
)

cd "$RUN_DIR"
"$VERL_NEURON_PYTHON" -m verl.trainer.main_ppo --config-path "$NKIRL_CONFIG_PATH" --config-name "$NKIRL_CONFIG_NAME" \
    trainer.default_local_dir="$RUN_DIR/checkpoints" "${overrides[@]}" \
    actor_rollout_ref.model.path="$MODEL_DIR" \
    data.train_files="$DATA_DIR/train_hermes.parquet" data.val_files="$DATA_DIR/val_hermes.parquet" \
    "$@" 2>&1 | tee "$RUN_DIR/train.log"
