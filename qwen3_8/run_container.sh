#!/usr/bin/env bash
# Launch the published three-step recipe on an exclusive trn2.48xlarge.
set -euo pipefail
: "${QWEN_IMAGE:?Set QWEN_IMAGE to the built Qwen3.8 image}"
: "${MODEL_PATH:?Set MODEL_PATH to the prepared checkpoint directory}"
: "${DATA_PATH:?Set DATA_PATH to the prepared GSM8K directory}"
: "${WORK_PATH:?Set WORK_PATH to a writable directory for runs and caches}"
test -f "$MODEL_PATH/model.safetensors.index.json"
test -f "$DATA_PATH/train.parquet"
test -f "$DATA_PATH/test.parquet"
mkdir -p "$WORK_PATH/runs" "$WORK_PATH/cache"
run_name=$(python3 -c 'import uuid; print("grpo-" + uuid.uuid4().hex)')
echo "Training output: $WORK_PATH/runs/$run_name"
devices=(--device /dev/infiniband)
for idx in {0..15}; do devices+=(--device "/dev/neuron$idx"); done
docker run --rm --init --network=host --shm-size=160g --ulimit memlock=-1:-1 "${devices[@]}" \
    --mount "type=bind,src=$(realpath "$MODEL_PATH"),dst=/data/Qwen3.8-27B,readonly" \
    --mount "type=bind,src=$(realpath "$DATA_PATH"),dst=/data/gsm8k,readonly" \
    --mount "type=bind,src=$(realpath "$WORK_PATH/cache"),dst=/cache" \
    --mount "type=bind,src=$(realpath "$WORK_PATH/runs"),dst=/runs" \
    -e RUN_DIR="/runs/$run_name" "$QWEN_IMAGE" bash -ec 'mkdir -m 0755 "$RUN_DIR"; exec bash run_grpo.sh'
python3 "$(dirname "$0")/verify_run.py" "$WORK_PATH/runs/$run_name/train.log"
