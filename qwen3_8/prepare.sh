#!/usr/bin/env bash
# Download public inputs into writable host-mounted directories.
set -euo pipefail
: "${MODEL_DIR:?}" "${DATA_DIR:?}"
mkdir -p "$MODEL_DIR" "$DATA_DIR"
: "${MODEL_REVISION:?}"
gsm8k_rev=740312add88f781978c0658806c59bc2815b9866
hf download Qwen/Qwen3.8-27B --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
hf download --repo-type dataset openai/gsm8k --revision "$gsm8k_rev" --local-dir "$DATA_DIR/raw"
python /opt/verl/examples/gsm8k/prepare_gsm8k.py --local_dataset_path "$DATA_DIR/raw" --local_save_dir "$DATA_DIR"
printf '%s\n' "$MODEL_REVISION" > "$MODEL_DIR/REVISION"
printf '%s\n' "$gsm8k_rev" > "$DATA_DIR/REVISION"
