#!/usr/bin/env bash
set -euo pipefail

prepare_gsm8k() {
    mkdir -p "$MODEL_DIR" "$DATA_DIR"
    # Mark only complete downloads, so an interrupted download is retried.
    if [[ ! -f "$MODEL_DIR/.verl-download-$MODEL_REVISION" ]]; then
        hf download Qwen/Qwen3-0.6B --revision "$MODEL_REVISION" --local-dir "$MODEL_DIR"
        touch "$MODEL_DIR/.verl-download-$MODEL_REVISION"
    fi
    if [[ ! -f "$DATA_DIR/train.parquet" || ! -f "$DATA_DIR/test.parquet" ]]; then
        python /workspace/gsm8k/prepare_gsm8k.py --local_save_dir "$DATA_DIR"
    fi
}

case "${1:-gsm8k}" in
    gsm8k)
        if [[ $# -gt 0 ]]; then shift; fi
        python "$PLUGIN_DIR/scripts/accelerators/neuron/check_neuron_environment.py" --hardware
        prepare_gsm8k
        export RUN_DIR=${RUN_DIR:-"/runs/$(date -u +%Y%m%d-%H%M%S)"}
        exec bash "$PLUGIN_DIR/scripts/accelerators/neuron/run_neuron.sh" "$@"
        ;;
    prepare)
        prepare_gsm8k
        ;;
    check)
        shift
        exec python "$PLUGIN_DIR/scripts/accelerators/neuron/check_neuron_environment.py" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
