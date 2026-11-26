#!/usr/bin/env bash
set -euo pipefail
recipe=$(cd "$(dirname "$0")" && pwd)
for arg in "$@"; do
    if [[ $arg == --dry-run || $arg == --help || $arg == -h ]]; then exec python3 "$recipe/benchmark.py" "$@"; fi
done
: "${BASE_IMAGE:?Set BASE_IMAGE to the image built with the root Dockerfile}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3.8-27B HF checkpoint directory}"
: "${TORCHTITAN_DIR:?Set TORCHTITAN_DIR to the ported TorchTitan-Neuron worktree}"
python3 "$recipe/benchmark.py" "$@" --dry-run >/dev/null
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || { echo "Missing HF checkpoint index in $MODEL_PATH" >&2; exit 1; }
[[ -f "$TORCHTITAN_DIR/torchtitan_neuron/models/qwen3_8/context_parallel.py" ]] || { echo "Missing state-parallel GDN in $TORCHTITAN_DIR" >&2; exit 1; }
storage=${VERL_DEV_STORAGE:-"$recipe/../../.docker-data"}
mkdir -p "$storage"/{cache,runs}
run=$(mktemp -d "$storage/runs/qwen38-cp-$(date -u +%Y%m%d-%H%M%S)-XXXXXX")
run=$(realpath "$run")
container="$(basename "$run")"
flags=()
for ((i=0; i<16; i++)); do flags+=(--device "/dev/neuron$i"); done
for repo in examples titan; do
    dir=$TORCHTITAN_DIR; if [[ $repo == examples ]]; then dir="$recipe/../.."; fi
    git -C "$dir" rev-parse HEAD > "$run/$repo.revision"
    git -C "$dir" status --short > "$run/$repo.status"
    git -C "$dir" diff HEAD > "$run/$repo.patch"
done
docker image inspect --format '{{.Id}}' "$BASE_IMAGE" > "$run/image.id"
printf 'Benchmark results: %s\n' "$run"
trap 'docker rm -f "$container" >/dev/null 2>&1 || true' EXIT
docker run -d --init --name "$container" "${flags[@]}" --device /dev/infiniband --shm-size 16g --ulimit memlock=-1:-1 \
    --mount "type=bind,src=$recipe,dst=/benchmark,readonly" \
    --mount "type=bind,src=$(realpath "$TORCHTITAN_DIR"),dst=/opt/verl/sources/torchtitan-neuron,readonly" \
    --mount "type=bind,src=$(realpath "$MODEL_PATH"),dst=/data/Qwen3.8-27B,readonly" \
    --mount "type=bind,src=$(realpath "$storage/cache"),dst=/cache" --mount "type=bind,src=$run,dst=/runs" \
    -e OMP_NUM_THREADS=1 -e TOKENIZERS_PARALLELISM=false -e HF_HUB_OFFLINE=1 \
    -e NEURON_USE_NKI_ALLTOALLV=0 -e NEURON_RT_MAP_HBM=1 -e NEURON_RT_ENABLE_P2P=0 -e FI_EFA_ENABLE_SHM_TRANSFER=0 \
    --entrypoint sleep "$BASE_IMAGE" infinity >/dev/null
docker exec -u root "$container" bash /benchmark/setup.sh > "$run/setup.log" 2>&1
docker exec "$container" neuron-ls -j > "$run/devices.json"
docker exec -e VERL_NEURON_PROCESS_ROLE=trainer -e TORCH_DEVICE_BACKEND_AUTOLOAD=1 "$container" python /benchmark/benchmark.py "$@"
