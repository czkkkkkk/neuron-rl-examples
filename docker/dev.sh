#!/usr/bin/env bash
# Keep a development container with writable editables and host Git checkouts.
set -euo pipefail
examples_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sources_dir=$(dirname "$examples_dir")
image=${VERL_DEV_IMAGE:?Set VERL_DEV_IMAGE to your own image name, for example alice-neuron-rl:dev}
# Docker image references can contain separators that container names cannot.
container=${VERL_DEV_CONTAINER:-"${image//[!a-zA-Z0-9_.-]/-}"}
storage=${VERL_DEV_STORAGE:-"$examples_dir/.docker-data"}
plugin_dir=${VERL_DEV_PLUGIN_DIR:-"$sources_dir/upstream-to-verl-hardware-plugin"}
torchtitan_dir=${VERL_DEV_TORCHTITAN_DIR:-"$sources_dir/torchtitan-neuron"}
vllm_dir=${VERL_DEV_VLLM_DIR:-"$sources_dir/vllm-neuron"}

for repository in "$plugin_dir" "$torchtitan_dir" "$vllm_dir"; do
    [[ -d "$repository/.git" && -f "$repository/pyproject.toml" ]] || {
        echo "Expected a Git clone with pyproject.toml at $repository." >&2
        exit 1
    }
done
plugin_dir=$(cd "$plugin_dir" && pwd)
torchtitan_dir=$(cd "$torchtitan_dir" && pwd)
vllm_dir=$(cd "$vllm_dir" && pwd)
mkdir -p "$storage"/{data,cache,runs}
storage=$(cd "$storage" && pwd)

# Refuse silently reusing a container with a different image or bind mounts.
configuration=$(printf '%s\n' "$examples_dir" "$plugin_dir" "$torchtitan_dir" \
    "$vllm_dir" "$storage" "$image" ubuntu | sha256sum | cut -d ' ' -f 1)
image_id=$(docker image inspect --format '{{.Id}}' "$image")
if docker container inspect "$container" >/dev/null 2>&1; then
    actual_configuration=$(docker inspect --format '{{index .Config.Labels "verl-neuron.dev.configuration"}}' "$container")
    actual_image=$(docker inspect --format '{{.Image}}' "$container")
    if [[ "$actual_configuration" != "$configuration" || "$actual_image" != "$image_id" ]]; then
        echo "$container has a different image or mount configuration." >&2
        echo "Use a new VERL_DEV_CONTAINER name, or remove the old container when finished with it." >&2
        exit 1
    fi
else
    docker create --init --name "$container" \
        --label "verl-neuron.dev.configuration=$configuration" \
        --device /dev/neuron0 --device /dev/neuron1 --device /dev/infiniband \
        --shm-size 16g --ulimit memlock=-1:-1 \
        --mount "type=bind,src=$examples_dir,dst=/workspace" \
        --mount "type=bind,src=$plugin_dir,dst=/opt/verl/sources/verl-hardware-plugin" \
        --mount "type=bind,src=$torchtitan_dir,dst=/opt/verl/sources/torchtitan-neuron" \
        --mount "type=bind,src=$vllm_dir,dst=/opt/verl/sources/vllm-neuron" \
        --mount "type=bind,src=$storage/data,dst=/data" \
        --mount "type=bind,src=$storage/cache,dst=/cache" \
        --mount "type=bind,src=$storage/runs,dst=/runs" \
        --workdir /workspace --entrypoint /usr/bin/sleep \
        "$image" infinity >/dev/null
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "$container")" != true ]]; then
    docker start "$container" >/dev/null
fi
docker exec "$container" test -w /opt/verl/venv/lib/python3.12/site-packages || {
    echo "The Python environment must be writable by ubuntu. Rebuild your image from this Dockerfile." >&2
    exit 1
}

# Ordinary Python edits are picked up without reinstalling. Refresh metadata
# explicitly with uv after changing package configuration or entry points.
echo "Using container $container (image $image)." >&2
interactive=(-i)
if [[ -t 0 && -t 1 ]]; then interactive=(-it); fi
if [[ $# == 0 ]]; then set -- bash; fi
exec docker exec "${interactive[@]}" --workdir /workspace "$container" "$@"
