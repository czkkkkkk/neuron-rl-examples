#!/usr/bin/env bash
set -euo pipefail
runtime=2.35.33.0-d2725ccb8
collectives=2.35.49.0-354ad3064
if [[ $(dpkg-query -W -f='${Version}' aws-neuronx-runtime-lib) != "$runtime" || $(dpkg-query -W -f='${Version}' aws-neuronx-collectives) != "$collectives" ]]; then
    apt-get update
    apt-get install -y --no-install-recommends "aws-neuronx-runtime-lib=$runtime" "aws-neuronx-collectives=$collectives"
    rm -rf /var/lib/apt/lists/*
fi
uv pip install --python "$VERL_NEURON_PYTHON" --no-deps --default-index https://pip.repos.neuron.amazonaws.com \
    'libtorch-neuronx-lite==2.13.0.1.0.4089+71ce3449'
fetch() {
    local repo=$1 sha=$2 dest=$3
    test "${#sha}" = 40
    mkdir -p "$dest"
    curl --fail --location --retry 3 "https://codeload.github.com/$repo/tar.gz/$sha" | tar -xz -C "$dest" --strip-components=1
    printf '%s %s\n' "$repo" "$sha" >> /opt/verl/qwen38-revisions.txt
}
sources=/opt/verl/sources
rm -rf "$sources/torchtitan-neuron" "$sources/vllm-neuron" "$PLUGIN_DIR"
fetch czkkkkkk/torchtitan-neuron "$TITAN_REV" "$sources/torchtitan-neuron"
fetch czkkkkkk/vllm-neuron "$VLLM_REV" "$sources/vllm-neuron"
fetch amazon-contributing/upstream-to-verl-hardware-plugin "$VERL_PLUGIN_REV" "$PLUGIN_DIR"
fetch aws-neuron/nki-library "$NKILIB_REV" /tmp/qwen-nki-library
uv pip install --python "$VERL_NEURON_PYTHON" --no-deps --no-build-isolation --reinstall \
    -e "$sources/torchtitan-neuron" -e "$sources/vllm-neuron" -e "$PLUGIN_DIR" /tmp/qwen-nki-library
rm -rf /tmp/qwen-nki-library
