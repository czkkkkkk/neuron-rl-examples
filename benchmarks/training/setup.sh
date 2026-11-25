#!/usr/bin/env bash
# Run as root inside the base container, before starting training workers.
set -euo pipefail
# Keep root-owned package metadata out of the training user's shared cache.
export UV_CACHE_DIR=/var/cache/uv
runtime=2.35.33.0-d2725ccb8
collectives=2.35.49.0-354ad3064
lite=2.13.0.1.0.4089+71ce3449
nki_revision=ad2f1b921d8e5949fa3b9a186e5f9130269353dc
if [[ $(dpkg-query -W -f='${Version}' aws-neuronx-runtime-lib) != "$runtime" || $(dpkg-query -W -f='${Version}' aws-neuronx-collectives) != "$collectives" ]]; then
    apt-get update -qq
    apt-get install -y --no-install-recommends "aws-neuronx-runtime-lib=$runtime" "aws-neuronx-collectives=$collectives"
fi
uv pip install --python "$VERL_NEURON_PYTHON" --no-deps --default-index https://pip.repos.neuron.amazonaws.com "libtorch-neuronx-lite==$lite"
if ! python -c 'import hashlib, importlib.metadata as m; p=m.distribution("nki-library").locate_file("nkilib_src/nkilib/experimental/gdn/gdn_cte_bwd.py"); assert hashlib.sha256(p.read_bytes()).hexdigest()=="d481f09cf3ef043181ffefff9f1b914b63c2e63d198373b815b92affae75c377"' 2>/dev/null; then
    uv pip install --python "$VERL_NEURON_PYTHON" --no-deps --no-build-isolation "https://github.com/aws-neuron/nki-library/archive/$nki_revision.tar.gz"
fi
