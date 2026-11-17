#!/usr/bin/env bash
set -euo pipefail
uv venv --managed-python --python 3.12.13 /opt/verl/venv

# This is the complete tested package set. --no-deps preserves the Neuron
# recipe's intentional exclusions of CUDA kernels, audio libraries, and cu13.
uv pip install --python "$VERL_NEURON_PYTHON" --torch-backend cpu \
    --index https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match \
    --no-deps -r /opt/verl/build/requirements.lock
uv pip install --python "$VERL_NEURON_PYTHON" --no-deps --no-build-isolation \
    -e /opt/verl/sources/torchtitan-neuron \
    -e /opt/verl/sources/vllm-neuron \
    -e "$PLUGIN_DIR"

"$VERL_NEURON_PYTHON" - <<'PY'
import sys
import sysconfig
from pathlib import Path

prefix = Path(sys.prefix)
links = prefix / "lib/nixl-cuda-stubs"
links.mkdir()
runtime = prefix / "lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
(links / "libcudart.so.12").symlink_to(runtime)
(links / "libcuda.so.1").symlink_to("/opt/aws/neuron/cuda-stubs/libcuda.so.1")
# The lite runtime imports TorchXLA, which links against shared libpython.
shared_python = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
for name in {shared_python.name, sysconfig.get_config_var("INSTSONAME")} - {None}:
    (prefix / "lib" / name).symlink_to(shared_python)
PY
"$VERL_NEURON_PYTHON" "$PLUGIN_DIR/scripts/accelerators/neuron/check_neuron_environment.py"
uv pip freeze --python "$VERL_NEURON_PYTHON" > /opt/verl/installed-packages.txt
