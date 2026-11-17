# syntax=docker/dockerfile:1
FROM ubuntu:26.04@sha256:d66b366dac61b575bee2aef07e9c26e8070096fbe84a3d4d3eb0da069b5d76db

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG TARGETARCH
RUN test "$TARGETARCH" = amd64 || { echo "Build with --platform linux/amd64 for Trainium2."; exit 1; }

COPY docker/install-system.sh /tmp/install-system.sh
RUN bash /tmp/install-system.sh && rm /tmp/install-system.sh

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/verl/python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/verl/venv \
    VERL_NEURON_PYTHON=/opt/verl/venv/bin/python \
    PLUGIN_DIR=/opt/verl/sources/verl-hardware-plugin \
    PATH=/opt/verl/venv/bin:/opt/aws/neuron/bin:/opt/amazon/efa/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/opt/verl/venv/lib:/opt/aws/neuron/lib:/opt/aws/neuron/cuda-stubs:/opt/amazon/efa/lib:/opt/amazon/efa/lib64 \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    VERL_PLATFORM=aws \
    VERL_NEURON_PROCESS_ROLE=controller \
    VLLM_PLUGINS="" \
    NEURON_LOGICAL_NC_CONFIG=2

# Immutable public source snapshots, matching the local GSM8K validation.
ARG PLUGIN_REVISION=5e5aaee1967e228ce116b303089bd55128859540
ARG TORCHTITAN_REVISION=4c01242f15d86a724110310847aae77526fe674b
ARG VLLM_NEURON_REVISION=eb6cb360f7cff6082e22a9437d8ce0ba3891304a
RUN <<'BASH'
set -euo pipefail
fetch_source() {
    local repository=$1 revision=$2 destination=$3
    mkdir -p "$destination"
    curl --fail --location --retry 3 "https://codeload.github.com/$repository/tar.gz/$revision" \
        | tar -xz --strip-components=1 -C "$destination"
}
fetch_source amazon-contributing/upstream-to-verl-hardware-plugin "$PLUGIN_REVISION" "$PLUGIN_DIR"
fetch_source czkkkkkk/torchtitan-neuron "$TORCHTITAN_REVISION" /opt/verl/sources/torchtitan-neuron
fetch_source czkkkkkk/vllm-neuron "$VLLM_NEURON_REVISION" /opt/verl/sources/vllm-neuron
printf 'verl-hardware-plugin %s\ntorchtitan-neuron %s\nvllm-neuron %s\n' \
    "$PLUGIN_REVISION" "$TORCHTITAN_REVISION" "$VLLM_NEURON_REVISION" > /opt/verl/source-revisions.txt
BASH

COPY docker/requirements.lock docker/install-python.sh /opt/verl/build/
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /opt/verl/build/install-python.sh \
    && chown -R ubuntu:ubuntu /opt/verl

# libarchive is required by neuronx-cc's native compiler. GitHub CLI and SSH
# support development against the mounted Git checkouts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libarchive13t64 gh openssh-client \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Check native compiler dependencies even on builders without Neuron hardware.
RUN python - <<'PY'
from pathlib import Path
import subprocess
import sys

compiler = Path(sys.prefix) / "lib/python3.12/site-packages/neuronxcc"
checked = 0
for binary in compiler.glob("**/bin/*"):
    if not binary.is_file():
        continue
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            continue
    result = subprocess.check_output(["ldd", str(binary)], text=True)
    if "not found" in result:
        raise SystemExit(f"Missing compiler dependencies for {binary}:\n{result}")
    checked += 1
assert checked, "No native Neuron compiler binaries found"
print(f"Checked shared libraries for {checked} Neuron compiler binaries.")
PY

COPY gsm8k/config/neuron_gsm8k.yaml /opt/verl/examples/gsm8k/config/neuron_gsm8k.yaml
# Vendored from verl 1876b06d0a3e4e71e06230be10af14492ca8a75b.
COPY gsm8k/prepare_gsm8k.py /opt/verl/examples/gsm8k/prepare_gsm8k.py
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/verl-neuron
ENV RAY_DEDUP_LOGS=1 \
    MODEL_DIR=/data/Qwen3-0.6B \
    DATA_DIR=/data/gsm8k \
    CONFIG_PATH=/opt/verl/examples/gsm8k/config \
    CONFIG_NAME=neuron_gsm8k \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache \
    NKI_COMPILE_CACHE_URL=/cache/nki \
    VLLM_CACHE_ROOT=/cache/vllm \
    TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor \
    PYTHONPYCACHEPREFIX=/cache/pycache \
    MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
RUN mkdir -p /data /cache /runs \
    && chown ubuntu:ubuntu /data /cache /runs
USER ubuntu
WORKDIR /opt/verl/examples/gsm8k
ENTRYPOINT ["/usr/local/bin/verl-neuron"]
CMD ["gsm8k"]
