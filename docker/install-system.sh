#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates curl git gnupg libnuma-dev libgomp1 procps \
    build-essential pkg-config python3

curl --fail --location --retry 3 \
    https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
    | gpg --dearmor -o /usr/share/keyrings/aws-neuron.gpg
echo "deb [signed-by=/usr/share/keyrings/aws-neuron.gpg] https://apt.repos.neuron.amazonaws.com resolute main" \
    > /etc/apt/sources.list.d/neuron.list
apt-get update
apt-get install -y --no-install-recommends \
    aws-neuronx-runtime-lib=2.35.30.0-d2725ccb8 \
    aws-neuronx-collectives=2.35.43.0-354ad3064 \
    aws-neuronx-tools=2.33.38.0-f874d5618

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT
curl --fail --location --retry 3 \
    https://efa-installer.amazonaws.com/aws-efa-installer-1.51.0.tar.gz \
    -o "$build_dir/efa.tar.gz"
echo "83993044fa4df3dc2f8b5c2f7051d61b9689441be023a2c92ea51d3f53f2465c  $build_dir/efa.tar.gz" \
    | sha256sum --check
tar -xzf "$build_dir/efa.tar.gz" -C "$build_dir"
(
    cd "$build_dir/aws-efa-installer"
    # Kernel modules belong on the host; no hardware is needed during build.
    bash efa_installer.sh --yes --skip-kmod --skip-limit-conf \
        --skip-mpi --skip-plugin --no-verify
)
/opt/amazon/efa/bin/fi_info --version

curl --fail --location --retry 3 \
    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
    -o "$build_dir/cuda-keyring.deb"
dpkg -i "$build_dir/cuda-keyring.deb"
apt-get update
apt-get install -y --no-install-recommends cuda-cudart-dev-12-6=12.6.77-1
mkdir -p /opt/aws/neuron/cuda-stubs
ln -s /usr/local/cuda-12.6/targets/x86_64-linux/lib/stubs/libcuda.so \
    /opt/aws/neuron/cuda-stubs/libcuda.so.1
ldconfig
apt-get clean
rm -rf /var/lib/apt/lists/*
