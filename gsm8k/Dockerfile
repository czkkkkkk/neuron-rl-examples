ARG BASE_IMAGE=neuron-rl-base:2.33
FROM ${BASE_IMAGE}

WORKDIR /workspace/gsm8k
COPY config/neuron_gsm8k.yaml config/neuron_gsm8k.yaml
# Vendored from verl 1876b06d0a3e4e71e06230be10af14492ca8a75b.
COPY prepare_gsm8k.py .
COPY --chmod=755 entrypoint.sh /usr/local/bin/verl-neuron
ENV MODEL_DIR=/data/Qwen3-0.6B \
    MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca \
    DATA_DIR=/data/gsm8k \
    CONFIG_PATH=/workspace/gsm8k/config \
    CONFIG_NAME=neuron_gsm8k
ENTRYPOINT ["/usr/local/bin/verl-neuron"]
CMD ["gsm8k"]
