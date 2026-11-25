# syntax=docker/dockerfile:1
# Build the base target of the root Dockerfile first.
ARG BASE_IMAGE=neuron-rl-base:2.33
FROM ${BASE_IMAGE}
USER root
ARG TITAN_REV=f7a0ca1e6999d9c08901535729f57b769a9f488d
ARG VLLM_REV=330691a3545685c36d4edf62882c0de052dd3e4d
ARG VERL_PLUGIN_REV=cfbdcf0c7c52f0fd783d7e0e86fa80f549aed378
ARG NKILIB_REV=ad2f1b921d8e5949fa3b9a186e5f9130269353dc
COPY qwen3_8/install.sh /tmp/install-qwen38.sh
RUN bash /tmp/install-qwen38.sh && rm /tmp/install-qwen38.sh
COPY gsm8k/prepare_gsm8k.py /opt/verl/examples/gsm8k/prepare_gsm8k.py
COPY --chown=ubuntu:ubuntu qwen3_8/ /opt/verl/examples/qwen3_8/
ENV VLLM_USE_V2_MODEL_RUNNER=1 \
    MODEL_REVISION=37561816e66b5b9153b1b64c10968c18c1e6f4a0 \
    MODEL_DIR=/data/Qwen3.8-27B DATA_DIR=/data/gsm8k
USER ubuntu
WORKDIR /opt/verl/examples/qwen3_8
RUN python check_sdk.py
CMD ["python", "check_sdk.py", "--hardware"]
