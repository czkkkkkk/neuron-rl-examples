# Qwen3.8-27B text GRPO on Neuron SDK 2.33.1

This example trains the full BF16 text decoder on GSM8K. One **trn2.48xlarge** runs both roles: 32 logical NeuronCores train the TorchTitan actor (TP4 × FSDP8), and 32 run four vLLM V2 rollout replicas (TP8 each). No GPU, profiler server, second host, private wheel, or sibling Git checkout is required.

The recipe uses 16 prompts per batch, two sampled responses per prompt, a 256-token prompt limit, a 256-token response limit and three optimizer updates. Each update streams the new actor weights over NIXL to all four rollout replicas. The actor ranks share a host-memory bucket. The complete configuration is in [qwen38_grpo.yaml](config/qwen38_grpo.yaml).

## 1. Build the image

Start in this repository's root. Use Docker with BuildKit/buildx for the shared base Dockerfile and network access to the public package indexes and GitHub. Running requires an exclusive Trainium2 host with all 16 Neuron devices, EFA, and loaded Neuron driver **2.31.7.0**. The image supplies the SDK 2.33.1 user-space libraries.

```bash
cat /sys/module/neuron/version
ls /dev/neuron{0..15} /dev/infiniband
export BASE_IMAGE="$USER-neuron-rl:base" QWEN_IMAGE="$USER-qwen38-grpo:2.33.1"
docker build --platform linux/amd64 -t "$BASE_IMAGE" .
docker build --platform linux/amd64 --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -f qwen3_8/Dockerfile -t "$QWEN_IMAGE" .
docker run --rm --device /dev/neuron0 "$QWEN_IMAGE"  # SDK pins, driver and a tensor addition
docker run --rm "$QWEN_IMAGE" cat /opt/verl/qwen38-revisions.txt
```

The [root Dockerfile](../Dockerfile) supplies Python, EFA and the shared RL packages, including verl 0.9.1, Transformers 5.10.4 and vLLM 0.29.0. If you already built the base for GSM8K, set `BASE_IMAGE` to the same tag and run just the Qwen build.

The Qwen image installs the SDK 2.33.1 runtime, collectives and `libtorch-neuronx-lite` updates. It includes the shared GSM8K preprocessing script and pins the following sources, including vLLM-Neuron 0.29.0.1.2.0.

| Component | Revision | What it provides |
| --- | --- | --- |
| [TorchTitan-Neuron](https://github.com/czkkkkkk/torchtitan-neuron/pull/1) | `f7a0ca1e6999d9c08901535729f57b769a9f488d` | Text actor, grouped-head GDN autograd and checkpoint import |
| [vLLM-Neuron](https://github.com/czkkkkkk/vllm-neuron/pull/1) | `330691a3545685c36d4edf62882c0de052dd3e4d` | Stable V2 hybrid model, recurrent state and token mask |
| [verl hardware plugin](https://github.com/amazon-contributing/upstream-to-verl-hardware-plugin/pull/5) | `cfbdcf0c7c52f0fd783d7e0e86fa80f549aed378` | GRPO integration and streamed weight updates |
| [NKI Library](https://github.com/aws-neuron/nki-library/tree/ad2f1b921d8e5949fa3b9a186e5f9130269353dc) | `ad2f1b921d8e5949fa3b9a186e5f9130269353dc` | GDN backward missing from stock 2.33.1; installed without upgrading the compiler |

## 2. Prepare the checkpoint and dataset

Run these commands once with internet access. Use an absolute path on a volume with room for the 27B BF16 checkpoint, the compile caches, and the logs. The container runs as `ubuntu` (UID 1000); make the writable directories accessible to that user. Keep this cache separate from other SDK versions.

```bash
export WORK_PATH="$PWD/qwen38-work"
export MODEL_PATH="$WORK_PATH/Qwen3.8-27B" DATA_PATH="$WORK_PATH/gsm8k"
mkdir -p "$MODEL_PATH" "$DATA_PATH" "$WORK_PATH/cache" "$WORK_PATH/runs"
docker run --rm \
    --mount "type=bind,src=$MODEL_PATH,dst=/data/Qwen3.8-27B" \
    --mount "type=bind,src=$DATA_PATH,dst=/data/gsm8k" \
    --mount "type=bind,src=$WORK_PATH/cache,dst=/cache" \
    "$QWEN_IMAGE" bash prepare.sh
```

[prepare.sh](prepare.sh) downloads [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0) and [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866) at fixed revisions. It converts the `main` train and test splits with the same preprocessing as the small GSM8K example. If you already have those exact inputs, set the variables to their paths and skip the download.

## 3. Run training

Run on the exclusive host with no other Ray cluster using the devices. Keep the environment variables from steps 1 and 2. The host launcher uses Python 3 to name each run directory.

```bash
bash qwen3_8/run_container.sh
```

The launcher uses the image's source snapshots and mounts the checkpoint and dataset read-only. It prints the run directory under `$WORK_PATH/runs` and returns the training container's exit status. That directory's `train.log` contains the step metrics and weight-transfer logs. The first run compiles training and rollout graphs. Subsequent runs reuse the compiler caches.

On driver `2.31.7.0`, the reference run completed all three updates and four policy deliveries with 32 logical cores for training and 32 for inference. Its mean batch rewards were 0.09375, 0.1875 and 0.25. Rewards vary with sampling.
