# Qwen3.8-27B text GRPO on Neuron SDK 2.33.1

This example trains the full BF16 text decoder on GSM8K. One **trn2.48xlarge** runs both roles: 32 logical NeuronCores train the TorchTitan actor (TP4 × FSDP8), and 32 run four vLLM V2 rollout replicas (TP8 each). No GPU, profiler server, second host, private wheel, or sibling Git checkout is required.

The recipe uses 16 prompts per batch, two sampled responses per prompt, a 256-token prompt limit, a 256-token response limit and three optimizer updates. Each update streams the new actor weights over NIXL to all four rollout replicas. The actor ranks share a host-memory bucket. The complete configuration is in [config/](config).

## 1. Build the image

Start in this repository's root. Use Docker with BuildKit/buildx for the shared base Dockerfile and network access to the public package indexes and GitHub. Running requires an exclusive Trainium2 host with all 16 Neuron devices, EFA, and loaded Neuron driver **2.31.8.0**. The image supplies user-space libraries; it does not change the host driver.

```bash
cat /sys/module/neuron/version
ls /dev/neuron{0..15} /dev/infiniband
export BASE_IMAGE="$USER-neuron-rl:base" QWEN_IMAGE="$USER-qwen38-grpo:2.33.1"
docker build --platform linux/amd64 -t "$BASE_IMAGE" .
docker build --platform linux/amd64 --build-arg RL_BASE_IMAGE="$BASE_IMAGE" \
    -t "$QWEN_IMAGE" qwen3_8/
docker run --rm --device /dev/neuron0 "$QWEN_IMAGE"  # SDK pins, driver and a tensor addition
docker run --rm "$QWEN_IMAGE" cat /opt/verl/qwen38-revisions.txt
```

The application uses verl 0.9.1, Transformers 5.10.4, vLLM 0.29.0 and vLLM-Neuron 0.29.0.1.2.0. The [Dockerfile](Dockerfile) fetches exact public source revisions; install them together because the stock SDK packages lack this combination:

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

## 3. Run and verify

Run on the exclusive host with no other Ray cluster using the devices. Keep the environment variables from steps 1 and 2. The host launcher and log verifier require Python 3.7 or newer.

```bash
bash qwen3_8/run_container.sh
```

The launcher uses the image's source snapshots and mounts the checkpoint and dataset read-only. It prints the new run directory under `$WORK_PATH/runs`, keeps compiler caches between runs, and returns a nonzero exit status if training or acceptance fails. The first run compiles training and rollout graphs.

For the pinned configuration, `train.log` must contain three numbered steps with finite, positive actor gradient norms and GSM8K rewards in [0, 1]. The initial policy and the three updated policies must each deliver all **401** buckets in order to **32** rollout workers. To recheck a completed log:

```bash
python3 qwen3_8/verify_run.py "$WORK_PATH/runs/<run>/train.log"
```

The reference run completed all three updates and four policy deliveries. Its mean batch rewards were 0.15625, 0.125 and 0.4375. Those stochastic numbers are examples, not thresholds or a held-out accuracy result.

For a separate greedy-generation check, [check_generation.py](check_generation.py) accepts `hf` (CPU FP32) and `v2` (BF16 TP8). Run the two commands independently on an idle host:

```bash
docker run --rm --mount "type=bind,src=$MODEL_PATH,dst=/model,readonly" \
    --mount "type=bind,src=$WORK_PATH,dst=/results" -e OMP_NUM_THREADS=12 \
    "$QWEN_IMAGE" python check_generation.py hf /model --output /results/reference.json
docker run --rm --device /dev/neuron0 --device /dev/neuron1 --device /dev/infiniband \
    --shm-size=16g --ulimit memlock=-1:-1 \
    --mount "type=bind,src=$MODEL_PATH,dst=/model,readonly" --mount "type=bind,src=$WORK_PATH,dst=/results" \
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e VERL_NEURON_PROCESS_ROLE=rollout \
    -e VLLM_PLUGINS=neuron -e VLLM_NEURON_BACKEND=neuron_native -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    "$QWEN_IMAGE" python check_generation.py v2 /model --reference /results/reference.json
```

The BF16 and FP32 references agreed on all eight generated token IDs for the two short prompts. This tests the start of decode; it does not measure convergence.
