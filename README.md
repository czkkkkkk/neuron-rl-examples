# RL examples on AWS Neuron

This repository provides examples for running reinforcement learning on AWS
Neuron with verl. The examples share an environment built around Neuron
TorchTitan for training, vLLM-Neuron for rollout generation, and
`upstream-to-verl-hardware-plugin` for integration with verl.

GSM8K is a typical starting example: it demonstrates GRPO training for math
reasoning with Qwen3-0.6B. Example-specific models, datasets, configurations,
launch instructions, and results live in their respective directories.

## Shared environment and development

The [Docker guide](docker/README.md) describes the shared RL environment, host
requirements, source checkout setup, and development workflow. Choose your own
unique base image name on the shared cluster, then build and open a development
shell from the repository root.

```bash
export BASE_IMAGE=alice-neuron-rl:base  # Replace alice with your own identifier.
docker build --platform linux/amd64 -t "$BASE_IMAGE" .
export VERL_DEV_IMAGE="$BASE_IMAGE"
bash docker/dev.sh
```

Each example below builds its own image with `FROM ${BASE_IMAGE}`. For a
development shell, `VERL_DEV_IMAGE` selects the image and determines the
default container name (`alice-neuron-rl-base` above).

The development launcher mounts the sibling `torchtitan-neuron`, `vllm-neuron`,
and `upstream-to-verl-hardware-plugin` Git checkouts at their editable-install
paths. Code edits, branches, and commits persist in the host repositories.

The environment is validated on Trainium2. Python and its dependencies are
installed inside the Docker image.

## Examples

| Example | Model | Method | What it demonstrates |
| --- | --- | --- | --- |
| [GSM8K](gsm8k/README.md) | Qwen3-0.6B | GRPO | Math reasoning with generation, rule-based rewards, training, and actor-to-rollout weight synchronization |
| [Qwen3.8-27B on SDK 2.33.1](qwen3_8/README.md) | Qwen3.8-27B text | GRPO | Three training steps with a full 27B actor and four V2 rollout replicas on one Trn2 |

The GSM8K guide includes a two-step smoke run to verify the environment and
instructions for a longer run with evaluation.

The standalone text-training benchmark compares TP, CP and DP sharding on 64 logical NeuronCores. [Qwen3.8 benchmark](benchmarks/training/README.md)

## Repository layout

- [Dockerfile](Dockerfile) and [docker/](docker/README.md): shared environment,
  editable development launcher, and container documentation.
- [gsm8k/](gsm8k/README.md): the GSM8K example, including its configuration,
  preprocessing, and validation results.
