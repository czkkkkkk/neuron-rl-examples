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
unique image name on the shared cluster, then build and open a development shell
from this repository's root:

```bash
export VERL_DEV_IMAGE=alice-neuron-rl:dev  # Replace alice with your own identifier.
docker build --platform linux/amd64 -t "$VERL_DEV_IMAGE" .
bash docker/dev.sh
```

`VERL_DEV_IMAGE` is required by the
launcher; its value also determines the default container name
(`alice-neuron-rl-dev` in this example).

The development launcher mounts the sibling `torchtitan-neuron`, `vllm-neuron`,
and `upstream-to-verl-hardware-plugin` Git checkouts at their editable-install
paths. Code edits, branches, and commits persist in the host repositories.

The environment is validated on Trainium2. Python and its dependencies are
installed inside the Docker image.

## Examples

| Example | Model | Method | What it demonstrates |
| --- | --- | --- | --- |
| [GSM8K](gsm8k/README.md) | Qwen3-0.6B | GRPO | Math reasoning with generation, rule-based rewards, training, and actor-to-rollout weight synchronization |
| [Qwen3.8-27B on SDK 2.33.1](qwen3_8/README.md) | Qwen3.8-27B text | GRPO | Full 27B actor and four V2 rollout replicas on one Trn2; public source pins and an automated three-step check |

The GSM8K guide includes a two-step smoke run to verify the environment and
instructions for a longer run with evaluation.

## Repository layout

- [Dockerfile](Dockerfile) and [docker/](docker/README.md): shared environment,
  editable development launcher, and container documentation.
- [gsm8k/](gsm8k/README.md): the GSM8K example, including its configuration,
  preprocessing, and validation results.
