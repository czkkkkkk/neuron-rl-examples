# GSM8K: GRPO with Qwen3-0.6B

GSM8K is a typical RL example in this repository. It trains Qwen3-0.6B for math
reasoning using GRPO, Neuron TorchTitan for training, and vLLM-Neuron for rollout
generation. The example follows the plugin's
[Neuron recipe](https://github.com/amazon-contributing/upstream-to-verl-hardware-plugin/blob/5e5aaee1967e228ce116b303089bd55128859540/docs/accelerators/neuron/README.md)
and uses the repository's [shared Docker environment](../docker/README.md).

## Run the example

Prepare the environment and sibling checkouts using the Docker guide, then set
your image name and run from the repository root:

```bash
export VERL_DEV_IMAGE=alice-neuron-rl:dev  # Use your own built image.
bash docker/dev.sh verl-neuron gsm8k
```

Or, with `VERL_DEV_IMAGE` set, open a development shell first:

```bash
bash docker/dev.sh
# Inside the container:
verl-neuron check --hardware
verl-neuron gsm8k
```

The launcher downloads Qwen3-0.6B at a pinned revision, prepares the GSM8K dataset,
and runs two GRPO steps by default. This smoke run verifies generation, rewards,
training, and weight synchronization; it is too short to assess model quality.

## Configuration

The [training configuration](config/neuron_gsm8k.yaml) uses four logical
NeuronCores: actor/reference TP=2 and rollout TP=2. Each step uses four prompts
and two responses per prompt, with prompt and response limits of 512 tokens each.
Sampling and reward computation run on CPU. Actor weights synchronize to rollout
through `neuron_nixl`.

The default run disables evaluation and disk checkpoints. For a longer run with
evaluation, use:

```bash
bash docker/dev.sh verl-neuron gsm8k \
  data.train_max_samples=-1 data.val_max_samples=32 \
  trainer.total_training_steps=20 trainer.test_freq=5
```

The development container reads the configuration from
`/workspace/gsm8k/config/neuron_gsm8k.yaml`, so host edits are available inside
the container. Restart a running job after changing its configuration or code.

## Data, caches, and results

With the Docker development launcher, artifacts persist on the host:

| Artifact | Host location |
| --- | --- |
| Model weights | `.docker-data/data/Qwen3-0.6B/` |
| Preprocessed GSM8K | `.docker-data/data/gsm8k/` |
| Compiler and download caches | `.docker-data/cache/` |
| Training logs | `.docker-data/runs/<UTC timestamp>/train.log` |

Cold compilation may take more than 15 minutes. Preserve the cache mount paths
inside the container because compiled graphs reference NKI binaries by absolute
path.

The [Docker validation results](../docker/RESULTS.md) record the successful
two-step run with editable host checkouts.
