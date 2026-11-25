# GSM8K: GRPO with Qwen3-0.6B

GSM8K is a typical RL example in this repository. It trains Qwen3-0.6B for math
reasoning using GRPO, Neuron TorchTitan for training, and vLLM-Neuron for rollout
generation. The example follows the plugin's
[Neuron recipe](https://github.com/amazon-contributing/upstream-to-verl-hardware-plugin/blob/5e5aaee1967e228ce116b303089bd55128859540/docs/accelerators/neuron/README.md)
and uses the repository's [shared Docker environment](../docker/README.md).

## Build the image

From the repository root, build the shared base and pass its tag to the GSM8K
Dockerfile. If the base is already built, set `BASE_IMAGE` to its tag and skip
its build command.

```bash
export BASE_IMAGE="$USER-neuron-rl:base" VERL_DEV_IMAGE="$USER-neuron-gsm8k:dev"
docker build --platform linux/amd64 -t "$BASE_IMAGE" .
docker build --platform linux/amd64 --build-arg BASE_IMAGE="$BASE_IMAGE" -t "$VERL_DEV_IMAGE" gsm8k
docker run --rm "$VERL_DEV_IMAGE" check
```

## Run the example

On a Trainium2 host with driver `2.31.7.0` and EFA, run the image with its
bundled sources. Run from the repository root with `VERL_DEV_IMAGE` set.

```bash
mkdir -p .docker-data/{data,cache,runs}
docker run --rm --init \
  --device /dev/neuron0 --device /dev/neuron1 --device /dev/infiniband \
  --shm-size 16g --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$PWD/.docker-data/data,dst=/data" \
  --mount "type=bind,src=$PWD/.docker-data/cache,dst=/cache" \
  --mount "type=bind,src=$PWD/.docker-data/runs,dst=/runs" \
  "$VERL_DEV_IMAGE"
```

The default command checks the hardware, downloads Qwen3-0.6B at a pinned
revision, prepares GSM8K, and runs two GRPO steps. Look for `step:1`, `step:2`,
and exit code 0. This checks generation, rewards, training, and weight synchronization.

For development with editable checkouts, follow the [source checkout setup](../docker/README.md#2-prepare-the-editable-source-checkouts)
and run the same image with the development launcher.

```bash
bash docker/dev.sh verl-neuron gsm8k
```

To run one stage at a time from a development shell, use

```bash
bash docker/dev.sh
# Inside the container
verl-neuron check --hardware
verl-neuron prepare
verl-neuron gsm8k
```

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

Both launch commands store artifacts on the host.

| Artifact | Host location |
| --- | --- |
| Model weights | `.docker-data/data/Qwen3-0.6B/` |
| Preprocessed GSM8K | `.docker-data/data/gsm8k/` |
| Compiler and download caches | `.docker-data/cache/` |
| Training logs | `.docker-data/runs/<UTC timestamp>/train.log` |

Cold compilation may take more than 15 minutes. Preserve the cache mount paths
inside the container because compiled graphs reference NKI binaries by absolute
path.

Set `RAY_DEDUP_LOGS=0` in the container to disable Ray log deduplication while debugging.

The [Docker validation results](RESULTS.md) record the successful
two-step run with editable host checkouts.
