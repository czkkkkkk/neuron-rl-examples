# Shared Docker environment for RL on Neuron

Follow these steps to build your own image, open a development container, and run
an RL example on Neuron. The environment is shared by the repository's examples;
[GSM8K](../gsm8k/README.md) is used below as a typical first run.

**Host** commands run in your normal SSH terminal on the cluster.
**Container** commands run in the shell opened by `docker/dev.sh` in step 4.

## 1. Check the host and enter the repository

Running the examples requires a Linux Trainium2 host with Docker, Neuron driver
`2.31.7.0`, and EFA drivers/devices already installed. The image supplies userspace
libraries; it does not install host kernel drivers. Building the image itself
does not require Neuron devices.

**Host:**

```bash
# Replace this with the location of your neuron-rl-examples checkout.
cd /path/to/neuron-rl-examples

docker info
cat /sys/module/neuron/version
ls -l /dev/neuron0 /dev/neuron1 /dev/infiniband
```

`docker info` must connect to the Docker daemon. The driver version should be
`2.31.7.0`, and the listed device paths must exist before starting the development
container. The recipe's hardware check enforces that driver version.

Use the repository root as the working directory for all following host commands.
The Docker build needs the root `Dockerfile`, `docker/`, and example files.

## 2. Prepare the editable source checkouts

The development launcher expects these sibling directories on the host:

```text
public_verl/
  neuron-rl-examples/
  upstream-to-verl-hardware-plugin/
  torchtitan-neuron/
  vllm-neuron/
```

If you already have these clones, keep your existing branches and edits and
continue to step 3. Otherwise, create them from the repository root.

**Host, for new clones:**

```bash
git clone https://github.com/amazon-contributing/upstream-to-verl-hardware-plugin.git ../upstream-to-verl-hardware-plugin
git clone https://github.com/czkkkkkk/torchtitan-neuron.git ../torchtitan-neuron
git clone https://github.com/czkkkkkk/vllm-neuron.git ../vllm-neuron

# Starting revisions validated with this environment:
git -C ../upstream-to-verl-hardware-plugin checkout 5e5aaee1967e228ce116b303089bd55128859540
git -C ../torchtitan-neuron checkout 4c01242f15d86a724110310847aae77526fe674b
git -C ../vllm-neuron checkout eb6cb360f7cff6082e22a9437d8ce0ba3891304a
```

The two Neuron backend clones come from the `czkkkkkk` forks. The pinned commits
above match the validated environment; create development branches from them
when needed. The launcher never checks out commits or changes branches.

Use ordinary Git clones. Linked Git worktrees need their external Git metadata
mounted too. For a different directory layout, set the source directory variables
listed under [Launcher options](#launcher-options) before starting the container.

## 3. Choose your image name and build it

Every user must choose a unique image name on the shared cluster. Replace `alice`
with your own identifier. Keep this value in the host shell for subsequent steps.

**Host:**

```bash
export VERL_DEV_IMAGE=alice-neuron-rl:dev
docker build --platform linux/amd64 --progress=plain -t "$VERL_DEV_IMAGE" .

# Confirm the image exists and check its installed package versions.
docker image inspect "$VERL_DEV_IMAGE" --format '{{.Id}}'
docker run --rm "$VERL_DEV_IMAGE" check
```

Wait for the build to finish successfully before continuing. It installs Python
3.12.13, pinned RL dependencies, Neuron runtime and collectives, EFA userspace
libraries, CUDA stubs for NIXL, Git, GitHub CLI, and pytest. Ubuntu 26.04 supplies
GLIBC 2.43 required by the TorchNeuron wheel. The image uses the standard `ubuntu`
user.

The build also checks package versions and native compiler shared-library
dependencies. It downloads pinned source snapshots from public repositories;
your local checkouts are mounted when you start the development container.

## 4. Start the container and open its shell

**Host, in the same shell with `VERL_DEV_IMAGE` set:**

```bash
bash docker/dev.sh
```

This creates and starts a persistent container, then opens Bash inside it. With
the example image name, it prints:

```text
Using container alice-neuron-rl-dev (image alice-neuron-rl:dev).
```

If the matching container already exists, the script reuses it and starts it if
it was stopped. The container name comes from the image name, replacing characters
such as `/` and `:` with `-`. `VERL_DEV_CONTAINER` can override it.

**Container:**

```bash
pwd
# Expected: /workspace
whoami
# Expected: ubuntu
command -v python
# Expected: /opt/verl/venv/bin/python
```

Python is already on `PATH`; no environment activation is needed. The launcher
mounts your host directories as follows:

| Host checkout | Container path |
| --- | --- |
| `../torchtitan-neuron` | `/opt/verl/sources/torchtitan-neuron` |
| `../vllm-neuron` | `/opt/verl/sources/vllm-neuron` |
| `../upstream-to-verl-hardware-plugin` | `/opt/verl/sources/verl-hardware-plugin` |
| This examples directory | `/workspace` |

`/workspace` is the same `neuron-rl-examples` directory you see on the host.
Edits from either side are immediately visible on the other. The three source
mounts include `.git` and replace the image's snapshots at their editable-install
paths. Branches and commits therefore persist in your host clones.

The launcher also passes through `/dev/neuron0`, `/dev/neuron1`, and
`/dev/infiniband`, with 16 GiB shared memory and unlimited locked memory. This
provides the four logical NeuronCores used by the GSM8K smoke configuration.
Coordinate use of these devices on the shared cluster; run one job at a time on
them.

## 5. Check the environment and run an example

Continue in the container shell. GSM8K is a typical first example; other examples
are listed in the [repository guide](../README.md#examples).

**Container:**

```bash
verl-neuron check --hardware

# Download Qwen3-0.6B and prepare GSM8K.
verl-neuron prepare

# Run the default two-step GRPO example.
verl-neuron gsm8k

# Check immediately after the training command returns.
echo $?
# Expected: 0
```

The first run may spend more than 15 minutes compiling kernels. Look for
`step:1` and `step:2` metrics in the output and a successful return to the shell.
The two-step run validates execution, not model quality.

`verl-neuron gsm8k` also performs preparation automatically, so subsequent runs
can call it directly. Downloads, preprocessed data, and compiler caches are reused.
See the [GSM8K guide](../gsm8k/README.md) for configuration changes and longer
runs with evaluation.

For later runs, you can launch a command through the container directly from the
host instead of opening an interactive shell.

**Host, with `VERL_DEV_IMAGE` set:**

```bash
bash docker/dev.sh verl-neuron gsm8k
```

## 6. Inspect logs and persistent files

The GSM8K run writes `/runs/<UTC timestamp>/train.log`.

The image enables Ray log deduplication by default (`RAY_DEDUP_LOGS=1`).
To disable it for debugging, run `RAY_DEDUP_LOGS=0 verl-neuron gsm8k` inside
the container. Ray still preserves repeated messages from the same source.

**Container, after the run:**

```bash
ls -lt /runs
latest_run=$(ls -1dt /runs/*/ | head -n 1)
tail -n 50 "$latest_run/train.log"
```

The following paths are bind mounts, so files survive container removal:

| Host directory | Container path |
| --- | --- |
| `.docker-data/data` | `/data` |
| `.docker-data/cache` | `/cache` |
| `.docker-data/runs` | `/runs` |

On the host, read the same log at
`.docker-data/runs/<UTC timestamp>/train.log`. GSM8K weights and data are stored
under `.docker-data/data/Qwen3-0.6B/` and `.docker-data/data/gsm8k/`.

Keep cache mount paths inside the container stable: compiled graphs refer to NKI
binaries by absolute path. `VERL_DEV_STORAGE` changes the storage location on the
host while preserving the container paths.

## 7. Leave, re-enter, or stop the container

**Container:**

```bash
exit
```

You are back in the host terminal. Exiting Bash leaves the container running.
To re-enter it from a new host terminal, set your image name again:

**Host:**

```bash
cd /path/to/neuron-rl-examples
export VERL_DEV_IMAGE=alice-neuron-rl:dev
bash docker/dev.sh
```

For an additional shell in an already running container, or to stop it when your
work is finished, use its printed container name.

**Host, using the example name from step 4:**

```bash
# Open another shell; exit it to return to the host.
docker exec -it alice-neuron-rl-dev bash

# Stop your container when no job is running in it.
docker stop alice-neuron-rl-dev
```

`bash docker/dev.sh` starts a stopped container again. Python packages installed
inside it and GitHub authentication persist across stops and starts.

## Develop and create PRs

Edit `/workspace` or the mounted source repositories from either the host or
container. New Python processes load source edits without rebuilding the image.
Restart a running RL job after editing its code.

The Python environment is writable. If package metadata, entry points, or
top-level packages change, refresh the editable installations.

**Container:**

```bash
uv pip install --no-deps --no-build-isolation \
  -e /opt/verl/sources/torchtitan-neuron \
  -e /opt/verl/sources/vllm-neuron \
  -e /opt/verl/sources/verl-hardware-plugin
```

Keep `--no-deps` for these installs: the pinned environment deliberately excludes
incompatible GPU/audio dependencies declared by upstream packages. Add other
development tools with `uv pip install`, and record dependency changes in the
Docker build inputs so teammates can reproduce them.

Use ordinary Git commands inside a mounted repository.

**Container:**

```bash
cd /opt/verl/sources/torchtitan-neuron
git switch -c my-change
git status
```

Configure your Git author, then run `gh auth login` and `gh auth setup-git` in the
container when you need GitHub access. Edit, test, commit, and push using your
usual workflow. Use `gh pr create --repo <target-repository>` for the intended
fork or upstream. Follow each repository's contributing instructions for checks.
The image and launcher do not include or copy host credentials.

## Rebuild after changing image dependencies

Source edits in the bind mounts do not need a rebuild. Changes to the Dockerfile
or installed dependencies do.

**Host, from the repository root with `VERL_DEV_IMAGE` set:**

```bash
docker build --platform linux/amd64 --progress=plain -t "$VERL_DEV_IMAGE" .
```

An existing container continues to use its original image. After a changed build,
the launcher refuses to reuse that container. Either choose a new container name:

```bash
# Host: replace alice with your own identifier.
export VERL_DEV_CONTAINER=alice-neuron-rl-dev-v2
bash docker/dev.sh
```

Or, when finished with the old container, stop and remove it, then run
`bash docker/dev.sh` again. Removing a container discards packages and credentials
added inside it. Mounted source checkouts, models, caches, and logs persist.

## Launcher options

Set these variables in the **host** shell before calling `docker/dev.sh`:

| Variable | Default |
| --- | --- |
| `VERL_DEV_IMAGE` | Required: your own image name, e.g. `alice-neuron-rl:dev` |
| `VERL_DEV_CONTAINER` | Derived from the image name, e.g. `alice-neuron-rl-dev` |
| `VERL_DEV_STORAGE` | `<examples directory>/.docker-data` |
| `VERL_DEV_PLUGIN_DIR` | `../upstream-to-verl-hardware-plugin` |
| `VERL_DEV_TORCHTITAN_DIR` | `../torchtitan-neuron` |
| `VERL_DEV_VLLM_DIR` | `../vllm-neuron` |

Changing mounts or the image requires a new container. `VERL_DEV_IMAGE` has no
default; the launcher exits with an instruction to set it when it is missing.

## Share or run without local source

Share this repository, including `Dockerfile`, `docker/`, and the example files
referenced by the build (currently the GSM8K configuration and preprocessor).
All image inputs are public. The build does not copy local checkouts, credentials,
models, or caches. Source revisions are Docker build arguments, and the complete
Python package set is in `docker/requirements.lock`.

The built image also runs with its included sources, without any development
checkouts. For example, run GSM8K with:

**Host, from the repository root with `VERL_DEV_IMAGE` set:**

```bash
mkdir -p .docker-data/{data,cache,runs}
docker run --rm --init \
  --device /dev/neuron0 --device /dev/neuron1 --device /dev/infiniband \
  --shm-size 16g --ulimit memlock=-1:-1 \
  --mount "type=bind,src=$PWD/.docker-data/data,dst=/data" \
  --mount "type=bind,src=$PWD/.docker-data/cache,dst=/cache" \
  --mount "type=bind,src=$PWD/.docker-data/runs,dst=/runs" \
  "$VERL_DEV_IMAGE" gsm8k
```

This command uses the image's included source snapshots. Use `docker/dev.sh`
when developing against host checkouts. The image's default command is currently
GSM8K; the development launcher opens a shell by default.

The development image completed the two-step GSM8K run with editable host
checkouts; see [RESULTS.md](RESULTS.md) for validation details.
