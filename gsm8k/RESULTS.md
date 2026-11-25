# Docker validation: 2026-11-16

The original `verl-neuron:2.33` image completed the default
two-step Qwen3-0.6B/GSM8K GRPO run on a `trn2.48xlarge`, with exit code **0**.
The run used `docker/dev.sh` with all three host Git checkouts mounted as
editable packages, using the Python environment installed inside the image.

Image ID:
`sha256:67dab12e66e315c80c94d37ce3aa6a92f4037fbddace7e5f6fccfd20d362bdca`

| Source | Commit |
| --- | --- |
| `czkkkkkk/torchtitan-neuron` | `4c01242f15d86a724110310847aae77526fe674b` |
| `czkkkkkk/vllm-neuron` | `eb6cb360f7cff6082e22a9437d8ce0ba3891304a` |
| `amazon-contributing/upstream-to-verl-hardware-plugin` | `5e5aaee1967e228ce116b303089bd55128859540` |

Validation covered:

- Standalone package/version checks without host source or Python mounts.
- Shared-library resolution for all eight native Neuron compiler binaries.
- Compilation and execution of a small Neuron tensor operation.
- Editable package paths resolving into the host mounts, with the expected Git
  commits and `czkkkkkk` remotes.
- Writable Python packages under host UID/GID `1000:1000`.
- Generation, reward computation, backward/optimizer execution, and NIXL
  transfer of all 23 weight buckets on both ranks at policy steps 0, 1, and 2.

| Step | Mean reward | Gradient norm | Step time |
| --- | ---: | ---: | ---: |
| 1 | 0 | 0 | 84.80 s |
| 2 | 0.375 | 2.12194 | 11.98 s |

The first batch received all-zero rewards, so its GRPO advantages and gradient
were zero. Step 2 had a nonzero gradient. These two steps validate execution;
they do not establish model quality or steady-state performance. Step times
exclude startup and initial weight synchronization.

The log includes multiprocessing `resource_tracker`/shared-memory cleanup
messages from vLLM, including `KeyError` messages. They did not prevent completion.
After the run, `neuron-ls` reported no Neuron processes and the development
container contained only its init and idle sleep processes. All three host Git
checkouts remained clean.

Local artifacts:

- Training log: `.docker-data/runs/20261116-225604/train.log`
- Build log: `setup/docker-build-final.log`
- Full launch output: `setup/docker-gsm8k.log`
- Image, mounts, versions, and metrics: `setup/docker-provenance.json`

These logs and caches are ignored by Git. The image includes `libarchive13t64`,
which is required by Neuron's `walrus_driver` compiler.
