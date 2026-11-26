# Qwen3.8-27B text training on Trainium2

Run from the root of `neuron-rl-examples` on a Trainium2 host. The launcher
uses the root Dockerfile and the Qwen3.8-27B HF checkpoint. It loads the text
weights and trains with BF16, AdamW, full activation checkpointing and the
bundled C4 sample. The sample loops, and context carries across documents in a sequence.

The commands assume a sibling `torchtitan-neuron` checkout with Qwen3.8
state-based CP support and a checkpoint under `.docker-data/data/Qwen3.8-27B`.
Set `TORCHTITAN_DIR`, `VERL_DEV_STORAGE`, or `MODEL_PATH` to use other locations.

```bash
export BASE_IMAGE="$USER-neuron-rl:base"
docker build --platform linux/amd64 --target base -t "$BASE_IMAGE" .
export TORCHTITAN_DIR="${TORCHTITAN_DIR:-../torchtitan-neuron}"
export VERL_DEV_STORAGE="${VERL_DEV_STORAGE:-$PWD/.docker-data}"
export MODEL_PATH="${MODEL_PATH:-$VERL_DEV_STORAGE/data/Qwen3.8-27B}"
benchmarks/training/run.sh --dry-run
benchmarks/training/run.sh
```

The default sweep uses 64 logical cores (16 Trainium2 devices at LNC2), 2,048
tokens per sequence, a local batch of one and a global batch of 16. Each
optimizer step processes 32,768 tokens across the host.

| TP | CP | DP shard | Accumulation | Logical cores |
|---:|---:|---:|---:|---:|
| 4 | 1 | 16 | 1 | 64 |
| 4 | 2 | 8 | 2 | 64 |
| 4 | 4 | 4 | 4 | 64 |
| 2 | 2 | 16 | 1 | 64 |

TorchTitan combines DP shard and CP for FSDP. Each CP rank owns one
contiguous slice of each sequence. The 48 GDN layers exchange convolution
history and recurrent state summaries. The 16 full-attention layers use SDPA CP.
For TP2 the launcher places each 32-rank FSDP group on two complete rows of
the chip torus and connects TP peers on adjacent rows. The per-case
`config.json` records the physical core for each logical rank.

```bash
benchmarks/training/run.sh --layout 4,2,8 --seq-len 8192
benchmarks/training/run.sh --layout 4,2,8 --seq-len 65536
benchmarks/training/run.sh --layout 4,4,4 --seq-len 131072
```

The last two commands probe 32,768 local tokens per rank, a shape at which
the pinned compiler rejected embedding backward in the long-context sweep.
`--capacity` appends both probes to the sweep. `--dry-run` prints resolved
cases and requires only host Python. The container installs the stable Qwen
runtime, collectives, libtorch wheel and pinned NKI Library before loading
the trainer. First-time setup requires access to the public package sources.

Every invocation prints its output directory under `$VERL_DEV_STORAGE/runs`.
`results.json` records each exit code, global tokens/s across the measured
steps, median step time, loss, gradient norm, rank-zero peak active memory
when the SDK reports it, and elapsed startup through the first optimizer step.
Three warmup steps precede ten measured steps by default. Startup includes HF
loading and compilation; cache contents affect it. Failed cases retain their
`train.log` and the sweep returns a failure after trying the remaining cases.

## Measured on one trn2.48xlarge

Each layout trained the 27B text model for 13 optimizer steps on all 64
logical cores. The table uses the ten steps after warmup. All four runs
processed 32,768 tokens per step.

| TP | CP | DP shard | Median step (s) | Global tokens/s | Rank-zero peak active HBM (GiB) |
|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 16 | 4.11 | 7,964 | 8.40 |
| 4 | 2 | 8 | 8.25 | 3,965 | 9.63 |
| 4 | 4 | 4 | 14.39 | 2,274 | 8.73 |
| 2 | 2 | 16 | 6.68 | 4,907 | 13.12 |

TP4/CP1 gave the highest throughput at 2,048 tokens. The C4 loader partitions
documents by DP rank, so changing DP also changes which tokens are packed
together.

## Long-context training on the same 64 cores

The long-context sweep uses the same checkpoint, BF16 training, global batch
of 16 sequences and ten measured optimizer steps after three warmup steps.
The training logs on all 64 ranks confirm 262,144, 524,288 and 1,048,576
valid tokens per step at 16K, 32K and 64K for each completed run.
Use `--tp4-cp-within-chip` to keep the two or four CP peers on one chip.
Each TP4 group then spans four rows of the chip torus. With the default TP4
placement, both CP2 and CP4 failed on a 32 MiB all-to-all during the first
16K backward pass. Changing only rank placement let both layouts finish.

```bash
benchmarks/training/run.sh --tp4-cp-within-chip --seq-len 16384
benchmarks/training/run.sh --tp4-cp-within-chip --seq-len 32768
benchmarks/training/run.sh --tp4-cp-within-chip --seq-len 65536 \
  --layout 4,4,4 --layout 4,2,8 --layout 2,2,16
```

| Context (tokens) | TP | CP | DP shard | Median step (s) | Global tokens/s | Rank-zero peak active HBM (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 4 | 1 | 16 | 18.18 | 14,411 | 11.64 |
| 16,384 | 4 | 2 | 8 | 24.18 | 10,843 | 11.58 |
| 16,384 | 4 | 4 | 4 | 29.78 | 8,806 | 9.65 |
| 32,768 | 4 | 2 | 8 | 45.27 | 11,580 | 11.92 |
| 32,768 | 4 | 4 | 4 | 53.07 | 9,880 | 10.92 |
| 65,536 | 4 | 4 | 4 | 107.46 | 9,757 | 12.02 |

TP2/CP2 failed to allocate device memory on step 2 at 16K and on step 1
at 32K. At 32K, TP4/CP1 also failed during the first step. The pinned
Neuron compiler lowered the embedding gradient to a `TopK` operation with
`k=32768` and reported a maximum of 16,384 (`NCC_EVRF014`).
At 64K, CP1 exceeds the separate 49,152-local-token limit of the pinned
GDN backward kernel. The launcher rejects that case before starting workers.

At 64K, TP4/CP2 failed to allocate device memory during the first step.
TP2/CP2 failed while loading a compiled graph with collectives. Neuron
reported a resource allocation failure without naming the resource.

We stopped TP4/CP1 and TP2/CP2 at 32K and TP4/CP2 at 64K after a compiler
rejection or device-memory errors stalled the distributed step.

### 128K capacity check

At 131,072 tokens, TP4/CP16/DP-shard1 completed two optimizer steps on the
same 64 cores. Each CP group spans one row of four Trainium2 chips. Each rank
processes 8,192 tokens per sequence. With a local batch of one, the trainer
accumulates 16 microbatches for 2,097,152 global tokens per step.

```bash
benchmarks/training/run.sh --layout 4,16,1 --seq-len 131072 --warmup 1 --steps 1
```

The single step measured after warmup took 392.86 s (5,338 global tokens/s).
Loss was 2.633 on step 1 and 7.129 on step 2. At the same sequence length
and global batch, TP4/CP4/DP-shard4 reached 32,768 tokens per CP rank and
failed device-memory allocation before its first optimizer update.

### 256K capacity check

At 262,144 tokens, TP4/CP16/DP-shard1 gives each CP rank 16,384 tokens per
sequence. With the same rank placement and batch settings as the 128K run,
all 64 ranks reported device-memory allocation failures on the first
microbatch. Neuron retried a 512 MiB allocation and reported failed OOM
recovery after cache cleanup. We stopped the stalled run before its first
optimizer update.

```bash
benchmarks/training/run.sh --layout 4,16,1 --seq-len 262144 --warmup 1 --steps 1
```

Both capacity checks use the Neuron backend with `fullgraph=True` for each
GDN block and for the feed-forward submodule of each full-attention block.
Ring attention runs outside those regions, and the trainer compiles the loss
separately.
