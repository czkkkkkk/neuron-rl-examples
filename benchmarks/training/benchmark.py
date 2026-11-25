"""Run the Qwen3.8 text trainer inside the benchmark container."""

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

DEFAULT_LAYOUTS = [(4, 1, 16), (4, 2, 8), (4, 4, 4), (2, 2, 16)]


def layout(value):
    try:
        tp, cp, dp = map(int, value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected TP,CP,DP_SHARD") from error
    if min(tp, cp, dp) < 1 or 4 % tp:
        raise argparse.ArgumentTypeError(
            "degrees must be positive and TP must divide the four KV heads"
        )
    if tp * cp * dp != 64:
        raise argparse.ArgumentTypeError(
            "TP * CP * DP_SHARD must equal 64 logical NeuronCores"
        )
    return tp, cp, dp


def metrics(run, warmup, steps, tokens, start):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    events = next(run.rglob("events.out.tfevents.*"))
    data = EventAccumulator(str(events.parent), size_guidance={"scalars": 0}).Reload()

    def series(name):
        return {x.step: x for x in data.Scalars(name)}

    times = series("time_metrics/end_to_end(s)")
    measured = range(warmup + 1, warmup + steps + 1)
    durations = [times[i].value for i in measured]
    if not all(math.isfinite(x) and x > 0 for x in durations):
        raise ValueError("invalid measured step duration")
    last = warmup + steps
    values = {
        "startup_through_first_step_s": times[1].wall_time - start,
        "median_step_s": statistics.median(durations),
        "global_tokens_per_s": steps * tokens / sum(durations),
        "loss": series("loss_metrics/global_avg_loss")[last].value,
        "grad_norm": series("grad_norm")[last].value,
    }
    if not all(math.isfinite(values[name]) for name in ("loss", "grad_norm")):
        raise ValueError("nonfinite training loss or gradient norm")
    if "memory/max_active(GiB)" in data.Tags()["scalars"]:
        hbm = series("memory/max_active(GiB)")
        values["rank0_peak_active_GiB"] = max(hbm[i].value for i in measured) or None
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--layout", type=layout, action="append", help="TP,CP,DP_SHARD (repeatable)"
    )
    p.add_argument(
        "--tp4-cp-within-chip",
        action="store_true",
        help="place each TP4 layout's CP2 or CP4 group on a single Trainium2 chip",
    )
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--global-batch", type=int, default=16)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=10, help="measured optimizer steps")
    p.add_argument(
        "--capacity", action="store_true", help="also run CP2 at 64K and CP4 at 128K"
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if min(args.seq_len, args.global_batch, args.steps) < 1 or args.warmup < 0:
        p.error(
            "sequence, batch and measured steps must be positive; warmup must be nonnegative"
        )
    cases = [
        (tp, cp, dp, args.seq_len) for tp, cp, dp in args.layout or DEFAULT_LAYOUTS
    ]
    if args.capacity:
        cases.extend([(4, 2, 8, 65536), (4, 4, 4, 131072)])
    specs = []
    for tp, cp, dp, length in dict.fromkeys(cases):
        if length % (2 * cp if cp > 1 else 1) or args.global_batch % dp:
            p.error(
                f"invalid local sequence or global batch for {(tp, cp, dp, length)}"
            )
        if length // cp > 49152:
            p.error(
                "GDN backward supports at most 49152 local tokens per CP rank. "
                f"sequence {length} at CP{cp} gives {length // cp}"
            )
        case = dict(
            tp=tp,
            cp=cp,
            dp_shard=dp,
            world_size=tp * cp * dp,
            seq_len=length,
            global_batch=args.global_batch,
            accumulation=args.global_batch // dp,
            measured_steps=args.steps,
            warmup_steps=args.warmup,
        )
        if tp == 2:
            # Each FSDP group occupies two complete rows of the chip torus.
            case["neuron_core_by_local_rank"] = [
                c for i in range(32) for c in (i, 32 + (i + 16) % 32)
            ]
        elif tp == 4 and cp > 1 and (args.tp4_cp_within_chip or cp == 16):
            if args.tp4_cp_within_chip and cp not in (2, 4):
                p.error("TP4 CP groups must have two or four ranks to fit on one chip")
            # Each FSDP group occupies one row. CP2/4 fit on a chip, CP16 uses the row.
            case["neuron_core_by_local_rank"] = [
                16 * t + i for i in range(16) for t in range(4)
            ]
        specs.append(case)
    if args.dry_run:
        print(json.dumps(specs, indent=2))
        return 0

    import torchtitan
    from nkilib.experimental.gdn import gdn_cte_bwd
    from torchtitan_neuron.models.qwen3_8.context_parallel import StateParallelGDN

    root = Path("/opt/verl/sources/torchtitan-neuron")
    assert Path(torchtitan.__file__).resolve().is_relative_to(root)
    assert gdn_cte_bwd and StateParallelGDN
    results = []
    for case in specs:
        run = Path("/runs") / "tp{tp}-cp{cp}-dp{dp_shard}-s{seq_len}".format(**case)
        run.mkdir()
        command = [
            "./run_train.sh",
            "--dump_folder",
            str(run),
            "--hf_assets_path",
            "/data/Qwen3.8-27B",
            "--training.steps",
            str(args.warmup + args.steps),
            "--training.seq_len",
            str(case["seq_len"]),
            "--training.global_batch_size",
            str(args.global_batch),
            "--training.dtype",
            "bfloat16",
            "--parallelism.tensor_parallel_degree",
            str(case["tp"]),
            "--parallelism.context_parallel_degree",
            str(case["cp"]),
            "--parallelism.data_parallel_shard_degree",
            str(case["dp_shard"]),
            "--dataloader.dataset",
            "c4_test",
            "--checkpoint.enable",
            "--checkpoint.initial_load_in_hf",
            "--checkpoint.load_only",
            "--metrics.enable_tensorboard",
            "--metrics.log_freq",
            "1",
            "--debug.seed",
            "42",
        ]
        env = {
            **os.environ,
            "NGPU": str(case["world_size"]),
            "MODULE": "torchtitan_neuron.models.qwen3_8",
            "CONFIG": "qwen38_27b",
        }
        if (cores := case.get("neuron_core_by_local_rank")) is not None:
            env["QWEN38_NEURON_CORE_ORDER"] = ",".join(map(str, cores))
            env["PYTHON_EXEC"] = "/benchmark/neuron_worker.sh"
        case["command"] = shlex.join(command)
        (run / "config.json").write_text(json.dumps(case, indent=2))
        print(f"Starting {run.name}", flush=True)
        start = time.time()
        with (run / "train.log").open("w") as log:
            code = subprocess.run(
                command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        row = {**case, "exit_code": code, "wall_s": time.time() - start}
        if code == 0:
            try:
                row.update(
                    metrics(
                        run,
                        args.warmup,
                        args.steps,
                        case["seq_len"] * args.global_batch,
                        start,
                    )
                )
            except (StopIteration, KeyError, ValueError) as exc:
                row["metrics_error"] = repr(exc)
        results.append(row)
        (Path("/runs") / "results.json").write_text(
            json.dumps(results, indent=2) + "\n"
        )
        print(json.dumps(row), flush=True)
    return int(any(row["exit_code"] or "metrics_error" in row for row in results))


if __name__ == "__main__":
    sys.exit(main())
