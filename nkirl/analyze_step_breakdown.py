#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Break an NKIRL run's wall-clock time into rollout, profiler, training and the rest.

    python analyze_step_breakdown.py <RUN_DIR> [--skip-steps 1] [--out <RUN_DIR>/report]

Reads what slurm/sbatch_container.sh and nkirl/run_nkirl.sh leave in RUN_DIR:

* ``metrics.jsonl``: verl's file logger, one line per step (``timing_s/*``, ``agent_loop/*``, lengths).
* ``telemetry/metrics_stamps.txt``: the wall-clock time each metrics line appeared, so each step's
  phases can be placed on the clock (the phases of a step run one after another).
* ``profiler_trace/*.jsonl``: one line per profiler request and per batch (profiler_trace.py).
* ``telemetry/neuron_monitor_<node>.jsonl`` and ``telemetry/host_<node>.txt``.
* ``driver.log``: ``driver_start_epoch`` and ``profiler_version`` lines.

Writes ``<out>.json`` (every number) and ``<out>.md`` (the tables). Steps in ``--skip-steps`` count
as warm-up: they are reported, but left out of the steady-state means.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from typing import Any

# The phases of verl 0.9.1's v1 sync trainer step, in execution order.
PHASES = [
    "gen",
    "reward",
    "old_log_prob",
    "ref",
    "values",
    "adv",
    "update_critic",
    "update_actor",
    "update_weights",
    "save_checkpoint",
    "testing",
    "dump_rollout_generations",
]


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _dist(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "min": min(values),
        "p50": _pct(values, 0.5),
        "p90": _pct(values, 0.9),
        "p99": _pct(values, 0.99),
        "max": max(values),
        "sum": sum(values),
    }


def load_steps(run_dir: str) -> list[dict[str, Any]]:
    steps = []
    path = os.path.join(run_dir, "metrics.jsonl")
    if os.path.exists(path):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    steps.append({"step": rec["step"], **rec["data"]})
    stamps = []
    stamp_path = os.path.join(run_dir, "telemetry", "metrics_stamps.txt")
    if os.path.exists(stamp_path):
        with open(stamp_path) as handle:
            stamps = [float(line.split()[0]) for line in handle if line.strip()]
    for i, step in enumerate(steps):
        step["_t_end"] = stamps[i] if i < len(stamps) else None
    return steps


def phase_windows(step: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Place each phase of one step on the clock, backward from the time its metrics were logged."""
    end = step.get("_t_end")
    if end is None:
        return {}
    total = step.get("timing_s/step", 0.0)
    start = end - total
    windows, t = {}, start
    for phase in PHASES:
        d = step.get(f"timing_s/{phase}")
        if d:
            windows[phase] = (t, t + d)
            t += d
    windows["step"] = (start, end)
    return windows


def load_trace(run_dir: str) -> tuple[list[dict], list[dict], list[dict]]:
    kernels, batches, gens = [], [], []
    for path in glob.glob(os.path.join(run_dir, "profiler_trace", "*.jsonl")):
        with open(path) as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = rec.get("kind")
                (kernels if kind == "kernel" else gens if kind == "generate" else batches).append(rec)
    return kernels, batches, gens


def _error_class(err: str | None) -> str:
    if not err:
        return "ok"
    e = err.lower()
    for tag in (
        "timed out",
        "submit_5",
        "submit_4",
        "connection",
        "compil",
        "mismatch",
        "incorrect",
        "nrt",
        "capacity",
        "thread exception",
        "async exception",
    ):
        if tag in e:
            return tag
    return re.sub(r"[0-9]+", "N", e[:40])


def profiler_summary(kernels: list[dict], batches: list[dict], windows: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_caller: dict[str, list[dict]] = defaultdict(list)
    for rec in kernels:
        by_caller[rec.get("caller", "unknown")].append(rec)
    for caller, recs in sorted(by_caller.items()):
        total = [r["t_end"] - r["t_start"] for r in recs]
        wait = [r["t_slot"] - r["t_start"] for r in recs if r.get("t_slot")]
        service = [r["t_end"] - r["t_slot"] for r in recs if r.get("t_slot")]
        out[caller] = {
            "requests": len(recs),
            "client_total_s": _dist(total),
            "client_queue_s": _dist(wait),
            "service_s": _dist(service),
            "outcome": dict(Counter(_error_class(r.get("error")) for r in recs).most_common(12)),
        }
    if kernels:
        t0 = min(r["t_start"] for r in kernels)
        t1 = max(r["t_end"] for r in kernels)
        events = sorted([(r["t_start"], 1) for r in kernels] + [(r["t_end"], -1) for r in kernels])
        inflight, peak, area, last = 0, 0, 0.0, events[0][0]
        for t, d in events:
            area += inflight * (t - last)
            inflight += d
            peak = max(peak, inflight)
            last = t
        out["overall"] = {
            "requests": len(kernels),
            "span_s": t1 - t0,
            "requests_per_min": 60 * len(kernels) / max(t1 - t0, 1e-9),
            "mean_in_flight": area / max(t1 - t0, 1e-9),
            "peak_in_flight": peak,
        }
    per_step = []
    for w in windows:
        if "step" not in w:
            continue
        s0, s1 = w["step"]
        inside = [r for r in kernels if s0 <= r["t_start"] < s1]
        per_step.append(
            {
                "step": w["_step"],
                "requests": len(inside),
                "by_caller": dict(Counter(r.get("caller") for r in inside)),
                "service_s_sum": sum((r["t_end"] - (r.get("t_slot") or r["t_start"])) for r in inside),
                "in_gen": sum(1 for r in inside if "gen" in w and w["gen"][0] <= r["t_start"] < w["gen"][1]),
            }
        )
    out["per_step"] = per_step
    return out


def _union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def generation_summary(gens: list[dict], windows: list[dict]) -> list[dict[str, Any]]:
    """Per step: trajectory timelines, per-replica busy share, tool-wait bubbles and the straggler tail."""
    out = []
    for w in windows:
        if "gen" not in w:
            continue
        g0, g1 = w["gen"]
        calls = [r for r in gens if r.get("t_end") and g0 - 1 <= r["t_start"] < g1 + 1]
        if not calls:
            continue
        wall = g1 - g0
        trajs: dict[str, list[dict]] = defaultdict(list)
        for r in calls:
            trajs[r["request_id"]].append(r)
        t_gen, t_gap, t_done, n_turns = [], [], [], []
        for turns in trajs.values():
            turns.sort(key=lambda r: r["t_start"])
            busy = sum(r["t_end"] - r["t_start"] for r in turns)
            span = turns[-1]["t_end"] - turns[0]["t_start"]
            t_gen.append(busy)
            t_gap.append(max(span - busy, 0.0))
            t_done.append(turns[-1]["t_end"] - g0)
            n_turns.append(len(turns))
        servers: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in calls:
            servers[str(r.get("server_id"))].append((max(r.get("t_acquired") or r["t_start"], g0), min(r["t_end"], g1)))
        busy_share = [_union(iv) / wall for iv in servers.values()] if wall > 0 else []
        # Time inside gen with no generation request in flight on ANY server: every live trajectory is in
        # a tool call (or between turns), so the whole rollout fleet is idle.
        any_busy = _union([(max(r["t_start"], g0), min(r["t_end"], g1)) for r in calls])
        first_start = min(r["t_start"] for r in calls)
        out.append(
            {
                "step": w["_step"],
                "gen_wall_s": wall,
                "generate_calls": len(calls),
                "trajectories": len(trajs),
                "output_tokens": sum(r.get("output_len") or 0 for r in calls),
                "output_tokens_per_s": sum(r.get("output_len") or 0 for r in calls) / wall if wall else None,
                "call_s": _dist([r["t_end"] - (r.get("t_acquired") or r["t_start"]) for r in calls]),
                "lb_wait_s": _dist([(r.get("t_acquired") or r["t_start"]) - r["t_start"] for r in calls]),
                "output_len": _dist([float(r.get("output_len") or 0) for r in calls]),
                "traj_generate_s": _dist(t_gen),
                "traj_between_turns_s": _dist(t_gap),
                "turns": _dist(n_turns),
                "traj_done_after_gen_start_s": _dist(t_done),
                "straggler_tail_s": wall - (_pct(t_done, 0.9) or 0.0),
                "servers": len(servers),
                "server_busy_share": _dist(busy_share),
                "fleet_idle_s": max(wall - any_busy, 0.0),
                "first_request_after_gen_start_s": first_start - g0,
            }
        )
    return out


def load_monitor(run_dir: str, period_s: float = 10.0) -> dict[str, list[tuple[float, float, float]]]:
    """Per node: (time, mean NeuronCore utilization %, device memory used GB) samples.

    A line is ``<epoch> <json>`` (slurm/sbatch_container.sh stamps each sample) or bare JSON from an
    older run, whose times are then derived backward from the file's mtime at ``period_s`` spacing.
    Every runtime reports the cores it sees; a core is counted once, at its highest reported value.
    """
    series: dict[str, list[tuple[float, float, float]]] = {}
    for path in glob.glob(os.path.join(run_dir, "telemetry", "neuron_monitor_*.jsonl")):
        node = os.path.basename(path)[len("neuron_monitor_") : -len(".jsonl")]
        raw = []
        with open(path) as handle:
            for line in handle:
                ts = None
                head, _, rest = line.partition(" ")
                if rest and head.replace(".", "", 1).isdigit():
                    ts, line = float(head), rest
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cores: dict[str, float] = {}
                mem = 0.0
                for rt in rec.get("neuron_runtime_data") or []:
                    report = rt.get("report", {})
                    for core, c in report.get("neuroncore_counters", {}).get("neuroncores_in_use", {}).items():
                        cores[core] = max(cores.get(core, 0.0), float(c.get("neuroncore_utilization") or 0.0))
                    used = report.get("memory_used", {}).get("neuron_runtime_used_bytes", {})
                    mem += float(used.get("neuron_device", 0) or 0)
                raw.append((ts, statistics.fmean(cores.values()) if cores else 0.0, mem / 1e9))
        if raw and raw[-1][0] is None:
            t_last = os.path.getmtime(path)
            raw = [(t_last - (len(raw) - 1 - i) * period_s, u, m) for i, (_, u, m) in enumerate(raw)]
        series[node] = [r for r in raw if r[0] is not None]
    return series


def monitor_by_phase(series: dict[str, list], windows: list[dict]) -> dict[str, Any]:
    acc: dict[str, list[float]] = defaultdict(list)
    mem_peak = 0.0
    for samples in series.values():
        for ts, util, mem in samples:
            mem_peak = max(mem_peak, mem)
            for w in windows:
                for phase, span in w.items():
                    if phase.startswith("_") or phase == "step":
                        continue
                    a, b = span
                    if a <= ts < b:
                        acc[phase].append(util)
    return {
        "neuroncore_util_pct_by_phase": {p: _dist(v) for p, v in acc.items()},
        "device_mem_used_gb_peak_per_node": mem_peak,
    }


def host_memory(run_dir: str) -> dict[str, Any]:
    out = {}
    for path in glob.glob(os.path.join(run_dir, "telemetry", "host_*.txt")):
        node = os.path.basename(path)[5:-4]
        used = []
        with open(path) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    used.append((float(parts[0]), float(parts[1]) / 1e9))
        if used:
            out[node] = {"first_gb": used[0][1], "last_gb": used[-1][1], "max_gb": max(u for _, u in used)}
    return out


def step_table(steps: list[dict]) -> list[dict]:
    rows = []
    for s in steps:
        total = s.get("timing_s/step", 0.0)
        row = {"step": s["step"], "step_s": total}
        accounted = 0.0
        for phase in PHASES:
            d = s.get(f"timing_s/{phase}")
            if d:
                row[phase] = d
                accounted += d
        row["other"] = max(total - accounted, 0.0)
        for key in (
            "agent_loop/generate_sequences/mean",
            "agent_loop/generate_sequences/max",
            "agent_loop/tool_calls/mean",
            "agent_loop/tool_calls/max",
            "agent_loop/compute_score/mean",
            "agent_loop/compute_score/max",
            "agent_loop/slowest/generate_sequences",
            "agent_loop/slowest/tool_calls",
            "agent_loop/slowest/compute_score",
            "agent_loop/slowest/response_length",
            "response_length/mean",
            "response_length/max",
            "prompt_length/mean",
            "num_turns/mean",
            "num_turns/max",
            "critic/score/mean",
            "global_seqlen/mean",
            "perf/throughput",
            "perf/total_num_tokens",
        ):
            for prefix in ("", "timing_s/"):
                if prefix + key in s:
                    row[key] = s[prefix + key]
                    break
        rows.append(row)
    return rows


def _fmt(v: Any, nd: int = 1) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def write_markdown(report: dict[str, Any], path: str) -> None:
    lines = [f"# NKIRL step-time breakdown: {report['run_dir']}", ""]
    rows = report["steps"]
    if rows:
        cols = ["step", "step_s"] + [p for p in PHASES if any(p in r for r in rows)] + ["other"]
        lines += ["## Per-step phase time (s)", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            lines.append("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |")
        ss = report["steady_state"]
        lines += [
            "",
            f"Steady state (steps {ss['steps']}): mean step {_fmt(ss.get('step_s'))} s",
            "",
            "| phase | mean s | share of step |",
            "|---|---|---|",
        ]
        for p, v in ss["phases"].items():
            lines.append(f"| {p} | {_fmt(v['mean_s'])} | {_fmt(100 * v['share'])}% |")
        lines += [
            "",
            "## Inside generation (per trajectory, s)",
            "",
            "| step | gen wall | traj gen mean | traj gen max | tool mean | tool max | score mean | "
            "score max | slowest traj gen / tool / score | resp len mean / max | turns mean |",
            "|" + "---|" * 11,
        ]
        for r in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _fmt(r["step"]),
                        _fmt(r.get("gen")),
                        _fmt(r.get("agent_loop/generate_sequences/mean")),
                        _fmt(r.get("agent_loop/generate_sequences/max")),
                        _fmt(r.get("agent_loop/tool_calls/mean")),
                        _fmt(r.get("agent_loop/tool_calls/max")),
                        _fmt(r.get("agent_loop/compute_score/mean")),
                        _fmt(r.get("agent_loop/compute_score/max")),
                        " / ".join(
                            _fmt(r.get(f"agent_loop/slowest/{key}"))
                            for key in ("generate_sequences", "tool_calls", "compute_score")
                        ),
                        f"{_fmt(r.get('response_length/mean'), 0)} / {_fmt(r.get('response_length/max'), 0)}",
                        _fmt(r.get("num_turns/mean"), 2),
                    ]
                )
                + " |"
            )
    gen = report.get("generation") or []
    if gen:
        lines += [
            "",
            "## Rollout timeline and bubbles (from per-turn generate calls)",
            "",
            "| step | gen wall s | trajectories | turns mean | output tok/s | traj generate mean s | "
            "traj between-turn (tool) mean s | traj done p50 / p90 / max s | straggler tail s | "
            "replica busy share mean / min | fleet idle s |",
            "|" + "---|" * 11,
        ]
        for g in gen:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(g["step"]),
                        _fmt(g["gen_wall_s"]),
                        str(g["trajectories"]),
                        _fmt(g["turns"].get("mean"), 2),
                        _fmt(g["output_tokens_per_s"]),
                        _fmt(g["traj_generate_s"].get("mean")),
                        _fmt(g["traj_between_turns_s"].get("mean")),
                        " / ".join(_fmt(g["traj_done_after_gen_start_s"].get(q)) for q in ("p50", "p90", "max")),
                        _fmt(g["straggler_tail_s"]),
                        f"{_fmt(100 * (g['server_busy_share'].get('mean') or 0))}% / "
                        f"{_fmt(100 * (g['server_busy_share'].get('min') or 0))}%",
                        _fmt(g["fleet_idle_s"]),
                    ]
                )
                + " |"
            )
        lines += [
            "",
            "Straggler tail: gen wall time after 90% of trajectories finished. Fleet idle: time inside "
            "gen with no generate call in flight on any replica (every live trajectory is waiting on a tool).",
        ]
    prof = report["profiler"]
    if prof:
        lines += [
            "",
            "## Profiler requests",
            "",
            "| caller | requests | client total p50 / p90 / max s | "
            "client queue p50 / p90 s | service p50 / p90 / max s | outcomes |",
            "|---|---|---|---|---|---|",
        ]
        for caller, v in prof.items():
            if caller in ("overall", "per_step"):
                continue
            t, q, s = v["client_total_s"], v["client_queue_s"], v["service_s"]
            lines.append(
                f"| {caller} | {v['requests']} | {_fmt(t.get('p50'))} / {_fmt(t.get('p90'))} / "
                f"{_fmt(t.get('max'))} | {_fmt(q.get('p50'))} / {_fmt(q.get('p90'))} | "
                f"{_fmt(s.get('p50'))} / {_fmt(s.get('p90'))} / {_fmt(s.get('max'))} | {v['outcome']} |"
            )
        if "overall" in prof:
            o = prof["overall"]
            lines += [
                "",
                f"Overall: {o['requests']} requests over {_fmt(o['span_s'])} s, "
                f"{_fmt(o['requests_per_min'])} per minute, mean {_fmt(o['mean_in_flight'])} in flight, "
                f"peak {o['peak_in_flight']}.",
            ]
        if prof.get("per_step"):
            lines += [
                "",
                "| step | requests | by caller | service time sum s | started during gen |",
                "|---|---|---|---|---|",
            ]
            for p in prof["per_step"]:
                lines.append(
                    f"| {p['step']} | {p['requests']} | {p['by_caller']} | {_fmt(p['service_s_sum'])} | {p['in_gen']} |"
                )
    mon = report.get("monitor") or {}
    if mon.get("neuroncore_util_pct_by_phase"):
        lines += [
            "",
            "## NeuronCore utilization by phase (neuron-monitor, all nodes)",
            "",
            "| phase | samples | mean % | p50 % | p90 % |",
            "|---|---|---|---|---|",
        ]
        for p, d in mon["neuroncore_util_pct_by_phase"].items():
            lines.append(
                f"| {p} | {d.get('n')} | {_fmt(d.get('mean'))} | {_fmt(d.get('p50'))} | {_fmt(d.get('p90'))} |"
            )
        lines.append(f"\nPeak device memory used on one node: {_fmt(mon.get('device_mem_used_gb_peak_per_node'))} GB")
    if report.get("host_memory"):
        lines += ["", "## Host memory (GB used)", "", "| node | first | last | max |", "|---|---|---|---|"]
        for n, v in sorted(report["host_memory"].items()):
            lines.append(f"| {n} | {_fmt(v['first_gb'])} | {_fmt(v['last_gb'])} | {_fmt(v['max_gb'])} |")
    if report.get("startup"):
        lines += ["", "## Startup", "", json.dumps(report["startup"], indent=1)]
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--skip-steps", type=int, default=1, help="warm-up steps left out of the means")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = os.path.abspath(args.run_dir)
    out = args.out or os.path.join(run_dir, "report")

    steps = load_steps(run_dir)
    windows = []
    for s in steps:
        w = phase_windows(s)
        w["_step"] = s["step"]
        windows.append(w)
    rows = step_table(steps)
    steady = [r for r in rows if r["step"] > args.skip_steps] or rows
    ss: dict[str, Any] = {"steps": [r["step"] for r in steady]}
    if steady:
        mean_step = statistics.fmean(r["step_s"] for r in steady)
        ss["step_s"] = mean_step
        ss["phases"] = {}
        for p in PHASES + ["other"]:
            vals = [r.get(p, 0.0) for r in steady]
            if any(vals):
                m = statistics.fmean(vals)
                ss["phases"][p] = {"mean_s": m, "share": m / mean_step if mean_step else 0.0}

    kernels, batches, gens = load_trace(run_dir)
    startup: dict[str, Any] = {}
    log = os.path.join(run_dir, "driver.log")
    if os.path.exists(log):
        with open(log, errors="replace") as handle:
            for line in handle:
                if line.startswith("driver_start_epoch"):
                    startup["driver_start_epoch"] = float(line.split()[1])
                elif line.startswith("profiler_version"):
                    startup.setdefault("profiler_version", []).append(line.strip())
    if steps and steps[0].get("_t_end") and "driver_start_epoch" in startup:
        first_step_start = steps[0]["_t_end"] - steps[0].get("timing_s/step", 0.0)
        startup["driver_to_first_step_s"] = first_step_start - startup["driver_start_epoch"]
    report = {
        "run_dir": run_dir,
        "steps": rows,
        "steady_state": ss,
        "profiler": profiler_summary(kernels, batches, windows) if kernels else {},
        "generation": generation_summary(gens, windows),
        "monitor": monitor_by_phase(load_monitor(run_dir), windows),
        "host_memory": host_memory(run_dir),
        "startup": startup,
    }
    with open(out + ".json", "w") as handle:
        json.dump(report, handle, indent=1, default=str)
    write_markdown(report, out + ".md")
    print(f"wrote {out}.json and {out}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
