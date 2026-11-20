# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""NKIRL's reward for verl 0.9.1: extract the graded kernel, re-profile it, judge it, score it.

Wiring (verl 0.9.1, v1 trainer):

    reward.custom_reward_function.path=/workspace/nkirl/nkirl_reward.py
    reward.custom_reward_function.name=compute_score

NOT the top-level ``custom_reward_function`` key, which the reward loop only migrates for backward
compatibility.

The whole pipeline — kernel extraction (including ``best_speedup``'s "fastest profiler-verified
kernel in the trajectory" tier chain), the copy gate, the LLM judge, milestone/speedup shaping, the
best-kernel artifact dump and the infra-abort breaker — already lives in
``nki_reward_profiler_local.compute_score_batch`` and is verl-shaped: its signature is verl's batch
reward signature and its extractor parses verl's own tool-call tags. So this file is a thin adapter
and deliberately contains no scoring logic of its own.

**What gets graded.** verl hands us ``solution_str`` = the decoded response, which with
``multi_turn.format=qwen3_coder`` already contains the ``<parameter=nki_code>`` / ``<tool_response>``
pairs the extractor wants. That is why no shadow transcript is needed here, unlike NKIRL's slime
port, which had to synthesise one because slime's native tags were unparseable.

**One sample at a time.** verl's reward loop calls per sample, concurrently, so
``compute_score_batch`` is called with a one-element batch — the same thing NKIRL's slime adapter
does. Profiler batching is lost, but the pool provides the concurrency instead.

**Profiling is a remote service.** ``compute_score_batch`` reaches it through NKIRL's own
``send_batch_profiler_request`` — HTTP to the endpoint in ``PROFILER_REMOTE_URL``. Nothing here
profiles locally, so the trainer keeps every NeuronCore on the node. The service's toolchain must
match this box's (see ``nkirl_tool.py``): the on-device time it reports is toolchain-bound, and the
dataset's baselines are only valid for the path that measured them.

**Two knobs are env vars, not arguments.** ``KERNEL_EXTRACT_STRATEGY`` (which kernel is graded) and
``TEST_CONFIG_MODE`` (which input shape it is measured on) are read from the environment by
``compute_score_batch``, because the tool must make the *same* two decisions and only an env var
reaches both. Passing them as kwargs would land them in ``**kwargs`` and be silently ignored — a bug
this file's slime counterpart used to have.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

__all__ = ["compute_score"]

from nkirl import nki_reward_profiler_local as _reward_core  # noqa: E402
from nkirl.nki_reward_profiler_local import compute_score_batch  # noqa: E402

from nkirl import profiler_trace  # noqa: E402

# NKIRL_PROFILER_TRACE_DIR set: the reward's re-profiles are timed (profiler_trace.py). Unset: no change.
profiler_trace.install(_reward_core)

#: Fields the reward returns that verl aggregates into run metrics. Scalars only: verl builds a
#: numpy array per key, so a dict or a list here would break the aggregation for the whole batch.
_SCALAR_FIELDS = (
    "speedup",
    "absolute_speedup",
    "is_correct",
    "is_copied",
    "is_non_nki",
    "is_infra_failure",
    "is_judge_failed",
    "is_judge_duplication",
    "graded_incomplete_kernel",
    "profiler_roundtrip_s",
    "profiler_poll_wait_s",
    "profiler_queue_s",
)


#: When set, every graded rollout is written here as one JSON line: the decoded response the policy
#: produced, plus the verdict it earned. Off by default -- it writes the full response text, which is
#: megabytes per step -- but it is the only place the generations are recoverable: verl dumps
#: generations for *validation* only, and the reward's own artifact dump keeps just the kernels that
#: scored, so a run where everything scores 0 leaves no trace of what the model actually wrote.
_DUMP_DIR = os.environ.get("NKIRL_RESPONSE_DUMP_DIR", "").strip()


def _dump_response(data_source: str, solution_str: str, extra_info: dict[str, Any], result: dict[str, Any]) -> None:
    """Append one graded rollout to the dump. Never raises: this is diagnostics, not the reward."""
    if not _DUMP_DIR:
        return
    try:
        os.makedirs(_DUMP_DIR, exist_ok=True)
        # One file per process: the reward loop runs many workers concurrently and appending from
        # all of them to one file interleaves partial lines.
        path = os.path.join(_DUMP_DIR, f"responses_pid{os.getpid()}.jsonl")
        record = {
            "data_source": data_source,
            "kernel_name": extra_info.get("kernel_name") or extra_info.get("example_name"),
            "num_turns": extra_info.get("num_turns"),
            "score": result.get("score"),
            "reason": result.get("reason"),
            "is_correct": result.get("is_correct"),
            "speedup": result.get("speedup"),
            # What the reward actually graded, verbatim.
            "response": solution_str,
            # Whether the tool-call tags the reward looks for are even present -- the first thing to
            # check when everything scores zero.
            "has_tool_call": "<parameter=nki_code>" in (solution_str or ""),
            "has_tool_response": "<tool_response>" in (solution_str or ""),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001
        logger.warning("[nkirl_reward] could not write the response dump", exc_info=True)


#: Tool-call grammar of the rollout (rollout.multi_turn.format). NKIRL's extractor reads only the
#: Qwen3-Coder XML form (``<parameter=nki_code>``). With ``hermes`` (general Qwen3, e.g. Qwen3-32B) every
#: call is ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``, which the extractor cannot see:
#: ``best_speedup`` would silently fall back to "the last code block". The calls are rewritten to the
#: XML form first, so the core grades exactly what the tool profiled.
_TOOL_FORMAT = os.environ.get("NKIRL_TOOL_FORMAT", "qwen3_coder").strip().lower()
_HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def hermes_to_qwen3_coder(text: str) -> str:
    """Rewrite Hermes JSON ``profile_kernel`` calls in ``text`` to the Qwen3-Coder XML grammar.

    A block that is not valid JSON, names another tool, or has no ``nki_code`` is left unchanged, the
    same calls verl's own hermes parser would not execute.
    """

    def _one(match: re.Match) -> str:
        try:
            call = json.loads(match.group(1))
        except (json.JSONDecodeError, ValueError):
            return match.group(0)
        args = call.get("arguments") if isinstance(call, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                return match.group(0)
        if not isinstance(args, dict) or call.get("name") != "profile_kernel" or "nki_code" not in args:
            return match.group(0)
        return (
            "<tool_call>\n<function=profile_kernel>\n<parameter=nki_code>\n"
            f"{args['nki_code']}\n</parameter>\n</function>\n</tool_call>"
        )

    return _HERMES_CALL.sub(_one, text)


def _reward_kwargs() -> dict[str, Any]:
    return dict(
        use_judge=os.environ.get("NKIRL_USE_JUDGE", "0") == "1",
        judge_concurrency=int(os.environ.get("NKIRL_JUDGE_CONCURRENCY", "16")),
        judge_reward_on_invalid=float(os.environ.get("NKIRL_JUDGE_INVALID", "-0.5")),
        reward_mode=os.environ.get("NKIRL_REWARD_MODE", "speedup"),
        speedup_reward_max=float(os.environ.get("NKIRL_SPEEDUP_MAX", "2.0")),
    )


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Score one rollout. Returns ``{"score": float, ...}``; the extra keys become run metrics.

    ``extra_info`` is the dataset row's own ``extra_info`` with two additions verl makes: the agent
    loop's ``num_turns``, and ``tool_extra_fields`` (our tool's per-call counters) merged in.
    """
    extra_info = dict(extra_info or {})
    if _TOOL_FORMAT == "hermes" and solution_str:
        solution_str = hermes_to_qwen3_coder(solution_str)

    try:
        results = compute_score_batch(
            # Passed through, not defaulted. The core only ever records `data_source` as metadata
            # (artifact JSON, infra-failure rows) and groups Pass@K by it, where an empty value is
            # deliberately skipped — so substituting a placeholder would invent a provenance label
            # and a metrics bucket matching none of the dataset's real data sources.
            data_sources=[data_source],
            solution_strs=[solution_str or ""],
            ground_truths=[ground_truth],
            extra_infos=[extra_info],
            **_reward_kwargs(),
        )
    except Exception:  # noqa: BLE001
        # A reward that raises kills the step. Score it as an infra failure instead: with
        # NKIRL_DROP_INFRA_FAILURES the sample contributes nothing rather than teaching a wrong
        # lesson, and the infra-abort breaker sees it.
        logger.exception("[nkirl_reward] compute_score_batch raised")
        return {"score": 0.0, "acc": False, "is_infra_failure": True, "reason_code": -1}

    result = results[0] if results else {}
    _dump_response(data_source, solution_str, extra_info, result)
    score = float(result.get("score", 0.0) or 0.0)

    out: dict[str, Any] = {
        "score": score,
        # verl logs `acc` by convention; correctness here means the profiler verified the kernel AND
        # the judge did not reject it. The reward leaves is_correct=True on a JUDGE_REJECTED
        # rollout (it only rewrites score/reason), so is_correct alone over-counts.
        "acc": bool(result.get("is_correct")) and result.get("reason") != "JUDGE_REJECTED",
    }
    for field in _SCALAR_FIELDS:
        if field in result:
            value = result[field]
            if isinstance(value, bool):
                out[field] = value
            elif isinstance(value, (int, float)):
                out[field] = float(value)
    # `reason` is the most useful per-sample diagnostic but is a string, so it is surfaced as a
    # string (verl keeps non-tensor batch keys) while the aggregate-friendly view is the booleans
    # above.
    if result.get("reason") is not None:
        out["reason"] = str(result["reason"])
    if result.get("num_turns") is None and extra_info.get("num_turns") is not None:
        out["num_turns"] = float(extra_info["num_turns"])
    return out
