# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""The ``profile_kernel`` tool for verl's ``ToolAgentLoop``: the profiler the policy calls mid-rollout.

The model writes a NKI kernel, calls this tool, reads the measured latency and profiler metrics, and
revises. verl owns the loop — stop tokens, turn caps, loss-masking the tool output, rendering the
tool schema into the prompt — so all that lives here is the tool body.

THE PARAMETER NAME IS LOAD-BEARING. It must be ``nki_code``, because with
``rollout.multi_turn.format=qwen3_coder`` verl renders a call as

    <tool_call><function=profile_kernel><parameter=nki_code>…</parameter></function></tool_call>

and NKIRL's reward re-reads the decoded response with
``_TOOL_CALL_KERNEL = <parameter=nki_code>(.*?)</parameter>``. Rename the parameter, or switch
``format`` to ``hermes`` (which renders arguments as JSON), and the reward finds zero verified
kernels — ``KERNEL_EXTRACT_STRATEGY=best_speedup`` then silently collapses to "the last block",
which is the exact failure NKIRL's slime port hit and documented at length.

THE OBSERVATION TEXT IS ALSO A CONTRACT, for the same reason. ``_tool_call_verdicts`` pairs each
``<parameter=nki_code>`` with the following ``<tool_response>`` and counts the call as
profiler-verified only if the response **starts with** ``"Kernel is correct"``; it reads the speedup
out of a ``speedup: <n>x`` substring. So those two strings are reproduced exactly rather than
paraphrased, and :data:`_TOOL_OK_PREFIX` is asserted against the reward core at import.

WHAT THE MODEL SEES AND THE REWARD GRADES MUST BE THE SAME NUMBER. The input shape comes from the
reward's own picker (``select_test_config_for_rollout``, honouring ``TEST_CONFIG_MODE``) and the seed
baseline from the dataset field or the shared cache — not from a local choice. NKIRL attributes a
broken run to getting this wrong.

PROFILING IS A REMOTE SERVICE. Every call here goes through NKIRL's own
``send_batch_profiler_request``, i.e. HTTP to the AWSNeuronGenAIProfiler named by
``PROFILER_REMOTE_URL``. That keeps every NeuronCore on this node available to the trainer, and it
is the only transport — there is no local profiler.

Two consequences worth knowing. The endpoint must be selected with ``PROFILER_REMOTE_URL`` and not
the ``PROFILING_SERVER_URL`` + ``PROFILER_BASE_PORT`` tunnel knobs: only the former folds
``get_inputs`` into ``src_code`` and switches the request to beta4, and without that every request
fails ``InputSpecError``. And the service's toolchain must match this box's ``nki`` / ``neuronx-cc``,
because the measured on-device time is toolchain-bound — the same seed kernels profile ~1.9x apart
between two compiler vintages, so a mismatch silently invalidates every baseline in the dataset.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

__all__ = ["ProfileKernelTool"]

from nkirl import nki_reward_profiler_local as _reward_core  # noqa: E402
from nkirl.nki_reward_profiler_local import (  # noqa: E402
    _TOOL_OK_PREFIX,
    _baseline_cache_get,
    _baseline_cache_put,
    extract_kernel_code_from_response,
    format_metrics_block,
    replace_kernel_in_file,
    select_test_config_for_rollout,
)

from nkirl import profiler_trace  # noqa: E402

# NKIRL_PROFILER_TRACE_DIR set: every profiler request is timed (profiler_trace.py). Unset: no change.
send_batch_profiler_request = profiler_trace.tool_sender(_reward_core, "tool")
# This module is imported in the agent-loop workers, which are the processes that call the rollout servers.
profiler_trace.install_generation_trace()
_send_baseline_request = profiler_trace.with_caller(
    getattr(_reward_core, "_nkirl_send_original", _reward_core.send_batch_profiler_request), "tool_baseline"
)

assert _TOOL_OK_PREFIX == "Kernel is correct", (
    f"the reward's verified-call prefix changed to {_TOOL_OK_PREFIX!r}; the observation text this "
    "tool emits must start with it or every tool call is read as unverified"
)


class ProfileKernelTool(BaseTool):
    """``profile_kernel``: compile, verify and benchmark the model's kernel on a NeuronCore.

    Constructed by ``verl.tools.tool_registry.initialize_tools_from_config`` as
    ``cls(config=<dict>, tool_schema=<OpenAIFunctionToolSchema>)``; the per-sample profiling context
    arrives through ``tools_kwargs["profile_kernel"]["create_kwargs"]``, which ``prepare_data.py``
    fills from the dataset row.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}
        self.num_workers = int(config.get("num_workers", 32))
        self.include_metrics = bool(config.get("include_metrics", True))
        self.max_response_chars = int(config.get("max_response_chars", 6000))
        # Built lazily: a Semaphore created here could bind to a loop other than the one the agent
        # loop runs on (the tool is constructed synchronously, before any loop is current).
        self._semaphore: Optional[asyncio.Semaphore] = None
        logger.info("Init ProfileKernelTool with config: %s", config)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.num_workers)
        return self._semaphore

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs):
        """Stash this row's profiling context.

        TWO CALLING CONVENTIONS, and getting this wrong fails *silently*. verl 0.9.1's
        ``ToolAgentLoop._call_tool`` passes the dict as ONE keyword argument::

            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))

        so this receives ``{"create_kwargs": {"extra_info": {...}}}`` — *not* the spread
        ``create(extra_info=...)`` that older verl used and that the docs (and this repo's first CPU
        tests) assumed. Unwrapped here rather than in ``prepare_data.py`` because the parquet shape is
        the thing the reward also reads, and the two must not drift.

        Why it matters that this is silent: without the unwrap, ``context`` became
        ``{"create_kwargs": ...}``, ``seed_kernel`` was absent, and every call returned "Kernel
        compiles, but the profiling context is missing" — a *valid-looking* observation. The tool
        fired, the model got a tool_response, `has_tool_call` was true, and no profiling happened at
        all. See nkirl/README.md, "Contracts that fail silently".
        """
        if instance_id is None:
            instance_id = str(uuid4())
        # verl 0.9.1 nests; older verl spreads. Accept both, nested first.
        nested = kwargs.get("create_kwargs")
        if isinstance(nested, dict):
            kwargs = {**{k: v for k, v in kwargs.items() if k != "create_kwargs"}, **nested}
        # Everything the profiler needs for this row. `extra_info` is accepted whole because that is
        # the shape the reward reads, so the tool and the reward cannot drift.
        context = kwargs.get("extra_info") or {k: v for k, v in kwargs.items() if k != "ground_truth"}
        if not (context.get("seed_kernel") or context.get("initial_code")):
            logger.warning(
                "profile_kernel.create got no profiling context (keys=%s); every call will report "
                "'the profiling context is missing' and nothing will be profiled",
                sorted(context)[:8],
            )
        self._instances[instance_id] = {"context": context, "calls": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Profile the submitted kernel. Never raises: a failure comes back as text the model can read.

        An exception here would abort the whole trajectory, throwing away every turn the policy had
        already spent — so an infrastructure fault is reported as an observation instead, and the
        reward's own grading pass decides what the rollout is worth.
        """
        instance = self._instances.get(instance_id)
        if instance is None:
            return ToolResponse(text="Internal error: this tool call has no session."), 0.0, {}

        nki_code = parameters.get("nki_code") if isinstance(parameters, dict) else parameters
        if not isinstance(nki_code, str):
            nki_code = "" if nki_code is None else str(nki_code)

        try:
            async with self._get_semaphore():
                # _profile is synchronous and blocks for seconds to minutes. One event loop drives
                # every concurrent rollout in this worker, so calling it directly would stall all of
                # them until the profile returned -- NKIRL records exactly this as a multi-x
                # step-time regression. Hand it to a thread instead.
                verdict = await asyncio.to_thread(self._profile, nki_code, instance["context"])
        except asyncio.CancelledError:
            raise  # an aborted rollout must stay cancelled, not become a tool result
        except Exception as exc:  # noqa: BLE001 - see the docstring
            logger.exception("[nkirl_tool] profile_kernel failed")
            verdict = {
                "text": f"Profiler error: {type(exc).__name__}: {exc}",
                "ok": False,
                "compiled": False,
                "speedup": 0.0,
                "recorded": False,
                "kernel": "",
            }

        if verdict["recorded"]:
            instance["calls"].append({"ok": bool(verdict["ok"]), "speedup": float(verdict["speedup"])})

        text = verdict["text"]
        if len(text) > self.max_response_chars:
            text = text[: self.max_response_chars] + "\n... (truncated)"

        # Reaches the reward as `tool_extra_fields`, merged into extra_info. Scalars only: verl
        # aggregates these into run metrics and a nested value would break that.
        calls = instance["calls"]
        metrics = {
            "nkirl_num_tool_calls": len(calls),
            "nkirl_num_tool_calls_ok": sum(1 for c in calls if c["ok"]),
            "nkirl_best_tool_speedup": max((c["speedup"] for c in calls), default=0.0),
            "nkirl_turn0_ok": bool(calls[0]["ok"]) if calls else False,
            "nkirl_turn0_speedup": float(calls[0]["speedup"]) if calls else 0.0,
        }
        return ToolResponse(text=text), 0.0, metrics

    # ------------------------------------------------------------------ the profiling itself
    def _profile(self, nki_code: str, context: dict[str, Any]) -> dict[str, Any]:
        """Blocking: substitute, profile, and render the observation. Ported from NKIRL's own tool."""
        out = {"text": "", "kernel": "", "ok": False, "compiled": False, "speedup": 0.0, "recorded": False}

        kernel_code = extract_kernel_code_from_response(nki_code) or nki_code
        if not kernel_code.strip():
            out["text"] = "No kernel code received. Pass your full kernel source as `nki_code`."
            return out
        out["kernel"] = kernel_code

        # The builtin `compile()`: a parse, not a run. Free, and it keeps an unparseable kernel from
        # spending a slot on the remote profiler (the step's bottleneck) to fail there anyway -- with
        # a neuronx-cc dump instead of the offending line number.
        try:
            compile(kernel_code, "<kernel>", "exec")
        except SyntaxError as exc:
            # A real kernel verdict: recorded, so it counts in the call metrics.
            out["recorded"] = True
            out["text"] = f"SyntaxError in your kernel at line {exc.lineno}: {exc.msg}"
            return out
        out["compiled"] = True  # "parsed", not "neuronx-cc accepted it"

        seed_kernel = context.get("seed_kernel") or context.get("initial_code")
        initial_code = context.get("initial_code")
        if not seed_kernel or not initial_code:
            # Config/infra fault: says nothing about the kernel, so it is NOT recorded.
            out["text"] = "Kernel compiles, but the profiling context is missing (no seed/initial code)."
            return out

        inputs_kwargs = self._inputs_kwargs(context)
        results = send_batch_profiler_request(
            [
                {
                    "src_code": replace_kernel_in_file(seed_kernel, kernel_code),
                    "initial_code": initial_code,
                    "benchmark_code": context.get("benchmark_code"),
                    "kernel_name": "kernel",
                    "get_inputs_kwargs": inputs_kwargs,
                }
            ]
        )
        if not results:
            out["text"] = "Profiler returned no results - try again."  # infra, not recorded
            return out

        result = results[0]
        if result.get("error"):
            out["recorded"] = True  # a kernel verdict: failed to compile, or wrong output
            out["text"] = (
                "Profiler error (your kernel failed to compile or produced the wrong output):\n"
                f"{str(result['error'])[:1500]}"
            )
            return out

        out["ok"] = True
        out["recorded"] = True
        metrics_block = format_metrics_block(result.get("profiler_raw")) if self.include_metrics else ""
        total_time = result.get("total_time")

        # Every branch below starts with _TOOL_OK_PREFIX so the reward reads the call as verified.
        if total_time is None:
            out["speedup"] = 1.0  # correct but unmeasurable: parity at worst, never a free win
            out["text"] = f"{_TOOL_OK_PREFIX}. (No latency data.)\n\n{metrics_block}"
            return out

        baseline = self._seed_baseline(context, seed_kernel, initial_code, inputs_kwargs)
        if not baseline or baseline <= 0:
            out["speedup"] = 1.0
            out["text"] = (
                f"{_TOOL_OK_PREFIX}. Latency: {total_time * 1000:.4f}ms "
                f"(no baseline to compute speedup).\n\n{metrics_block}"
            )
            return out

        speedup = baseline / total_time
        out["speedup"] = float(speedup)
        out["text"] = (
            f"{_TOOL_OK_PREFIX}.\n"
            f"Latency: {total_time * 1000:.4f}ms  |  original: {baseline * 1000:.4f}ms  "
            f"|  speedup: {speedup:.3f}x\n\n{metrics_block}"
        )
        return out

    @staticmethod
    def _inputs_kwargs(context: dict[str, Any]) -> dict[str, Any]:
        """The input shape every call in this rollout measures, chosen by the reward's own picker.

        Through ``select_test_config_for_rollout`` (which honours ``TEST_CONFIG_MODE``) rather than a
        local choice, so the shape the model is shown and the shape the reward grades agree.
        """
        try:
            return select_test_config_for_rollout(context).get("config") or {}
        except Exception:  # noqa: BLE001 - a malformed row must not abort the rollout
            return {}

    @staticmethod
    def _seed_baseline(
        context: dict[str, Any], seed_kernel: str, initial_code: Any, inputs_kwargs: dict[str, Any]
    ) -> Optional[float]:
        """Seed latency in seconds: the dataset field, then the shared cache, then profile the seed.

        The dataset field is the normal path and is why a run does not re-profile the seed on every
        rollout. On this toolchain it must be a *locally re-measured* value — see
        ``remeasure_baselines.py`` for why the shipped numbers do not apply.
        """
        recorded = context.get("initial_kernel_total_time")
        if recorded is not None:
            try:
                return float(recorded)
            except (TypeError, ValueError):
                pass

        name = context.get("example_name") or context.get("kernel_name") or ""
        if name:
            cached = _baseline_cache_get(name)
            if cached and cached > 0:
                return cached

        results = _send_baseline_request(
            [
                {
                    "src_code": seed_kernel,
                    "initial_code": initial_code,
                    "benchmark_code": context.get("benchmark_code"),
                    "kernel_name": "kernel",
                    "get_inputs_kwargs": inputs_kwargs,
                }
            ]
        )
        if not results:
            return None
        seed_time = results[0].get("total_time")
        if seed_time and seed_time > 0:
            if name:
                _baseline_cache_put(name, seed_time)
            return seed_time
        return None

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
