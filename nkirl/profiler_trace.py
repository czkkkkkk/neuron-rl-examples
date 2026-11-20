# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Per-request timing trace of the profiler calls that the NKIRL tool and reward make.

Set ``NKIRL_PROFILER_TRACE_DIR`` to turn it on. Each process then appends JSON lines to
``<dir>/profiler_calls.<host>.<pid>.jsonl``:

* ``kind="batch"``: one ``send_batch_profiler_request`` call. It records the caller (``tool``,
  ``tool_baseline`` or ``reward``), the wall-clock start and end, and the number of kernels.
* ``kind="kernel"``: one kernel inside a batch. It records the time the call entered the client,
  the time it got a slot of the client semaphore (``PROFILER_MAX_PARALLEL_CHUNKS``), the end time,
  the error, and the numeric timing fields of the result.

The reward core is not changed. ``install`` replaces two attributes of the core module, which the
core looks up at call time. The caller travels inside each config dict, because the core runs a
batch on a plain ``threading.Thread`` when an event loop is running, and context variables do not
pass into such a thread. The request builder reads named keys only, so the extra key does not
reach the service.
"""

from __future__ import annotations

import functools
import json
import os
import socket
import threading
import time
from typing import Any, Callable

_CALLER_KEY = "_nkirl_trace_caller"
_LOCK = threading.Lock()


def _trace_dir() -> str | None:
    return os.getenv("NKIRL_PROFILER_TRACE_DIR") or None


def _write(record: dict[str, Any]) -> None:
    directory = _trace_dir()
    if not directory:
        return
    path = os.path.join(directory, f"profiler_calls.{socket.gethostname()}.{os.getpid()}.jsonl")
    line = json.dumps(record, default=str)
    with _LOCK, open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _result_fields(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"error": f"non-dict result: {type(result).__name__}"}
    out: dict[str, Any] = {"error": str(result["error"])[:300] if result.get("error") else None}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if any(tag in key for tag in ("time", "_s", "_ms", "_ns", "latency", "speedup")):
            out[key] = value
    return out


class _TimedSemaphore:
    """Proxy for the client semaphore that records when this kernel got its slot."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.acquired_at: float | None = None

    async def __aenter__(self) -> _TimedSemaphore:
        await self._inner.acquire()
        self.acquired_at = time.time()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self._inner.release()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def with_caller(send: Callable[..., Any], caller: str) -> Callable[..., Any]:
    """Wrap a ``send_batch_profiler_request`` so that each config and the batch record carry ``caller``.

    Parameters
    ----------
    send : callable
        The core's ``send_batch_profiler_request``.
    caller : str
        The label written to the trace.

    Returns
    -------
    callable
        ``send`` itself when tracing is off, else a wrapper with the same signature.
    """
    if not _trace_dir():
        return send

    @functools.wraps(send)
    def wrapper(configs, *args, **kwargs):
        tagged = []
        for config in configs:
            if isinstance(config, dict) and _CALLER_KEY not in config:
                config = {**config, _CALLER_KEY: caller}
            tagged.append(config)
        start = time.time()
        error = None
        try:
            return send(tagged, *args, **kwargs)
        except Exception as exc:
            error = repr(exc)[:300]
            raise
        finally:
            _write(
                {
                    "kind": "batch",
                    "caller": caller,
                    "t_start": start,
                    "t_end": time.time(),
                    "n": len(tagged),
                    "pid": os.getpid(),
                    "error": error,
                }
            )

    return wrapper


def install(core: Any) -> None:
    """Trace every profiler request that the core module ``core`` sends. Idempotent.

    Parameters
    ----------
    core : module
        ``nki_reward_profiler_local``.
    """
    directory = _trace_dir()
    if not directory or getattr(core, "_nkirl_trace_installed", False):
        return
    os.makedirs(directory, exist_ok=True)
    core._nkirl_send_original = core.send_batch_profiler_request
    core.send_batch_profiler_request = with_caller(core._nkirl_send_original, "reward")

    single = getattr(core, "_profile_single_kernel_local", None)
    if single is not None:

        @functools.wraps(single)
        async def traced(session, config, semaphore, sample_idx, *args, **kwargs):
            timed = _TimedSemaphore(semaphore)
            start = time.time()
            result: Any = None
            try:
                result = await single(session, config, timed, sample_idx, *args, **kwargs)
                return result
            finally:
                caller = config.get(_CALLER_KEY, "unknown") if isinstance(config, dict) else "unknown"
                kernel = config.get("kernel_name") if isinstance(config, dict) else None
                _write(
                    {
                        "kind": "kernel",
                        "caller": caller,
                        "kernel": kernel,
                        "pid": os.getpid(),
                        "t_start": start,
                        "t_slot": timed.acquired_at,
                        "t_end": time.time(),
                        **_result_fields(result),
                    }
                )

        core._profile_single_kernel_local = traced
    core._nkirl_trace_installed = True


def install_generation_trace() -> None:
    """Trace every generation call that verl's agent loop sends to a rollout server. Idempotent.

    Wraps ``verl.workers.rollout.llm_server.LLMServerClient.generate`` (and the fully-async
    subclass, which overrides it). One ``kind="generate"`` line per assistant turn: the trajectory's
    sticky ``request_id`` (the same for every turn of one trajectory), the server that served it,
    the start, the time the load balancer returned a server, the end, and the prompt and output
    token counts. The gaps between consecutive turns of one ``request_id`` are the tool calls, and
    the per-server intervals show when a replica had nothing to decode.
    """
    if not _trace_dir():
        return
    try:
        from verl.workers.rollout import llm_server
    except Exception:  # noqa: BLE001 - another verl layout: skip the trace, never the run
        return
    for cls_name in ("LLMServerClient", "FullyAsyncLLMServerClient"):
        cls = getattr(llm_server, cls_name, None)
        if cls is None or "generate" not in cls.__dict__ or getattr(cls, "_nkirl_gen_traced", False):
            continue
        inner_generate = cls.__dict__["generate"]
        inner_acquire = cls.__dict__.get("_acquire_server")

        def _make(inner_generate=inner_generate):
            @functools.wraps(inner_generate)
            async def generate(self, request_id, *args, prompt_ids=None, **kwargs):
                record = {
                    "kind": "generate",
                    "request_id": str(request_id),
                    "pid": os.getpid(),
                    "t_start": time.time(),
                    "prompt_len": len(prompt_ids) if prompt_ids is not None else None,
                }
                acquired = getattr(self, "_nkirl_acquired", None)
                if acquired is None:
                    acquired = self._nkirl_acquired = {}
                output = None
                error = None
                try:
                    output = await inner_generate(self, request_id, *args, prompt_ids=prompt_ids, **kwargs)
                    return output
                except BaseException as exc:
                    error = repr(exc)[:200]
                    raise
                finally:
                    server_id, t_acq = acquired.pop(str(request_id), (None, None))
                    record.update(
                        {
                            "t_end": time.time(),
                            "t_acquired": t_acq,
                            "server_id": server_id,
                            "output_len": len(getattr(output, "token_ids", None) or []) if output else None,
                            "stop_reason": getattr(output, "stop_reason", None),
                            "error": error,
                        }
                    )
                    _write(record)

            return generate

        cls.generate = _make()
        if inner_acquire is not None:

            def _make_acquire(inner_acquire=inner_acquire):
                @functools.wraps(inner_acquire)
                async def _acquire_server(self, request_id, **extra):
                    result = await inner_acquire(self, request_id, **extra)
                    acquired = getattr(self, "_nkirl_acquired", None)
                    if acquired is None:
                        acquired = self._nkirl_acquired = {}
                    acquired[str(request_id)] = (str(result[0]) if isinstance(result, tuple) else None, time.time())
                    return result

                return _acquire_server

            cls._acquire_server = _make_acquire()
        cls._nkirl_gen_traced = True


def tool_sender(core: Any, caller: str) -> Callable[..., Any]:
    """Return the core's untraced sender, wrapped with ``caller``, for a module that imported it by name."""
    install(core)
    return with_caller(getattr(core, "_nkirl_send_original", core.send_batch_profiler_request), caller)
