# Vendored from NKIRL (src/nki_reward_profiler_local.py) at commit 9e59523, unmodified below this header,
# so this recipe needs no other repository. Update it by copying the file again; do not edit it here.
"""
NKI Kernel Profiler-based Reward Function (v2).

Paired dataset builder: `src/create_rl_dataset_nkibench.py` (also v2).
Legacy versions kept for reference (see those files for full detail):
  - `nki_reward_profiler_local_v1.py` — earlier reward function that
    pre-extracted input shapes locally and applied profiler-side wrapper
    rewrites; replaced by the server-side `get_inputs()` flow here.
  - `nki_reward_profiler_v3.py` — older reward function targeting an NKI
    beta1 profiler; core helpers were ported into this file.

Changes vs v1 (`nki_reward_profiler_local_v1.py`):
  - Server-side input generation. Reads `test_configs` (TEST_CONFIGS dicts +
    per-config cached baselines from --profiling_results) from extra_info
    and sends `inputs_fn="get_inputs"` + `get_inputs_kwargs=<config>` to the
    profiler. The profiler executes `get_inputs(**config)` in its sandbox.
    Works because the dataset uses `benchmarks_clean/` (nki imports stripped).
  - Per-rollout config selection via the TEST_CONFIG_MODE / TEST_CONFIG_INDICES
    env vars (NOT reward_kwargs — the agent loops pick a config too and cannot
    see reward_kwargs; see _resolve_test_config_mode):
      TEST_CONFIG_MODE=largest TEST_CONFIG_INDICES=0,1
    Modes: first (default), random, cycle, largest. See _select_test_config.
  - AST-based `extract_kernel_function` / `replace_kernel_in_file` — handles
    multi-line `def kernel(...) -> ...:` signatures (a v1 regression class
    truncated such bodies).
  - Drops profiler-side wrapper rewrites (`_prepare_code_for_profiler`,
    `_align_kernel_params_to_forward`, `_patch_get_inputs_to_return_dict`,
    `_rewrite_forward_varargs`, `_rewrite_forward_to_kwargs`) — clean
    benchmarks already have correct forward()/get_inputs() signatures.
    See `nki_reward_profiler_local_v1.py` for the legacy implementations.
  - Optional LLM-judge gating via `use_judge` reward_kwarg (Bedrock Claude).
    Both functional-validity and seed-duplication checks; reason field is
    overridden to `JUDGE_FUNC_FAIL` / `JUDGE_DUPLICATION` on reject.
  - Train/val scoping: replay-buffer mutate + harvest skipped for rows with
    `extra_info["split"] == "val"` (prevents val leakage into the buffer).

================================================================================
WHAT THIS FILE CONTAINS
================================================================================

Kernel extraction (from model response)
  - `extract_kernel_code_from_response` — DSPy markers + profiler-verified tool
    calls + markdown blocks + raw AST scan fallback. A tool-calling / multi-turn
    rollout emits many kernels; WHICH one gets graded is selectable via
    `_KERNEL_EXTRACT_STRATEGIES` (best_speedup default, latest_verified, latest,
    legacy), set per run with the KERNEL_EXTRACT_STRATEGY env var — one knob
    shared by the reward path AND both agent loops, deliberately not a
    reward_kwarg (which the loops cannot see). All but `legacy` share one
    precondition: never grade a budget-truncated block while a complete kernel
    exists in the rollout.
  - `extract_kernel_function` (AST-based) / `replace_kernel_in_file`
    (AST-based) — locate and substitute the @nki.jit kernel block.

Profiler request preparation
  - `strip_blocked_imports`, `replace_kernel_in_file` — build the src_code
    sent to the profiler.
  - Profiler request: src_code (kernel file with model's kernel swapped in),
    initial_code (helpers + benchmark for the reference path), `inputs_fn`
    + `get_inputs_kwargs` (chosen TEST_CONFIGS entry).

Cheating / hygiene detection (per-sample, pre-profile)
  - `_is_copied_kernel` — SequenceMatcher similarity vs the seed kernel
    (threshold 0.95). Hits → COPIED_KERNEL_REWARD.
  - `_is_non_nki_kernel` — kernel body scan for numpy/torch-only patterns.
    Hits → NON_NKI_KERNEL_REWARD.

Profiler transport (local aiohttp)
  - `_profile_single_kernel_local`, `_run_async_profiling_batch`,
    `send_batch_profiler_request` — round-robin against
    http://localhost:<8002..8017> (SSH-tunneled trn2 workers).
  - Retry round for transient infra failures
    (PROFILER_MAX_RETRY_ROUNDS env, default 1).

Failure classification
  - `classify_failure_reason` — taxonomy: kernel_extraction_failed,
    compilation_error, accuracy_error, infra (timeout / NRT / http /
    results_dropped), copied_kernel, non_nki_kernel, success_correct.

Reward computation
  - `compute_single_reward` — failure-reason-driven discrete reward:
      kernel_extraction_failed → -1.0
      copied/non_nki/compile/accuracy/infra → -0.5
      correct → 1.0
      beats_initial (5% faster than baseline) → 2.0
      beats_sft (5% faster than optional SFT reference) → 2.0

Initial kernel re-profiling cache
  - `_initial_latency_cache` (per-process dict + lock). Populated lazily;
    once an `example_name` is cached it's free. Skipped when the dataset
    already provides a cached baseline (test_configs[i].baseline_p50_s).

LLM judge gating (optional; opt-in via `use_judge=true` reward_kwarg)
  - Calls `judge_batch` from `src/llm_judge.py` on `is_correct` rollouts.
  - Override: `score`, `abs_reward` → judge_reward_on_invalid (default -0.5).
  - `reason` → `JUDGE_FUNC_FAIL` (functional fail) or `JUDGE_DUPLICATION`
    (cosmetic-only seed refactor). Both `is_judge_failed` and
    `is_judge_duplication` boolean fields ride on the reward dict for wandb.

Replay buffer (optional; opt-in via REPLAY_BUFFER_ENABLED=1)
  - `ReplayBuffer` class — per-problem top-N (sorted by latency), in-memory
    + JSON persist under $REPLAY_BUFFER_PERSIST_DIR.
  - Sampling mixes original-seed / best-from-buffer / intermediate with
    annealing toward best as training progresses.
  - mutate/harvest gated to `split=="train"` rows only.

Best-kernel snapshots
  - `_save_best_kernels` — per-problem frontier. Writes kernel.py +
    meta.json + _summary.json under $KERNEL_ARTIFACTS_DIR/best/.

Infra-failure dumps
  - `_save_infra_failures` — per-process JSONL of infra-failed samples
    (timeout / NRT crash / http error / results_dropped). Files at
    $KERNEL_ARTIFACTS_DIR/infra_failures/pid_<PID>.jsonl.

Reward dict return shape (per sample, consumed by the RL trainer)
  - score, abs_reward — final reward (judge override applied)
  - is_correct, is_infra_failure, is_copied, is_non_nki, is_mutated,
    is_judge_failed, is_judge_duplication, frontier_advance — boolean flags
  - speedup, absolute_speedup — speedup vs (possibly mutated) baseline and
    vs original seed
  - reason — final reason string (post-judge override if applicable)
  - category, nki_source, judge_confidence, judge_concerns — diagnostics

Full-batch aggregates (`reward/*`, `speedup/*`, `pass_at_k/*`, etc.) are
computed by the trainer-side metrics hook, NOT here.
The bundled `_log_batch_aggregates` helper is kept for reference but not
called in the async-rollout / RewardLoopWorker flow.

================================================================================
USAGE
================================================================================

Training script:
    custom_reward_function.path=$NKIRL_DIR/src/nki_reward_profiler_local.py
    custom_reward_function.name=compute_score

    # reward_kwargs (Hydra):
    +custom_reward_function.reward_kwargs.use_judge=true
    +custom_reward_function.reward_kwargs.judge_concurrency=16
    +custom_reward_function.reward_kwargs.judge_reward_on_invalid=-0.5

Required env (set by start_ray_cluster_qwen3_5_with_profiler.sh):
    PROFILING_SERVER_URL=http://localhost
    PROFILER_BASE_PORT=8002
    PROFILER_NUM_WORKERS=16

Optional env:
    KERNEL_EXTRACT_STRATEGY       which kernel of a multi-kernel rollout gets
                                  graded (default best_speedup). See
                                  _KERNEL_EXTRACT_STRATEGIES.
    TEST_CONFIG_MODE              which TEST_CONFIGS input shape it is measured
                                  on (default first). See
                                  _resolve_test_config_mode.
    TEST_CONFIG_INDICES           restrict the pool that mode picks from, e.g.
                                  "0,1". Default: all configs.
    PROFILER_MAX_PARALLEL_CHUNKS  per-worker concurrency cap on profiler
                                  requests (default 16; set to 1-2 to avoid
                                  saturating trn2 workers).
    REWARD_LOG_FILE               dedicated reward log path (FileHandler).
    KERNEL_ARTIFACTS_DIR          root for best/ + infra_failures/ subdirs
                                  (BEST_KERNELS_DIR accepted as legacy alias).
    REPLAY_BUFFER_ENABLED         "1" to turn on curriculum amplification.
    REPLAY_BUFFER_PERSIST_DIR     persisted buffer JSON path.
    JUDGE_MODEL_ID                Bedrock model ID for the judge.
    JUDGE_REGION                  Bedrock region.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    aiohttp = None


# ============================================
# Logging setup
# ============================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Reward-only file logger (separate from veRL's logging).
# Set REWARD_LOG_FILE env var to enable. Each reward worker appends to the same file.
_reward_log_file = os.environ.get("REWARD_LOG_FILE")
if _reward_log_file:
    _reward_handler = logging.FileHandler(_reward_log_file, mode="a")
    _reward_handler.setFormatter(logging.Formatter("%(asctime)s [pid=%(process)d] %(message)s"))
    logger.addHandler(_reward_handler)


# ============================================
# Profiler configuration (local server, multi-worker)
# ============================================
PROFILER_BASE_URL = os.environ.get("PROFILING_SERVER_URL", "http://localhost")
PROFILER_BASE_PORT = int(os.environ.get("PROFILER_BASE_PORT", "8002"))
PROFILER_NUM_WORKERS = int(os.environ.get("PROFILER_NUM_WORKERS", "16"))

# --- OPTIONAL remote profiler endpoint (knob) -------------------------------
# Default "" → UNCHANGED localhost:8002..8002+N-1 multi-worker behavior reached
# through the per-container SSH tunnel (verl parity — must stay byte-identical
# when unset). When set (e.g. https://<your-profiler-endpoint>),
# ALL /profile requests route to this SINGLE endpoint instead of the localhost
# workers, with NO port suffix appended (the remote load-balancer fans out
# server-side, so port rotation is meaningless). PROFILER_NUM_WORKERS then only
# governs display; concurrency is still bounded by MAX_PARALLEL_CHUNKS + the
# semaphore, exactly as in localhost mode. The launcher skips the SSH tunnel and
# probes THIS url's /health preflight when it is set.
PROFILER_REMOTE_URL = os.environ.get("PROFILER_REMOTE_URL", "").strip().rstrip("/")

# Remote auth mode:
#   ""      (default) → no auth (open endpoint / localhost). No-op.
#   "sigv4"           → sign every request with AWS SigV4 using the DEFAULT
#                       botocore credential chain (IMDS instance role on the
#                       cluster → env/~/.aws locally). Needed for an API Gateway
#                       endpoint fronted by IAM (403 MissingAuthenticationToken).
#                       NO credential export required on a role-attached node.
#   "header"          → static header (PROFILER_REMOTE_AUTH_HEADER/VALUE), for a
#                       bearer/api-key endpoint.
# Back-compat: if AUTH_MODE is unset but a static header+value ARE set, behave as
# "header" (the original knob).
PROFILER_REMOTE_AUTH_MODE = os.environ.get("PROFILER_REMOTE_AUTH_MODE", "").strip().lower()
# Static-header auth (bearer/api-key). BOTH must be set; never logged verbatim.
_PROFILER_REMOTE_AUTH_HEADER = os.environ.get("PROFILER_REMOTE_AUTH_HEADER", "").strip()
_PROFILER_REMOTE_AUTH_VALUE = os.environ.get("PROFILER_REMOTE_AUTH_VALUE", "")
# SigV4 params. Service = execute-api for API Gateway. Region MUST match where the
# API is deployed (a custom domain hides it), so it is an explicit
# knob with env fallbacks; if it is wrong the signature is rejected with a clear
# region error. NO hardcoded instance-region default (that is usually NOT the API
# region) — resolve from the knob, then AWS_REGION/AWS_DEFAULT_REGION.
PROFILER_REMOTE_AUTH_SERVICE = os.environ.get("PROFILER_REMOTE_AUTH_SERVICE", "execute-api").strip()
PROFILER_REMOTE_AUTH_REGION = (
    os.environ.get("PROFILER_REMOTE_AUTH_REGION")
    or os.environ.get("AWS_REGION")
    or os.environ.get("AWS_DEFAULT_REGION")
    or ""
).strip()

# Resolve effective auth mode (with the header back-compat shortcut above).
if PROFILER_REMOTE_AUTH_MODE:
    _PROFILER_AUTH_MODE = PROFILER_REMOTE_AUTH_MODE
elif _PROFILER_REMOTE_AUTH_HEADER and _PROFILER_REMOTE_AUTH_VALUE:
    _PROFILER_AUTH_MODE = "header"
else:
    _PROFILER_AUTH_MODE = ""
_PROFILER_SIGV4_ON = bool(PROFILER_REMOTE_URL) and _PROFILER_AUTH_MODE == "sigv4"

# Diagnostic knob: when set, log the SHAPE of every outgoing profiler request
# (nki_version, whether get_inputs is present in src_code, src_code length,
# tolerance mode, get_inputs_kwargs keys) at submit time, and re-log it alongside
# the server error on any non-"completed" poll result. Quiet by default so normal
# runs are unaffected; turn on to debug remote InputSpecError / compile skew and
# prove the src_code get_inputs fold reached the wire. On (1/true) OR whenever the
# remote endpoint is in use (remote failures are exactly what this exists to trace).
_PROFILER_DEBUG_REQUEST = (
    os.environ.get("PROFILER_DEBUG_REQUEST", "").strip().lower() in ("1", "true", "yes")
    or bool(PROFILER_REMOTE_URL)
)

# Lazily-built, reused SigV4 signer. Credentials from the botocore default chain
# auto-refresh (RefreshableCredentials) so the temporary IMDS creds staying fresh
# across a long run is handled for us — add_auth() re-freezes each call.
_SIGV4_SIGNER = None
_SIGV4_INIT_DONE = False


def _get_sigv4_signer():
    """Build (once) a botocore SigV4Auth for the remote profiler, or None.

    None (with a logged reason) if sigv4 is off, botocore is missing, no
    credentials resolve, or no region resolves — the caller then sends the
    request unsigned (which the endpoint will 403), making the misconfig loud.
    """
    global _SIGV4_SIGNER, _SIGV4_INIT_DONE
    if _SIGV4_INIT_DONE:
        return _SIGV4_SIGNER
    _SIGV4_INIT_DONE = True
    if not _PROFILER_SIGV4_ON:
        return None
    try:
        import botocore.session
        from botocore.auth import SigV4Auth
        sess = botocore.session.get_session()
        creds = sess.get_credentials()
        region = PROFILER_REMOTE_AUTH_REGION or (sess.get_config_variable("region") or "")
        if creds is None:
            logger.error("profiler sigv4: NO AWS credentials resolved (IMDS/env/~/.aws) "
                         "— remote calls will 403. Check the node's instance role.")
            return None
        if not region:
            logger.error("profiler sigv4: NO region resolved — set PROFILER_REMOTE_AUTH_REGION "
                         "to the API's region (a custom domain hides it).")
            return None
        _SIGV4_SIGNER = SigV4Auth(creds, PROFILER_REMOTE_AUTH_SERVICE, region)
        logger.info("profiler sigv4: signing enabled (service=%s region=%s, creds via default chain)",
                    PROFILER_REMOTE_AUTH_SERVICE, region)
    except Exception as e:
        logger.error("profiler sigv4: signer init failed (%s) — remote calls will be unsigned", e)
        _SIGV4_SIGNER = None
    return _SIGV4_SIGNER


def _sigv4_headers(method: str, url: str, body: bytes) -> Dict[str, str]:
    """SigV4 auth headers for one request (empty dict if signing unavailable)."""
    signer = _get_sigv4_signer()
    if signer is None:
        return {}
    try:
        from botocore.awsrequest import AWSRequest
        req = AWSRequest(method=method, url=url, data=body or b"")
        signer.add_auth(req)  # mutates req.headers: Authorization, X-Amz-Date, token
        return dict(req.headers)
    except Exception as e:
        logger.warning("profiler sigv4: signing this request failed (%s)", e)
        return {}


def _static_auth_headers() -> Dict[str, str]:
    """Static bearer/api-key header (empty dict if not in header mode)."""
    if PROFILER_REMOTE_URL and _PROFILER_AUTH_MODE == "header":
        return {_PROFILER_REMOTE_AUTH_HEADER: _PROFILER_REMOTE_AUTH_VALUE}
    return {}


# Static headers for the "header"/localhost path (empty = no-op, verl parity).
_PROFILER_HEADERS = _static_auth_headers()

TIMEOUT_SECONDS = int(os.environ.get("PROFILER_TIMEOUT_SECONDS", "200"))
# Ceiling for per-request timeout overrides (a caller may pass a larger
# `timeout_seconds` in the config for big shapes that are slow to compile/run;
# it is clamped to this). The aiohttp session total is sized to this so an
# override never trips the client-level socket timeout.
MAX_TIMEOUT_SECONDS = int(os.environ.get("PROFILER_MAX_TIMEOUT_SECONDS", "900"))
POLL_INTERVAL = float(os.environ.get("PROFILER_POLL_INTERVAL", "2.0"))
POLL_MAX_WAIT = float(os.environ.get("PROFILER_POLL_MAX_WAIT", "300.0"))

MAX_PARALLEL_CHUNKS = int(os.environ.get("PROFILER_MAX_PARALLEL_CHUNKS", "16"))
MAX_RETRY_ROUNDS = int(os.environ.get("PROFILER_MAX_RETRY_ROUNDS", "1"))

# --- capacity-503 backpressure handling (opt-in, default OFF) ---------------
# A profiler "Worker at capacity" 503 is the fleet SHEDDING LOAD (all workers
# busy) — backpressure, NOT an infra fault: the server is up and answering, the
# kernel is fine, the fleet is just saturated for a moment. Counting it as an
# infra failure lets a transient load spike (e.g. an eval burst tripling the
# profiler load) push the rolling infra-rate over NKIRL_INFRA_ABORT_THRESHOLD and
# self-kill the run — exactly how job 732 died. NKIRL_PROFILER_503_SOFT=1 turns on
# option (A): (1) back off and re-submit a capacity-503 on a FRESH worker (so most
# resolve to a real profile with no throughput loss), and (2) if it STILL fails
# after the retries, EXCLUDE it from the infra-abort window (so backpressure can't
# self-kill the job). It stays a failed/dropped kernel for the reward — only the
# breaker ignores it. A genuine outage still trips the breaker via the OTHER error
# classes (connection refused, timeouts, 5xx that are not "Worker at capacity").
# Default "0" → this whole path is inert and every existing verl/slime run is
# byte-identical.
#
# SCOPE NOTE: NKIRL_PROFILER_503_SOFT also gates TUNNEL-BLIP GRACE (see the submit
# and poll loops in _profile_single_kernel_local). A connection refused/reset —
# the autossh localhost tunnel dropping mid-run and re-establishing, which makes
# ALL of 8002-8017 refuse at once (so ejection can't route around it) — is RETRIED
# with the SAME knobs (_503_TRIES/_BACKOFF_S/_JITTER_S, but EXPONENTIAL backoff to
# span autossh's ~45-60s reconnect) instead of instantly counting every rollout as
# an infra failure. That is how the 18h run died at 20:42: the tunnel blipped, the
# rolling window filled with "connection error" in seconds, and NKIRL_INFRA_ABORT
# fired before autossh could reconnect. Grace holds each call across the reconnect;
# a TRULY dead tunnel still fails after the retries and trips the breaker (correct,
# fast). No new knob — raise NKIRL_PROFILER_503_TRIES to widen the grace window.
_NKIRL_503_SOFT = os.environ.get("NKIRL_PROFILER_503_SOFT", "0") == "1"
_NKIRL_503_TRIES = int(os.environ.get("NKIRL_PROFILER_503_TRIES", "5"))
_NKIRL_503_BACKOFF_S = float(os.environ.get("NKIRL_PROFILER_503_BACKOFF_S", "2.0"))
_NKIRL_503_JITTER_S = float(os.environ.get("NKIRL_PROFILER_503_JITTER_S", "1.0"))

# Per-socket limits, distinct from the session `total` above. `total` has to stay
# large because a slow kernel legitimately keeps the request open for minutes;
# these two cap a SINGLE socket operation, which is always fast when the tunnel is
# healthy. Without them a half-open SSH tunnel leaves every request hanging until
# TCP gives up (minutes per call), which can blow a ~15 min rollout up to hours
# while reward/actor timings stay normal. The retry that follows each stall
# succeeds in seconds, so the wait buys nothing.
#
# Safe against slow profiling because /profile is ASYNC: the POST returns a
# poll_url immediately and the wait happens in the poll loop (2 s sleeps, bounded
# by `timeout_seconds`), so no individual request carries the profiling time. The
# 11 successful batches over 120 s in that run (max 882 s) were all poll-loop
# waits, and the slowest single response in 35,598 batches was still well under
# 30 s. Raise PROFILER_SOCK_READ_SECONDS if a future server buffers responses.
SOCK_CONNECT_SECONDS = float(os.environ.get("PROFILER_SOCK_CONNECT_SECONDS", "10"))
SOCK_READ_SECONDS = float(os.environ.get("PROFILER_SOCK_READ_SECONDS", "30"))


def _client_timeout() -> "aiohttp.ClientTimeout":
    """Session timeout: unchanged `total`, plus per-socket connect/read caps."""
    return aiohttp.ClientTimeout(
        total=MAX_TIMEOUT_SECONDS + 30,
        sock_connect=SOCK_CONNECT_SECONDS,
        sock_read=SOCK_READ_SECONDS,
    )
MAX_RETRIES = int(os.environ.get("PROFILER_MAX_RETRIES", "2"))


# ============================================
# Infra-failure circuit breaker (default OFF — verl parity)
# ============================================
# When the profiler backend goes globally sick (NRT crashes, dead SSH tunnel,
# remote endpoint 5xx/timeout after auth) the retry round above can't recover and
# EVERY kernel comes back is_infra_failure=True. With advantage-zeroing
# (NKIRL_DROP_INFRA_FAILURES) those samples are then dropped, so training silently
# burns GPU-hours producing zero learning signal instead of failing loudly. This
# breaker watches the recent infra-failure rate and, once it stays above a
# threshold, kills the whole job so a human notices immediately.
#
# DEFAULT OFF: NKIRL_INFRA_ABORT_THRESHOLD unset/"0"/"" => disabled, so this file
# stays byte-for-byte inert for verl (which never sets it). Set to a fraction in
# (0,1] to enable, e.g. 0.9 = abort when ≥90% of the recent window is infra-failed.
#
# Rate is measured over a ROLLING WINDOW of the last N profiled kernels, NOT
# per-batch: slime calls the reward per-sample (batch size 1-2), so a per-batch
# rate is pure noise there, whereas verl batches large. A window smooths both. The
# window must also have seen at least MIN_SAMPLES kernels so a run can't die on its
# first unlucky call. Each reward worker process keeps its own independent window
# (they don't share memory); because a global profiler outage hits ALL of them,
# whichever crosses the threshold first trips the breaker — fast detection.
_INFRA_ABORT_THRESHOLD = float(os.environ.get("NKIRL_INFRA_ABORT_THRESHOLD", "0") or "0")
_INFRA_ABORT_WINDOW = int(os.environ.get("NKIRL_INFRA_ABORT_WINDOW", "64"))
_INFRA_ABORT_MIN_SAMPLES = int(os.environ.get("NKIRL_INFRA_ABORT_MIN_SAMPLES", "48"))
# Rolling window of recent per-kernel infra outcomes (True=infra failure). Bounded
# by _INFRA_ABORT_WINDOW. Guarded by a lock because a worker may run several
# asyncio batches, but the reward path itself is single-threaded per process.
_infra_window: "list" = []
_infra_window_lock = threading.Lock()
_infra_abort_tripped = False


def _record_infra_outcomes(final_results: List[Dict[str, Any]]) -> None:
    """Feed one profiler batch's infra outcomes into the rolling window and, if the
    breaker is enabled and the recent infra-failure rate stays too high, terminate
    the job. No-op unless NKIRL_INFRA_ABORT_THRESHOLD is set to a positive fraction.

    Called from the batch summary point so it sees post-retry outcomes (transient
    blips already filtered) across BOTH transports (localhost + remote)."""
    global _infra_abort_tripped
    if _INFRA_ABORT_THRESHOLD <= 0.0 or _infra_abort_tripped:
        return

    if _NKIRL_503_SOFT:
        # Option (A): a "Worker at capacity" 503 is load-shedding backpressure, not
        # an infra fault — keep it OUT of the breaker window so a transient
        # saturation spike can't self-kill the job. It is still an infra failure
        # everywhere else (reward, drop path) via _is_infra_failure; only the abort
        # rate ignores it. Real outages still trip the breaker (connection/timeout/
        # non-capacity 5xx all remain flagged).
        flags = [bool(_is_infra_failure(r)) and not _is_capacity_503(r) for r in final_results]
    else:
        flags = [bool(_is_infra_failure(r)) for r in final_results]
    if not flags:
        return

    with _infra_window_lock:
        _infra_window.extend(flags)
        if len(_infra_window) > _INFRA_ABORT_WINDOW:
            del _infra_window[:-_INFRA_ABORT_WINDOW]
        n = len(_infra_window)
        if n < _INFRA_ABORT_MIN_SAMPLES:
            return
        rate = sum(_infra_window) / n
        if rate < _INFRA_ABORT_THRESHOLD:
            return
        # Latch so we abort exactly once even if several batches race in.
        _infra_abort_tripped = True

    _abort_job_infra(rate, n)


def _abort_job_infra(rate: float, window_n: int) -> None:
    """Loudly terminate the training job because the profiler backend is dead.

    Metric hooks (nkirl_metrics) wrap their bodies in try/except and swallow
    exceptions, so raising from the reward path would NOT stop training — we must
    kill the process tree. Best-effort `scancel` of our own Slurm job (so sibling
    ranks/actors also die and the allocation frees), then a guaranteed os._exit."""
    job = os.environ.get("SLURM_JOB_ID", "").strip()
    msg = (
        f"[NKIRL_INFRA_ABORT] profiler infra-failure rate {rate:.1%} over the last "
        f"{window_n} kernels ≥ threshold {_INFRA_ABORT_THRESHOLD:.1%} — the profiler "
        f"backend looks globally down (dead tunnel / NRT crash / endpoint 5xx). "
        f"Aborting job{f' {job}' if job else ''} so it doesn't burn GPU-hours on "
        f"zero-signal rollouts. Fix the profiler and resubmit."
    )
    logger.error(msg)
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass
    if _reward_log_file:
        try:
            with open(_reward_log_file, "a") as f:
                f.write(msg + "\n")
        except Exception:
            pass
    # Kill the whole Slurm allocation if we can (so the driver + all ranks exit,
    # not just this reward worker). Best-effort: the container may lack a slurm
    # client, in which case os._exit below still tears down this process and slime's
    # next gather on the dead actor fails the driver.
    if job:
        try:
            subprocess.run(
                ["scancel", job],
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            logger.error(f"[NKIRL_INFRA_ABORT] scancel {job} failed ({e}); hard-exiting anyway")
    # Guaranteed teardown of this process. Non-zero code so the job records failure.
    os._exit(42)


def _get_profiler_servers() -> List[str]:
    """Build the list of profiler server URLs.

    Default (PROFILER_REMOTE_URL unset): localhost:8002..8002+N-1, one per
    tunneled worker (verl behavior — unchanged).

    Remote knob set: a SINGLE entry (the remote base URL, no port suffix). The
    round-robin then trivially always returns that one URL; the remote
    load-balancer handles fan-out server-side.
    """
    if PROFILER_REMOTE_URL:
        return [PROFILER_REMOTE_URL]
    from urllib.parse import urlparse
    parsed = urlparse(PROFILER_BASE_URL.rstrip("/"))
    base = f"{parsed.scheme}://{parsed.hostname}"
    return [f"{base}:{PROFILER_BASE_PORT + i}" for i in range(PROFILER_NUM_WORKERS)]


_profiler_servers = _get_profiler_servers()
_round_robin_idx = 0
_round_robin_lock = threading.Lock()

# ============================================
# Dead-worker ejection (default OFF in code — enabled via env, e.g. slime)
# ============================================
# When one worker port goes hard-down — its NeuronCores leaked into a ghost
# reservation (a SIGKILLed worker never released them; only reboot/modprobe
# clears them), or the process is wedged on a hung child — EVERY request the
# round-robin sends there fails with an infra error. The server list is frozen
# at import (_get_profiler_servers), so without ejection the client keeps
# routing a fixed 1/N share to the dead port for the whole run: those rollouts
# are wasted and the steady infra-failure rate is dragged toward the
# NKIRL_INFRA_ABORT_THRESHOLD breaker. Ejection makes the client "give up fast"
# on a bad port: after N consecutive infra failures on one server it is dropped
# from the round-robin for a cooldown, and traffic redistributes to the live
# workers. A success (or a post-cooldown re-probe that succeeds — e.g. the
# worker got restarted) restores it.
#
# Thread-safety: slime runs each profiling batch in its own OS thread
# (send_batch_profiler_request offloads to a thread when inside a running loop),
# so the ejection state is guarded by the SAME threading lock as the round-robin
# index. State is per-process, mirroring the infra-breaker window design.
#
# Code default OFF so a caller that never sets the knob is byte-identical to the
# pre-ejection behavior. Also inert in remote mode (single LB endpoint — client
# ejection is meaningless there) and when there is only one server.
_EJECT_ENABLE = (
    os.environ.get("NKIRL_PROFILER_EJECT", "0").strip() not in ("", "0", "false", "False")
    and not PROFILER_REMOTE_URL
    and len(_profiler_servers) > 1
)
# Consecutive infra failures on one server before it is ejected. >1 so a single
# unlucky timeout / transient blip never ejects a healthy worker.
_EJECT_THRESHOLD = int(os.environ.get("NKIRL_PROFILER_EJECT_THRESHOLD", "3"))
# How long an ejected server stays out before it is re-probed. A ghost
# reservation won't recover until reboot (it just re-ejects next cooldown), but a
# worker that was merely restarted will pass the re-probe and rejoin.
_EJECT_COOLDOWN_S = float(os.environ.get("NKIRL_PROFILER_EJECT_COOLDOWN_S", "300"))
_server_fails: "Dict[str, int]" = {}            # server_url -> consecutive infra failures
_server_ejected_until: "Dict[str, float]" = {}  # server_url -> epoch until which it is ejected


def _next_server() -> str:
    """Get next profiler server URL in round-robin order (thread-safe).

    With ejection enabled, skip servers currently ejected for being hard-down.
    If EVERY server is ejected, fall back to the full list rather than hard-stop:
    a true global outage is the infra-abort breaker's job, not this router's."""
    global _round_robin_idx
    with _round_robin_lock:
        pool = _profiler_servers
        if _EJECT_ENABLE and _server_ejected_until:
            now = time.time()
            live = [s for s in _profiler_servers if _server_ejected_until.get(s, 0.0) <= now]
            if live:
                pool = live
        server = pool[_round_robin_idx % len(pool)]
        _round_robin_idx += 1
    return server


def _mark_server_ok(server_url: str) -> None:
    """A terminal success (or non-infra kernel error) from this server — it is
    alive, so clear its failure streak and any ejection."""
    if not _EJECT_ENABLE:
        return
    with _round_robin_lock:
        if _server_fails.get(server_url):
            _server_fails[server_url] = 0
        _server_ejected_until.pop(server_url, None)


def _mark_server_bad(server_url: str) -> None:
    """An infra failure attributable to this server. Count it; eject once the
    streak reaches the threshold."""
    if not _EJECT_ENABLE:
        return
    with _round_robin_lock:
        f = _server_fails.get(server_url, 0) + 1
        _server_fails[server_url] = f
        if f >= _EJECT_THRESHOLD:
            _server_ejected_until[server_url] = time.time() + _EJECT_COOLDOWN_S
            _server_fails[server_url] = 0
            logger.warning(
                "[NKIRL_EJECT] profiler %s ejected for %.0fs after %d consecutive infra "
                "failures (leaked cores / wedged worker); routing around it.",
                server_url, _EJECT_COOLDOWN_S, _EJECT_THRESHOLD,
            )


def _record_server_result(server_url: str, result: "Dict[str, Any]") -> None:
    """Feed one request's terminal outcome into the ejection tracker. A capacity
    503 is backpressure (server is up) → neutral; any other infra failure →
    bad; anything else (success, or a kernel compile/accuracy error) → ok."""
    if not _EJECT_ENABLE:
        return
    if _is_capacity_503(result):
        return
    if _is_infra_failure(result):
        _mark_server_bad(server_url)
    else:
        _mark_server_ok(server_url)


# ============================================
# Reward values (discrete milestones, GRPO-friendly)
# ============================================
NO_KERNEL_REWARD = -1.0
COMPILE_FAILURE_REWARD = -0.5
COMPILE_WRONG_REWARD = -0.5
COPIED_KERNEL_REWARD = -0.5
NON_NKI_KERNEL_REWARD = -0.5
DEGENERATE_KERNEL_REWARD = -0.5
INFRA_FAILURE_REWARD = -0.5
CORRECT_REWARD = 1.0
BEATS_INITIAL_REWARD = 2.0
BEATS_SFT_REWARD = 2.0
# BEATS bar: fraction of baseline latency a CORRECT kernel must undercut for
# BEATS_INITIAL. 0.95 = ≥5% faster; lower it to make BEATS harder (0.90 = ≥10%).
BEATS_INITIAL_MAX_LATENCY_FRAC = float(os.environ.get("BEATS_INITIAL_MAX_LATENCY_FRAC", "0.95"))
COPY_SIMILARITY_THRESHOLD = float(os.environ.get("COPY_SIMILARITY_THRESHOLD", "0.95"))

# ============================================
# Reward shaping mode (set per-run via reward_kwargs; see compute_score_batch)
# ============================================
# "milestone" (default): discrete -1 / -0.5 / 1.0 / 2.0 ladder (constants above).
# "speedup": every failure (no-kernel, compile, accuracy, copy, non-NKI, infra)
#   → 0, and a CORRECT rollout earns its measured speedup vs baseline, capped
#   at `speedup_reward_max`.
DEFAULT_REWARD_MODE = "milestone"
DEFAULT_SPEEDUP_REWARD_MAX = 2.0
SPEEDUP_FAIL_REWARD = 0.0


def _correct_reward(total_time: Optional[float],
                    baseline_latency: Optional[float],
                    reward_mode: str = DEFAULT_REWARD_MODE,
                    speedup_reward_max: float = DEFAULT_SPEEDUP_REWARD_MAX,
                    ) -> "tuple[float, str, Optional[float]]":
    """Reward for a CORRECT rollout. Returns (reward, milestone_label, speedup).

    milestone mode: 1.0, or 2.0 when ≥5% faster than baseline (BEATS_INITIAL).
    speedup mode:   reward = speedup (baseline/latency), clipped to
                    [0, speedup_reward_max]; gates to 0 when no usable
                    latency/baseline is available (no free CORRECT).
    """
    speedup = None
    if total_time and total_time > 0 and baseline_latency and baseline_latency > 0:
        speedup = baseline_latency / total_time

    if reward_mode == "speedup":
        if speedup is None:
            # No measurable speedup — no seed baseline (e.g. seed profiling
            # failed, no baked initial_kernel_total_time) or the kernel's own
            # latency wasn't measured. In speedup mode there is nothing to
            # reward, so gate to 0 rather than hand out a free CORRECT (1.0),
            # which would reward every rollout on a baseline-less problem.
            return SPEEDUP_FAIL_REWARD, "SPEEDUP_NO_BASELINE", None
        capped = max(0.0, min(speedup, speedup_reward_max))
        return capped, "SPEEDUP", speedup

    # milestone (default)
    if speedup is not None and total_time < baseline_latency * BEATS_INITIAL_MAX_LATENCY_FRAC:
        return BEATS_INITIAL_REWARD, "BEATS_INITIAL", speedup
    return CORRECT_REWARD, "CORRECT", speedup


def _fail_reward(milestone_reward: float, reward_mode: str = DEFAULT_REWARD_MODE) -> float:
    """Reward for a non-correct rollout under the active mode.

    speedup mode gates all failures to 0; milestone mode uses the discrete
    penalty constant passed in.
    """
    return SPEEDUP_FAIL_REWARD if reward_mode == "speedup" else milestone_reward


def _reward_from_correct_speedup(total_time: Optional[float],
                                 baseline_latency: Optional[float],
                                 reward_mode: str = DEFAULT_REWARD_MODE,
                                 speedup_reward_max: float = DEFAULT_SPEEDUP_REWARD_MAX,
                                 ) -> float:
    """Map a CORRECT rollout's (latency, baseline_latency) → reward.

    Pure function, no logging, no SFT. Used to compute `abs_reward` against
    the un-mutated original-seed baseline in parallel with the gradient-bearing
    `score` (which uses the possibly-mutated baseline driven by the replay
    buffer). Follows the active reward_mode.
    """
    reward, _, _ = _correct_reward(total_time, baseline_latency,
                                   reward_mode, speedup_reward_max)
    return reward


# ============================================
# Initial kernel latency cache
# ============================================
# Caches seed-kernel ON-DEVICE latency (seconds) keyed by example_name. Two layers:
#   1. per-process in-memory dict (fast, avoids repeat disk reads)
#   2. shared on-disk dir (one JSON per example) so the reward workers AND the
#      multi-turn agent-loop workers — separate processes — share baselines.
#      Whichever path profiles a seed first writes the file; the other reads it.
# After every example is measured once, no extra seed profiling happens.
# Stores on-device kernel time (matches the speedup numerator's metric).
_initial_latency_cache: Dict[str, float] = {}
_initial_latency_cache_lock = threading.Lock()


# ── Shared profiler-metric rendering ────────────────────────────────────────
# Used by BOTH the multi-turn agent loop (turn feedback) and the dataset builder
# (seed_profiler_metrics in schema='seed'), so single-turn seed prompts and
# multi-turn feedback show the model the SAME format. Keep them here, the shared
# module both import from, so the two never drift.

def format_metrics_block(profiler_raw) -> str:
    """Render the profiler payload (JSON str or dict) as a grouped, readable
    metric list. Includes the actionable fields (engine utilization across all
    engines, stall/idle share, DMA fragmentation, memory traffic, spill, compute
    intensity, per-engine instruction counts) — not just MFU/MBU. Falls back
    gracefully if empty/unparseable."""
    if not profiler_raw:
        return "(profiler metrics unavailable)"
    try:
        d = json.loads(profiler_raw) if isinstance(profiler_raw, str) else profiler_raw
        # New raw format keys the perf block as "summary_metrics"; older dumps
        # used "summary". Accept either.
        s = (d.get("summary_metrics") or d.get("summary") or {})
    except Exception:
        return "(profiler metrics unavailable)"
    def _pct(k):
        x = s.get(k)
        return f"{x*100:.1f}%" if isinstance(x, (int, float)) else "n/a"
    def _num(k):
        x = s.get(k)
        return f"{x:,.0f}" if isinstance(x, (int, float)) else "n/a"
    def _f(k, nd=2):
        x = s.get(k)
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "n/a"
    mfu = s.get("mfu_hlo_estimated_percent") or s.get("mfu_inst_estimated_percent")
    lines = [
        "[Utilization — is any engine actually busy?]",
        f"- total active (non-idle): {_pct('total_active_time_percent')}   "
        f"(100% minus this = pure stall/idle — the headroom)",
        f"- MFU (tensor/compute): {mfu*100:.2f}%" if isinstance(mfu, (int, float)) else "- MFU: n/a",
        f"- MBU (HBM bandwidth): {(s.get('mbu_min_read_util_percent') or 0)*100:.3f}%",
        f"- tensor engine active: {_pct('tensor_engine_active_time_percent')}",
        f"- vector engine active: {_pct('vector_engine_active_time_percent')}",
        f"- scalar engine active: {_pct('scalar_engine_active_time_percent')}",
        f"- gpsimd engine active: {_pct('gpsimd_engine_active_time_percent')}",
        f"- sync engine active: {_pct('sync_engine_active_time_percent')}",
        "[Data movement — DMA cost & fragmentation]",
        f"- DMA active: {_pct('dma_active_time_percent')}",
        f"- DMA queues: {_num('dma_queue_count')}, "
        f"software packets: {_num('software_dynamic_dma_packet_count')}",
        f"- static vs dynamic DMA size: {_pct('static_dma_size_percent')} / {_pct('dynamic_dma_size_percent')}",
        "[Memory traffic — bytes moved]",
        f"- HBM read / write bytes: {_num('hbm_read_bytes')} / {_num('hbm_write_bytes')}",
        f"- SBUF read / write bytes: {_num('sbuf_read_bytes')} / {_num('sbuf_write_bytes')}",
        f"- spill reload / save bytes: {_num('spill_reload_bytes')} / {_num('spill_save_bytes')}",
        "[Compute characterization]",
        f"- matmul instruction count: {_num('matmul_instruction_count')}",
        f"- arithmetic intensity: {_f('mm_arithmetic_intensity')}",
        f"- neuroncore cycle count (total work): {_num('neuroncore_cycle_count')}",
        "[Per-engine instruction counts — where the work is]",
        f"- tensor / vector / scalar / gpsimd: "
        f"{_num('tensor_engine_instruction_count')} / {_num('vector_engine_instruction_count')} / "
        f"{_num('scalar_engine_instruction_count')} / {_num('gpsimd_engine_instruction_count')}",
    ]
    return "\n".join(lines)


# "How to read" guide appended after the metric block (full feedback + seed
# prompt) so the model gets the same interpretation frame in both paths.
METRICS_GUIDE = """What each metric measures (you diagnose the bottleneck and decide the fix from THESE numbers — do not assume a default):
- total active (non-idle): 100% minus it is pure stall/idle time.
- engine-active rows show WHERE work runs: tensor=matmul/conv, vector=elementwise/reduction, scalar=activation/pointwise scaling, gpsimd=irregular/gather-scatter, sync=engine coordination (not compute).
- MFU: fraction of peak tensor-engine FLOP/s used. MBU: fraction of peak HBM bandwidth used. Both can read ~0 on small kernels.
- DMA active: fraction of time DMA engines move data. DMA queues / software packets: count of DMA queues and small software-issued transfers.
- spill reload/save bytes: bytes spilled between SBUF and HBM.
- arithmetic intensity: compute per byte moved (FLOPs/byte).
- HBM/SBUF bytes and cycle count: raw work/traffic — lower for the same correct result is a genuine win."""


def _baseline_cache_dir() -> Optional[str]:
    """Shared baseline-cache dir (sibling of KERNEL_ARTIFACTS_DIR). None if unset."""
    kad = os.environ.get("KERNEL_ARTIFACTS_DIR")
    if not kad:
        return None
    return os.path.join(os.path.dirname(kad.rstrip("/")), "baseline_cache")


def _baseline_cache_get(example_name: str) -> Optional[float]:
    """Read seed on-device baseline (seconds): in-memory first, then shared disk."""
    if not example_name:
        return None
    with _initial_latency_cache_lock:
        v = _initial_latency_cache.get(example_name)
    if v:
        return v
    d = _baseline_cache_dir()
    if not d:
        return None
    path = os.path.join(d, f"{example_name}.json")
    try:
        with open(path) as f:
            on_device_s = float(json.load(f)["on_device_s"])
        if on_device_s > 0:
            with _initial_latency_cache_lock:
                _initial_latency_cache[example_name] = on_device_s  # promote to memory
            return on_device_s
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return None
    return None


def _baseline_cache_put(example_name: str, on_device_s: float) -> None:
    """Store seed on-device baseline in memory + shared disk, WRITE-ONCE.

    One fixed value per example: turn-0 and the final reward both divide by it,
    so re-measuring/overwriting would diverge their speedups. First write wins."""
    if not example_name or not on_device_s or on_device_s <= 0:
        return
    # Skip if already known (memory or shared disk) — write-once.
    if _baseline_cache_get(example_name) is not None:
        return
    with _initial_latency_cache_lock:
        # Re-check under lock; another thread may have set it meanwhile.
        if example_name in _initial_latency_cache:
            return
        _initial_latency_cache[example_name] = on_device_s
    d = _baseline_cache_dir()
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{example_name}.json")
        if os.path.exists(path):
            return  # another process already persisted it
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"on_device_s": on_device_s}, f)
        os.replace(tmp, path)  # atomic; first writer wins, others are redundant
    except OSError:
        pass


# ============================================
# Kernel extraction
# ============================================

import ast as _ast


def _extract_kernel_via_ast(code_str: str) -> Optional[str]:
    """Find `def kernel(...)` (sync or async) via AST and return its source
    block including any decorators. Returns None if code doesn't parse or
    no matching function found.
    """
    try:
        tree = _ast.parse(code_str)
    except SyntaxError:
        return None
    lines = code_str.split('\n')
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == 'kernel':
            # Start at the earliest decorator (or the def itself when no
            # decorators). AST line numbers are 1-based.
            start_line = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
            end_line = getattr(node, 'end_lineno', None)
            if end_line is None:
                end_line = len(lines)
            return '\n'.join(lines[start_line:end_line])
    return None


def extract_kernel_function(code_str: str) -> Optional[str]:
    """Extract the @nki.jit kernel function from code.

    Uses AST (handles multi-line signatures, decorators, varied whitespace).
    Falls back to a line-based scan only when the code can't be parsed —
    useful for truncated model outputs where AST fails. The fallback uses
    paren-depth tracking so a multi-line `def kernel(\\n ... \\n) -> ...:`
    signature is not mistaken for the function body's end.
    """
    via_ast = _extract_kernel_via_ast(code_str)
    if via_ast:
        return via_ast

    # Fallback: paren-depth-aware line scan for unparseable code.
    lines = code_str.split('\n')
    kernel_lines: List[str] = []
    state = "search"     # search → in_sig → in_body
    paren_depth = 0
    found_decorator = False

    for line in lines:
        if state == "search":
            if '@nki.jit' in line and not found_decorator:
                found_decorator = True
                kernel_lines = [line]
                continue
            if 'def kernel(' in line:
                if not found_decorator:
                    kernel_lines = [line]
                else:
                    kernel_lines.append(line)
                paren_depth = line.count('(') - line.count(')')
                state = "in_sig" if paren_depth > 0 else "in_body"
                continue
            # Reset decorator if we wandered onto unrelated content.
            if found_decorator and line.strip() and 'kernel' not in line:
                found_decorator = False
                kernel_lines = []
        elif state == "in_sig":
            kernel_lines.append(line)
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0:
                state = "in_body"
        elif state == "in_body":
            stripped = line.strip()
            if stripped == '':
                kernel_lines.append(line)
            elif line[0] in ' \t' or stripped.startswith('#'):
                kernel_lines.append(line)
            else:
                break   # dedent → function ended

    return '\n'.join(kernel_lines) if kernel_lines else None


_DSPY_OPEN = re.compile(r'\[\[\s*##\s*edited_nki_code\s*##\s*\]\][ \t]*\n')
_DSPY_CLOSE = re.compile(r'\[\[\s*##\s*completed\s*##\s*\]\]')
# The original pattern, kept ONLY for strategy="legacy" so the A/B baseline
# stays bit-exact — including its bug (see _dspy_blocks).
_DSPY_BLOCK_LEGACY = re.compile(
    r'\[\[\s*##\s*edited_nki_code\s*##\s*\]\]\s*\n(.*?)(?:\[\[\s*##\s*completed\s*##\s*\]\]|$)',
    re.DOTALL,
)
# Tool-calling trajectories only. The closing </parameter> is REQUIRED: an
# unterminated tail is a kernel the budget cut mid-emit, and grading half a
# function is meaningless.
_TOOL_CALL_KERNEL = re.compile(r'<parameter=nki_code>(.*?)</parameter>', re.DOTALL)
_TOOL_RESPONSE = re.compile(r'<tool_response>(.*?)</tool_response>', re.DOTALL)
# The profiler tool's success header / speedup line.
_TOOL_OK_PREFIX = 'Kernel is correct'
_TOOL_SPEEDUP = re.compile(r'speedup:\s*([0-9.]+)x')


def _dspy_blocks(response_str: str) -> "List[Tuple[int, str, bool]]":
    """Split the response into [(body_start, body, is_complete)] marker blocks.

    A block runs from one `[[ ## edited_nki_code ## ]]` to whichever comes first:
    the next such marker, a `[[ ## completed ## ]]`, or the end of the response.
    `is_complete` is True only in the closed case — the model deliberately
    finalized rather than being cut off by the response-token budget.

    Deliberately NOT one regex. `...edited_nki_code...(.*?)...completed...` looks
    equivalent but its non-greedy body happily spans an intervening
    `edited_nki_code` marker, so a draft block plus a closed answer block collapse
    into ONE match starting at the draft — and the AST scan then returns the
    DRAFT. That silently defeated the last-first scan on exactly the multi-block
    responses it exists for (a tool-calling rollout often carries 2+ blocks).
    """
    opens = list(_DSPY_OPEN.finditer(response_str))
    if not opens:
        return []
    closes = [m.start() for m in _DSPY_CLOSE.finditer(response_str)]
    blocks: List[Tuple[int, str, bool]] = []
    for i, open_match in enumerate(opens):
        start = open_match.end()
        next_open = opens[i + 1].start() if i + 1 < len(opens) else len(response_str)
        close = next((c for c in closes if c >= start), None)
        if close is not None and close < next_open:
            blocks.append((start, response_str[start:close], True))
        else:
            blocks.append((start, response_str[start:next_open], False))
    return blocks


def _norm_ws(code: str) -> str:
    """Whitespace-insensitive key for comparing two kernel sources."""
    return re.sub(r'\s+', '', code or '')


def _kernel_from_block(code_block: str) -> Optional[str]:
    """Shared tail of every extraction tier: AST-extract, else accept the raw
    block when it plainly holds a kernel."""
    code_block = (code_block or '').strip()
    if not code_block:
        return None
    kernel = extract_kernel_function(code_block)
    if kernel:
        return kernel
    if 'def kernel(' in code_block or '@nki.jit' in code_block:
        return code_block
    return None


def _extract_dspy_edited_nki_code(
    response_str: str,
    complete_only: bool = False,
    reject: "Optional[Dict[str, bool]]" = None,
) -> Optional[str]:
    """Extract kernel code from DSPy-style [[ ## edited_nki_code ## ]] marker.

    Scans blocks LAST-first. In a tool-calling rollout the model often restates
    the skeleton (including this marker) before each tool call, so a rollout can
    carry 2+ blocks and the original `re.search` locked onto the earliest —
    usually the pre-iteration draft.

    complete_only=True accepts only a block closed by `[[ ## completed ## ]]`.
    The caller uses it to split "the model finalized" (highest priority) from
    "the block was cut off mid-emit" (lower priority than a verified kernel).

    `reject` holds kernels the profiler tested and REJECTED in this same rollout.
    A block matching one of them is skipped in favour of an older block: the model
    restates the skeleton before each tool call, so a mid-trajectory draft can
    carry a complete `completed` marker and still be a version it already saw fail.
    Skipping is only ever a fallback — if every block is rejected, the last one is
    returned anyway so a deliberate final answer is never silently dropped.
    """
    blocks = _dspy_blocks(response_str)
    if not blocks:
        return None
    # Complete blocks first even without complete_only: a closed block is an
    # intentional answer, an unclosed one is wherever the budget ran out.
    tiers = ([b for b in blocks if b[2]],) if complete_only else (
        [b for b in blocks if b[2]], blocks)
    reject = reject or {}
    for tier in tiers:
        fallback = None
        for _start, body, _complete in reversed(tier):
            kernel = _kernel_from_block(body)
            if not kernel:
                continue
            if fallback is None:
                fallback = kernel
            if reject.get(_norm_ws(kernel), False):
                continue
            return kernel
        if fallback is not None:
            return fallback
    return None


def _tool_call_verdicts(response_str: str) -> "Dict[str, Optional[float]]":
    """Map each profiled kernel -> speedup if the profiler accepted it, else None.

    Pairs `<parameter=nki_code>` i with `<tool_response>` i: the ReAct loop appends
    them strictly alternating, so the indices line up. Calls with no response yet
    (budget cut the rollout mid-call) are dropped. Keys are whitespace-normalized
    so a re-indented resubmission of the same kernel still matches.

    Empty when the trajectory has no tool calls, which makes every caller a no-op
    for single-turn and fixed-multi-turn responses.
    """
    kernels = _TOOL_CALL_KERNEL.findall(response_str)
    if not kernels:
        return {}
    responses = _TOOL_RESPONSE.findall(response_str)
    verdicts: Dict[str, Optional[float]] = {}
    for idx, raw in enumerate(kernels):
        if idx >= len(responses):
            break
        kernel = _kernel_from_block(raw)
        if not kernel:
            continue
        text = responses[idx].lstrip()
        key = _norm_ws(kernel)
        if not text.startswith(_TOOL_OK_PREFIX):
            # Compile error / wrong output. Don't let it overwrite an earlier
            # PASS for the same source — profiling is not perfectly repeatable.
            verdicts.setdefault(key, None)
            continue
        match = _TOOL_SPEEDUP.search(text)
        # No speedup line ("no baseline to compute speedup") still means correct;
        # rank it at parity so a measured win outranks it.
        verdicts[key] = float(match.group(1)) if match else 1.0
    return verdicts


def _tool_call_candidates(
    response_str: str,
    verdicts: "Optional[Dict[str, Optional[float]]]" = None,
) -> "List[Tuple[int, str, Optional[float]]]":
    """Every kernel the model handed to the profiler: [(call_index, code, speedup)].

    `speedup` is None when the profiler REJECTED that source (compile error /
    wrong output) or never answered it, so a picker only has to test
    `speedup is not None` to mean "confirmed correct". Empty for non-tool-calling
    responses. Pass `verdicts` to reuse an already-parsed map.
    """
    raws = _TOOL_CALL_KERNEL.findall(response_str)
    if not raws:
        return []
    if verdicts is None:
        verdicts = _tool_call_verdicts(response_str)
    candidates: List[Tuple[int, str, Optional[float]]] = []
    for idx, raw in enumerate(raws):
        kernel = _kernel_from_block(raw)
        if kernel:
            candidates.append((idx, kernel, verdicts.get(_norm_ws(kernel))))
    return candidates


def _extract_fallback_tail(response_str: str) -> Optional[str]:
    """Markdown code blocks, then a bare AST scan over the whole response.

    Shared last resort for every strategy: a response with neither markers nor
    tool calls lands here, and this is exactly the pre-registry behaviour.
    """
    code_block_patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```nki\n(.*?)```',
    ]
    for pattern in code_block_patterns:
        for match in re.findall(pattern, response_str, re.DOTALL):
            if 'def kernel(' in match or '@nki.jit' in match:
                kernel = extract_kernel_function(match)
                if kernel:
                    return kernel
    return extract_kernel_function(response_str)


# ============================================
# Extraction strategies (which of N kernels gets graded)
# ============================================
# A tool-calling rollout emits up to max_assistant_turns kernels and a multi-turn
# rollout one per turn, but only ONE can be graded — and the choice moves the
# reward a lot (numbers in extract_kernel_code_from_response). Selectable so a run
# can A/B them instead of hard-coding one:
#
#   best_speedup     (default) fastest kernel the profiler CONFIRMED correct
#   latest_verified  newest profiler-confirmed kernel
#   latest           newest COMPLETE kernel, profiler verdict ignored
#   legacy           original behaviour (FIRST marker block); A/B baseline
#
# Completeness is NOT one of the axes these trade off. A block the response-token
# budget cut mid-emit is not a submission, so no strategy but `legacy` grades one
# while a finished kernel exists anywhere in the rollout. What they actually
# differ on is how to rank the COMPLETE candidates.
#
# Adding one: write `_extract_<name>(response_str) -> Optional[str]` and register
# it below. Build on `_tool_call_candidates` (per-call kernels + verdicts) and
# `_extract_fallback_tail` (markdown/AST tail) so a new strategy expresses only
# the CHOICE, not the parsing. `_extract_verified_tiers` takes a picker if the
# new rule differs from best/latest only in how it ranks verified calls.


def _pick_best_verified(
    candidates: "List[Tuple[int, str, Optional[float]]]",
) -> Optional[str]:
    """Highest-speedup profiler-confirmed kernel; ties -> LATEST such call, since
    the model kept iterating from there."""
    best = None          # (speedup, call_index, kernel)
    for idx, kernel, speedup in candidates:
        if speedup is None:
            continue
        if best is None or (speedup, idx) >= (best[0], best[1]):
            best = (speedup, idx, kernel)
    return best[2] if best else None


def _pick_latest_verified(
    candidates: "List[Tuple[int, str, Optional[float]]]",
) -> Optional[str]:
    """Newest profiler-confirmed kernel, ignoring how fast it was."""
    for _idx, kernel, speedup in reversed(candidates):
        if speedup is not None:
            return kernel
    return None


def _extract_verified_tiers(response_str: str, pick) -> Optional[str]:
    """Turn selection for the profiler-verified strategies (best_speedup /
    latest_verified). `pick` ranks the profiler-VERIFIED candidates across ALL
    turns; the tiers under it are shared fallbacks that fire only when NO verified
    kernel exists anywhere in the rollout, so they never separate one strategy
    from another.

    1. `pick`'s choice among the profiler-VERIFIED candidates — the strategy's
       whole point (fastest, or latest, verified kernel over every turn)
    2. A COMPLETE [[ ## edited_nki_code ## ]] block: the model's deliberate final
       answer, returned even if the profiler rejected it (nothing verified to rank)
    3. The last complete tool-call kernel
    4. A partial (budget-truncated) marker block
    5. Markdown blocks, then a raw AST scan (_extract_fallback_tail)

    HISTORY. The picker used to sit BEHIND a "tier 1" that returned the committed
    final block first and yielded to the picker only when the profiler had PROVED
    that block broken. On the TOOLCALL shape (final turn IS profiled) that
    final-first tier fired on most rollouts, and on FIXED MULTI-TURN on nearly all
    of them — so best_speedup and latest_verified both collapsed to "the last
    verified kernel" and KERNEL_EXTRACT_STRATEGY barely moved the graded kernel,
    defeating its only purpose as an A/B axis. The knob must do exactly what its
    name says, so the picker is now FIRST. The old final-block priority was a
    deploy-faithful "grade what the model committed" bias, NOT a correctness
    requirement; if that training-shaping property is ever wanted back it belongs
    behind its own explicit strategy, not baked into every verified pick.

    This only changes the TOOLCALL shape. On FIXED MULTI-TURN there are no
    <tool_response> verdicts, so `_tool_call_candidates` is empty, the picker
    returns None, and tier 2 (last complete block) still decides — exactly as
    before.

    Tier 1 cannot be farmed by profiling the seed and never answering: the seed
    scores ~1.0x speedup (near-zero reward under reward_mode=speedup) and both
    `_is_copied_kernel` and the judge's duplication check still gate it.
    """
    # What did the profiler already test, and how did it go? Empty for
    # non-tool-calling responses, so the picker is empty and selection degrades to
    # the marker/tail fallbacks below.
    verdicts = _tool_call_verdicts(response_str)
    rejected = {key: (speedup is None) for key, speedup in verdicts.items()}

    # 1. The strategy's own rule over every profiler-verified candidate, across
    #    ALL turns. This is exactly what best_speedup / latest_verified promise.
    kernel = pick(_tool_call_candidates(response_str, verdicts))
    if kernel:
        return kernel

    # 2+. No verified candidate anywhere → shared fallbacks (no strategy
    #     difference below). A COMPLETE finalized block, even one the profiler
    #     rejected, is the model's real answer and beats the raw tiers under it;
    #     _extract_dspy_edited_nki_code returns the last complete block, preferring
    #     a non-rejected one but falling back to a rejected one rather than nothing.
    kernel = _extract_dspy_edited_nki_code(response_str, complete_only=True, reject=rejected)
    if kernel:
        return kernel

    for raw in reversed(_TOOL_CALL_KERNEL.findall(response_str)):
        kernel = _kernel_from_block(raw)
        if kernel:
            return kernel

    kernel = _extract_dspy_edited_nki_code(response_str, reject=rejected)
    if kernel:
        return kernel

    return _extract_fallback_tail(response_str)


def _extract_latest(response_str: str) -> Optional[str]:
    """Newest COMPLETE kernel in the trajectory, whatever the profiler said about it.

    "Latest" and "complete" are separate axes, and only the first is a preference.
    Completeness is a precondition every strategy shares: a block the response-token
    budget cut mid-emit is not a kernel the model chose to submit, so it is never
    graded while a finished one exists. Compile/correctness verdicts are what this
    strategy ignores — a closed block that fails to compile IS the model's answer and
    keeps its penalty (see the (a)/(b) split in kernel_was_finalized).

    A source counts as complete when it is a `[[ ## completed ## ]]`-terminated
    marker block, or ANY `<parameter=nki_code>` argument: the ReAct loop only
    dispatches a call once the closing tag arrives, so a tool argument that exists
    at all is finished. Partial blocks are the last resort, matching the verified
    strategies' partial-block fallback — returning nothing instead would earn
    NO_KERNEL_REWARD (-1.0), harsher than the compile failure a truncated kernel
    actually deserves.

    Recency is measured at the START of the code, so a tool-call argument and a
    marker block compete on one axis: whichever the model wrote last wins. (Using
    the END would let a truncated block that runs to \\Z swallow the tool calls
    nested inside it and win by construction.)

    Still measured worse than best_speedup — it grades kernels the profiler already
    proved broken, on purpose. See extract_kernel_code_from_response.
    """
    spans = [(m.start(1), m.group(1), True)
             for m in _TOOL_CALL_KERNEL.finditer(response_str)]
    spans += [(start, body, is_complete)
              for start, body, is_complete in _dspy_blocks(response_str)]
    latest = None        # (code_start_offset, kernel) among COMPLETE sources
    partial = None       # same, among the rest — only used if nothing is complete
    for start, body, is_complete in spans:
        kernel = _kernel_from_block(body)
        if not kernel:
            continue
        slot = latest if is_complete else partial
        if slot is None or start >= slot[0]:
            if is_complete:
                latest = (start, kernel)
            else:
                partial = (start, kernel)
    if latest:
        return latest[1]
    if partial:
        return partial[1]
    return _extract_fallback_tail(response_str)


def _extract_legacy(response_str: str) -> Optional[str]:
    """The original behaviour, kept BIT-EXACT as the A/B baseline: the FIRST
    `_DSPY_BLOCK_LEGACY` match, then the markdown/AST tail.

    Position-based, so on a tool-calling rollout it grades the opening draft and
    discards every later repair. It also keeps the old regex's span bug (a body
    that swallows later markers) on purpose — this is the original behaviour
    verbatim, and a "fixed legacy" would not be the A/B baseline it exists to be.
    """
    match = _DSPY_BLOCK_LEGACY.search(response_str)
    if match:
        kernel = _kernel_from_block(match.group(1))
        if kernel:
            return kernel
    return _extract_fallback_tail(response_str)


_KERNEL_EXTRACT_STRATEGIES = {
    "best_speedup": lambda s: _extract_verified_tiers(s, _pick_best_verified),
    "latest_verified": lambda s: _extract_verified_tiers(s, _pick_latest_verified),
    "latest": _extract_latest,
    "legacy": _extract_legacy,
}
DEFAULT_KERNEL_EXTRACT_STRATEGY = "best_speedup"
# THE knob — env var, and deliberately the ONLY one. Multiple call sites extract
# kernels: this file's reward path and the rollout loops (fixed multi-turn with
# per-turn feedback, and tool-calling). The loops bind
# extract_kernel_code_from_response directly with no arguments, so a
# reward_kwargs-only knob would silently move the REWARD while leaving per-turn
# FEEDBACK on the default — two settings for one decision, and an A/B that grades
# with one rule while it steers with another. An env var is the only channel they
# all share. Set it in your run config; the launcher forwards it into the
# Ray workers' runtime_env.env_vars so they see it too.
_STRATEGY_LOGGED = set()


def _resolve_extract_strategy(name: Optional[str] = None) -> str:
    """KERNEL_EXTRACT_STRATEGY, else the default. `name` is for tests only.

    Read on EVERY call rather than cached at import: the reward function and the
    agent loops are imported inside long-lived Ray workers, so an import-time
    snapshot would freeze whatever the env happened to be at actor construction.

    An unknown name logs and falls back instead of raising — killing a training
    step over a typo'd knob is worse than one loud line per batch.
    """
    candidate = name or os.environ.get("KERNEL_EXTRACT_STRATEGY", "").strip()
    if not candidate:
        return DEFAULT_KERNEL_EXTRACT_STRATEGY
    if candidate in _KERNEL_EXTRACT_STRATEGIES:
        return candidate
    logger.error(
        f"Unknown KERNEL_EXTRACT_STRATEGY '{candidate}' — falling back to "
        f"'{DEFAULT_KERNEL_EXTRACT_STRATEGY}'. "
        f"Valid: {sorted(_KERNEL_EXTRACT_STRATEGIES)}"
    )
    return DEFAULT_KERNEL_EXTRACT_STRATEGY


def extract_kernel_code_from_response(
    response_str: str,
    strategy: Optional[str] = None,
) -> Optional[str]:
    """Extract the ONE kernel to grade from a model response.

    Shared by all three rollout shapes — single-turn, fixed multi-turn
    (per-turn feedback) and tool-calling — and selected by
    KERNEL_EXTRACT_STRATEGY. `strategy` overrides it for tests; production callers
    pass nothing so one env var controls every path.

    The profiler-verified strategies apply their picker FIRST (see
    _extract_verified_tiers), so on the TOOLCALL shape best_speedup and
    latest_verified genuinely grade the best / latest verified kernel across ALL
    turns and differ as their names imply.

    On FIXED MULTI-TURN the injected feedback is concatenated into the response
    string but never in the `<tool_response>` shape `_tool_call_verdicts` reads, so
    verdicts come back empty and the profiler-verified strategies collapse to the
    last COMPLETE block — i.e. the last turn's kernel. `latest` collapses onto them
    too (the axis it gives up, the verdict, does not exist here), so the real choice
    on this shape is those three vs `legacy` (FIRST block = turn 0). A per-turn
    "best" on this shape has to come from the reward path, which is the only place
    that knows the final turn's verdict — this function only sees the response text.

    best_speedup is the default: `legacy` is position-based and grades the opening
    draft, so when the token budget ends before the final block it lands on the
    first tool call and discards every later repair — often a kernel the profiler
    already reported broken while a working one exists in the same rollout.
    best_speedup instead ranks the profiler-verified candidates and takes the
    fastest; latest_verified takes the newest verified kernel, `latest` the newest
    COMPLETE one regardless of verdict. All stay registered so a run can A/B them.
    """
    return _KERNEL_EXTRACT_STRATEGIES[_resolve_extract_strategy(strategy)](response_str)


def kernel_was_finalized(response_str: str, kernel_code: Optional[str]) -> bool:
    """Did the model FINISH the kernel it is about to be graded on?

    True when `kernel_code` is carried by a source the model closed on purpose: a
    `[[ ## completed ## ]]`-terminated marker block, or a `<parameter=nki_code>`
    tool-call argument (the ReAct loop only dispatches a call once the closing tag
    arrives, so a truncated call is never profiled). False only when the sole
    carrier is a marker block the response-token budget cut off mid-emit —
    _extract_verified_tiers' partial-block fallback, reachable only when nothing
    complete exists anywhere in the rollout.

    This distinction is NOT used to pick a different kernel, on purpose. Two
    outcomes look identical in the reward log and are not alike:

      (a) the model closed the block and the kernel doesn't compile — a real
          mistake, and the compile penalty IS the signal;
      (b) the budget ended mid-kernel — the model never got to decide.

    (a) must stay in the reward. Dropping it would teach the model that an
    answer which merely LOOKS unfinished is free. And (b) must not be dropped
    either: returning nothing at tier 5 yields FAILURE_KERNEL_EXTRACTION
    (NO_KERNEL_REWARD -1.0), which is HARSHER than the compile failure the
    partial kernel earns — the wrong direction for a rollout that was cut off
    through no decision of its own. So extraction is unchanged and this flag
    exists to make (b) *countable*: it rides along as `finalized` /
    `graded_incomplete_kernel` per sample so "bad kernel" and "out of budget"
    stop being indistinguishable in the metrics.

    In practice most rollouts with no complete block still land on a tool-call
    argument, which is a finished kernel; only a small tail is graded off a partial
    block, and those are usually logged `compilation_error` — i.e. today they are
    silently pooled with genuinely broken kernels.
    """
    if not kernel_code:
        return False
    key = _norm_ws(kernel_code)
    for raw in _TOOL_CALL_KERNEL.findall(response_str):
        candidate = _kernel_from_block(raw)
        if candidate and _norm_ws(candidate) == key:
            return True
    for _start, body, is_complete in _dspy_blocks(response_str):
        candidate = _kernel_from_block(body)
        if candidate and _norm_ws(candidate) == key and is_complete:
            return True
    # No closed carrier. A markdown-fence / raw-AST fallback (tiers 6-7) has no
    # completion marker to check, so it counts as not-finalized rather than
    # claiming a guarantee the source can't give.
    return False


# ============================================
# Code preprocessing for profiler
# ============================================

def strip_blocked_imports(src: str) -> str:
    """Remove torch/numpy imports from src_code (local profiler sandbox blocks them)."""
    lines = src.splitlines()
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("import torch", "import numpy")):
            continue
        if stripped.startswith(("from torch ", "from numpy ")):
            continue
        out.append(line)
    return "\n".join(out)


def replace_kernel_in_file(original_code: str, new_kernel: str) -> str:
    """Replace the kernel function in original_code with new_kernel.

    Uses AST to locate the exact span of `def kernel(...)` (and any
    preceding decorators) so multi-line signatures like
    `def kernel(\\n  x,\\n  ...,\\n) -> nl.ndarray:` are handled correctly.
    Falls back to line-scan when the original file isn't parseable.
    """
    marker = "# ===== GENERATED NKI KERNEL ====="
    if marker in original_code:
        before = original_code.split(marker, 1)[0]
        return before + marker + "\n" + new_kernel

    lines = original_code.split('\n')

    try:
        tree = _ast.parse(original_code)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and node.name == 'kernel':
                start_line = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
                end_line = getattr(node, 'end_lineno', None) or len(lines)
                new_lines = lines[:start_line] + [new_kernel] + lines[end_line:]
                return '\n'.join(new_lines)

    # Fallback: paren-depth-aware line scan when original_code is unparseable.
    kernel_start = None
    kernel_end = None
    found_decorator = False
    state = "search"
    paren_depth = 0

    for i, line in enumerate(lines):
        if state == "search":
            if '@nki.jit' in line and kernel_start is None:
                kernel_start = i
                found_decorator = True
                continue
            if 'def kernel(' in line:
                if kernel_start is None:
                    kernel_start = i
                paren_depth = line.count('(') - line.count(')')
                state = "in_sig" if paren_depth > 0 else "in_body"
        elif state == "in_sig":
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0:
                state = "in_body"
        elif state == "in_body":
            stripped = line.strip()
            if stripped and line[0] not in ' \t' and not stripped.startswith('#'):
                kernel_end = i
                break

    if kernel_start is not None and kernel_end is None:
        kernel_end = len(lines)

    if kernel_start is None:
        logger.warning("No kernel found in original code, appending new kernel")
        return original_code + '\n\n' + new_kernel

    new_lines = lines[:kernel_start] + [new_kernel] + lines[kernel_end:]
    return '\n'.join(new_lines)


# ============================================
# Local profiler client (raw aiohttp, no SDK)
# ============================================

async def _profile_single_kernel_local(
    session: "aiohttp.ClientSession",
    config: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    sample_idx: int,
) -> Dict[str, Any]:
    """Profile a single kernel via raw aiohttp against a localhost worker.

    Returns the legacy dict format expected by compute_single_reward:
      {"error": str|None, "total_time": float|None, ...}
    """
    server_url = _next_server()

    # ---- REMOTE-ONLY request composition (single choke point for ALL callers:
    # reward, in-rollout tool feedback, dataset build). The remote profiler resolves
    # get_inputs ONLY from src_code and is a beta4 backend, so remote-correctness is
    # normalized HERE, not at each call site — a site that forgot it (nkirl_generate)
    # is exactly what broke run 606. Callers pass kernel-only src_code + initial_code;
    # a caller that already pre-folded (create_rl reconstructs kernel+benchmark) is
    # detected by the get_inputs guard and left as-is, so its request is unchanged.
    #
    # SKIPPED ENTIRELY when PROFILER_REMOTE_URL is unset — the LOCAL profiler path
    # (localhost/trn2: kernel-only src_code + beta3 + fixed tolerance) runs the
    # original request build below BYTE-FOR-BYTE. Local behavior is untouched.
    #
    # Fold get_inputs into src_code (the remote profiler resolves get_inputs ONLY
    # from src_code). extra_info now carries raw benchmark_code, so prefer it —
    # yielding the canonical `kernel + benchmark` (each symbol once). initial_code
    # (= helpers + benchmark) is the fallback for older datasets that predate the
    # benchmark_code field; folding it duplicates the kernel's helpers and lets the
    # originals shadow any helper the model rewrote.
    if PROFILER_REMOTE_URL:
        _src = config["src_code"]
        if "def get_inputs" not in _src:
            _fold = config.get("benchmark_code") or config.get("initial_code")
            if _fold:
                _src = _src + "\n\n" + _fold
        config = {
            **config,
            "src_code": _src,
            "nki_version": config.get("nki_version", "beta4"),
            "server_default_tolerance": config.get("server_default_tolerance", True),
        }

    # Build request mirroring tools/profiler/run_all_nkibench_detail.py:
    # full kernel file as src_code, helpers+benchmark as initial_code, and
    # inputs_fn=get_inputs + get_inputs_kwargs=test_config so the profiler
    # runs get_inputs(**config) server-side from initial_code at reward time.
    request = {
        "src_code": config["src_code"],
        "kernel_fn": config.get("kernel_name", "kernel"),
        "num_cores": 1,
        "warmup": 5,
        "iters": 50,
        # Frontend defaults to beta3 for reward parity. A caller (e.g. the seed
        # sweep via send_batch_profiler_request_v2) may pass nki_version="beta4"
        # — the frontend NKIBench kernels were hardware-validated on — to avoid
        # spurious incompatibility signals from a frontend they never ran against.
        "nki_version": config.get("nki_version", "beta3"),
    }
    if config.get("initial_code"):
        request["initial_code"] = config["initial_code"]
        request["reference_fn"] = "forward"
        request["reference_type"] = "pytorch"
        # Legacy fixed tolerance by default (reward parity). When a caller sets
        # server_default_tolerance=True we omit `tolerance` entirely so the
        # server applies its dtype-aware defaults (fp16-appropriate, contraction-
        # dim scaled) — matching the NKIBench remote probe, which sends none.
        if not config.get("server_default_tolerance"):
            request["tolerance"] = {"atol": 1e-5, "rtol": 0.02, "auto_adjust_for_dtype": False}
    if config.get("get_inputs_kwargs"):
        # v2: get_inputs_kwargs is the TEST_CONFIGS dict; profiler runs
        # get_inputs(**kwargs) server-side via inputs_fn=get_inputs.
        request["inputs_fn"] = "get_inputs"
        request["get_inputs_kwargs"] = config["get_inputs_kwargs"]
    if config.get("multi_seed_seeds"):
        # Optional (off in the reward path): profile the same shape under
        # several input RNG seeds; the server returns worst-case correctness so
        # a kernel that only passes on one seed can't get a free pass. Used by
        # the cheat sweep (tools/sweep_cheat_tsv.py) to catch value-dependent
        # cheats. No effect on reward calls, which never set this key.
        request["multi_seed_seeds"] = list(config["multi_seed_seeds"])

    if _PROFILER_DEBUG_REQUEST:
        # Prove the ACTUAL wire request shape — most importantly whether get_inputs
        # made it into src_code (the remote profiler resolves get_inputs ONLY from
        # src_code; a kernel-only src_code triggers InputSpecError on the config's
        # dtypes sub-dict). src_len jumping up on remote confirms the initial_code
        # fold landed, without dumping the whole kernel. No secrets (kernel source).
        _src = request.get("src_code", "") or ""
        _kw = request.get("get_inputs_kwargs") or {}
        logger.info(
            "[NKIRL_VERIFY] component=profiler_request sample=%s server=%s "
            "nki_version=%s src_has_get_inputs=%s src_len=%d has_initial_code=%s "
            "inputs_fn=%s tolerance_mode=%s get_inputs_kwargs_keys=%s",
            sample_idx, server_url, request.get("nki_version"),
            ("def get_inputs" in _src), len(_src), ("initial_code" in request),
            request.get("inputs_fn"),
            ("server_default" if config.get("server_default_tolerance") else
             ("fixed" if "tolerance" in request else "none")),
            list(_kw.keys()) if isinstance(_kw, dict) else type(_kw).__name__,
        )

    _t_q0 = time.perf_counter()
    async with semaphore:
        _t_acq = time.perf_counter()  # semaphore acquired → queue_s = _t_acq - _t_q0
        _n_polls = 0
        _t_sub = None  # set once the submit round-trip completes (below)

        def _mk_timing():
            # Observational timing for the EARLY-ERROR returns (submit non-200,
            # no_poll_url, polling timeout, connection error, async exception).
            # The success + completed-with-error returns build `_timing` explicitly
            # from `_t_done` below; this mirrors it for the failure paths so a slow
            # or TIMED-OUT remote profile still reports its poll_wait_s. That is the
            # single most important datapoint for the capacity argument — a timeout
            # is the SLOWEST case — yet dropping its timing collapsed the whole
            # profiler/* metric to empty on remote runs (every value was None →
            # filtered). This only shapes the RETURN dict; the request build above
            # (and the LOCAL path) stay byte-identical.
            _now = time.perf_counter()
            _sub = _t_sub if _t_sub is not None else _now
            return {
                "queue_s": _t_acq - _t_q0,
                "submit_s": _sub - _t_acq,
                "poll_wait_s": (_now - _sub) if _t_sub is not None else 0.0,
                "roundtrip_s": _now - _t_acq,
                "n_polls": _n_polls,
            }

        try:
            # Submit. SigV4 must sign the EXACT bytes + URL, so in sigv4 mode we
            # serialize the JSON ourselves and send it as `data=` with the signed
            # headers; otherwise the original `json=request` + static/no headers
            # path is byte-identical to before (localhost/verl parity).
            submit_data = None
            for _sub_attempt in range(_NKIRL_503_TRIES if _NKIRL_503_SOFT else 1):
                submit_url = f"{server_url}/profile"
                if _PROFILER_SIGV4_ON:
                    _body = json.dumps(request).encode()
                    _hdrs = _sigv4_headers("POST", submit_url, _body)
                    _hdrs["Content-Type"] = "application/json"
                    _post_ctx = session.post(submit_url, data=_body, headers=_hdrs)
                else:
                    _post_ctx = session.post(submit_url, json=request, headers=_PROFILER_HEADERS)
                try:
                    async with _post_ctx as resp:
                        if resp.status != 200:
                            text = await resp.text()
                            # A "Worker at capacity" 503 is the profiler fleet shedding
                            # load (all workers busy) — backpressure, NOT an infra fault.
                            # Back off and re-submit on a FRESH worker (round-robin) rather
                            # than burning the kernel. Gated on NKIRL_PROFILER_503_SOFT:
                            # when OFF the condition is False on the first (only) loop pass,
                            # so the original surface-and-return path below runs byte-for-
                            # byte as before.
                            if (_NKIRL_503_SOFT and resp.status == 503
                                    and "worker at capacity" in text.lower()
                                    and _sub_attempt < _NKIRL_503_TRIES - 1):
                                await asyncio.sleep(
                                    _NKIRL_503_BACKOFF_S * (_sub_attempt + 1)
                                    + random.uniform(0.0, _NKIRL_503_JITTER_S)
                                )
                                server_url = _next_server()  # rotate off the busy worker
                                continue
                            # Previously silent — a non-200 at SUBMIT (auth/gateway/route
                            # errors, e.g. 403 MissingAuthenticationToken, 5xx) is the one
                            # remote failure the poll-path logs never see, so surface it.
                            logger.warning(
                                "Sample %s: profiler submit HTTP %s from %s: %s",
                                sample_idx, resp.status, submit_url, text[:300],
                            )
                            _ret = {"error": f"submit_{resp.status}: {text[:200]}", "total_time": None, **_mk_timing()}
                            _record_server_result(server_url, _ret)
                            return _ret
                        submit_data = await resp.json()
                    break
                except aiohttp.ClientConnectionError as _conn_e:
                    # TUNNEL-BLIP GRACE (submit): connection refused/reset — the autossh
                    # localhost tunnel dropped mid-run and is re-establishing (all
                    # 8002-8017 refuse at once, so ejection can't route around a whole-
                    # tunnel death). Same soft-handling as a capacity-503: back off and
                    # retry rather than counting a RECOVERABLE ~1min tunnel gap as an
                    # infra failure that trips NKIRL_INFRA_ABORT and kills a multi-hour
                    # run (this is exactly how the 18h run died at 20:42). Reuses the
                    # 503 knobs — NO new knob — but EXPONENTIAL backoff so _503_TRIES
                    # attempts span autossh's ~45-60s reconnect; raise
                    # NKIRL_PROFILER_503_TRIES to widen. Catches only the connection
                    # family (ClientConnectorError / ServerDisconnectedError / server
                    # timeouts), never ClientResponseError, so HTTP-status handling
                    # above is untouched. 503_SOFT OFF → loop is range(1) → re-raises on
                    # the first pass → the outer handler's "connection error" return is
                    # byte-identical to before.
                    if _NKIRL_503_SOFT and _sub_attempt < _NKIRL_503_TRIES - 1:
                        _bk = (_NKIRL_503_BACKOFF_S * (2 ** _sub_attempt)
                               + random.uniform(0.0, _NKIRL_503_JITTER_S))
                        logger.warning(
                            "[NKIRL_TUNNEL_GRACE] sample=%s submit connection blip (%s) "
                            "attempt=%d/%d backoff=%.1fs — retrying (tunnel re-establishing?)",
                            sample_idx, str(_conn_e)[:120], _sub_attempt + 1,
                            _NKIRL_503_TRIES, _bk,
                        )
                        await asyncio.sleep(_bk)
                        server_url = _next_server()
                        continue
                    raise

            _t_sub = time.perf_counter()  # submit round-trip done → submit_s = _t_sub - _t_acq
            poll_url = submit_data.get("poll_url")
            if not poll_url:
                _ret = {"error": "no_poll_url", "total_time": None, **_mk_timing()}
                _record_server_result(server_url, _ret)
                return _ret
            # Localhost workers return a RELATIVE poll_url ("/status/<id>") joined
            # onto server_url. A remote LB may hand back an ABSOLUTE url
            # ("https://.../status/<id>") — use it verbatim so we don't produce
            # "https://host/https://host/...". No effect on the relative case.
            poll_target = poll_url if poll_url.startswith(("http://", "https://")) \
                else f"{server_url}{poll_url}"

            # Poll. A caller may raise the timeout for a specific config (big
            # shapes are slow to compile/run and otherwise hit "polling timed
            # out" at the default 200s); it is clamped to MAX_TIMEOUT_SECONDS.
            req_timeout = config.get("timeout_seconds")
            timeout_s = TIMEOUT_SECONDS
            if req_timeout:
                try:
                    timeout_s = min(int(req_timeout), MAX_TIMEOUT_SECONDS)
                except (TypeError, ValueError):
                    timeout_s = TIMEOUT_SECONDS
            max_polls = int(timeout_s / POLL_INTERVAL)
            result = None
            status = None
            _conn_blips = 0  # consecutive connection-error polls tolerated (tunnel blip)
            for _ in range(max_polls):
                _n_polls += 1
                await asyncio.sleep(POLL_INTERVAL)
                # Re-sign each GET (SigV4 headers carry a per-request timestamp and
                # the credential token, both of which must be fresh); no body.
                _get_hdrs = _sigv4_headers("GET", poll_target, b"") if _PROFILER_SIGV4_ON \
                    else _PROFILER_HEADERS
                try:
                    async with session.get(poll_target, headers=_get_hdrs) as resp:
                        result = await resp.json()
                except aiohttp.ClientConnectionError as _conn_e:
                    # TUNNEL-BLIP GRACE (poll): the tunnel dropped WHILE polling an
                    # in-flight job. The profiler worker kept running server-side —
                    # only the autossh hop blipped — and poll_target is still valid, so
                    # tolerate a BOUNDED burst of connection errors and keep polling the
                    # SAME job; a recovered tunnel resumes it. Bounded by _503_TRIES
                    # (reused knob, exponential backoff ≈30s) so a TRULY dead tunnel
                    # still fails fast and trips the breaker, instead of every kernel
                    # limping to its full timeout_s. 503_SOFT OFF → re-raises
                    # immediately → byte-identical "connection error".
                    if _NKIRL_503_SOFT and _conn_blips < _NKIRL_503_TRIES:
                        _bk = (_NKIRL_503_BACKOFF_S * (2 ** _conn_blips)
                               + random.uniform(0.0, _NKIRL_503_JITTER_S))
                        logger.warning(
                            "[NKIRL_TUNNEL_GRACE] sample=%s poll connection blip (%s) "
                            "%d/%d backoff=%.1fs — still polling (tunnel re-establishing?)",
                            sample_idx, str(_conn_e)[:120], _conn_blips + 1,
                            _NKIRL_503_TRIES, _bk,
                        )
                        _conn_blips += 1
                        await asyncio.sleep(_bk)
                        continue
                    raise
                _conn_blips = 0  # a good GET clears the blip counter
                status = result.get("status")
                if status in ("completed", "failed", "error"):
                    break
            else:
                _ret = {"error": "polling timed out", "total_time": None, **_mk_timing()}
                _record_server_result(server_url, _ret)
                return _ret

            # Profiler round-trip wall-clock split (obs-only; the request-build block
            # above is untouched, so the LOCAL path stays byte-identical). queue_s =
            # our-side semaphore wait (CHUNKS throttle), submit_s = POST round-trip,
            # poll_wait_s = backend compile+run+queue, roundtrip_s = submit+poll
            # (backend-attributable). Merged into BOTH the success and
            # completed-with-error returns so compile/accuracy fails — which still
            # burn profiler time — are counted, not only correct kernels.
            _t_done = time.perf_counter()
            _timing = {
                "queue_s": _t_acq - _t_q0,
                "submit_s": _t_sub - _t_acq,
                "poll_wait_s": _t_done - _t_sub,
                "roundtrip_s": _t_done - _t_acq,
                "n_polls": _n_polls,
            }

            if status != "completed":
                error_msg = result.get("error", "")[:300] if result else "unknown"
                if _PROFILER_DEBUG_REQUEST:
                    # Co-locate the server rejection with the request shape that
                    # caused it — the whole point is to tie an InputSpecError back
                    # to "did get_inputs reach src_code?" in one grep-able line.
                    _src = request.get("src_code", "") or ""
                    logger.info(
                        "[NKIRL_VERIFY] component=profiler_reject sample=%s status=%s "
                        "nki_version=%s src_has_get_inputs=%s src_len=%d error=%s",
                        sample_idx, status, request.get("nki_version"),
                        ("def get_inputs" in _src), len(_src), error_msg,
                    )
                _ret = {"error": error_msg or "completed_with_error", "total_time": None, **_timing}
                _record_server_result(server_url, _ret)
                return _ret

            _record_server_result(server_url, {})  # empty error dict => healthy → mark ok
            return {**_local_response_to_reward_format(result), **_timing}

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Sample {sample_idx}: connection error: {str(e)[:200]}")
            _ret = {"error": f"connection error: {str(e)[:200]}", "total_time": None, **_mk_timing()}
            _record_server_result(server_url, _ret)
            return _ret
        except Exception as e:
            logger.warning(f"Sample {sample_idx}: profiler exception: {str(e)[:200]}")
            _ret = {"error": f"async exception: {str(e)[:200]}", "total_time": None, **_mk_timing()}
            _record_server_result(server_url, _ret)
            return _ret


# One-shot guard for the RAW summary_metrics diagnostic log (see below).
_RAW_SUMMARY_LOGGED = False


def _to_pct(frac: Optional[float]) -> Optional[float]:
    """Scale a profiler fraction (0-1) to percent (0-100). None stays None."""
    return frac * 100.0 if frac is not None else None


def _sum_or_none(*vals) -> Optional[float]:
    """Sum non-None numeric values; return None if all are None."""
    present = [v for v in vals if v is not None]
    return float(sum(present)) if present else None


# Bulky nested blobs to drop wherever they appear (not just top level).
_JSON_SAFE_DROP = {"raw_metrics", "sbuf_analysis"}


def _json_safe(obj, _depth=0):
    """Recursively coerce a profiler result into a JSON-serializable structure.
    Keeps EVERYTHING the profiler returned — dicts/lists recursed, scalars kept,
    anything else stringified — except the known-bulky blobs in _JSON_SAFE_DROP
    (dropped at any nesting level). Guards against runaway depth. This is what
    makes `profiler_raw` actually raw instead of a hand-picked subset."""
    if _depth > 12:
        return None
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth + 1)
                for k, v in obj.items() if k not in _JSON_SAFE_DROP}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v, _depth + 1) for v in obj]
    # pydantic model or similar → dict; else stringify
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn(), _depth + 1)
            except Exception:
                break
    return str(obj)


def _json_safe_metrics(pr: Dict[str, Any]) -> str:
    """Serialize the WHOLE profiler result (`pr` = profiling_results[0]) to JSON
    for the rollout dump — latency, summary_metrics, correctness, status, config,
    and any other field the profiler returned. No hand-picked subset (an earlier
    version kept only summary+correctness and dropped `latency`, which is exactly
    the speedup denominator, so the rollout couldn't show what speedup divided
    by). Drops only the known-bulky `raw_metrics`/`sbuf_analysis` blobs; recurses
    everything else. Returns '{}' on failure."""
    try:
        payload = {k: _json_safe(v) for k, v in (pr or {}).items()
                   if k not in _JSON_SAFE_DROP}
        return json.dumps(payload)
    except Exception:
        return "{}"


def _local_response_to_reward_format(response: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the profiler's raw HTTP response into the dict shape that
    the reward path expects.

    Downstream consumers — `compute_single_reward`, `classify_failure_reason`,
    `_is_infra_failure`, the reward-loop side effects — all read a flat dict
    with these keys:
        error               (str | None)   non-None means this sample failed
        total_time          (float | None) measured latency in seconds
        max_abs_error
        l2_error
        cosine_similarity
        failure_count
        allclose_passed

    The profiler's HTTP response packs the same information in a nested
    structure (`result.profiling_results[0].{correctness,latency,error}`)
    and reports correctness with separate booleans; this function flattens
    it, picks the most reliable latency field, and synthesizes an
    `accuracy error: ...` string when the server's correctness flags say
    the kernel ran but produced wrong output. Accuracy failures are
    propagated as-is (no client-side override)."""
    profiling_results = response.get("result", {}).get("profiling_results", [])
    if not profiling_results:
        server_error = response.get("result", {}).get("error") or response.get("error") or "empty profiling_results"
        return {"error": str(server_error)[:300], "total_time": None}

    pr = profiling_results[0]
    error = pr.get("error")

    correctness = pr.get("correctness", {}) or {}
    latency = pr.get("latency", {}) or {}
    summary = pr.get("summary_metrics", {}) or {}

    # One-shot diagnostic (train log only, NOT rollout): dump RAW summary_metrics
    # to confirm units (fraction 0-1 vs already-percent) before _to_pct scales.
    # Tells us whether stalled_rate≈1.0 is real low utilization or double-scaling.
    # Remove once confirmed.
    global _RAW_SUMMARY_LOGGED
    if not _RAW_SUMMARY_LOGGED and summary:
        logger.info(f"[RAW summary_metrics] {summary}")
        _RAW_SUMMARY_LOGGED = True

    # Extract correctness metrics for logging
    max_abs = correctness.get("max_abs_error")
    l2_err = correctness.get("l2_error")
    cosine = correctness.get("cosine_similarity")
    failure_count = correctness.get("failure_count")
    allclose_passed = correctness.get("allclose_passed")
    global_max_passed = correctness.get("global_max_passed")
    all_passed = correctness.get("all_passed")

    # Determine if there's an accuracy failure (no override applied)
    if not error:
        # Server didn't set an explicit error string. Check correctness flags.
        # passed = True if global_max_passed OR cosine_similarity >= 0.9999 (fp16 tolerance)
        passed = global_max_passed
        if cosine is not None and cosine >= 0.9999:
            passed = True
        # If correctness object has all_passed=False, treat as accuracy failure
        if not passed and (all_passed is False or allclose_passed is False or global_max_passed is False):
            details = []
            if failure_count:
                details.append(f"failure_count={failure_count}")
            if l2_err is not None:
                details.append(f"l2_error={l2_err:.6e}")
            if max_abs is not None:
                details.append(f"max_abs_error={max_abs:.6e}")
            error = f"accuracy error: {'; '.join(details)}" if details else "accuracy error: correctness check failed"

    # Latency = on-device kernel time ONLY (from the NTFF trace). No p50/mean
    # fallback and no >1000ns threshold, deliberately:
    #  - speedup = baseline/kernel must divide TWO measurements of the SAME kind;
    #    letting each kernel independently pick on_device-or-p50 could compare a
    #    seed's on_device against a custom kernel's p50 (apples vs oranges). Using
    #    only on_device guarantees seed and custom are always the same metric.
    #  - on_device excludes host/launch overhead, so it reflects true kernel perf
    #    even for small kernels (where E2E p50 is dominated by overhead). It is
    #    also highly reproducible (~1% run-to-run on identical kernels).
    #  - the old >1000ns threshold was backwards: the smallest kernels (which
    #    need on_device most) would fall through to overhead-laden p50.
    # If on_device is missing, leave total_time=None → speedup is simply not
    # computed for that rollout (better than a mixed-metric speedup).
    total_time = None
    on_device_ns = latency.get("on_device_time_ns")
    if on_device_ns is not None and on_device_ns > 0:
        total_time = on_device_ns / 1e9

    # Real E2E p50 (`latency.p50_ms`) from the SAME latency dict — a distinct
    # measurement from on_device (typically ~5x larger; includes host/DMA/launch
    # overhead). Observation-only: NOT used by the reward (speedup stays strictly
    # on-device per the note above). Carried through so dataset-build can bake a
    # genuine p50 into baseline_p50_s (instead of an on-device proxy) and offline
    # analysis can read it. None when the profiler omits/zeroes it.
    p50_ms = latency.get("p50_ms")
    p50_time_s = (p50_ms / 1000.0) if (p50_ms is not None and p50_ms > 0) else None

    return {
        "error": error,
        "total_time": total_time,
        "p50_time_s": p50_time_s,
        "max_abs_error": max_abs,
        "l2_error": l2_err,
        "cosine_similarity": cosine,
        "failure_count": failure_count,
        "allclose_passed": allclose_passed,
        # Hardware perf metrics (observation-only; NOT used in reward). Kept as
        # None when the profiler omits summary_metrics (filtered out of the
        # wandb aggregate). NOTE: despite the `_percent` suffix these fields are
        # FRACTIONS (0-1) — profiler cli.py prints them as `value * 100`. We
        # scale to percent here so downstream (and _STALLED_MAX_UTIL_PCT) is in
        # %. MFU: hlo estimate is often 0, so fall back to the inst estimate
        # (mirrors profiler cli.py).
        "mfu": _to_pct(summary.get("mfu_hlo_estimated_percent")
                       or summary.get("mfu_inst_estimated_percent")),
        "mbu": _to_pct(summary.get("mbu_min_read_util_percent")),
        "dma_active_pct": _to_pct(summary.get("dma_active_time_percent")),
        # Work volume (raw, not %). Proxy for "genuine optimization": honest
        # data-movement wins (HBM round-trip elim, DMA hoist, 128-tiling) reduce
        # hbm_bytes; work-reducing wins reduce cycles. Caveat: the dominant honest
        # optimization (SBUF buffer-reuse / alloc-hoisting) is pure runtime
        # overhead → does NOT move these. And a work↓ here can be honest OR a
        # cheat (dropped_op/incomplete_reduction) — profiler can't tell which.
        # Observation-only; read as a trend.
        "hbm_bytes": _sum_or_none(summary.get("hbm_read_bytes"),
                                  summary.get("hbm_write_bytes")),
        "cycles": summary.get("neuroncore_cycle_count"),
        # Arithmetic intensity (FLOPs/byte). Observation-only. Central to the
        # diagnosis analysis (model mislabels high-AI kernels "memory-bound"),
        # so surfaced as a per-sample field for
        # within-group diversity tracking, not just buried in profiler_raw.
        "arith_intensity": summary.get("mm_arithmetic_intensity"),
        # Full profiler payload for post-hoc analysis (rollout dump only, NOT
        # wandb): the complete summary_metrics dict + the correctness section,
        # so fields we don't surface individually (sqnr_db, engine_ns, neff_size,
        # per-dir hbm bytes, mfu_inst, ...) are still available in the jsonl.
        # JSON string keeps the parquet/non_tensor schema stable across kernels.
        "profiler_raw": _json_safe_metrics(pr),
    }


_LOOP_SEMAPHORES: "Dict[int, asyncio.Semaphore]" = {}


def _get_loop_semaphore() -> asyncio.Semaphore:
    """Per-event-loop global profiler semaphore.

    A single semaphore shared by ALL concurrent _run_async_profiling_batch calls
    on the same event loop, so total in-flight profiler requests are capped at
    MAX_PARALLEL_CHUNKS regardless of how many callers there are. This is what
    prevents the agent loop from flooding the profiler: hundreds of rollouts
    (across all their turns) run in one event loop and would otherwise each open
    their own request. asyncio.Semaphore binds to the loop it's created on, so
    it's keyed by loop id (reward workers and agent-loop actors are separate
    loops/processes and get separate semaphores — fine, they don't overlap).
    """
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _LOOP_SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(MAX_PARALLEL_CHUNKS)
        _LOOP_SEMAPHORES[key] = sem
    return sem


async def _run_async_profiling_batch(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Profile a batch of kernels using local aiohttp (round-robin across workers)."""
    if aiohttp is None:
        return [{"error": "aiohttp not installed", "total_time": None}] * len(configs)

    total = len(configs)
    if total == 0:
        return []

    semaphore = _get_loop_semaphore()
    logger.info(
        f"Local profiler batch: {total} kernels, {len(_profiler_servers)} workers, "
        f"concurrency={MAX_PARALLEL_CHUNKS}, timeout={TIMEOUT_SECONDS}s"
    )

    connector = aiohttp.TCPConnector(limit=MAX_PARALLEL_CHUNKS)
    async with aiohttp.ClientSession(
        timeout=_client_timeout(),
        connector=connector,
    ) as session:
        tasks = [_profile_single_kernel_local(session, cfg, semaphore, i) for i, cfg in enumerate(configs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results: List[Dict[str, Any]] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            logger.error(f"Sample {i}: unhandled exception: {r}")
            final_results.append({"error": f"async exception: {r}", "total_time": None})
        else:
            final_results.append(r)

    # Retry round for infra failures
    for retry_round in range(MAX_RETRY_ROUNDS):
        retry_indices = [
            i for i, r in enumerate(final_results)
            if _is_infra_failure(r) or _is_dropped_result(r)
        ]
        if not retry_indices:
            break
        logger.info(f"Retry round {retry_round + 1}: retrying {len(retry_indices)} infra-failed samples")

        connector = aiohttp.TCPConnector(limit=MAX_PARALLEL_CHUNKS)
        async with aiohttp.ClientSession(
            timeout=_client_timeout(),
            connector=connector,
        ) as session:
            retry_tasks = [
                _profile_single_kernel_local(session, configs[i], semaphore, i)
                for i in retry_indices
            ]
            retry_results = await asyncio.gather(*retry_tasks, return_exceptions=True)

        for j, idx in enumerate(retry_indices):
            rr = retry_results[j]
            if isinstance(rr, Exception):
                final_results[idx] = {"error": f"retry exception: {rr}", "total_time": None}
            elif not _is_infra_failure(rr) and not _is_dropped_result(rr):
                final_results[idx] = rr

    ok = sum(1 for r in final_results if r.get("error") is None)
    infra = sum(1 for r in final_results if _is_infra_failure(r))
    comp = sum(1 for r in final_results if r.get("error") and not _is_infra_failure(r))
    logger.info(f"Local profiler batch complete: {total} total, {ok} ok, {comp} compile/accuracy_err, {infra} infra_err")

    # Circuit breaker: if the profiler backend is globally down, abort the job
    # rather than burn GPU-hours on zero-signal rollouts. No-op unless
    # NKIRL_INFRA_ABORT_THRESHOLD is set (default OFF → verl parity).
    _record_infra_outcomes(final_results)

    return final_results


def send_batch_profiler_request(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Synchronous entry-point for batch profiling (handles event-loop context)."""
    total = len(configs)
    if total == 0:
        return []

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        # We're inside an existing event loop (e.g., Ray async actor).
        result_container: List[Optional[List[Dict[str, Any]]]] = [None]
        exception_container: List[Optional[Exception]] = [None]

        def _run_in_thread():
            try:
                result_container[0] = asyncio.run(_run_async_profiling_batch(configs))
            except Exception as e:
                exception_container[0] = e

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join()

        if exception_container[0] is not None:
            logger.error(f"Async profiling thread failed: {exception_container[0]}")
            return [{"error": f"thread exception: {exception_container[0]}", "total_time": None}] * total

        return result_container[0] or [{"error": "no results", "total_time": None}] * total
    else:
        return asyncio.run(_run_async_profiling_batch(configs))


def send_batch_profiler_request_v2(configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Probe-aligned batch profiling: beta4 frontend + server dtype-aware tolerance.

    Identical transport/polling to send_batch_profiler_request, but each request
    is submitted with nki_version=beta4 (the frontend NKIBench kernels were
    hardware-validated on) and no explicit tolerance (so the server applies its
    dtype-aware defaults). This matches the NKIBench remote probe, under which
    the corpus profiles as 100% compatible. Used by the seed-dataset sweep; the
    reward path keeps send_batch_profiler_request (beta3 + fixed tolerance) until
    this is promoted. Per-config overrides already present are preserved.
    """
    patched = []
    for c in configs:
        c2 = dict(c)
        c2.setdefault("nki_version", "beta4")
        c2.setdefault("server_default_tolerance", True)
        patched.append(c2)
    return send_batch_profiler_request(patched)


# ============================================
# Failure classification
# ============================================

FAILURE_KERNEL_EXTRACTION = "kernel_extraction_failed"
FAILURE_MISSING_ORIGINAL_CODE = "missing_original_code"
FAILURE_TIMEOUT = "timeout"
FAILURE_HTTP_ERROR = "http_error"
FAILURE_COMPILE = "compilation_error"
FAILURE_ACCURACY = "accuracy_error"
FAILURE_INFRA = "infrastructure_error"
FAILURE_RESULTS_DROPPED = "results_dropped"
SUCCESS_CORRECT = "correct"
SUCCESS_COMPILE_ONLY = "compiles_but_incorrect"
FAILURE_COPIED_KERNEL = "copied_kernel"
FAILURE_NON_NKI_KERNEL = "non_nki_kernel"


def _is_dropped_result(result: Dict[str, Any]) -> bool:
    error = str(result.get("error", "") or "").lower()
    return "result dropped" in error


def _is_infra_failure(result: Dict[str, Any]) -> bool:
    error = str(result.get("error", "") or "").lower()
    if not error:
        return False
    return any(p in error for p in [
        "timeout", "timed out", "result dropped", "max retries exceeded",
        "http ", "server error:", "503", "502", "403", "401", "429",
        "forbidden", "unauthorized",
        "connection", "async exception", "retry exception",
        "thread exception", "could not connect", "polling timed out",
        "submit_5", "submit_4", "no_poll_url",
        # Trainium runtime crashes — server/worker side, not the kernel's fault
        "nrt error", "nrt_failure", "nrterror",
        "no cores available", "nocoresavailable",
    ])


def _is_capacity_503(result: Dict[str, Any]) -> bool:
    """A profiler "Worker at capacity" 503 — the fleet is saturated (backpressure),
    NOT an infra fault (the server is up, the kernel is fine). Detected from the
    submit error string set in _profile_single_kernel_local
    (`submit_503: {"detail":"Worker at capacity"}`). Used to keep load-shedding out
    of the infra-abort breaker when NKIRL_PROFILER_503_SOFT is on; every other 503
    (real gateway/backend 5xx) still counts as infra."""
    error = str(result.get("error", "") or "").lower()
    return "503" in error and "worker at capacity" in error


def classify_failure_reason(profiling_result: Dict[str, Any]) -> str:
    error_message = str(profiling_result.get("error", "") or "")
    if not error_message:
        return SUCCESS_CORRECT

    error_lower = error_message.lower()
    if "timeout" in error_lower or "timed out" in error_lower or "polling timed out" in error_lower:
        return FAILURE_TIMEOUT
    if "result dropped" in error_lower:
        return FAILURE_RESULTS_DROPPED
    if (error_lower.startswith("http ") or "server error:" in error_lower
            or "max retries exceeded" in error_lower
            or any(code in error_lower for code in ('403', '401', '429', '502', '503', '504'))
            or "forbidden" in error_lower or "unauthorized" in error_lower
            or error_lower.startswith("submit_")):
        return FAILURE_HTTP_ERROR
    if "connection" in error_lower or "could not connect" in error_lower:
        return FAILURE_INFRA
    if "async exception" in error_lower or "retry exception" in error_lower or "thread exception" in error_lower:
        return FAILURE_INFRA
    # Trainium runtime crashes — worker died on server side, kernel never compiled
    if "nrt error" in error_lower or "nrt_failure" in error_lower or "nrterror" in error_lower:
        return FAILURE_INFRA
    if "no cores available" in error_lower or "nocoresavailable" in error_lower:
        return FAILURE_INFRA
    if "mismatched elements" in error_lower or "accuracy error" in error_lower or "max_abs_error" in error_lower:
        return FAILURE_ACCURACY
    return FAILURE_COMPILE


# ============================================
# Code similarity / cheating detection
# ============================================

def _normalize_code(code: str) -> str:
    lines = code.split('\n')
    normalized = []
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    continue
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue
        if stripped.startswith('#'):
            continue
        code_part = stripped
        in_str = False
        str_char = None
        for ci, c in enumerate(stripped):
            if not in_str:
                if c in ('"', "'"):
                    in_str = True
                    str_char = c
                elif c == '#':
                    code_part = stripped[:ci].rstrip()
                    break
            else:
                if c == str_char and (ci == 0 or stripped[ci - 1] != '\\'):
                    in_str = False
        if code_part:
            normalized.append(code_part)

    return '\n'.join(normalized)


def _is_copied_kernel(generated_kernel: str, initial_kernel: str, threshold: float = None) -> bool:
    if threshold is None:
        threshold = COPY_SIMILARITY_THRESHOLD
    if not generated_kernel or not initial_kernel:
        return False
    norm_gen = _normalize_code(generated_kernel)
    norm_orig = _normalize_code(initial_kernel)
    if norm_gen == norm_orig:
        return True
    if not norm_gen or not norm_orig:
        return False
    from difflib import SequenceMatcher
    similarity = SequenceMatcher(None, norm_gen, norm_orig).ratio()
    if similarity >= 0.80:
        logger.info(f"Copy similarity: {similarity:.3f} (threshold={threshold}) — {'COPIED' if similarity >= threshold else 'PASS'}")
    return similarity >= threshold


def _is_non_nki_kernel(kernel_code: str) -> bool:
    """Detect numpy/torch-only kernel (cheat that won't run on Neuron hardware)."""
    if not kernel_code:
        return False

    lines = kernel_code.split('\n')
    body_lines = []
    in_body = False
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()
        if not in_body:
            if stripped.startswith('def kernel('):
                in_body = True
            continue

        if not stripped:
            continue

        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                docstring_char = stripped[:3]
                if stripped.count(docstring_char) >= 2:
                    continue
                in_docstring = True
                continue
        else:
            if docstring_char and docstring_char in stripped:
                in_docstring = False
            continue

        if stripped.startswith('#'):
            continue

        code_part = stripped
        in_str = False
        str_char = None
        for ci, c in enumerate(stripped):
            if not in_str:
                if c in ('"', "'"):
                    in_str = True
                    str_char = c
                elif c == '#':
                    code_part = stripped[:ci].rstrip()
                    break
            else:
                if c == str_char and (ci == 0 or stripped[ci - 1] != '\\'):
                    in_str = False

        if code_part:
            body_lines.append(code_part)

    if not body_lines:
        return False

    body_text = '\n'.join(body_lines)

    nki_patterns = [
        r'\bnl\.\w+', r'\bnki\.\w+', r'\bnisa\.\w+',
        r'\baffine_range\b', r'\bsequential_range\b',
        r'\bpar_dim\b', r'\bfree_dim\b',
    ]
    has_nki_ops = any(re.search(p, body_text) for p in nki_patterns)

    non_nki_patterns = [
        r'\bnp\.\w+', r'\bnumpy\.\w+', r'\btorch\.\w+',
        r'\b\.cpu\(\)', r'\b\.numpy\(\)',
    ]
    has_non_nki_ops = any(re.search(p, body_text) for p in non_nki_patterns)

    if has_non_nki_ops and not has_nki_ops:
        non_nki_lines = []
        for line in body_lines:
            if any(re.search(p, line) for p in non_nki_patterns):
                non_nki_lines.append(line.strip()[:80])
        logger.debug(f"Non-NKI kernel detected. Offending lines: {non_nki_lines[:5]}")
        return True

    return False


# ============================================
# Source category classification (l1, l2, aten)
# ============================================

def _classify_source_category(data_source: str, example_name: str, extra_info: dict) -> str:
    """Bucket samples for per-category metric rollups.

    Priority:
      1. Explicit `category` in extra_info (allows override at dataset level)
      2. NKIBench `difficulty` (1-5) → "difficulty_1" .. "difficulty_5"
      3. Fallback: "other"
    """
    cat = extra_info.get('category')
    if cat is not None:
        cat = str(cat).lower().strip()
        if cat:
            return cat

    diff = extra_info.get('difficulty')
    if diff is not None:
        try:
            return f"difficulty_{int(diff)}"
        except (ValueError, TypeError):
            pass

    return 'other'


# ============================================
# Reward computation
# ============================================

def compute_single_reward(
    kernel_code: Optional[str],
    original_code: str,
    original_latency: Optional[float],
    profiling_result: Dict[str, Any],
    sample_idx: int = -1,
    initial_kernel: Optional[str] = None,
    sft_reference_latency: Optional[float] = None,
    reward_mode: str = DEFAULT_REWARD_MODE,
    speedup_reward_max: float = DEFAULT_SPEEDUP_REWARD_MAX,
) -> float:
    if kernel_code is None:
        r = _fail_reward(NO_KERNEL_REWARD, reward_mode)
        logger.info(f"Sample {sample_idx}: reward={r:.2f} | reason=kernel_extraction_failed")
        return r

    if initial_kernel and _is_copied_kernel(kernel_code, initial_kernel):
        r = _fail_reward(COPIED_KERNEL_REWARD, reward_mode)
        logger.info(f"Sample {sample_idx}: reward={r:.2f} | reason=COPIED_KERNEL")
        return r

    error_message = profiling_result.get("error")
    total_time = profiling_result.get("total_time")
    failure_reason = classify_failure_reason(profiling_result)

    if failure_reason in (FAILURE_TIMEOUT, FAILURE_HTTP_ERROR, FAILURE_INFRA, FAILURE_RESULTS_DROPPED):
        r = _fail_reward(INFRA_FAILURE_REWARD, reward_mode)
        logger.warning(
            f"Sample {sample_idx}: reward={r:.2f} "
            f"| reason={failure_reason.upper()} | error={str(error_message)[:500]}"
        )
        return r

    if error_message and failure_reason != FAILURE_ACCURACY:
        r = _fail_reward(COMPILE_FAILURE_REWARD, reward_mode)
        logger.info(f"Sample {sample_idx}: reward={r:.2f} | reason=COMPILE_FAILED | error={str(error_message)[:500]}")
        return r

    if failure_reason == FAILURE_ACCURACY:
        r = _fail_reward(COMPILE_WRONG_REWARD, reward_mode)
        logger.info(f"Sample {sample_idx}: reward={r:.2f} | reason=ACCURACY_FAILED | error={str(error_message)[:500]}")
        return r

    # CORRECT — reward depends on the active mode (milestone ladder or
    # capped speedup). SFT-reference bonus only applies in milestone mode.
    reward, milestone, speedup = _correct_reward(
        total_time, original_latency, reward_mode, speedup_reward_max)
    if (reward_mode != "speedup" and milestone == "BEATS_INITIAL"
            and sft_reference_latency and sft_reference_latency > 0
            and total_time and total_time <= sft_reference_latency * BEATS_INITIAL_MAX_LATENCY_FRAC):
        reward = BEATS_SFT_REWARD
        milestone = "BEATS_SFT"

    speedup_str = f", speedup={speedup:.2f}x" if speedup else ""
    latency_str = f", latency={total_time:.6f}s" if total_time else ""
    sft_ref_str = f", sft_ref={sft_reference_latency:.6f}s" if sft_reference_latency else ""
    logger.info(f"Sample {sample_idx}: reward={reward:.2f} | reason={milestone}{latency_str}{speedup_str}{sft_ref_str}")
    return reward


# ============================================
# Helpers for data parsing
# ============================================

def _parse_json_if_string(val):
    if val is None:
        return {}
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(val, dict):
        return val
    return {}


def _get_initial_code(extra_info: dict, ground_truth: dict) -> str:
    code = extra_info.get('initial_code', '') or extra_info.get('original_code', '')
    if not code:
        code = ground_truth.get('initial_code', '') or ground_truth.get('original_code', '')
    if not code:
        helper = extra_info.get('helper_functions', '')
        kernel = extra_info.get('initial_kernel', '')
        if helper or kernel:
            code = f"{helper}\n\n{kernel}" if helper and kernel else (helper or kernel)
    return code


def _get_initial_latency(extra_info: dict, ground_truth: dict) -> Optional[float]:
    latency = extra_info.get('initial_kernel_total_time') or extra_info.get('original_latency')
    if latency is None:
        latency = ground_truth.get('initial_kernel_total_time') or ground_truth.get('original_latency')
    if latency is not None:
        try:
            return float(latency)
        except (ValueError, TypeError):
            return None
    return None


def _best_earlier_turn(extra_info: "Optional[Dict[str, Any]]") -> "Optional[Dict[str, Any]]":
    """The fastest profiler-AND-judge-passing earlier turn of a multi-turn rollout.

    Returns `{"turn", "kernel", "speedup"}` for the best such turn, or None. None
    for every other rollout shape: the discriminator is `extra_info["turns"]`,
    which ONLY the fixed multi-turn agent loop writes, so single-turn and
    tool-calling are untouched by construction.

    NO RE-PROFILING. The agent loop already profiled these turns through the same
    profiler, on the shape `select_test_config_for_rollout` picks for both the
    feedback and the reward, and recorded the verdict. Re-measuring would only add
    load and sampling noise, so the recorded `speedup` is taken as-is.

    The LAST turn is deliberately absent from this comparison: the loop does not
    profile it (`status='last_turn'` carries no `speedup`
    key), which is exactly why the caller has to do the comparison — the graded
    kernel's own speedup is only known after the reward profiles it.

    `correct` in the record is already judge-gated (profiler pass AND judge pass),
    so a turn the judge rejected for cheating/duplication can't be resurrected
    here. Turns with a missing or non-positive speedup are skipped.
    """
    if not extra_info:
        return None
    raw = extra_info.get("turns")
    if not raw:
        return None
    try:
        turns = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return None
    if not isinstance(turns, list):
        return None

    best = None
    for idx, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        # No 'speedup' key = the unprofiled final turn, not a slow one.
        if "speedup" not in turn or not turn.get("correct"):
            continue
        try:
            speedup = float(turn.get("speedup") or 0.0)
        except (ValueError, TypeError):
            continue
        if speedup <= 0:
            continue
        if best is not None and speedup <= best["speedup"]:
            continue
        # `latest` (not the env strategy): within ONE turn's text there are no
        # tool calls and no verdicts, so the only sensible rule is the last
        # complete block that turn emitted. Using the env strategy here would
        # make this recursive and, on `legacy`, grade a mid-turn draft.
        kernel = extract_kernel_code_from_response(turn.get("response") or "", "latest")
        if not kernel:
            continue
        best = {"turn": idx, "kernel": kernel, "speedup": speedup}
    return best


def _parse_test_configs(extra_info: dict) -> List[Dict]:
    """Decode test_configs (list of TEST_CONFIGS dicts) from extra_info.

    v2 dataset stores TEST_CONFIGS as a JSON string (so parquet schema is
    stable across kernels with heterogeneous config keys).
    """
    tc = extra_info.get('test_configs', [])
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except json.JSONDecodeError:
            tc = []
    return list(tc) if tc else []


# Per-process round-robin counter for `mode=cycle`.
_cycle_counter_lock = threading.Lock()
_cycle_counter = {"i": 0}


def _parse_indices_arg(val) -> Optional[List[int]]:
    """Accept indices as list, comma-string '0,1,2', or None."""
    if val is None or val == "":
        return None
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, str):
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    return None


# THE test-config knobs — env vars, and deliberately the ONLY ones, for exactly the
# reason KERNEL_EXTRACT_STRATEGY is (see _KERNEL_EXTRACT_STRATEGIES). Three places
# pick a config: this file's reward path, NkirlAgentLoop._get_inputs_kwargs (fixed
# multi-turn) and ProfileKernelTool._get_inputs_kwargs (tool-calling). Both loops
# used to HARDCODE mode="first" while the reward read
# reward_kwargs.test_config_mode, so any mode other than "first" would have
# profiled the model on one input shape and graded it on another — feedback and
# reward describing different measurements. An env var is the only channel all
# three share.
#
# `first` stays the default: it is what every run so far used, and the alternatives
# change what the reward MEANS, so they must be opted into per experiment rather
# than inherited silently. In this dataset index 0 is the smallest shape (e.g.
# batch_size=2) and the last index the largest (batch_size=512) — so `first` is
# also the cheapest and least representative config.
_TEST_CONFIG_MODES = ("first", "largest", "random", "cycle")
DEFAULT_TEST_CONFIG_MODE = "first"


def _resolve_test_config_mode(mode: Optional[str] = None) -> str:
    """TEST_CONFIG_MODE env var, else the default. `mode` is for tests only.

    Read on EVERY call rather than cached at import, because the reward function
    and both agent loops are imported inside long-lived Ray workers — an
    import-time snapshot would freeze whatever the env was at actor construction.

    An unknown mode logs and falls back instead of raising: killing a training step
    over a typo'd knob is worse than one loud line per batch.
    """
    candidate = (mode or os.environ.get("TEST_CONFIG_MODE", "")).strip()
    if not candidate:
        return DEFAULT_TEST_CONFIG_MODE
    if candidate in _TEST_CONFIG_MODES:
        return candidate
    logger.error(
        f"Unknown TEST_CONFIG_MODE '{candidate}' — falling back to "
        f"'{DEFAULT_TEST_CONFIG_MODE}'. Valid: {sorted(_TEST_CONFIG_MODES)}"
    )
    return DEFAULT_TEST_CONFIG_MODE


def _resolve_test_config_indices(indices=None) -> Optional[List[int]]:
    """TEST_CONFIG_INDICES env var ('0,1'), else `indices`. Restricts the pool the
    mode picks from — e.g. mode=random with indices=[2,3] samples only the two
    largest shapes."""
    env = os.environ.get("TEST_CONFIG_INDICES", "").strip()
    return _parse_indices_arg(env) if env else _parse_indices_arg(indices)


# ============================================================================
# PER-CONFIG PROFILER TIMEOUT — size-based (NOT on_device, NOT config_idx)
# ============================================================================
# The profiler timeout is a POLLING CEILING: a request returns the instant the
# result is ready, so a LARGER ceiling costs a fast config NOTHING — it only sets
# how long a genuinely stuck/broken kernel is waited on before being declared a
# failure. Its dominant cost is COMPILE time, which scales with the problem's
# ELEMENT COUNT (shape / tiling complexity), NOT with how fast the compiled
# kernel then runs. Hence:
#   * baseline on_device_s (μs–ms run latency) is a POOR predictor and is
#     deliberately NOT used — empirically it is even anti-correlated across size
#     tiers (the largest-shape configs often have the smallest on_device time).
#   * config_idx is a per-kernel RELATIVE rung, so absolute size varies by >1e10x
#     WITHIN a single idx (some config_idx=0 rows exceed 1e11 elements while most
#     are <1e7). Gating by idx would under-time the giants that sit in low idx, so
#     it is NOT used either.
# The element-count thresholds below are empirical: at dataset-build time measured
# timeouts clustered at ~1e8+ elements, so small shapes fail fast (default 200s)
# and bigger shapes get progressively more room, clamped by the profiler to
# MAX_TIMEOUT_SECONDS (900s) when the value is read at request time. This mirrors
# create_rl_dataset_nkibench.py's identical logic (kept as an independent copy so
# the runtime profiler has no build-time dependency).
#
# TUNE HERE (one place): thresholds + seconds. First tier whose threshold the
# config's element-count meets wins; below the smallest ⇒ profiler default (200s).
_CONFIG_TIMEOUT_TIERS = (
    (1e11, 900),   # >= 1e11 elements  → 900s
    (1e9, 600),    # >= 1e9  elements  → 600s
    (1e7, 400),    # >= 1e7  elements  → 400s
    # < 1e7 elements → None → PROFILER_TIMEOUT_SECONDS (default 200s)
)


def _config_size_proxy(cfg) -> int:
    """Rough element-count proxy for a config: product of integer leaves > 1
    (shape dims), skipping `dtypes` and trivial 0/1 values. Same definition as the
    dataset builder's _config_size_proxy."""
    ints = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            if v > 1:
                ints.append(v)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for k, x in v.items():
                if k == "dtypes":
                    continue
                walk(x)

    walk(cfg)
    p = 1
    for x in ints:
        p *= x
    return p


def _timeout_for_config(cfg):
    """Size-scaled profiler polling timeout (seconds) for a config, or None to
    keep the profiler default. See the _CONFIG_TIMEOUT_TIERS block for the
    rationale (size, not on_device, not config_idx)."""
    try:
        size = _config_size_proxy(cfg)
    except Exception:
        return None
    for threshold, seconds in _CONFIG_TIMEOUT_TIERS:
        if size >= threshold:
            return seconds
    return None


def select_test_config_for_rollout(extra_info: dict) -> Dict:
    """The ONE entry point both agent loops use to pick an input shape.

    Wraps parse+resolve+select so the loops cannot drift from the reward path:
    they call this with no mode argument, so TEST_CONFIG_MODE decides for all
    three. Returns the full entry ({"config", "baseline_p50_s",
    "baseline_on_device_s"}); callers wanting just the shape take ["config"].

    Note `random`/`cycle` pick INDEPENDENTLY per call, so on those modes a
    rollout's per-turn feedback and its final reward may land on different
    shapes — inherent to sampling per call, not a wiring bug. Use `first` or
    `largest` when feedback and reward must agree exactly.
    """
    return _select_test_config(
        _parse_test_configs(extra_info),
        mode=_resolve_test_config_mode(),
        indices=_resolve_test_config_indices(),
    )


def _select_test_config(
    test_configs: List[Dict],
    mode: str = "first",
    indices: Optional[List[int]] = None,
) -> Dict:
    """Pick one TEST_CONFIGS entry per rollout based on mode + indices.

    v2 dataset stores entries as {"config": <dict>, "baseline_p50_s": <float|None>}.
    For backward compatibility we also accept bare config dicts (older datasets);
    in that case the returned entry has `baseline_p50_s = None`.

    Args:
        test_configs: list parsed from extra_info["test_configs"]
        mode: "first" (default), "random", "cycle", "largest"
        indices: optional subset of valid indices into test_configs

    Returns: {"config": <dict>, "baseline_p50_s": <float|None>} (empty dicts on miss).
    """
    if not test_configs:
        return {"config": {}, "baseline_p50_s": None, "baseline_on_device_s": None}
    n = len(test_configs)
    if indices:
        candidate_idxs = [i for i in indices if 0 <= i < n]
        if not candidate_idxs:
            candidate_idxs = list(range(n))
    else:
        candidate_idxs = list(range(n))

    if mode == "random":
        import random as _rand
        chosen = _rand.choice(candidate_idxs)
    elif mode == "cycle":
        with _cycle_counter_lock:
            chosen = candidate_idxs[_cycle_counter["i"] % len(candidate_idxs)]
            _cycle_counter["i"] += 1
    elif mode == "largest":
        chosen = max(candidate_idxs)
    else:  # "first" (default)
        chosen = min(candidate_idxs)
    entry = test_configs[chosen]
    # New schema: {"config": ..., "baseline_p50_s": ..., "baseline_on_device_s": ...}
    if isinstance(entry, dict) and "config" in entry:
        cfg = dict(entry.get("config") or {})
        # Size-based profiler timeout for big shapes (see _timeout_for_config).
        # Only add it when the config does not already carry one (a dataset may
        # bake it in) and the tier is non-default. This just populates a field the
        # request path already forwards by design (the profiler reads
        # config["timeout_seconds"] at poll time and clamps to MAX_TIMEOUT_SECONDS);
        # it does NOT alter request-build serialization, and is identical on the
        # local and remote profiler paths.
        if "timeout_seconds" not in cfg:
            tmo = _timeout_for_config(cfg)
            if tmo:
                cfg["timeout_seconds"] = tmo
        return {"config": cfg,
                "baseline_p50_s": entry.get("baseline_p50_s"),
                "baseline_on_device_s": entry.get("baseline_on_device_s")}
    # Legacy schema: bare config dict
    if isinstance(entry, dict):
        return {"config": dict(entry), "baseline_p50_s": None, "baseline_on_device_s": None}
    return {"config": {}, "baseline_p50_s": None, "baseline_on_device_s": None}


def _parse_get_inputs_kwargs(extra_info: dict) -> Dict:
    """v1-compat shim: v2 uses _parse_test_configs + _select_test_config in
    compute_score_batch. Returns {} so any legacy caller short-circuits.
    """
    return {}


# ============================================
# Best kernel tracking
# ============================================
_global_best_kernels: Dict[str, Dict[str, Any]] = {}
# Lock guards _global_best_kernels for concurrent reward threads inside the
# RewardLoopWorker process (verl runs reward in a thread pool to avoid
# blocking asyncio; without this, dict mutation during iteration crashes).
import threading as _threading
_global_best_kernels_lock = _threading.Lock()

# Root directory for all kernel artifacts. Layout:
#   $KERNEL_ARTIFACTS_DIR/best/<data_source>.py + .json   (best kernel per problem)
#   $KERNEL_ARTIFACTS_DIR/best/_summary.json              (ranked list)
#   $KERNEL_ARTIFACTS_DIR/infra_failures/pid_<PID>.jsonl  (per-worker infra fails)
_kernel_artifacts_dir = os.environ.get(
    'KERNEL_ARTIFACTS_DIR',
    os.environ.get(
        'BEST_KERNELS_DIR',  # legacy fallback
        os.path.join(os.environ.get('NKIRL_DIR', '/tmp'), 'artifacts', 'kernels')
    )
)
_best_kernels_dir = os.path.join(_kernel_artifacts_dir, 'best')


def _config_id(config: Optional[Dict[str, Any]]) -> str:
    """Canonical, deterministic string for a test config (memory-key use).
    Key order and whitespace normalized so identical configs collide."""
    return json.dumps(config or {}, sort_keys=True, separators=(',', ':'))


def _config_hash(config: Optional[Dict[str, Any]]) -> str:
    """Short deterministic hash of a config, for collision-free filenames.
    The full config (495+ chars) can't go in a filename; this can."""
    return hashlib.md5(_config_id(config).encode()).hexdigest()[:8]


# Best kernels are tracked per (example_name, config). A given problem
# profiled under a different input-shape config is effectively a different
# problem, so it gets its own all-time best. The memory key is the tuple
# (example_name, _config_id(config)); the on-disk filename uses the short
# config hash. Selection is by RUNTIME (latency_ms), not speedup: runtime is
# an absolute measurement immune to baseline re-profiling drift across resumes.
_BEST_LOADED = False


def _maybe_load_global_best():
    """Lightweight resume load: populate _global_best_kernels from the on-disk
    _summary.json so `frontier_advance` and best-overwrite decisions stay
    continuous across a resume. Only the fields needed for the latency-based
    overwrite decision are loaded (kernel_code stays on disk). Runs once."""
    global _BEST_LOADED
    if _BEST_LOADED:
        return
    _BEST_LOADED = True
    summary_file = os.path.join(_best_kernels_dir, "_summary.json")
    if not os.path.exists(summary_file):
        return
    try:
        with open(summary_file) as f:
            summary = json.load(f)
        with _global_best_kernels_lock:
            for _skey, meta in summary.items():
                en = meta.get("example_name")
                cfg = meta.get("config")
                lat = meta.get("latency_ms")
                if en is None or lat is None:
                    continue
                _global_best_kernels[(en, _config_id(cfg))] = dict(meta)
        logger.info(f"Loaded {len(_global_best_kernels)} best-kernel records "
                    f"from {summary_file} (resume continuity)")
    except Exception as e:
        logger.warning(f"Failed to load best-kernel summary: {e}")


def _save_best_kernels(batch_best: Dict[Any, Dict[str, Any]]):
    """Persist per-(problem, config) all-time best kernels.

    Keys of `batch_best` and `_global_best_kernels` are tuples
    (example_name, config_id). Best is decided by RUNTIME (latency_ms, lower
    is better) — an absolute measurement, unlike speedup which depends on a
    baseline that can drift when re-profiled across resumes."""
    # Opt-out: NKIRL_SAVE_BEST_KERNELS=0 disables the per-problem best-kernel
    # frontier dump. It is heavy on a long run: every frontier advance rewrites the
    # ENTIRE snapshot (a .py + .json per best kernel + a full _summary.json), so cost
    # grows with the frontier size. Default "1" keeps it ON, so every existing
    # verl/slime run is byte-identical; only a config that explicitly opts out skips.
    if os.environ.get("NKIRL_SAVE_BEST_KERNELS", "1") == "0":
        return
    global _global_best_kernels
    _maybe_load_global_best()
    with _global_best_kernels_lock:
        updated = False
        for key, info in batch_best.items():
            prev_lat = _global_best_kernels.get(key, {}).get("latency_ms")
            if prev_lat is None or info["latency_ms"] < prev_lat:
                _global_best_kernels[key] = info
                updated = True
                logger.info(f"New best kernel for '{key[0]}' cfg={info.get('config_hash')}: "
                            f"{info['latency_ms'] * 1e6:.1f}us ({info['speedup']:.2f}x)")
        if not updated:
            return
        # Snapshot to a list under the lock so iteration below is safe even
        # if another thread enters _save_best_kernels next.
        snapshot = list(_global_best_kernels.items())
    try:
        os.makedirs(_best_kernels_dir, exist_ok=True)
        for (en, _cid), info in snapshot:
            cfg_hash = info.get("config_hash") or _config_hash(info.get("config"))
            base = re.sub(r'[^a-zA-Z0-9_-]', '_', str(en))[:100]
            safe_name = f"{base}__cfg{cfg_hash}"
            kernel_file = os.path.join(_best_kernels_dir, f"{safe_name}.py")
            meta_file = os.path.join(_best_kernels_dir, f"{safe_name}.json")
            if info.get("kernel_code"):
                with open(kernel_file, 'w') as f:
                    f.write(f"# Best kernel for: {en} (config {cfg_hash})\n"
                            f"# Latency: {info['latency_ms'] * 1e6:.1f}us  Speedup: {info['speedup']:.2f}x\n\n")
                    f.write(info["kernel_code"])
            meta = {k: v for k, v in info.items() if k != "kernel_code"}
            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2, default=str)
        summary_file = os.path.join(_best_kernels_dir, "_summary.json")
        summary = {}
        # Rank by speedup (most impressive first) for human readability.
        for (en, _cid), info in sorted(snapshot, key=lambda x: -x[1].get("speedup", 0)):
            cfg_hash = info.get("config_hash") or _config_hash(info.get("config"))
            summary[f"{en}::{cfg_hash}"] = {
                "example_name": en,
                "config": info.get("config"),
                "config_hash": cfg_hash,
                "speedup": info.get("speedup"),
                "reward": info.get("reward"),
                "latency_ms": info.get("latency_ms"),
                "original_latency_ms": info.get("original_latency_ms"),
                "data_source": info.get("data_source"),
                "nki_source": info.get("nki_source"),
                "difficulty": info.get("difficulty"),
                "num_ops": info.get("num_ops"),
                "op_categories": info.get("op_categories"),
            }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved {len(snapshot)} best kernels to {_best_kernels_dir}")
    except Exception as e:
        logger.warning(f"Failed to save best kernels: {e}")


# ============================================
# Infra-failure kernel logging
# ============================================
# Saves kernels that hit infrastructure failures (timeouts, NRT crashes, http
# errors, results-dropped) so we can inspect what triggers them. Per-slice
# (no aggregation needed) — every infra failure across the cluster gets its
# own line in a per-process JSONL file. To find them all:
#   ls $KERNEL_ARTIFACTS_DIR/infra_failures/*.jsonl
_infra_failures_dir = os.path.join(_kernel_artifacts_dir, "infra_failures")
_infra_failures_path: Optional[str] = None  # lazy-init per process


def _save_infra_failures(
    batch_size, rewards, per_sample, extracted_kernels, solution_strs,
    data_sources, extra_infos, original_latencies, results_map,
):
    """Append infra-failed kernels from this slice to a per-process JSONL.

    `per_sample[i]['is_infra_failure']` already classifies each sample.
    """
    global _infra_failures_path
    if _infra_failures_path is None:
        try:
            os.makedirs(_infra_failures_dir, exist_ok=True)
        except Exception:
            pass
        _infra_failures_path = os.path.join(
            _infra_failures_dir, f"pid_{os.getpid()}.jsonl"
        )

    rows = []
    for i in range(batch_size):
        if i >= len(per_sample) or not per_sample[i].get('is_infra_failure'):
            continue
        ei = extra_infos[i] if i < len(extra_infos) else {}
        result = results_map.get(i, {})
        total_time = result.get('total_time')
        orig_lat = original_latencies[i] if i < len(original_latencies) else None
        rows.append({
            'pid': os.getpid(),
            'example_name': ei.get('example_name') or ei.get('kernel_name', ''),
            'data_source': data_sources[i] if i < len(data_sources) else '',
            'reward': float(rewards[i]) if i < len(rewards) else None,
            'reason': per_sample[i].get('reason'),
            'error': str(result.get('error') or '')[:600],
            'latency_ms': float(total_time) if total_time else None,
            'original_latency_ms': float(orig_lat) if orig_lat else None,
            'kernel_extracted': extracted_kernels[i] is not None if i < len(extracted_kernels) else False,
            'kernel_code': extracted_kernels[i][:50000] if i < len(extracted_kernels) and extracted_kernels[i] else None,
            'response': solution_strs[i][:50000] if i < len(solution_strs) else None,
        })

    if not rows:
        return

    try:
        with open(_infra_failures_path, 'a') as f:
            for row in rows:
                f.write(json.dumps(row, default=str) + '\n')
        logger.info(f"Saved {len(rows)} infra-failure kernel(s) to {_infra_failures_path}")
    except Exception as e:
        logger.warning(f"Failed to save infra failures: {e}")


# ============================================
# MLflow / WandB metric logging
# ============================================
def _log_custom_metrics(metrics: Dict[str, float]):
    """Stage metrics for the current wandb step.

    Uses commit=False so subsequent log calls (from veRL's standard logger)
    accumulate onto the same step, keeping reward/* aligned with actor/* etc.
    """
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, commit=False)
    except Exception as e:
        logger.debug(f"Could not log custom metrics to WandB: {e}")
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metrics(metrics)
    except Exception as e:
        logger.debug(f"Could not log custom metrics to MLflow: {e}")


def _pass_at_k_for_groups(
    prompt_groups: Dict[str, List[Dict]],
    prefix: str = "pass_at_k",
) -> Dict[str, Any]:
    """Compute pass@K over a set of prompt groups.

    For each prompt, take the first K non-infra rollouts and check whether
    ANY passed (reward >= 1.0 → CORRECT or BEATS_INITIAL). Returns flat
    metrics keyed under `<prefix>/pass@K_*`.
    """
    if not prompt_groups:
        return {}
    sizes = [len([s for s in v if not s.get('is_infra_failure')])
             for v in prompt_groups.values()]
    sizes = [s for s in sizes if s > 0]
    if not sizes:
        return {}
    max_size = max(sizes)
    k_values = sorted({1} | {k for k in (5, 8, 10) if k <= max_size} | {max_size})

    metrics: Dict[str, Any] = {f"{prefix}/num_prompts": len(prompt_groups)}
    for k in k_values:
        eligible = [v for v in prompt_groups.values()
                    if len([s for s in v if not s.get('is_infra_failure')]) >= k]
        if not eligible:
            continue
        passes = 0
        max_speedups = []
        for samples in eligible:
            non_infra = [s for s in samples if not s.get('is_infra_failure')][:k]
            any_pass = any(s.get('is_correct') and s.get('reward_score', 0) >= 1.0 for s in non_infra)
            if any_pass:
                passes += 1
                spds = [s.get('speedup') for s in non_infra if s.get('is_correct') and s.get('speedup')]
                if spds:
                    max_speedups.append(max(spds))
        metrics[f"{prefix}/pass@{k}"] = passes / len(eligible)
        metrics[f"{prefix}/pass@{k}_num_prompts"] = len(eligible)
        if max_speedups:
            metrics[f"{prefix}/pass@{k}_avg_max_speedup"] = sum(max_speedups) / len(max_speedups)
    return metrics


def _compute_pass_at_k(prompt_groups: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Backward-compatible alias: overall pass@K under `pass_at_k/*`."""
    return _pass_at_k_for_groups(prompt_groups, prefix="pass_at_k")


def _log_batch_aggregates(
    rewards: List[float],
    per_sample: List[Dict[str, Any]],
    batch_size: int,
    data_sources: Optional[List[str]] = None,
    extra_infos: Optional[List[Dict[str, Any]]] = None,
):
    """Compute and log group-wise / full-batch metrics.

    Called once per BatchRewardManager.compute_score call (full batch visible).
    Logs reward distribution, speedup, correctness, fresh-vs-mutated split,
    pass@K (overall + per_source + per_category), per-category aggregates,
    failure breakdown (overall + per_category).
    """
    if batch_size == 0:
        return

    if data_sources is None:
        data_sources = [""] * batch_size
    if extra_infos is None:
        extra_infos = [{}] * batch_size

    # Group samples by example_name (= prompt UID) for pass@K
    prompt_groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, info in enumerate(per_sample):
        rec = dict(info)
        rec['reward_score'] = float(rewards[i])
        rec['data_source'] = data_sources[i] if i < len(data_sources) else ''
        prompt_groups.setdefault(info['example_name'], []).append(rec)

    # ---- Per-sample arrays ----
    is_infra = [info.get('is_infra_failure', False) for info in per_sample]
    is_correct = [info.get('is_correct', False) for info in per_sample]
    is_mutated = [info.get('is_mutated', False) for info in per_sample]
    speedups = [info.get('speedup') for info in per_sample
                if info.get('is_correct') and info.get('speedup')]
    abs_speedups = [info.get('absolute_speedup') for info in per_sample
                    if info.get('is_correct') and info.get('absolute_speedup')]
    fresh_correct = sum(1 for c, m in zip(is_correct, is_mutated) if c and not m)
    mutated_correct = sum(1 for c, m in zip(is_correct, is_mutated) if c and m)
    fresh_rewards = [r for r, m, infra in zip(rewards, is_mutated, is_infra)
                     if not m and not infra]
    mutated_rewards = [r for r, m, infra in zip(rewards, is_mutated, is_infra)
                       if m and not infra]
    fresh_speedups = [info.get('speedup') for info in per_sample
                      if info.get('is_correct') and info.get('speedup') and not info.get('is_mutated')]
    mutated_speedups = [info.get('speedup') for info in per_sample
                        if info.get('is_correct') and info.get('speedup') and info.get('is_mutated')]
    frontier_advances = sum(1 for info in per_sample if info.get('frontier_advance'))
    non_infra_rewards = [r for r, infra in zip(rewards, is_infra) if not infra]
    eval_count = sum(1 for x in is_infra if not x)
    correct_count = sum(is_correct)
    correctness_rate = correct_count / eval_count if eval_count > 0 else 0.0

    # Failure breakdown (per-sample reasons)
    reasons = [info.get('reason') for info in per_sample]
    is_copied = [info.get('is_copied', False) for info in per_sample]
    is_non_nki = [info.get('is_non_nki', False) for info in per_sample]
    is_compile_fail = [r == FAILURE_COMPILE for r in reasons]
    is_accuracy_fail = [r in (FAILURE_ACCURACY, SUCCESS_COMPILE_ONLY) for r in reasons]
    is_kernel_extract_fail = [r == FAILURE_KERNEL_EXTRACTION for r in reasons]
    is_missing_orig = [r == FAILURE_MISSING_ORIGINAL_CODE for r in reasons]

    counts = {
        "correct": correct_count,
        "copied": sum(is_copied),
        "non_nki": sum(is_non_nki),
        "infra": sum(is_infra),
        "compile_fail": sum(is_compile_fail),
        "accuracy_fail": sum(is_accuracy_fail),
        "kernel_extraction_fail": sum(is_kernel_extract_fail),
        "missing_original_code": sum(is_missing_orig),
    }

    metrics: Dict[str, Any] = {
        "reward/total_samples": batch_size,
        "reward/correctness_rate": correctness_rate,
        "reward/correct_count": correct_count,
        "reward/evaluated_count": eval_count,
        "reward/infra_fail_count": sum(is_infra),
        "reward/non_nki_kernel_count": counts["non_nki"],
        "reward/mutated_sample_rate": sum(is_mutated) / batch_size,
        "reward/frontier_advances": frontier_advances,
    }
    if non_infra_rewards:
        metrics["reward/mean"] = sum(non_infra_rewards) / len(non_infra_rewards)
        metrics["reward/min"] = min(non_infra_rewards)
        metrics["reward/max"] = max(non_infra_rewards)
        non_infra_sorted = sorted(non_infra_rewards)
        metrics["reward/median"] = non_infra_sorted[len(non_infra_sorted) // 2]
    if speedups:
        sorted_spd = sorted(speedups)
        metrics["speedup/mean"] = sum(speedups) / len(speedups)
        metrics["speedup/median"] = sorted_spd[len(sorted_spd) // 2]
        metrics["speedup/max"] = max(speedups)
        if len(sorted_spd) >= 10:
            metrics["speedup/p90"] = sorted_spd[int(len(sorted_spd) * 0.9)]
        metrics["speedup/num_correct_with_speedup"] = len(speedups)
    # Absolute speedup: vs original baseline (before replay-buffer mutation)
    if abs_speedups:
        metrics["reward/avg_abs_speedup"] = sum(abs_speedups) / len(abs_speedups)
        metrics["reward/max_abs_speedup"] = max(abs_speedups)

    # Fresh vs mutated split — counts, avg reward, avg/max speedup
    metrics["fresh/correct_count"] = fresh_correct
    metrics["mutated/correct_count"] = mutated_correct
    if fresh_rewards:
        metrics["fresh/avg_reward"] = sum(fresh_rewards) / len(fresh_rewards)
    if mutated_rewards:
        metrics["mutated/avg_reward"] = sum(mutated_rewards) / len(mutated_rewards)
    if fresh_speedups:
        metrics["fresh/avg_speedup"] = sum(fresh_speedups) / len(fresh_speedups)
        metrics["fresh/max_speedup"] = max(fresh_speedups)
    if mutated_speedups:
        metrics["mutated/avg_speedup"] = sum(mutated_speedups) / len(mutated_speedups)
        metrics["mutated/max_speedup"] = max(mutated_speedups)

    # Failure breakdown (overall counts)
    for name, val in counts.items():
        metrics[f"breakdown/{name}_count"] = val

    # Per-category aggregates (incl. avg_reward + per-category failure breakdown)
    cat_groups: Dict[str, List[int]] = {}
    for i, info in enumerate(per_sample):
        cat_groups.setdefault(info.get('category', 'other'), []).append(i)
    for cat, idxs in cat_groups.items():
        cat_correct_n = sum(per_sample[i].get('is_correct', False) for i in idxs)
        cat_infra_n = sum(per_sample[i].get('is_infra_failure', False) for i in idxs)
        cat_eval_n = len(idxs) - cat_infra_n
        cat_speedups = [per_sample[i].get('speedup') for i in idxs
                        if per_sample[i].get('is_correct') and per_sample[i].get('speedup')]
        cat_eval_rewards = [rewards[i] for i in idxs if not per_sample[i].get('is_infra_failure')]
        metrics[f"category/{cat}/total"] = len(idxs)
        metrics[f"category/{cat}/correct"] = cat_correct_n
        metrics[f"category/{cat}/infra_fail"] = cat_infra_n
        if cat_eval_n > 0:
            metrics[f"category/{cat}/correctness_rate"] = cat_correct_n / cat_eval_n
        if cat_eval_rewards:
            metrics[f"category/{cat}/avg_reward"] = sum(cat_eval_rewards) / len(cat_eval_rewards)
        if cat_speedups:
            metrics[f"category/{cat}/avg_speedup"] = sum(cat_speedups) / len(cat_speedups)
            metrics[f"category/{cat}/max_speedup"] = max(cat_speedups)
        # Per-category failure breakdown
        metrics[f"category/{cat}/copied_count"] = sum(is_copied[i] for i in idxs)
        metrics[f"category/{cat}/non_nki_count"] = sum(is_non_nki[i] for i in idxs)
        metrics[f"category/{cat}/compile_fail_count"] = sum(is_compile_fail[i] for i in idxs)
        metrics[f"category/{cat}/accuracy_fail_count"] = sum(is_accuracy_fail[i] for i in idxs)
        metrics[f"category/{cat}/kernel_extract_fail_count"] = sum(is_kernel_extract_fail[i] for i in idxs)

    # Pass@K — overall
    metrics.update(_pass_at_k_for_groups(prompt_groups, prefix="pass_at_k"))

    # Pass@K — per source (data_source, e.g., "nkibench/<name>")
    source_groups: Dict[str, Dict[str, List[Dict]]] = {}
    for example_name, samples in prompt_groups.items():
        # All samples in a group share the same data_source
        ds = samples[0].get('data_source', '') if samples else ''
        source_groups.setdefault(ds, {})[example_name] = samples
    for ds, sub_groups in source_groups.items():
        if not ds:
            continue
        # Sanitize key (replace / with _ since wandb treats / as namespace)
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', ds)[:80]
        metrics.update(_pass_at_k_for_groups(sub_groups, prefix=f"pass_at_k/source/{sanitized}"))

    # Pass@K — per category (difficulty bucket)
    cat_prompt_groups: Dict[str, Dict[str, List[Dict]]] = {}
    for example_name, samples in prompt_groups.items():
        cat = samples[0].get('category', 'other') if samples else 'other'
        cat_prompt_groups.setdefault(cat, {})[example_name] = samples
    for cat, sub_groups in cat_prompt_groups.items():
        metrics.update(_pass_at_k_for_groups(sub_groups, prefix=f"pass_at_k/category/{cat}"))

    # Console summary headline
    pass_log = ", ".join(
        f"{k.split('/')[-1]}={v:.2%}"
        for k, v in metrics.items()
        if k.startswith("pass_at_k/pass@") and not k.endswith("_num_prompts") and not k.endswith("_avg_max_speedup")
    )
    headline_parts = [
        f"correctness={correctness_rate:.1%} ({correct_count}/{eval_count})",
    ]
    if speedups:
        headline_parts.append(f"median_spd={metrics['speedup/median']:.2f}x")
        headline_parts.append(f"max_spd={metrics['speedup/max']:.2f}x")
    headline_parts.append(f"frontier_advances={frontier_advances}")
    logger.info(
        f"=== REWARD BATCH ({batch_size} samples, {len(prompt_groups)} prompts): "
        f"{' | '.join(headline_parts)} | pass@K: {pass_log} ==="
    )

    _log_custom_metrics(metrics)


# ============================================
# Replay Buffer
# ============================================
REPLAY_BUFFER_ENABLED = os.environ.get('REPLAY_BUFFER_ENABLED', '0') == '1'
REPLAY_BUFFER_MAX_PER_PROBLEM = int(os.environ.get('REPLAY_BUFFER_MAX_PER_PROBLEM', '10'))
# Harvest bar tracks the BEATS milestone (== "beat the initial seed").
REPLAY_BUFFER_MIN_REWARD = float(os.environ.get('REPLAY_BUFFER_MIN_REWARD', str(BEATS_INITIAL_REWARD)))
REPLAY_BUFFER_P_ORIG = float(os.environ.get('REPLAY_BUFFER_P_ORIG', '0.4'))
REPLAY_BUFFER_P_BEST = float(os.environ.get('REPLAY_BUFFER_P_BEST', '0.4'))
REPLAY_BUFFER_P_INTER = float(os.environ.get('REPLAY_BUFFER_P_INTER', '0.2'))
REPLAY_BUFFER_ANNEAL = os.environ.get('REPLAY_BUFFER_ANNEAL', '1') == '1'
REPLAY_BUFFER_PERSIST_DIR = os.environ.get(
    'REPLAY_BUFFER_PERSIST_DIR',
    os.path.join(os.environ.get('NKIRL_DIR', '/tmp'), 'artifacts', 'replay_buffer')
)
REPLAY_BUFFER_SOFTMAX_TEMP = float(os.environ.get('REPLAY_BUFFER_SOFTMAX_TEMP', '1.0'))


class ReplayBuffer:
    """Per-problem replay buffer for online curriculum amplification."""

    def __init__(self, max_per_problem=REPLAY_BUFFER_MAX_PER_PROBLEM,
                 min_reward=REPLAY_BUFFER_MIN_REWARD, p_orig=REPLAY_BUFFER_P_ORIG,
                 p_best=REPLAY_BUFFER_P_BEST, p_inter=REPLAY_BUFFER_P_INTER,
                 anneal=REPLAY_BUFFER_ANNEAL, softmax_temp=REPLAY_BUFFER_SOFTMAX_TEMP,
                 persist_dir=REPLAY_BUFFER_PERSIST_DIR):
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._max_per_problem = max_per_problem
        self._min_reward = min_reward
        self._p_orig_base = p_orig
        self._p_best_base = p_best
        self._p_inter_base = p_inter
        self._anneal = anneal
        self._softmax_temp = softmax_temp
        self._persist_dir = persist_dir
        self._total_harvested = 0
        self._total_rejected = 0
        self._epoch = 0
        self._total_epochs = 1
        if persist_dir:
            self._try_load(persist_dir)

    def set_training_progress(self, epoch, total_epochs):
        self._epoch = epoch
        self._total_epochs = max(total_epochs, 1)

    @property
    def progress(self):
        return min(self._epoch / self._total_epochs, 1.0)

    def get_sampling_probs(self):
        if not self._anneal:
            return self._p_orig_base, self._p_best_base, self._p_inter_base
        progress = self.progress
        p_inter = self._p_inter_base
        p_orig = self._p_orig_base * (1.0 - progress)
        p_best = 1.0 - p_orig - p_inter
        return p_orig, p_best, p_inter

    def add(self, problem_id, kernel_code, latency, reward, initial_code="",
            helper_functions="", metadata=None):
        if reward < self._min_reward or not kernel_code or latency is None or latency <= 0:
            self._total_rejected += 1
            return False
        code_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:16]
        entry = {"kernel_code": kernel_code, "latency": latency, "reward": reward,
                 "code_hash": code_hash, "initial_code": initial_code,
                 "helper_functions": helper_functions, "harvested_epoch": self._epoch,
                 "metadata": metadata or {}}
        with self._lock:
            if problem_id not in self._buffer:
                self._buffer[problem_id] = []
            buf = self._buffer[problem_id]
            if any(e["code_hash"] == code_hash for e in buf):
                self._total_rejected += 1
                return False
            buf.append(entry)
            buf.sort(key=lambda x: x["latency"])
            if len(buf) > self._max_per_problem:
                buf.pop()
            self._total_harvested += 1
            return True

    def sample(self, problem_id):
        p_orig, p_best, p_inter = self.get_sampling_probs()
        with self._lock:
            buf = self._buffer.get(problem_id, [])
            if not buf:
                return None
            roll = random.random()
            if roll < p_orig:
                return None
            if roll < p_orig + p_best:
                return dict(buf[0])
            if len(buf) <= 1:
                return dict(buf[0])
            candidates = buf[1:]
            if not candidates:
                return dict(buf[0])
            latencies = [c["latency"] for c in candidates]
            neg_lats = [-lat / self._softmax_temp for lat in latencies]
            max_neg = max(neg_lats)
            exp_weights = [math.exp(nl - max_neg) for nl in neg_lats]
            total_weight = sum(exp_weights)
            probs = [w / total_weight for w in exp_weights]
            chosen_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            return dict(candidates[chosen_idx])

    def get_best(self, problem_id):
        with self._lock:
            buf = self._buffer.get(problem_id, [])
            return dict(buf[0]) if buf else None

    def get_stats(self):
        with self._lock:
            num_problems = len(self._buffer)
            total_entries = sum(len(b) for b in self._buffer.values())
            avg_per_problem = total_entries / num_problems if num_problems > 0 else 0
            all_latencies = [e["latency"] for b in self._buffer.values() for e in b]
            return {
                "num_problems": num_problems, "total_entries": total_entries,
                "avg_per_problem": avg_per_problem, "total_harvested": self._total_harvested,
                "total_rejected": self._total_rejected,
                "min_latency": min(all_latencies) if all_latencies else 0,
                "max_latency": max(all_latencies) if all_latencies else 0,
                "epoch": self._epoch, "progress": self.progress,
                "p_orig": self.get_sampling_probs()[0], "p_best": self.get_sampling_probs()[1],
                "p_inter": self.get_sampling_probs()[2],
            }

    def _apply_mutation(self, extra_info, sampled):
        mutated = dict(extra_info)
        if "_original_initial_kernel_total_time" not in mutated:
            mutated["_original_initial_kernel_total_time"] = mutated.get("initial_kernel_total_time")
        mutated["initial_kernel"] = sampled["kernel_code"]
        mutated["initial_kernel_total_time"] = sampled["latency"]
        mutated["_replay_buffer_mutated"] = True
        # Judge dup-check compares against current seed, not frozen original.
        mutated["_mutated_seed_kernel"] = sampled["kernel_code"]
        helper = mutated.get("helper_functions", "")
        if helper:
            mutated["initial_code"] = helper + "\n\n" + sampled["kernel_code"]
        elif sampled.get("initial_code"):
            mutated["initial_code"] = replace_kernel_in_file(sampled["initial_code"], sampled["kernel_code"])
        return mutated

    def mutate_extra_info(self, problem_id, extra_info):
        sampled = self.sample(problem_id)
        if sampled is None:
            return extra_info
        return self._apply_mutation(extra_info, sampled)

    def mutate_batch_extra_infos(self, problem_ids, extra_infos):
        assert len(problem_ids) == len(extra_infos)
        mutation_cache = {}
        for pid in problem_ids:
            if pid not in mutation_cache:
                mutation_cache[pid] = self.sample(pid)
        mutated_infos = []
        for pid, extra_info in zip(problem_ids, extra_infos):
            sampled = mutation_cache[pid]
            if sampled is None:
                mutated_infos.append(extra_info)
            else:
                mutated_infos.append(self._apply_mutation(extra_info, sampled))
        return mutated_infos

    def persist(self):
        if not self._persist_dir:
            return
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            save_data = {}
            with self._lock:
                for problem_id, entries in self._buffer.items():
                    save_data[problem_id] = [
                        {"kernel_code": e["kernel_code"], "latency": e["latency"],
                         "reward": e["reward"], "code_hash": e["code_hash"],
                         "harvested_epoch": e.get("harvested_epoch", 0),
                         "metadata": e.get("metadata", {})}
                        for e in entries
                    ]
            filepath = os.path.join(self._persist_dir, "replay_buffer.json")
            with open(filepath, 'w') as f:
                json.dump({"buffer": save_data, "stats": {
                    "total_harvested": self._total_harvested,
                    "total_rejected": self._total_rejected,
                    "epoch": self._epoch, "total_epochs": self._total_epochs,
                }}, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist replay buffer: {e}")

    def _try_load(self, persist_dir):
        filepath = os.path.join(persist_dir, "replay_buffer.json")
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            with self._lock:
                self._buffer = {}
                for problem_id, entries in data.get("buffer", {}).items():
                    self._buffer[problem_id] = [
                        {"kernel_code": e["kernel_code"], "latency": e["latency"],
                         "reward": e["reward"], "code_hash": e["code_hash"],
                         "initial_code": "", "helper_functions": "",
                         "harvested_epoch": e.get("harvested_epoch", 0),
                         "metadata": e.get("metadata", {})}
                        for e in entries
                    ]
                stats = data.get("stats", {})
                self._total_harvested = stats.get("total_harvested", 0)
                self._total_rejected = stats.get("total_rejected", 0)
                self._epoch = stats.get("epoch", 0)
                self._total_epochs = stats.get("total_epochs", 1)
            total_entries = sum(len(b) for b in self._buffer.values())
            logger.info(f"Loaded replay buffer from {filepath}: {len(self._buffer)} problems, {total_entries} entries")
        except Exception as e:
            logger.warning(f"Failed to load replay buffer: {e}")


_replay_buffer: Optional[ReplayBuffer] = None


def get_replay_buffer() -> ReplayBuffer:
    global _replay_buffer
    if _replay_buffer is None:
        _replay_buffer = ReplayBuffer()
        logger.info(f"Initialized replay buffer (enabled={REPLAY_BUFFER_ENABLED})")
    return _replay_buffer


# ============================================
# Main batch reward function (called by veRL)
# ============================================

def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[Any],
    extra_infos: List[Dict[str, Any]],
    use_judge: bool = False,
    judge_concurrency: int = 4,
    judge_reward_on_invalid: float = -0.5,
    reward_mode: str = DEFAULT_REWARD_MODE,
    speedup_reward_max: float = DEFAULT_SPEEDUP_REWARD_MAX,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Batch reward computation. Returns list of dicts {"score": float, ...}.

    Pipeline per call:
        (optional) replay-buffer mutation     ← only when REPLAY_BUFFER_ENABLED=1
        kernel extraction + src_code build
        batched profiling (rollouts + seed kernels) over local aiohttp
        per-sample reward derivation
        (optional) LLM-judge gating            ← only when use_judge=True
        (optional) replay-buffer harvest       ← only when REPLAY_BUFFER_ENABLED=1
                                                  and split != "val"

    Default path (replay buffer off, judge off) is just: extract → profile →
    score → return. Profiler transport is local aiohttp (round-robin across
    SSH-tunneled workers); no client-side accuracy override. Pass@K and other
    full-batch aggregates are computed at the trainer level, where the
    gathered batch is visible.

    Two decisions here are env vars rather than reward_kwargs, because the agent
    loops make the SAME two decisions and only an env var reaches all three call
    sites: WHICH kernel gets graded (KERNEL_EXTRACT_STRATEGY, see
    _KERNEL_EXTRACT_STRATEGIES) and WHICH input shape it is measured on
    (TEST_CONFIG_MODE / TEST_CONFIG_INDICES, see _resolve_test_config_mode).
    """
    batch_size = len(solution_strs)
    extract_strategy = _resolve_extract_strategy()
    test_config_mode = _resolve_test_config_mode()
    test_config_indices = _resolve_test_config_indices()
    # Once per worker process per setting: which rules are actually in force is
    # invisible otherwise, and the extraction strategy alone changes which kernel
    # gets graded.
    _knobs = (extract_strategy, test_config_mode, tuple(test_config_indices or ()))
    if _knobs not in _STRATEGY_LOGGED:
        _STRATEGY_LOGGED.add(_knobs)
        logger.info(
            f"Kernel extraction strategy: {extract_strategy} | "
            f"test config: mode={test_config_mode} "
            f"indices={test_config_indices if test_config_indices else 'all'}"
        )

    if extra_infos is None:
        extra_infos = [{}] * batch_size
    else:
        extra_infos = [_parse_json_if_string(e) for e in extra_infos]
    while len(extra_infos) < batch_size:
        extra_infos.append({})

    if ground_truths is None:
        ground_truths = [{}] * batch_size
    else:
        ground_truths = [_parse_json_if_string(g) for g in ground_truths]
    while len(ground_truths) < batch_size:
        ground_truths.append({})

    # Replay buffer mutation (must happen before building configs).
    # Train-only: val rows (extra_info["split"] == "val") are passed through
    # untouched so they don't get harvested into / sampled from the buffer
    # and contaminate val metrics across training steps.
    if REPLAY_BUFFER_ENABLED:
        rb = get_replay_buffer()
        problem_ids = [
            (extra_infos[i].get('example_name') or extra_infos[i].get('kernel_name') or f'unknown_{i}')
            for i in range(batch_size)
        ]
        train_idxs = [i for i in range(batch_size)
                      if extra_infos[i].get("split") != "val"]
        if train_idxs:
            train_ids = [problem_ids[i] for i in train_idxs]
            train_infos = [extra_infos[i] for i in train_idxs]
            mutated_train = rb.mutate_batch_extra_infos(train_ids, train_infos)
            for j, i in enumerate(train_idxs):
                extra_infos[i] = mutated_train[j]
        mutated_in_batch = sum(1 for ei in extra_infos if ei.get('_replay_buffer_mutated', False))
        _rb_stats = rb.get_stats()
        logger.info(
            f"Replay buffer mutation: {mutated_in_batch}/{batch_size} samples mutated "
            f"(buffer has {_rb_stats['num_problems']} problems, "
            f"{_rb_stats['total_entries']} entries)"
        )
        # Optional diagnostic hook: confirm the replay buffer actually engaged in
        # this reward worker. Only runs when REPLAY_BUFFER_ENABLED — off by
        # default → zero impact. verify_once: one line per worker (config→worker-
        # read + initial stats). verify_event: per reward call (mutated count for
        # this call).
        try:
            from nkirl_verify_log import verify_once, verify_event
            verify_once(
                "replay_buffer",
                detail=f"ENABLED buffer_problems={_rb_stats['num_problems']} "
                       f"buffer_entries={_rb_stats['total_entries']}",
            )
            verify_event(
                "replay_buffer_mutate",
                detail=f"mutated={mutated_in_batch}/{batch_size} "
                       f"buffer_problems={_rb_stats['num_problems']} "
                       f"buffer_entries={_rb_stats['total_entries']}",
            )
        except Exception:
            pass

    # Build profiling configurations
    profiling_configs = []
    extracted_kernels = []
    original_latencies = []
    original_codes = []
    valid_indices = []
    # Per-sample chosen config (dict, not list: the loop skips samples).
    chosen_configs: Dict[int, Dict[str, Any]] = {}

    initial_kernel_to_profile: Dict[str, Tuple[str, Dict]] = {}

    for i in range(batch_size):
        extra_info = extra_infos[i]
        ground_truth = ground_truths[i]

        kernel_code = extract_kernel_code_from_response(solution_strs[i], extract_strategy)
        extracted_kernels.append(kernel_code)

        original_code = _get_initial_code(extra_info, ground_truth)
        original_codes.append(original_code)

        original_latency = _get_initial_latency(extra_info, ground_truth)
        original_latencies.append(original_latency)

        if kernel_code is None:
            logger.warning(f"Sample {i}: Could not extract kernel code from response")
            continue
        if not original_code:
            logger.warning(f"Sample {i}: No original_code provided, skipping")
            continue

        # src_code = full original kernel file with the model's kernel
        # swapped in. Must have imports (`import nki...`) — a bare snippet
        # breaks the profiler with `NameError: name 'nki'`. The v2 dataset
        # stores this as `seed_kernel`. Both `seed_kernel` and
        # `initial_code` are required at reward time — raise loudly if
        # missing rather than silently producing speedup=None.
        seed_kernel_file = extra_info.get('seed_kernel')
        if not seed_kernel_file:
            raise RuntimeError(
                f"Sample {i} ({extra_info.get('kernel_name', '?')}): "
                "extra_info has no `seed_kernel` (full kernel file with "
                "imports). Rebuild the dataset with the current "
                "create_rl_dataset_nkibench.py."
            )
        if not original_code:
            raise RuntimeError(
                f"Sample {i} ({extra_info.get('kernel_name', '?')}): "
                "extra_info has no `initial_code` (benchmark + helpers). "
                "Rebuild the dataset."
            )
        src_code = replace_kernel_in_file(seed_kernel_file, kernel_code)
        original_code_for_profiler = original_code

        # Pick one TEST_CONFIGS entry for this rollout, using the mode/indices
        # resolved once above from TEST_CONFIG_MODE / TEST_CONFIG_INDICES — the
        # same env vars select_test_config_for_rollout hands the agent loops, so
        # the shape the model got feedback on is the shape it is graded on.
        # The returned entry carries both the input-shape config and the
        # cached seed baseline (baseline_p50_s) when the dataset was built
        # with --profiling_results.
        test_configs = _parse_test_configs(extra_info)
        chosen_entry = _select_test_config(
            test_configs, mode=test_config_mode, indices=test_config_indices,
        )
        chosen_config = chosen_entry.get("config") or {}
        chosen_configs[i] = chosen_config
        # Use the on-device baseline only — speedup divides custom on-device by
        # this, so they must be the same metric. The dataset's p50 baseline is
        # NOT used for speedup. If on-device baseline is absent, leave it unset
        # and the seed gets live-profiled (on-device) below.
        chosen_baseline_on_device_s = chosen_entry.get("baseline_on_device_s")
        is_mutated_for_baseline = extra_info.get('_replay_buffer_mutated', False)
        if chosen_baseline_on_device_s and chosen_baseline_on_device_s > 0 and not is_mutated_for_baseline:
            original_latencies[i] = chosen_baseline_on_device_s

        # src_code stays kernel-only. The profiler choke point
        # (_profile_single_kernel_local) folds initial_code into src_code for the
        # REMOTE backend (which resolves get_inputs only from src_code) and leaves it
        # kernel-only for LOCAL — one place, so the reward and in-rollout tool-feedback
        # paths compose remote requests identically. See the choke point for the
        # option-C TODO on the duplicated-helper fold. (Was pre-folded here; for LOCAL
        # the else-branch already passed kernel-only src_code, so local is unchanged.)
        config = {
            "src_code": src_code,
            "initial_code": original_code_for_profiler,
            "benchmark_code": extra_info.get("benchmark_code"),
            "kernel_name": "kernel",
            "get_inputs_kwargs": chosen_config,
        }

        profiling_configs.append(config)
        valid_indices.append(i)

        # Initial kernel re-profiling: skip if mutated, already cached, OR the
        # dataset provided an on-device baseline. Otherwise live-profile the seed
        # (result cached on-device) — this is the primary path when the dataset
        # ships no baselines.
        example_name = extra_info.get('example_name') or extra_info.get('kernel_name', '')
        is_mutated = extra_info.get('_replay_buffer_mutated', False)
        if example_name and not is_mutated and not (chosen_baseline_on_device_s and chosen_baseline_on_device_s > 0):
            already_cached = _baseline_cache_get(example_name) is not None
            if not already_cached and example_name not in initial_kernel_to_profile:
                # Re-profile the SEED kernel: reuse the full kernel file
                # (`seed_kernel_file` was validated above so it's guaranteed
                # non-empty here).
                initial_kernel_to_profile[example_name] = (
                    seed_kernel_file, original_code, chosen_config,
                    extra_info.get("benchmark_code"),
                )

    # Build initial kernel profiling configs
    initial_profile_keys = list(initial_kernel_to_profile.keys())
    initial_profiling_configs = []
    for example_name in initial_profile_keys:
        seed_src, init_ref_code, init_kwargs, init_bench_code = \
            initial_kernel_to_profile[example_name]
        # Kernel-only src_code; the profiler choke point folds the benchmark in for
        # the REMOTE backend (prefers benchmark_code, falls back to init_ref_code;
        # same single-place remote composition as the rollout config above),
        # kernel-only for LOCAL (unchanged from before).
        initial_profiling_configs.append({
            "src_code": seed_src,
            "initial_code": init_ref_code,  # benchmark file for reference
            "benchmark_code": init_bench_code,
            "kernel_name": "kernel",
            "get_inputs_kwargs": init_kwargs,
        })

    # Combine into one batch
    all_configs = profiling_configs + initial_profiling_configs
    n_rollout = len(profiling_configs)
    n_initial = len(initial_profiling_configs)

    if all_configs:
        logger.info(
            f"Sending {n_rollout} rollout + {n_initial} initial kernels to local profiler "
            f"(total={len(all_configs)}, cached_initial={len(_initial_latency_cache)})"
        )
        # The remote profiler backend is a beta4 backend: a beta3 reward request
        # trips InputSpecError + NKIFrontend.compile signature skew and profiles as
        # ~all-COMPILE_FAILED (see run 604). v2 sets nki_version=beta4 +
        # server_default_tolerance=True (matching the frontend the corpus was
        # validated on) then defers to the same transport. Local trn2 workers are
        # beta3-compatible, so keep v1 there unchanged (reward parity).
        if PROFILER_REMOTE_URL:
            # Grep-able proof of which reward dispatch path actually ran (line-number
            # forensics are the only alternative — see run 605). Confirms remote =
            # v2/beta4 request + get_inputs folded into src_code.
            logger.info(
                f"[NKIRL_VERIFY] component=profiler_dispatch applied=remote "
                f"detail=v2/beta4+src_code_folded(get_inputs) url={PROFILER_REMOTE_URL} "
                f"n_configs={len(all_configs)}"
            )
            all_results = send_batch_profiler_request_v2(all_configs)
        else:
            logger.info(
                f"[NKIRL_VERIFY] component=profiler_dispatch applied=local "
                f"detail=v1/beta3+kernel_only_src_code n_configs={len(all_configs)}"
            )
            all_results = send_batch_profiler_request(all_configs)

        if len(all_results) != len(all_configs):
            logger.error(
                f"CRITICAL: profiler returned {len(all_results)} results "
                f"for {len(all_configs)} configs"
            )
            all_results.extend([{'error': 'Result dropped', 'total_time': None}] * (len(all_configs) - len(all_results)))

        profiling_results = all_results[:n_rollout]
        initial_results = all_results[n_rollout:]
    else:
        profiling_results = []
        initial_results = []

    # Cache initial kernel on-device latencies
    for j, example_name in enumerate(initial_profile_keys):
        if j >= len(initial_results):
            break
        res = initial_results[j]
        on_device_s = res.get('total_time')
        if on_device_s and on_device_s > 0 and not _is_infra_failure(res):
            _baseline_cache_put(example_name, on_device_s)
            logger.info(f"Cached initial kernel on-device for '{example_name}': {on_device_s * 1e6:.1f}µs")
        else:
            logger.warning(f"Initial kernel profiling failed for '{example_name}': {str(res.get('error', 'unknown'))[:100]}")

    # Override original_latencies with cached p50s where available
    for i in range(batch_size):
        extra_info = extra_infos[i]
        example_name = extra_info.get('example_name') or extra_info.get('kernel_name', '')
        is_mutated = extra_info.get('_replay_buffer_mutated', False)
        if example_name and not is_mutated:
            cached_on_device = _baseline_cache_get(example_name)
            if cached_on_device:
                original_latencies[i] = cached_on_device

    # Build results map
    results_map = {}
    for result_idx, batch_idx in enumerate(valid_indices):
        if result_idx < len(profiling_results):
            results_map[batch_idx] = profiling_results[result_idx]
        else:
            results_map[batch_idx] = {
                'error': f'Result dropped: profiler returned {len(profiling_results)} for {len(valid_indices)}',
                'total_time': None,
            }

    # ------------------------------------------------------------------
    # Per-sample reward computation
    # ------------------------------------------------------------------
    # No slice-level aggregation here. Each sample gets a reward and a set of
    # per-sample fields (reason, speedup, category, etc.) that the trainer
    # collects via `non_tensor_batch` and aggregates over the full gathered
    # batch (at the trainer level).
    rewards: List[float] = []
    per_sample: List[Dict[str, Any]] = []
    best_kernels: Dict[str, Dict[str, Any]] = {}

    for i in range(batch_size):
        initial_kernel_for_check = extra_infos[i].get('initial_kernel', '')
        ds_i = data_sources[i] if i < len(data_sources) else ''
        en_i = extra_infos[i].get('example_name') or extra_infos[i].get('kernel_name', f'unknown_{i}')
        cat_i = _classify_source_category(ds_i, en_i, extra_infos[i])
        is_mutated_sample = bool(extra_infos[i].get('_replay_buffer_mutated', False))

        info = {
            'example_name': en_i,
            'category': cat_i,
            'is_infra_failure': False,
            'reason': None,
            'speedup': None,
            'absolute_speedup': None,
            'abs_reward': None,  # set on the correct-rollout path below; defaults to `score` afterwards
            'is_mutated': is_mutated_sample,
            'is_correct': False,
            'is_copied': False,
            'is_non_nki': False,
            'frontier_advance': False,
            # Did the model FINISH the kernel being graded, or did the response
            # budget cut it off mid-emit? Observation only — it changes no reward
            # and no extraction (see kernel_was_finalized for why both cases must
            # stay in the reward). It exists so a truncated rollout stops being
            # indistinguishable from a genuinely broken kernel in the metrics.
            'graded_incomplete_kernel': (
                extracted_kernels[i] is not None
                and not kernel_was_finalized(solution_strs[i], extracted_kernels[i])
            ),
        }

        # Cheating detection: copied kernel
        if extracted_kernels[i] is not None and initial_kernel_for_check and _is_copied_kernel(extracted_kernels[i], initial_kernel_for_check):
            reward = _fail_reward(COPIED_KERNEL_REWARD, reward_mode)
            info['reason'] = FAILURE_COPIED_KERNEL
            info['is_copied'] = True
            logger.info(f"Sample {i}: reward={reward:.2f} | reason=COPIED_KERNEL")
            rewards.append(reward)
            per_sample.append(info)
            continue

        # Cheating detection: non-NKI kernel
        if extracted_kernels[i] is not None and _is_non_nki_kernel(extracted_kernels[i]):
            reward = _fail_reward(NON_NKI_KERNEL_REWARD, reward_mode)
            info['reason'] = FAILURE_NON_NKI_KERNEL
            info['is_non_nki'] = True
            logger.info(f"Sample {i}: reward={reward:.2f} | reason=NON_NKI_KERNEL (uses numpy/torch, no NKI APIs)")
            rewards.append(reward)
            per_sample.append(info)
            continue

        if i in results_map:
            sft_ref_latency = None
            gt_i = ground_truths[i]
            if gt_i:
                sft_ref_val = gt_i.get('optimized_kernel_total_time')
                if sft_ref_val is not None:
                    try:
                        sft_ref_latency = float(sft_ref_val)
                        if sft_ref_latency <= 0:
                            sft_ref_latency = None
                    except (ValueError, TypeError):
                        sft_ref_latency = None

            reward = compute_single_reward(
                extracted_kernels[i], original_codes[i], original_latencies[i],
                results_map[i], sample_idx=i, initial_kernel=initial_kernel_for_check,
                sft_reference_latency=sft_ref_latency,
                reward_mode=reward_mode, speedup_reward_max=speedup_reward_max,
            )
            reason = classify_failure_reason(results_map[i])
            info['reason'] = reason

            # Reward-stage profiler round-trip wall-clock (obs-only). Set for EVERY
            # profiled sample — incl. compile/accuracy fails, which still consume
            # profiler time — so the wandb aggregate reflects true profiler load, not
            # just correct kernels. None on the rare infra/submit-error paths that
            # return before the poll completes (nkirl_metrics filters those out).
            info['profiler_roundtrip_s'] = results_map[i].get('roundtrip_s')
            info['profiler_poll_wait_s'] = results_map[i].get('poll_wait_s')
            info['profiler_queue_s'] = results_map[i].get('queue_s')

            if reason in (FAILURE_TIMEOUT, FAILURE_HTTP_ERROR, FAILURE_INFRA, FAILURE_RESULTS_DROPPED):
                info['is_infra_failure'] = True

            if reason == SUCCESS_CORRECT:
                info['is_correct'] = True
                # Perf metrics only meaningful for correct kernels (obs-only).
                info['mfu'] = results_map[i].get("mfu")
                info['mbu'] = results_map[i].get("mbu")
                info['dma_active_pct'] = results_map[i].get("dma_active_pct")
                info['hbm_bytes'] = results_map[i].get("hbm_bytes")
                info['cycles'] = results_map[i].get("cycles")
                info['arith_intensity'] = results_map[i].get("arith_intensity")
                info['profiler_raw'] = results_map[i].get("profiler_raw")
                total_time = results_map[i].get("total_time")
                orig_lat = original_latencies[i]
                # Absolute ON-DEVICE latencies (seconds), obs-only. `total_time`
                # here is on_device_time_ns/1e9 (see _local_response_to_reward_format)
                # — the SAME metric the speedup numerator/denominator use, NOT E2E
                # total time. Named on_device_* to avoid the total_time confusion.
                # Not used by the reward; the offline eval (src/pass_at_k_eval.py)
                # reads them to render exact latency strings in multi-turn feedback,
                # matching the agent-loop template. Harmless for training.
                info['kernel_on_device_s'] = total_time
                info['original_on_device_s'] = orig_lat
                if total_time and orig_lat and total_time > 0 and orig_lat > 0:
                    spd = orig_lat / total_time
                    info['speedup'] = spd
                    abs_orig_lat = extra_infos[i].get('_original_initial_kernel_total_time')
                    info['absolute_speedup'] = (abs_orig_lat / total_time) if (abs_orig_lat and abs_orig_lat > 0) else spd
                    # Reward measured against the UN-MUTATED original-seed baseline.
                    # When replay buffer mutates the seed, `reward`/`score` (above)
                    # uses the mutated baseline (= harder over time, on purpose).
                    # `abs_reward` keeps the original baseline so we can see
                    # absolute progress vs the trivial reference.
                    info['abs_reward'] = _reward_from_correct_speedup(
                        total_time,
                        abs_orig_lat if (abs_orig_lat and abs_orig_lat > 0) else orig_lat,
                        reward_mode=reward_mode,
                        speedup_reward_max=speedup_reward_max,
                    )

                    # Track best per (example_name, config): a different input
                    # config is effectively a different problem. Best is by
                    # RUNTIME (lower latency), immune to baseline drift.
                    ds = data_sources[i] if i < len(data_sources) else ""
                    cfg = chosen_configs.get(i, {})
                    bk_key = (en_i, _config_id(cfg))
                    prev_lat = best_kernels.get(bk_key, {}).get("latency_ms")
                    if prev_lat is None or total_time < prev_lat:
                        _maybe_load_global_best()
                        prev_global = _global_best_kernels.get(bk_key, {}).get("latency_ms")
                        if prev_global is None or total_time < prev_global:
                            info['frontier_advance'] = True
                        best_kernels[bk_key] = {
                            "speedup": spd, "reward": reward,
                            "latency_ms": total_time, "original_latency_ms": orig_lat,
                            "config": cfg, "config_hash": _config_hash(cfg),
                            "max_abs_error": results_map[i].get('max_abs_error'),
                            "l2_error": results_map[i].get('l2_error'),
                            "cosine_similarity": results_map[i].get('cosine_similarity'),
                            "kernel_code": extracted_kernels[i],
                            "response": solution_strs[i][:50000],
                            "data_source": ds,
                            "example_name": en_i,
                            # Preserve raw NKIBench provenance fields (from
                            # extra_info) so each best-kernel JSON has full
                            # context: source, difficulty, num_ops, op_categories.
                            "nki_source": extra_infos[i].get('nki_source'),
                            "difficulty": extra_infos[i].get('difficulty'),
                            "num_ops": extra_infos[i].get('num_ops'),
                            "op_categories": extra_infos[i].get('op_categories'),
                        }
        else:
            if extracted_kernels[i] is None:
                info['reason'] = FAILURE_KERNEL_EXTRACTION
                reward = _fail_reward(NO_KERNEL_REWARD, reward_mode)
                logger.info(f"Sample {i}: reward={reward:.2f} | reason=kernel_extraction_failed")
            elif not original_codes[i]:
                info['reason'] = FAILURE_MISSING_ORIGINAL_CODE
                reward = _fail_reward(NO_KERNEL_REWARD, reward_mode)
                logger.info(f"Sample {i}: reward={reward:.2f} | reason=missing_original_code")
            else:
                info['reason'] = FAILURE_RESULTS_DROPPED
                info['is_infra_failure'] = True
                reward = _fail_reward(INFRA_FAILURE_REWARD, reward_mode)
                logger.error(f"Sample {i}: reward={reward:.2f} | reason=RESULTS_DROPPED")

        rewards.append(reward)
        per_sample.append(info)

    # ------------------------------------------------------------------
    # Per-slice side effects (best kernels, batch samples, replay buffer)
    # These are intentionally per-slice; aggregation across the cluster
    # happens via the file artifacts and the replay buffer's persist file.
    # ------------------------------------------------------------------
    _save_best_kernels(best_kernels)
    _save_infra_failures(
        batch_size, rewards, per_sample, extracted_kernels, solution_strs,
        data_sources, extra_infos, original_latencies, results_map,
    )

    # Replay buffer harvest — train rows only.
    if REPLAY_BUFFER_ENABLED:
        rb = get_replay_buffer()
        harvest_count = 0
        for i in range(batch_size):
            if i not in results_map:
                continue
            if extra_infos[i].get("split") == "val":
                continue
            reason = classify_failure_reason(results_map[i])
            if reason != SUCCESS_CORRECT:
                continue
            reward_i = rewards[i] if isinstance(rewards[i], (int, float)) else rewards[i].get("score", 0)
            total_time_i = results_map[i].get("total_time")
            if not total_time_i or total_time_i <= 0 or not extracted_kernels[i]:
                continue
            example_name = extra_infos[i].get('example_name') or extra_infos[i].get('kernel_name', f'unknown_{i}')
            baseline_lat = original_latencies[i] if i < len(original_latencies) else None
            added = rb.add(
                problem_id=example_name, kernel_code=extracted_kernels[i],
                latency=total_time_i, reward=reward_i, initial_code=original_codes[i],
                helper_functions=extra_infos[i].get('helper_functions', ''),
                metadata={"data_source": data_sources[i] if i < len(data_sources) else "",
                          "baseline_latency": baseline_lat,
                          "speedup": baseline_lat / total_time_i if baseline_lat else None},
            )
            if added:
                harvest_count += 1
        rb_stats = rb.get_stats()
        logger.info(f"Replay buffer: harvested={harvest_count}, problems={rb_stats['num_problems']}, entries={rb_stats['total_entries']}")
        rb.persist()
        # Optional diagnostic hook: per reward call, record that the harvest half
        # of the buffer loop ran and how many new kernels entered the buffer this
        # call. Gated by REPLAY_BUFFER_ENABLED (off by default → no verl-parity
        # impact) + NKIRL_VERIFY_LOG inside verify_event.
        try:
            from nkirl_verify_log import verify_event
            verify_event(
                "replay_buffer_harvest",
                detail=f"harvested={harvest_count} "
                       f"buffer_problems={rb_stats['num_problems']} "
                       f"buffer_entries={rb_stats['total_entries']}",
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Group-wise / full-batch aggregation — done at the trainer level.
    #
    # We use async rollout + agent loop, which routes reward computation
    # through veRL's experimental RewardLoopWorker (per-sample, single-data
    # at a time). BatchRewardManager (which delivers the whole batch to
    # `compute_score_batch` in one call) is incompatible with this path —
    # the experimental reward registry doesn't expose "batch", and even if
    # it did, the per-sample async semantics required by multi-turn
    # rollouts wouldn't permit accumulating into a single batch call.
    #
    # The trainer-side metrics hook sees
    # the entire gathered batch via DataProto.non_tensor_batch and computes
    # all the aggregates we used to compute here:
    #   reward/*, speedup/*, pass_at_k/*, category/*, breakdown/*,
    #   fresh/*, mutated/*, frontier_advances
    # The reward function still writes per-sample fields into the returned
    # dict so the trainer has everything it needs.
    #
    # Implementation kept in this file (see _log_batch_aggregates and
    # _compute_pass_at_k / _pass_at_k_for_groups) for reference / when we
    # eventually move to a sync rollout path that supports BatchRewardManager.
    # ------------------------------------------------------------------
    # _log_batch_aggregates(rewards, per_sample, batch_size, data_sources, extra_infos)

    # ------------------------------------------------------------------
    # MULTI-TURN ONLY: grade the rollout's BEST turn, not its last one.
    #
    # WHY THIS LIVES HERE and not in the extraction strategy. `best_speedup` /
    # `latest_verified` pick among `<parameter=nki_code>`/`<tool_response>` pairs
    # in `_extract_verified_tiers`. Fixed multi-turn injects its feedback as a plain
    # `user` turn, so those tags never appear and the picker is EMPTY — selection
    # falls through to the last complete block, so
    # `KERNEL_EXTRACT_STRATEGY=best_speedup` graded the last turn even when that
    # turn regressed or broke a working kernel. (Unchanged by the tier-1 removal:
    # that only reordered the TOOLCALL path, where the picker has real candidates;
    # fixed multi-turn still has none, so this reward-path regrade is still needed.)
    #
    # It cannot be fixed inside the extractor: the agent loop deliberately leaves
    # the final turn unprofiled, so its verdict is unknown until reward time. The
    # comparison is only possible AFTER the final turn is profiled — i.e. right here.
    #
    # SCOPE. Gated on `extra_info["turns"]`, which only the fixed multi-turn agent
    # loop writes, and on the strategy asking for "best". Single-turn and
    # tool-calling take no branch here at all; tool-calling selects in the
    # extractor's picker (now FIRST after the tier-1 removal), which genuinely works
    # there.
    #
    # NO RE-PROFILING: the earlier turn's speedup comes from the agent loop's own
    # record, measured by the same profiler on the same shape.
    # ------------------------------------------------------------------
    if extract_strategy in ("best_speedup", "latest_verified"):
        n_regraded = 0
        for i in range(batch_size):
            if i >= len(per_sample):
                break
            info = per_sample[i]
            if info.get('is_infra_failure'):
                continue
            best = _best_earlier_turn(extra_infos[i])
            if best is None:
                continue  # not multi-turn, or no verified earlier turn
            # What did the graded (final) turn actually score? A broken final turn
            # has is_correct False, so 0.0 — any verified earlier turn beats it.
            final_speedup = info.get('speedup') if info.get('is_correct') else 0.0
            try:
                final_speedup = float(final_speedup or 0.0)
            except (ValueError, TypeError):
                final_speedup = 0.0
            if best["speedup"] <= final_speedup:
                continue

            # The earlier turn wins. Re-derive the reward from ITS speedup with
            # the same mode/cap as the normal path so the two are comparable.
            # `_correct_reward` takes latency + baseline rather than a speedup, so
            # feed it a latency consistent with the recorded speedup
            # (speedup = baseline / latency) — the ratio is all either mode uses,
            # and this keeps the milestone-mode BEATS_INITIAL threshold working.
            baseline_i = original_latencies[i] if i < len(original_latencies) else None
            if not baseline_i or baseline_i <= 0:
                continue  # no baseline -> speedup mode would gate to 0 anyway
            new_reward, _milestone, _spd = _correct_reward(
                baseline_i / best["speedup"], baseline_i,
                reward_mode=reward_mode, speedup_reward_max=speedup_reward_max,
            )
            rewards[i] = new_reward
            extracted_kernels[i] = best["kernel"]   # judge below sees the graded kernel
            info['is_correct'] = True
            info['reason'] = SUCCESS_CORRECT
            info['speedup'] = best["speedup"]
            info['abs_reward'] = new_reward
            # Provenance + the counterfactual, so a dump can separate "the model
            # improved" from "we graded an earlier turn".
            info['regraded_from_turn'] = best["turn"]
            info['regraded_final_speedup'] = final_speedup
            # Everything below was measured on the FINAL turn's kernel, which is
            # no longer the graded one. Drop rather than carry: a stale hardware
            # payload next to a different kernel's speedup is worse than a gap,
            # and the earlier turn's own metrics are in the `turns` record.
            # `absolute_speedup` specifically feeds speedup/absolute_* aggregates,
            # so leaving the final turn's value there would mix two kernels.
            for _k in ('mfu', 'mbu', 'dma_active_pct', 'hbm_bytes', 'cycles',
                       'arith_intensity', 'profiler_raw', 'absolute_speedup',
                       'kernel_on_device_s', 'original_on_device_s'):
                info.pop(_k, None)
            n_regraded += 1
        if n_regraded:
            logger.info(
                f"[multiturn regrade] graded an earlier turn on {n_regraded}/{batch_size} "
                f"rollouts (strategy={extract_strategy})"
            )

    # ------------------------------------------------------------------
    # Return per-sample reward dicts.
    # ------------------------------------------------------------------
    reward_dicts = []
    for i, reward in enumerate(rewards):
        info = per_sample[i] if i < len(per_sample) else {}
        d = {
            "score": float(reward),
            "is_infra_failure": float(info.get('is_infra_failure', False)),
            "is_correct": float(info.get('is_correct', False)),
            "is_copied": float(info.get('is_copied', False)),
            "is_non_nki": float(info.get('is_non_nki', False)),
            "is_mutated": float(info.get('is_mutated', False)),
            "frontier_advance": float(info.get('frontier_advance', False)),
            # Graded kernel came off a block the model never closed (budget cut it
            # off). Obs-only provenance, NOT a quality signal and not used to
            # rescore — it exists so truncation stops hiding inside
            # `compilation_error`. See kernel_was_finalized.
            "graded_incomplete_kernel": float(info.get('graded_incomplete_kernel', False)),
            "speedup": float(info.get('speedup') or 0.0),
            "absolute_speedup": float(info.get('absolute_speedup') or 0.0),
            # Multi-turn regrade provenance (obs-only). `regraded_from_turn` is
            # -1 when the graded kernel IS the final turn's (the normal case, and
            # every single-turn / tool-calling rollout); >=0 names the earlier
            # turn that won. `regraded_final_speedup` is what the final turn
            # scored, so the cost of the last turn is measurable directly.
            "regraded_from_turn": float(info.get('regraded_from_turn', -1)),
            "regraded_final_speedup": float(info.get('regraded_final_speedup') or 0.0),
            # Hardware perf (obs-only). NaN = not measured (non-correct kernels
            # or profiler omitted summary_metrics); patcher filters NaN out of
            # the aggregate so it doesn't skew the mean.
            "mfu": float(info['mfu']) if info.get('mfu') is not None else float('nan'),
            "mbu": float(info['mbu']) if info.get('mbu') is not None else float('nan'),
            "dma_active_pct": float(info['dma_active_pct']) if info.get('dma_active_pct') is not None else float('nan'),
            "hbm_bytes": float(info['hbm_bytes']) if info.get('hbm_bytes') is not None else float('nan'),
            "cycles": float(info['cycles']) if info.get('cycles') is not None else float('nan'),
            "arith_intensity": float(info['arith_intensity']) if info.get('arith_intensity') is not None else float('nan'),
            # Full profiler payload (JSON str) for post-hoc analysis in the rollout dump.
            "profiler_raw": info.get('profiler_raw') or "",
            # Reward-stage profiler round-trip timing (obs-only). Stamped onto `info`
            # above (from results_map[i]) for every PROFILED sample incl.
            # compile/accuracy fails; None on samples never sent to the profiler
            # (extraction/copy/non-nki fails). Carried here so nkirl_reward's
            # metadata whitelist can forward them and nkirl_metrics can aggregate
            # into the profiler/reward_* wandb keys. Kept as None (NOT nan) on the
            # miss path because nkirl_metrics._vals filters `is not None` — a nan
            # would slip through and pollute the mean/max.
            "profiler_roundtrip_s": info.get('profiler_roundtrip_s'),
            "profiler_poll_wait_s": info.get('profiler_poll_wait_s'),
            "profiler_queue_s": info.get('profiler_queue_s'),
            # Reward computed against un-mutated original-seed baseline (vs `score`,
            # which uses the mutated baseline when replay buffer is on). Falls back
            # to `score` when there's no replay-buffer mutation or no baseline info.
            "abs_reward": float(info['abs_reward']) if info.get('abs_reward') is not None else float(reward),
            "category": str(info.get('category') or 'other'),
            "nki_source": str(extra_infos[i].get('nki_source') or 'other'),
            "reason": str(info.get('reason') or 'unknown'),
            # WHICH input shape this rollout was measured on. Without it a dump
            # tells you the speedup but not what it was a speedup *of*, and under
            # TEST_CONFIG_MODE=random/cycle two rows of the same problem can be
            # measured on different shapes — indistinguishable after the fact.
            # Empty for skipped samples (no kernel / no seed file).
            "test_config": json.dumps(chosen_configs.get(i, {}), default=str),
            # Judge fields — populated below when use_judge=True, default 0/"".
            "is_judge_failed": 0.0,
            "is_judge_duplication": 0.0,
            "judge_confidence": "",
            "judge_concerns": "",
            "judge_cheat_categories": "",
            "extracted_kernel": extracted_kernels[i] or "",
        }
        reward_dicts.append(d)

    # Optional LLM-judge gating. Only `is_correct` rollouts are judged
    # (negative rewards already rejected — judging them wastes bedrock).
    # If the judge rules INVALID, score is overridden to
    # `judge_reward_on_invalid`. Judge errors leave the v2 reward untouched.
    # When `seed_kernel` is present in extra_info, the judge also runs the
    # duplication check (kernel must be a meaningful change over the seed).
    if use_judge:
        judge_targets = [
            i for i, d in enumerate(reward_dicts)
            if d["is_correct"] >= 1.0 and d["is_infra_failure"] < 1.0
        ]
        if judge_targets:
            from llm_judge import judge_batch
            items = [
                {
                    "kernel_name": (extra_infos[i].get("kernel_name")
                                    or extra_infos[i].get("example_name")
                                    or f"sample_{i}"),
                    "kernel_code": extracted_kernels[i] or "",
                    "benchmark_code": (extra_infos[i].get("benchmark_code")
                                       or extra_infos[i].get("initial_code", "") or ""),
                    "seed_code": (extra_infos[i].get("_mutated_seed_kernel")
                                  or extra_infos[i].get("seed_kernel") or None),
                }
                for i in judge_targets
            ]
            verdicts = judge_batch(items, concurrency=judge_concurrency)
            n_valid = n_func_fail = n_dup = n_err = 0
            for j, batch_idx in enumerate(judge_targets):
                v = verdicts[j]
                if v is None:
                    continue
                d = reward_dicts[batch_idx]
                d["judge_confidence"] = str(v.get("confidence") or "")
                concerns = v.get("concerns") or []
                if not isinstance(concerns, list):
                    concerns = [str(concerns)]
                d["judge_concerns"] = ";".join(str(c)[:200] for c in concerns[:5])
                cheats = v.get("cheat_categories") or []
                if not isinstance(cheats, list):
                    cheats = [str(cheats)]
                d["judge_cheat_categories"] = ";".join(str(c) for c in cheats)

                fv = v.get("functionally_valid")
                is_dup = bool(v.get("is_duplication"))
                # Flag both independently (wandb tracking); reject if either.
                if fv is False:
                    d["is_judge_failed"] = 1.0
                if is_dup:
                    d["is_judge_duplication"] = 1.0

                # Sweep-style mutually-exclusive label for counters/log.
                if fv is None:
                    n_err += 1
                elif fv is False:
                    n_func_fail += 1
                elif is_dup:
                    n_dup += 1
                else:
                    n_valid += 1

                # Override score AND reason if either failure mode fires.
                # Functional fail takes priority over duplication (same as
                # sweep-tool labeling) — a functionally broken kernel is the
                # more severe problem.
                if v.get("valid") is False:
                    prev_reason = d.get("reason", "")
                    prev_score = d["score"]
                    # In speedup mode all fails/cheats gate to 0, so route the
                    # judge override through _fail_reward too (milestone → -0.5,
                    # speedup → 0) instead of the raw kwarg.
                    judge_fail = _fail_reward(float(judge_reward_on_invalid), reward_mode)
                    d["score"] = judge_fail
                    d["abs_reward"] = judge_fail
                    d["reason"] = "JUDGE_REJECTED"
                    logger.info(
                        f"Sample {batch_idx}: judge override "
                        f"reward {prev_score:.2f}->{d['score']:.2f} | "
                        f"reason={prev_reason}->JUDGE_REJECTED | "
                        f"func_fail={fv is False} dup={is_dup} | "
                        f"cheats={d['judge_cheat_categories']} | "
                        f"confidence={d['judge_confidence']} | "
                        f"concerns={d['judge_concerns'][:150]}"
                    )
            logger.info(
                f"Judge gating: judged={len(judge_targets)}/{len(reward_dicts)} "
                f"valid={n_valid} dup={n_dup} func_fail={n_func_fail} err={n_err}"
            )

    return reward_dicts


# Alias for compatibility
compute_score_batched = compute_score_batch


def compute_score(data_source: str, solution_str: str, ground_truth=None, extra_info=None,
                  use_judge: bool = False, judge_concurrency: int = 4,
                  judge_reward_on_invalid: float = -0.5,
                  reward_mode: str = DEFAULT_REWARD_MODE,
                  speedup_reward_max: float = DEFAULT_SPEEDUP_REWARD_MAX, **kwargs):
    """Single-sample reward for veRL's experimental reward_loop (naive manager).

    Returns the full per-sample dict (with `score` plus all the boolean/numeric
    fields the trainer needs). reward_kwargs forwarded to compute_score_batch:
        - use_judge: gate correct rollouts via LLM judge (Bedrock Claude)
        - judge_concurrency / judge_reward_on_invalid: judge knobs
        - reward_mode / speedup_reward_max: milestone ladder vs capped speedup
    The two decisions the agent loops ALSO make are env vars, not reward_kwargs:
    which kernel gets graded (KERNEL_EXTRACT_STRATEGY) and which input shape it
    is measured on (TEST_CONFIG_MODE / TEST_CONFIG_INDICES).
    """
    results = compute_score_batch(
        data_sources=[data_source],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info if extra_info else {}],
        use_judge=use_judge,
        judge_concurrency=judge_concurrency,
        judge_reward_on_invalid=judge_reward_on_invalid,
        reward_mode=reward_mode,
        speedup_reward_max=speedup_reward_max,
    )
    return results[0]
