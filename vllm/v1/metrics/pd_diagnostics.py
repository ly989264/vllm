# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, process-local TTFT summaries; never synchronize business work."""

import contextlib
import contextvars
import itertools
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any, NamedTuple

_LOG = logging.getLogger(__name__)
_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "pd_diagnostics", default=None
)
_BATCH: contextvars.ContextVar["Batch | None"] = contextvars.ContextVar(
    "pd_compact_batch", default=None
)
_ADD_REQUEST_SEND: contextvars.ContextVar["AddRequestSend | None"] = (
    contextvars.ContextVar("pd_add_request_send", default=None)
)
_RUN = uuid.uuid4().hex
_HOST = socket.gethostname()
_COUNTER = itertools.count()
_MAX_ITEMS = 128
_SUMMARIES = {
    "api_summary",
    "core_summary",
    "worker_summary",
    "pool_task",
    "pd_transfer_summary",
}
_SPANS = {
    "api_resolve": "req_resolve",
    "req_resolve": "req_resolve",
    "render": "render",
    "tokenize": "tokenize",
    "add_request": "add_request",
    "scheduler_schedule": "schedule",
    "schedule": "schedule",
    "runner.forward": "forward",
    "forward": "forward",
    "runner.sampling": "sampling",
    "sampling": "sampling",
    "kv.load_submit": "submit",
    "submit": "submit",
    "kv.save_wait": "wait_for_save",
    "wait_for_save": "wait_for_save",
    "wait_for_transfer": "wait_for_transfer",
    "runner.prepare_inputs": "prepare_inputs",
    "runner.lm_head": "lm_head",
    "runner.draft": "draft",
    "runner.bookkeeping": "bookkeeping",
}
_INTERNAL_SPANS = {"scheduler_external_lookup"}
_LINK_FIELDS = {
    "observation_id",
    "api_request_id",
    "external_req_id",
    "request_id",
    "request_ids",
    "parent_request_id",
    "choice_index",
    "prompt_index",
    "execution_id",
    "task_id",
    "store_job_id",
    "engine_id",
    "remote_engine_id",
    "remote_request_id",
    "transfer_request_id",
    "transfer_engine_id",
    "transfer_role",
    "kv_role",
    "dp_rank",
    "pp_rank",
    "tp_rank",
    "finished_sending",
    "finished_recving",
    "failed_recving",
}
_LINK_EVENTS = {
    "api_request_link",
    "request_id_link",
    "request_child_link",
    "frontend_request_registered",
    "pd_request_link",
    "pd_transfer_offer",
    "kv.completion_observed",
    "engine_execution",
    "scheduler_execution",
}


def is_pd_decode(vllm_config: Any) -> bool:
    """Whether this instance is a PD decode instance (D), for the D-only records.

    It judges the configuration: vllm-ascend's own condition
    is_pd_decode_recompute_scheduler_enabled(vllm_config) (top-level KV role
    kv_consumer and not kv_producer, recompute scheduler enabled), the
    condition under which vllm-ascend selects its recompute scheduler. Read
    from vllm_ascend.utils only when that module is already loaded; never
    imports vllm_ascend. Not loaded, or any error: False (not D).
    Call once at initialization and keep the result, never per step.
    """
    try:
        utils = sys.modules.get("vllm_ascend.utils")
        check = getattr(utils, "is_pd_decode_recompute_scheduler_enabled", None)
        return bool(check(vllm_config)) if callable(check) else False
    except Exception:
        return False


class DiagLevel(NamedTuple):
    level: int
    source: str


_DIAG_LEVEL: DiagLevel | None = None
_DIAG_LEVEL_LOGGED: set[str] = set()
_RECORDS_KV_LOAD_SUBMIT: bool | None = None


def parse_diag_level(raw: str | None) -> DiagLevel:
    """PD_DIAG_LEVEL to (level, source). Unset or empty: 3, "default". Exactly
    "1", "2" or "3": that level, "env". Anything else ("0", "4", "abc", " 2 ",
    "2x"): 3, "invalid"."""
    if raw is None or raw == "":
        return DiagLevel(3, "default")
    if raw in ("1", "2", "3"):
        return DiagLevel(int(raw), "env")
    return DiagLevel(3, "invalid")


def diag_level() -> DiagLevel:
    """The diagnostics level in effect in this process, from PD_DIAG_LEVEL.

    3 (the default) records everything; 2 leaves out the level-3 segments and 1
    keeps only the level-1 segments (the scheduler's sub-stages are recorded at
    every level). A component's own switches may record less than the level
    allows, never more. Read on first use and kept for the life of the process:
    a change needs a restart.
    """
    global _DIAG_LEVEL
    value = _DIAG_LEVEL
    if value is None:
        value = _DIAG_LEVEL = parse_diag_level(os.environ.get("PD_DIAG_LEVEL"))
    return value


def diag_level_fields() -> dict[str, int]:
    """The diag_level field for the per-request summaries: written only below
    level 3, so records at the default level are unchanged."""
    level = diag_level().level
    return {"diag_level": level} if level < 3 else {}


def log_diag_level(log: logging.Logger, role: str) -> None:
    """Plain startup log line, not a [PD_DIAG] record: the level in effect. Once
    per process for each role (EngineCore, Worker); never raises."""
    try:
        if role in _DIAG_LEVEL_LOGGED:
            return
        _DIAG_LEVEL_LOGGED.add(role)
        level, source = diag_level()
        log.info("PD_DIAG_STARTUP diag level=%d source=%s", level, source)
    except Exception:
        pass


def decide_kv_load_submit(vllm_config: Any) -> bool:
    """Decide, for this process, whether the "kv.load_submit" span
    (worker_summary stages.submit) is recorded: not on a PD decode instance
    (is_pd_decode) at level 1, yes everywhere else; any error keeps it. The
    worker calls this once, after init_device."""
    global _RECORDS_KV_LOAD_SUBMIT
    try:
        records = not (diag_level().level == 1 and is_pd_decode(vllm_config))
    except Exception:
        records = True
    _RECORDS_KV_LOAD_SUBMIT = records
    return records


def kv_load_submit_span(vllm_config: Any = None) -> contextlib.AbstractContextManager:
    """The "kv.load_submit" span around a KV load submission, or a no-op where
    decide_kv_load_submit() left it out. If the process has not decided yet, it
    decides now from vllm_config."""
    records = _RECORDS_KV_LOAD_SUBMIT
    if records is None:
        records = decide_kv_load_submit(vllm_config)
    if not records:
        return contextlib.nullcontext()
    return span(
        "kv.load_submit", completion="submission_return_not_transfer_completion"
    )


def new_id(kind: str) -> str:
    return f"{kind}-{_RUN}-{os.getpid()}-{next(_COUNTER)}"


def clock_domain() -> dict[str, Any]:
    """Identity of the process-local monotonic clock used by frontend carriers."""
    return {"host": _HOST, "pid": os.getpid(), "run_id": _RUN}


def current() -> dict[str, Any]:
    return dict(_CONTEXT.get() or {})


def current_batch() -> "Batch | None":
    value = _BATCH.get()
    return value if value is not None and not value.closed else None


class AddRequestSend:
    """API-local observation of ADD's ZMQ future, never part of the wire request."""

    __slots__ = ("future", "end_ns", "status", "error_type")

    def __init__(self):
        self.future = None
        self.end_ns = None
        self.status = None
        self.error_type = None

    def watch(self, future) -> None:
        self.future = future
        if future.done():
            self._done(future)
        else:
            # DP may await FIRST_REQ before awaiting this future. Observe its
            # completion independently, without moving or wrapping either await.
            future.add_done_callback(self._done)

    def _done(self, future) -> None:
        if self.end_ns is not None:
            return
        self.end_ns = time.monotonic_ns()
        if future.cancelled():
            self.status, self.error_type = "cancelled", "CancelledError"
        else:
            error = future.exception()
            self.status = "ok" if error is None else "error"
            self.error_type = type(error).__name__ if error is not None else None

    def close(self) -> None:
        if self.future is not None:
            # An already-complete future can still have a queued callback.
            if self.future.done():
                self._done(self.future)
            self.future.remove_done_callback(self._done)
            self.future = None


def current_add_request_send() -> AddRequestSend | None:
    return _ADD_REQUEST_SEND.get()


@contextlib.contextmanager
def track_add_request_send() -> Iterator[AddRequestSend]:
    value = AddRequestSend()
    token = _ADD_REQUEST_SEND.set(value)
    try:
        yield value
    finally:
        value.close()
        _ADD_REQUEST_SEND.reset(token)


@contextlib.contextmanager
def scope(**fields: Any) -> Iterator[None]:
    values = current()
    values.update(fields)
    token = _CONTEXT.set(values)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


@contextlib.contextmanager
def collect(value: "Batch | None") -> Iterator["Batch | None"]:
    """Temporarily restore an explicit owner, including a shared output task."""
    token = _BATCH.set(value)
    try:
        yield value
    finally:
        _BATCH.reset(token)


# ---- Fields written (record format 1.6.0) -------------------------------------
# Each record kind keeps the fields in _KEEP and nothing else; _write applies it
# just before the line is written, so values are never changed, only left out.
# _KEEP_ALL keeps a value as it is; a dict keeps the keys it lists (each by its
# own rule); a one-item list applies that rule to every item of a list. The
# pool_task kinds not in _KEEP are no longer written. Other kinds are written
# as they are.
_KEEP_ALL = True
_STAGE_FIELDS = dict.fromkeys(
    (
        "duration_ns",
        "start_ns",
        "end_ns",
        "status",
        "calls",
        "measured_calls",
        "skipped_calls",
        "unmeasured_calls",
        "device_duration_ns",
    ),
    _KEEP_ALL,
)
_KEEP_STAGES: dict[str, Any] = dict.fromkeys(
    (
        "req_resolve",
        "render",
        "tokenize",
        "add_request",
        "detokenize",
        "queue",
        "hash",
        "kv_ready_wait",
        "lookup",
        "alloc",
        "schedule",
        "other",
        "forward",
        "prepare_inputs",
        "lm_head",
        "submit",
        "wait_for_put",
        "sampling",
        "draft",
        "build_key",
        "prepare",
        "sync",
        "put",
        "wait_for_transfer",
    ),
    _STAGE_FIELDS,
)
_KEEP_STAGES["queue"] = {
    **_STAGE_FIELDS,
    "initial_wait_ns": _KEEP_ALL,
    "ranges": [{"execution_id": _KEEP_ALL}],
}
_KEEP_STAGES["hash"] = {
    **_STAGE_FIELDS,
    "dropped_ranges": _KEEP_ALL,
    "ranges": [dict.fromkeys(("execution_id", "start_ns", "end_ns"), _KEEP_ALL)],
}
_KEEP_STAGES["detokenize"] = {
    **_STAGE_FIELDS,
    "dropped_ranges": _KEEP_ALL,
    "ranges": [
        dict.fromkeys(
            ("execution_id", "duration_ns", "start_ns", "end_ns"), _KEEP_ALL
        )
    ],
}
_KEEP_STAGES["kv_ready_wait"] = {
    **_STAGE_FIELDS,
    "ranges": [{"scheduled_execution_id": _KEEP_ALL}],
}
_KEEP_STAGES["bookkeeping"] = _KEEP_ALL
_KEEP_COMMON: dict[str, Any] = dict.fromkeys(
    (
        "event",
        "version",
        "host",
        "pid",
        "native_tid",
        "run_id",
        "timestamp_ns",
        "record_id",
        "status",
        "scope",
        "phase",
        "operation",
        "transfer_role",
        "tp_rank",
        "request_id",
        "request_ids",
        "execution_id",
        "start_ns",
        "end_ns",
        "duration_ns",
        "task_id",
        "request_generation",
        "queued_ns",
        "request_done_ns",
    ),
    _KEEP_ALL,
)
_KEEP_COMMON["stages"] = _KEEP_STAGES
# The request key of the kinds the node filter writes out.
_KEEP_WRITTEN_IDS = dict.fromkeys(
    ("observation_id", "external_req_id", "api_request_id"), _KEEP_ALL
)
_KEEP_POOL = {
    **_KEEP_COMMON,
    **dict.fromkeys(
        ("member_count", "members_dropped", "queue_enter_ns", "counters"),
        _KEEP_ALL,
    ),
}
_KEEP_POOL_WRITTEN = {
    **_KEEP_POOL,
    **_KEEP_WRITTEN_IDS,
    "members": [{"request_id": _KEEP_ALL}],
    "native_calls_dropped": _KEEP_ALL,
    "native_calls": [
        dict.fromkeys(
            (
                "operation",
                "native_tid",
                "start_wall_ns",
                "end_wall_ns",
                "submitted_keys",
                "submitted_bytes",
                "duration_ns",
            ),
            _KEEP_ALL,
        )
    ],
}
_KEEP: dict[str, Any] = {
    "api_summary[http]": {
        **_KEEP_COMMON,
        **_KEEP_WRITTEN_IDS,
        **dict.fromkeys(
            (
                "http_status",
                "stream",
                "response_start",
                "first_body",
                "first_generated",
            ),
            _KEEP_ALL,
        ),
        "links": [dict.fromkeys(("kind", "external_req_id"), _KEEP_ALL)],
    },
    "api_summary[engine_submission]": {
        **_KEEP_COMMON,
        **_KEEP_WRITTEN_IDS,
        "submission_id": _KEEP_ALL,
        "links": [
            dict.fromkeys(("kind", "request_id", "start_ns", "end_ns"), _KEEP_ALL)
        ],
    },
    "api_summary[request]": {
        **_KEEP_COMMON,
        **_KEEP_WRITTEN_IDS,
        "submission_id": _KEEP_ALL,
        "diag_level": _KEEP_ALL,
        "official": dict.fromkeys(
            (
                "e2e_latency",
                "time_to_first_token",
                "mean_time_per_output_token",
                "num_prompt_tokens",
                "num_cached_tokens",
                "num_generation_tokens",
                "status",
                "queued_time",
                "prefill_time",
                "decode_time",
            ),
            _KEEP_ALL,
        ),
        "terminal": dict.fromkeys(("detokenize_duration_ns", "status"), _KEEP_ALL),
    },
    "core_summary[request]": {
        **_KEEP_COMMON,
        **_KEEP_WRITTEN_IDS,
        "diag_level": _KEEP_ALL,
        "dp_rank": _KEEP_ALL,
        "members": [dict.fromkeys(("request_id", "executions"), _KEEP_ALL)],
    },
    "core_summary[execution]": _KEEP_COMMON,
    "worker_summary[execute]": {
        **_KEEP_COMMON,
        "no_forward": _KEEP_ALL,
        "cudagraph_runtime_mode": _KEEP_ALL,
    },
    "worker_summary[sample]": {**_KEEP_COMMON, "step_device_span_ns": _KEEP_ALL},
    "pool_task[task]:load": _KEEP_POOL_WRITTEN,
    "pool_task[task]:save": _KEEP_POOL_WRITTEN,
    "pool_task[task]:load_submit": _KEEP_POOL,
    "pool_task[task]:save_wait": _KEEP_POOL,
    "pool_task[task]:unknown": _KEEP_POOL,
    "pd_transfer_summary[task]": _KEEP_COMMON,
    "pd_transfer_summary[link]": _KEEP_COMMON,
    "pd_transfer_summary[request_generation]": {
        **_KEEP_COMMON,
        **_KEEP_WRITTEN_IDS,
        **dict.fromkeys(
            (
                "error_type",
                "native_calls_dropped",
                "kv_load_envelope_ns",
                "transfer_envelope_ns",
                "task_max_queue_wait_ns",
                "task_max_meta_fetch_ns",
                "task_max_prepare_ns",
                "submitted_tasks",
                "succeeded_tasks",
                "failed_tasks",
                "skipped_tasks",
                "submitted_bytes",
                "succeeded_bytes",
                "failed_bytes",
            ),
            _KEEP_ALL,
        ),
    },
}


def _kind(values: dict[str, Any]) -> str:
    event = values.get("event")
    sub = values.get("scope")
    if sub is None:
        sub = values.get("phase")
    name = f"{event}[{sub}]" if sub is not None else str(event)
    if event == "pool_task":
        return f"{name}:{values.get('operation')}"
    return name


def _kept(value: Any, rule: Any) -> Any:
    if rule is _KEEP_ALL:
        return value
    if isinstance(rule, list):
        if not isinstance(value, list):
            return value
        return [_kept(item, rule[0]) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: _kept(item, rule[key]) for key, item in value.items() if key in rule
    }


def _keep_fields(values: dict[str, Any]) -> dict[str, Any] | None:
    """The record as written: only the fields its kind keeps; None when the
    kind is no longer written (record format 1.6.0)."""
    rule = _KEEP.get(_kind(values))
    if rule is None:
        return None if values.get("event") == "pool_task" else values
    return _kept(values, rule)


def _number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _write(event: str, fields: dict[str, Any]) -> None:
    try:
        values = dict(fields)
        values.update(
            event=event,
            version=2,
            host=_HOST,
            pid=os.getpid(),
            native_tid=threading.get_native_id(),
            run_id=_RUN,
            timestamp_ns=time.monotonic_ns(),
            record_id=new_id("record"),
        )
        kept = _keep_fields(values)
        if kept is None:
            return
        _LOG.info("[PD_DIAG] %s", json.dumps(kept, separators=(",", ":")))
    except Exception:
        # Diagnostics must not replace a serving failure or fail a response.
        pass


class Batch:
    """One original lifecycle's bounded timing/identity accumulator."""

    def __init__(self, event: str, **identity: Any):
        if event not in _SUMMARIES:
            raise ValueError(f"Not a compact summary type: {event}")
        self.event = event
        self.fields = {**current(), **identity}
        self.stages: dict[str, dict[str, Any]] = {}
        self.links: list[dict[str, Any]] = []
        self.closed = False
        self._lock = threading.RLock()
        self._dropped = 0

    def update(self, **fields: Any) -> None:
        with self._lock:
            if not self.closed:
                self.fields.update(fields)

    def add_stage(
        self,
        name: str,
        *,
        start_ns: int | None = None,
        end_ns: int | None = None,
        duration_ns: int | None = None,
        calls: int = 1,
        status: str = "ok",
        **attributes: Any,
    ) -> None:
        if duration_ns is None and _number(start_ns) and _number(end_ns):
            duration_ns = end_ns - start_ns
        exceptions = attributes.pop("exceptions", None)
        failed = status in ("error", "failed", "cancelled", "exception")
        with self._lock:
            if self.closed:
                return
            if name not in self.stages and len(self.stages) >= _MAX_ITEMS:
                self._dropped += 1
                return
            value = self.stages.setdefault(
                name,
                {
                    "calls": 0,
                    "duration_ns": 0,
                    "start_ns": None,
                    "end_ns": None,
                    "status": status,
                    "exceptions": 0,
                    "measured_calls": 0,
                    "skipped_calls": 0,
                    "unmeasured_calls": 0,
                },
            )
            count = calls if _number(calls) else 1
            value["calls"] += count
            if status in ("skipped", "fused"):
                value["skipped_calls"] += count
                if not value["measured_calls"]:
                    value["status"] = status
                    value["duration_ns"] = None
                elif not value["unmeasured_calls"] and not value["exceptions"]:
                    value["status"] = "mixed"
            elif status in ("missing", "invalid") or not _number(duration_ns):
                value["unmeasured_calls"] += max(count, 1)
                value["status"] = (
                    status
                    if status in ("missing", "invalid")
                    else "invalid"
                    if duration_ns is not None
                    else "missing"
                )
                value["duration_ns"] = None
            else:
                value["measured_calls"] += count
                if not value["unmeasured_calls"]:
                    value["duration_ns"] = (value["duration_ns"] or 0) + duration_ns
                    if failed or not value["exceptions"]:
                        value["status"] = (
                            "mixed"
                            if status == "ok" and value["skipped_calls"]
                            else status
                        )
            if _number(start_ns):
                old = value["start_ns"]
                value["start_ns"] = start_ns if old is None else min(old, start_ns)
            if _number(end_ns):
                old = value["end_ns"]
                value["end_ns"] = end_ns if old is None else max(old, end_ns)
            value["exceptions"] += (
                max(exceptions, int(failed)) if _number(exceptions) else int(failed)
            )
            value.update(attributes)

    def add_link(self, kind: str, **identity: Any) -> None:
        link = {"kind": kind, **identity}
        with self._lock:
            if self.closed or link in self.links:
                return
            if len(self.links) >= _MAX_ITEMS:
                self._dropped += 1
            else:
                self.links.append(link)

    def capture(self, event: str, **fields: Any) -> None:
        """Merge retained observations, not old leaf-record stdout."""
        stage = _SPANS.get(event)
        if stage is not None:
            values = dict(fields)
            values.pop("event", None)
            self.add_stage(stage, **values)
        elif event == "request_summary":
            self.update(official=dict(fields))
        elif event == "frontend_request_terminal":
            duration = fields.get("detokenize_duration_ns")
            if duration is not None:
                calls = fields.get("detokenize_calls", 1)
                self.add_stage(
                    "detokenize",
                    duration_ns=duration,
                    calls=calls,
                    status=(
                        "skipped"
                        if calls == 0
                        else "ok"
                        if fields.get("error_type") is None
                        else "error"
                    ),
                )
            self.update(terminal=dict(fields))
        elif event in _LINK_EVENTS:
            self.add_link(
                event, **{k: v for k, v in fields.items() if k in _LINK_FIELDS}
            )

    def finish(self, status: str = "ok", **fields: Any) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            values = {
                **self.fields,
                **fields,
                "stages": self.stages,
                "links": self.links,
                "status": status,
                "complete": (
                    fields.get("complete", self.fields.get("complete", True))
                    and not self._dropped
                ),
                "dropped_observations": self._dropped,
            }
        _write(self.event, values)


@contextlib.contextmanager
def batch(event: str, **identity: Any) -> Iterator[Batch]:
    value = Batch(event, **identity)
    status = "ok"
    error_type = None
    try:
        with collect(value):
            yield value
    except BaseException as exc:
        status = (
            "cancelled"
            if isinstance(exc, (KeyboardInterrupt, GeneratorExit))
            or type(exc).__name__ == "CancelledError"
            else "error"
        )
        error_type = type(exc).__name__
        raise
    finally:
        value.finish(status=status, error_type=error_type)


def emit(event: str, **fields: Any) -> None:
    if event in _SUMMARIES:
        _write(event, {**current(), **fields})
        return
    value = current_batch()
    if value is not None:
        value.capture(event, **fields)


@contextlib.contextmanager
def span(event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Measure retained host calls and necessary scheduler formula inputs."""
    if event not in _SPANS and event not in _INTERNAL_SPANS:
        yield {}
        return
    values = dict(fields)
    start = time.monotonic_ns()
    values.update(start_ns=start, status="ok")
    try:
        yield values
    except BaseException as exc:
        values["status"] = (
            "cancelled"
            if isinstance(exc, (KeyboardInterrupt, GeneratorExit))
            or type(exc).__name__ == "CancelledError"
            else "error"
        )
        values["error_type"] = type(exc).__name__
        raise
    finally:
        end = time.monotonic_ns()
        values.update(end_ns=end, duration_ns=end - start)
        value = current_batch()
        if value is not None and event in _SPANS:
            value.capture(event, **values)
