# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import contextlib
import threading
from collections.abc import Callable, Iterator
from functools import wraps
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar

import torch
import torch.nn as nn

import vllm.ir
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tracing import instrument
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.utils.system_utils import update_environment_variables
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    record_kv_cache_layout,
)
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.metrics import pd_diagnostics as diag

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
else:
    SchedulerOutput = object
    GrammarOutput = object
    AsyncModelRunnerOutput = object
    ModelRunnerOutput = object

logger = init_logger(__name__)

_R = TypeVar("_R")


# ---- T51 C14 (A2): a runner may take over writing this phase's worker_summary ------------------------------------
_PD_CLAIMS_LOCK = threading.Lock()
_PD_CLAIMS: dict[int, "PdFinishClaim"] = {}
_PD_HELD: dict[int, "PdFinishClaim"] = {}


class PdFinishClaim:
    """Holds one worker_summary record open past the wrapped method's return, until release()."""

    def __init__(self, batch: "diag.Batch"):
        self._batch = batch
        self._lock = threading.Lock()
        self._result: tuple[str, str | None] | None = None  # (status, error_type) once the method returned
        self._released = False
        self._written = False

    def set_stage_fields(self, stage: str, **fields: Any) -> bool:
        """Add fields to stages[stage] of the held record; False when the stage is absent or it was written."""
        batch = self._batch
        with batch._lock:
            if batch.closed or stage not in batch.stages:
                return False
            batch.stages[stage].update(fields)
            return True

    def release(self) -> None:
        """Idempotent, any thread. Before the wrapped method returned: written at its return, as without a claim.
        After: written now, with the status the wrapper determined."""
        with self._lock:
            if self._released or self._written:
                return
            self._released = True
            result = self._result
        if result is not None:
            self._write(result)

    def _returned(self, status: str, error_type: str | None) -> None:
        with self._lock:
            if self._written:
                return
            self._result = (status, error_type)
            write_now = self._released
        if write_now:
            self._write((status, error_type))
        else:
            with _PD_CLAIMS_LOCK:
                _PD_HELD[id(self)] = self

    def _write(self, result: tuple[str, str | None]) -> None:
        with self._lock:
            if self._written:
                return
            self._written = True
        with _PD_CLAIMS_LOCK:
            _PD_HELD.pop(id(self), None)
        self._batch.finish(status=result[0], error_type=result[1])


def pd_claim_finish() -> PdFinishClaim | None:
    """Inside a method wrapped by pd_execution or pd_sample (same thread): keep this phase's worker_summary open after
    the method returns, until the returned claim is released. None when no worker_summary record is open here.
    Calling it again for the same record returns the same claim. If the method raises, the record is written at once
    with the error status, as without a claim."""
    batch = diag.current_batch()
    if batch is None or batch.event != "worker_summary":
        return None
    with _PD_CLAIMS_LOCK:
        claim = _PD_CLAIMS.get(id(batch))
        if claim is None:
            claim = _PD_CLAIMS[id(batch)] = PdFinishClaim(batch)
        return claim


def pd_release_claims() -> None:
    """Write every claimed record that is still held (process exit)."""
    with _PD_CLAIMS_LOCK:
        held = list(_PD_HELD.values())
    for claim in held:
        claim.release()


atexit.register(pd_release_claims)


@contextlib.contextmanager
def _pd_worker_batch(phase: str, context: dict[str, Any]) -> Iterator["diag.Batch"]:
    """diag.batch("worker_summary", phase=...) that honours pd_claim_finish(); same record, same statuses."""
    value = diag.Batch("worker_summary", phase=phase, **context)
    status = "ok"
    error_type = None
    try:
        with diag.collect(value):
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
        with _PD_CLAIMS_LOCK:
            claim = _PD_CLAIMS.pop(id(value), None)
        if claim is None or status != "ok":
            if claim is not None:
                with claim._lock:
                    claim._written = True
            value.finish(status=status, error_type=error_type)
        else:
            claim._returned(status, error_type)


def pd_execution_context(owner: Any, scheduler_output: Any) -> dict[str, Any]:
    """Snapshot CPU identity, never retain a mutable scheduler/batch object."""
    from vllm.distributed import get_pp_group, get_tp_group

    return {
        "execution_id": scheduler_output.pd_execution_id,
        "request_id": None,
        "request_ids": list(scheduler_output.num_scheduled_tokens),
        "tp_rank": get_tp_group().rank_in_group,
        "pp_rank": get_pp_group().rank_in_group,
        "dp_rank": owner.parallel_config.data_parallel_index,
        "runner": type(owner).__module__ + "." + type(owner).__qualname__,
        "timing_domain": "host_monotonic",
    }


def pd_execution(method):
    """Retain explicit execution identity across the execute/sample split."""

    @wraps(method)
    def wrapped(self, scheduler_output, *args, **kwargs):
        dummy_run = kwargs.get("dummy_run", args[1] if len(args) > 1 else False)
        if dummy_run:
            with diag.scope(dummy_run=True):
                return method(self, scheduler_output, *args, **kwargs)
        # An Ascend override can enter the shared method on the same call.
        if getattr(self, "_pd_executing", False):
            return method(self, scheduler_output, *args, **kwargs)
        context = pd_execution_context(self, scheduler_output)
        self._pd_execution_context = context
        self._pd_executing = True
        try:
            with (
                diag.scope(**context),
                _pd_worker_batch("execute", context),
            ):
                return method(self, scheduler_output, *args, **kwargs)
        except BaseException:
            self._pd_execution_context = None
            raise
        finally:
            self._pd_executing = False

    return wrapped


def pd_sample(method):
    """Restore the originating execute identity, including deferred MTP work."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        if getattr(self, "_pd_sampling", False):
            return method(self, *args, **kwargs)
        context = getattr(self, "_pd_execution_context", None)
        self._pd_sampling = True
        try:
            with (
                diag.scope(**(context or {})),
                _pd_worker_batch("sample", context or {}),
            ):
                return method(self, *args, **kwargs)
        finally:
            self._pd_sampling = False
            self._pd_execution_context = None

    return wrapped


_PD_STARTUP_RUNNER_LOGGED = False
_PD_STARTUP_TIMING_LOGGED = False


def _pd_startup_tp_rank() -> str:
    try:
        from vllm.distributed import get_tp_group

        rank = get_tp_group().rank_in_group
        return str(rank) if isinstance(rank, int) else "?"
    except Exception:
        return "?"


def _pd_startup_log_runner(worker: Any) -> None:
    """Plain startup log line, not a [PD_DIAG] record: the model runner class built by
    init_device. Once per process; never raises."""
    global _PD_STARTUP_RUNNER_LOGGED
    try:
        if _PD_STARTUP_RUNNER_LOGGED:
            return
        _PD_STARTUP_RUNNER_LOGGED = True
        runner = getattr(worker, "model_runner", None)
        name = (
            f"{type(runner).__module__}.{type(runner).__qualname__}"
            if runner is not None
            else "?"
        )
        logger.info(
            "PD_DIAG_STARTUP worker model_runner=%s tp_rank=%s",
            name,
            _pd_startup_tp_rank(),
        )
    except Exception:
        pass


def _pd_startup_log_timing(worker: Any, context: Any, scheduler_output: Any) -> None:
    """Plain log line, not a [PD_DIAG] record, after the first execute_model of a real request
    returned: whether vllm-ascend's device timing is on ("?" if not readable). vllm-ascend
    decides it in that step; a step that ran no forward and left it undecided does not count.
    Once per process; never raises."""
    global _PD_STARTUP_TIMING_LOGGED
    try:
        if _PD_STARTUP_TIMING_LOGGED:
            return
        timing = getattr(getattr(worker, "model_runner", None), "_pd_device_timing", None)
        if timing is None and not getattr(scheduler_output, "total_num_scheduled_tokens", 0):
            return
        _PD_STARTUP_TIMING_LOGGED = True
        enabled = getattr(timing, "enabled", None)
        tp_rank = context.get("tp_rank") if isinstance(context, dict) else None
        logger.info(
            "PD_DIAG_STARTUP worker device_timing=%s tp_rank=%s",
            "on" if enabled is True else "off" if enabled is False else "?",
            tp_rank if isinstance(tp_rank, int) else "?",
        )
    except Exception:
        pass


class CompilationTimes(NamedTuple):
    language_model: float
    encoder: float


class WorkerBase:
    """Worker interface that allows vLLM to cleanly separate implementations for
    different hardware. Also abstracts control plane communication, e.g., to
    communicate request metadata to other workers.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        """
        Initialize common worker components.

        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver
                responsibilities
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config
        self.kv_transfer_config = vllm_config.kv_transfer_config
        self.compilation_config = vllm_config.compilation_config

        from vllm.platforms import current_platform

        self.current_platform = current_platform

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: torch.device | None = None
        self.model_runner: nn.Module | None = None

        # IR op priority and torch-wrap state are constant for the worker's
        # lifetime.
        vllm_config.kernel_config.ir_op_priority.set_default()
        vllm.ir.set_default_torch_wrap(
            vllm_config.compilation_config.ir_enable_torch_wrap
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def get_supported_kv_cache_layouts(self) -> list[str]:
        """Layout names every attention backend supports, most preferred first."""
        backends = get_current_attn_backends(self.vllm_config)
        return [layout.name for layout in get_supported_kv_cache_layouts(backends)]

    def set_kv_cache_layout(self, kv_cache_layout: str) -> None:
        """Adopt the KV cache layout resolved by the engine core."""
        record_kv_cache_layout(self.vllm_config.cache_config, kv_cache_layout)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        """Prepare model for execution through compilation/warmup.

        Returns:
            Compilation times (language_model, encoder) in seconds.
        """
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return

    def synchronize_device(self) -> None:
        """Block until in-flight device work completes; backends outside
        ``torch.accelerator`` must override with their own wait."""
        if torch.accelerator.is_available():
            torch.accelerator.synchronize()

    def init_device(self) -> None:
        """Initialize device state, such as loading the model or other on-device
        memory allocations.
        """
        raise NotImplementedError

    def reset_mm_cache(self) -> None:
        reset_fn = getattr(self.model_runner, "reset_mm_cache", None)
        if callable(reset_fn):
            reset_fn()

    def get_model(self) -> nn.Module:
        raise NotImplementedError

    def supports_draft_weight_updates(self) -> bool:
        """Whether this worker can update its configured speculative model."""
        return False

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        """Apply a function on the model inside this worker."""
        return fn(self.get_model())

    def get_model_inspection(self) -> str:
        """Return a transformers-style hierarchical view of the model."""
        from vllm.model_inspection import format_model_inspection

        return format_model_inspection(self.get_model())

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load model onto target device."""
        raise NotImplementedError

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.
        """
        raise NotImplementedError

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """Should be called immediately after execute_model iff it returned None."""
        raise NotImplementedError

    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError

    def list_loras(self) -> set[int]:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size from model configuration."""
        return self.model_config.get_vocab_size()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        return


class WorkerWrapperBase:
    """
    This class represents one process in an executor/engine. It is responsible
    for lazily initializing the worker and handling the worker's lifecycle.
    We first instantiate the WorkerWrapper, which remembers the worker module
    and class name. Then, when we call `update_environment_variables`, and the
    real initialization happens in `init_worker`.
    """

    def __init__(
        self,
        rpc_rank: int = 0,
        global_rank: int | None = None,
    ) -> None:
        """
        Initialize the worker wrapper with the given vllm_config and rpc_rank.
        Note: rpc_rank is the rank of the worker in the executor. In most cases,
        it is also the rank of the worker in the distributed group. However,
        when multiple executors work together, they can be different.
        e.g. in the case of SPMD-style offline inference with TP=2,
        users can launch 2 engines/executors, each with only 1 worker.
        All workers have rpc_rank=0, but they have different ranks in the TP
        group.
        """
        self.rpc_rank: int = rpc_rank
        self.global_rank: int = self.rpc_rank if global_rank is None else global_rank

        # Initialized after init_worker is called
        self.worker: WorkerBase
        self.vllm_config: VllmConfig

    def shutdown(self) -> None:
        if self.worker is not None:
            self.worker.shutdown()

    def update_environment_variables(
        self,
        envs_list: list[dict[str, str]],
    ) -> None:
        envs = envs_list[self.rpc_rank]
        update_environment_variables(envs)

    @instrument(span_name="Worker init")
    def init_worker(self, all_kwargs: list[dict[str, Any]]) -> None:
        """
        Here we inject some common logic before initializing the worker.
        Arguments are passed to the worker class constructor.
        """
        kwargs = all_kwargs[self.rpc_rank]

        vllm_config: VllmConfig | None = kwargs.get("vllm_config")
        assert vllm_config is not None, (
            "vllm_config is required to initialize the worker"
        )
        self.vllm_config = vllm_config

        vllm_config.enable_trace_function_call_for_thread()

        from vllm.plugins import load_general_plugins

        load_general_plugins()

        parallel_config = vllm_config.parallel_config
        if isinstance(parallel_config.worker_cls, str):
            worker_class: type[WorkerBase] = resolve_obj_by_qualname(
                parallel_config.worker_cls
            )
        else:
            raise ValueError(
                "passing worker_cls is no longer supported. "
                "Please pass keep the class in a separate module "
                "and pass the qualified name of the class as a string."
            )

        if parallel_config.worker_extension_cls:
            worker_extension_cls = resolve_obj_by_qualname(
                parallel_config.worker_extension_cls
            )
            extended_calls = []
            if worker_extension_cls not in worker_class.__bases__:
                # check any conflicts between worker and worker_extension_cls
                for attr in dir(worker_extension_cls):
                    if attr.startswith("__"):
                        continue
                    assert not hasattr(worker_class, attr), (
                        f"Worker class {worker_class} already has an attribute"
                        f" {attr}, which conflicts with the worker"
                        f" extension class {worker_extension_cls}."
                    )
                    if callable(getattr(worker_extension_cls, attr)):
                        extended_calls.append(attr)
                # dynamically inherit the worker extension class
                worker_class.__bases__ = worker_class.__bases__ + (
                    worker_extension_cls,
                )
                logger.info(
                    "Injected %s into %s for extended collective_rpc calls %s",
                    worker_extension_cls,
                    worker_class,
                    extended_calls,
                )

        assigned_physical_gpu_ids = kwargs.pop("assigned_physical_gpu_ids", None)
        if assigned_physical_gpu_ids is not None:
            vllm_config.parallel_config.assigned_physical_gpu_ids = (
                assigned_physical_gpu_ids
            )

        shared_worker_lock = kwargs.pop("shared_worker_lock", None)
        if shared_worker_lock is None:
            msg = (
                "Missing `shared_worker_lock` argument from executor. "
                "This argument is needed for mm_processor_cache_type='shm'."
            )

            mm_config = vllm_config.model_config.multimodal_config
            if mm_config and mm_config.mm_processor_cache_type == "shm":
                raise ValueError(msg)
            else:
                logger.warning_once(msg)

            self.mm_receiver_cache = None
        else:
            self.mm_receiver_cache = (
                MULTIMODAL_REGISTRY.worker_receiver_cache_from_config(
                    vllm_config,
                    shared_worker_lock,
                )
            )

        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during worker initialization
            self.worker = worker_class(**kwargs)

    def initialize_from_config(self, kv_cache_configs: list[Any]) -> None:
        kv_cache_config = kv_cache_configs[self.global_rank]
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            self.worker.initialize_from_config(kv_cache_config)  # type: ignore

    def init_device(self):
        assert self.vllm_config is not None
        with set_current_vllm_config(self.vllm_config):
            # To make vLLM config available during device initialization
            self.worker.init_device()  # type: ignore
        _pd_startup_log_runner(self.worker)

    def __getattr__(self, attr: str):
        return getattr(self.worker, attr)

    def _apply_mm_cache(self, scheduler_output: SchedulerOutput) -> None:
        mm_cache = self.mm_receiver_cache
        if mm_cache is None:
            return

        for req_data in scheduler_output.scheduled_new_reqs:
            req_data.mm_features = mm_cache.get_and_update_features(
                req_data.mm_features
            )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        context = pd_execution_context(self.worker, scheduler_output)
        self._pd_worker_context = context
        try:
            with diag.scope(**context):
                self._apply_mm_cache(scheduler_output)
                output = self.worker.execute_model(scheduler_output)
        except BaseException:
            self._pd_worker_context = None
            raise
        _pd_startup_log_timing(self.worker, context, scheduler_output)
        return output

    def sample_tokens(
        self, grammar_output: GrammarOutput, *args: Any, **kwargs: Any
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        context = getattr(self, "_pd_worker_context", None)
        try:
            with diag.scope(**(context or {})):
                return self.worker.sample_tokens(grammar_output, *args, **kwargs)
        finally:
            self._pd_worker_context = None

    def reset_mm_cache(self) -> None:
        mm_receiver_cache = self.mm_receiver_cache
        if mm_receiver_cache is not None:
            mm_receiver_cache.clear_cache()

        self.worker.reset_mm_cache()
