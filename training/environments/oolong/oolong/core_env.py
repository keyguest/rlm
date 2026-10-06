"""Verifiers adapter that evaluates OOLONG with the core recursive RLM."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, cast

import verifiers as vf
from datasets import Dataset
from verifiers.clients import Client
from verifiers.types import AssistantMessage, RolloutInput, SamplingArgs, State

from rlm import RLM
from rlm.core.types import ClientBackend, EnvironmentType, RLMChatCompletion
from rlm.logger import RLMLogger
from rlm.utils.prompts import RLM_SYSTEM_PROMPT
from rlm_train import RLMTrainRubric


def trajectory_stats(metadata: dict[str, Any] | None) -> dict[str, int]:
    """Summarize the complete nested RLM trajectory."""
    stats = {
        "rlm_total_iterations": 0,
        "rlm_total_repl_calls": 0,
        "rlm_sub_llm_calls": 0,
        "rlm_recursive_subcalls": 0,
        "rlm_max_depth_reached": 0,
    }

    def visit(node: dict[str, Any], depth: int) -> None:
        stats["rlm_max_depth_reached"] = max(stats["rlm_max_depth_reached"], depth)
        iterations = node.get("iterations") or []
        stats["rlm_total_iterations"] += len(iterations)
        for iteration in iterations:
            code_blocks = iteration.get("code_blocks") or []
            stats["rlm_total_repl_calls"] += len(code_blocks)
            for code_block in code_blocks:
                result = code_block.get("result") or {}
                for subcall in result.get("rlm_calls") or []:
                    stats["rlm_sub_llm_calls"] += 1
                    child_metadata = subcall.get("metadata")
                    if isinstance(child_metadata, dict):
                        stats["rlm_recursive_subcalls"] += 1
                        visit(child_metadata, depth + 1)

    if metadata:
        visit(metadata, 0)
    return stats


class CoreRLMRubric(RLMTrainRubric):
    """OOLONG correctness plus metrics specific to nested core RLM calls."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.add_metric(self.rlm_recursive_subcalls)
        self.add_metric(self.rlm_total_iterations)
        self.add_metric(self.rlm_total_repl_calls)
        self.add_metric(self.rlm_max_depth_reached)

    async def rlm_recursive_subcalls(self, state: State) -> int:
        return int(state.get("rlm_recursive_subcalls") or 0)

    async def rlm_total_iterations(self, state: State) -> int:
        return int(state.get("rlm_total_iterations") or 0)

    async def rlm_total_repl_calls(self, state: State) -> int:
        return int(state.get("rlm_total_repl_calls") or 0)

    async def rlm_max_depth_reached(self, state: State) -> int:
        return int(state.get("rlm_max_depth_reached") or 0)


class CoreRLMEnv(vf.Environment):
    """Run each verifier rollout through :class:`rlm.RLM`.

    The verifier client is still initialized by ``vf-eval`` so its normal CLI
    configuration remains valid, while the core RLM creates its own synchronous
    client from environment variables. This is necessary because recursive RLM
    children own independent LM handlers and REPL environments.
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        rubric: vf.Rubric,
        max_depth: int = 2,
        max_iterations: int = 12,
        max_concurrent_subcalls: int = 4,
        sub_max_tokens: int = 4096,
        sub_model: str | None = None,
        rlm_backend: str = "openai",
        rlm_environment: str = "local",
        api_key_var: str = "OPENAI_API_KEY",
        base_url_var: str = "OPENAI_BASE_URL",
        base_url: str | None = None,
        request_timeout: float = 300.0,
        max_timeout: float | None = None,
        max_tokens: int | None = None,
        max_errors: int | None = None,
        require_recursive_subcall: bool = False,
        verbose: bool = False,
        backend_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if max_concurrent_subcalls < 1:
            raise ValueError("max_concurrent_subcalls must be at least 1")
        if sub_model is not None and (not isinstance(sub_model, str) or not sub_model.strip()):
            raise ValueError("sub_model must be a non-empty string")
        if require_recursive_subcall and max_depth < 2:
            raise ValueError("require_recursive_subcall requires max_depth >= 2")
        if max_depth > 1 and rlm_environment not in {"local", "ipython", "docker"}:
            raise ValueError(
                "Recursive core evaluation requires rlm_environment to be local, ipython, or docker"
            )

        super().__init__(dataset=dataset, rubric=rubric, **kwargs)
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_concurrent_subcalls = max_concurrent_subcalls
        self.sub_max_tokens = sub_max_tokens
        self.sub_model = sub_model
        self.rlm_backend = rlm_backend
        self.rlm_environment = rlm_environment
        self.api_key_var = api_key_var
        self.base_url_var = base_url_var
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.max_timeout = max_timeout
        self.max_tokens = max_tokens
        self.max_errors = max_errors
        self.require_recursive_subcall = require_recursive_subcall
        self.verbose = verbose
        self.backend_kwargs = dict(backend_kwargs or {})

    def build_rlm(
        self,
        model: str,
        sampling_args: SamplingArgs | None,
    ) -> RLM:
        api_key = os.environ.get(self.api_key_var)
        if not api_key and self.rlm_backend != "vllm":
            raise ValueError(f"Missing API key in environment variable {self.api_key_var!r}")

        backend_kwargs = dict(self.backend_kwargs)
        backend_kwargs["model_name"] = model
        backend_kwargs["timeout"] = self.request_timeout
        if api_key:
            backend_kwargs["api_key"] = api_key

        base_url = self.base_url or os.environ.get(self.base_url_var)
        if base_url:
            backend_kwargs["base_url"] = base_url.rstrip("/")

        root_sampling_args = dict(sampling_args or {})
        root_sampling_args.pop("n", None)
        system_prompt = RLM_SYSTEM_PROMPT
        if self.require_recursive_subcall:
            system_prompt += (
                "\n\nRecursive evaluation requirement: before finalizing, use "
                "`rlm_query()` or `rlm_query_batched()` at least once for a substantive "
                "subtask. Do not substitute `llm_query()` for this required recursive call. "
                "When program logic consumes the result, pass a documented `response_schema`."
            )

        return RLM(
            backend=cast(ClientBackend, self.rlm_backend),
            backend_kwargs=backend_kwargs,
            environment=cast(EnvironmentType, self.rlm_environment),
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_timeout=self.max_timeout,
            max_tokens=self.max_tokens,
            max_errors=self.max_errors,
            custom_system_prompt=system_prompt,
            max_concurrent_subcalls=self.max_concurrent_subcalls,
            sampling_args=root_sampling_args,
            sub_sampling_args={"max_tokens": self.sub_max_tokens},
            sub_model=self.sub_model,
            logger=RLMLogger(),
            verbose=self.verbose,
        )

    def run_rlm(
        self,
        *,
        context: str,
        root_prompt: str,
        model: str,
        sampling_args: SamplingArgs | None,
    ) -> RLMChatCompletion:
        rlm = self.build_rlm(model, sampling_args)
        try:
            return rlm.completion(context, root_prompt=root_prompt)
        finally:
            rlm.close()

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        state = await self.init_state(input, client, model, sampling_args)
        started = time.perf_counter()
        try:
            info = state.get("info")
            if not isinstance(info, dict):
                raise ValueError("Core OOLONG evaluation requires dictionary metadata in 'info'")
            context = info.get("context")
            root_prompt = info.get("root_prompt")
            if not isinstance(context, str) or not isinstance(root_prompt, str):
                raise ValueError("OOLONG metadata must contain string 'context' and 'root_prompt'")

            result = await asyncio.to_thread(
                self.run_rlm,
                context=context,
                root_prompt=root_prompt,
                model=model,
                sampling_args=sampling_args,
            )
            metadata = result.metadata or {}
            stats = trajectory_stats(metadata)
            if self.require_recursive_subcall and stats["rlm_recursive_subcalls"] < 1:
                raise RuntimeError(
                    "The rollout did not create a child RLM; expected at least one "
                    "rlm_query() or rlm_query_batched() call"
                )
            root_iterations = len(metadata.get("iterations") or [])
            root_repl_calls = sum(
                len(iteration.get("code_blocks") or [])
                for iteration in metadata.get("iterations") or []
            )

            state["completion"] = [AssistantMessage(content=result.response)]
            state["rlm_final_answer"] = result.response
            state["rlm_metadata"] = metadata
            state["rlm_usage_summary"] = result.usage_summary.to_dict()
            state["rlm_iterations"] = root_iterations
            state["rlm_repl_calls"] = root_repl_calls
            state["rlm_has_final_answer"] = 1 if result.response else 0
            state.update(stats)
            self.increment_state_usage(
                state,
                input_tokens=result.usage_summary.total_input_tokens,
                output_tokens=result.usage_summary.total_output_tokens,
            )
            state["is_completed"] = True
            state["is_truncated"] = False
            state["stop_condition"] = "has_final_answer"
        except Exception as exc:  # noqa: BLE001
            state["error"] = vf.ModelError(str(exc))
            state["completion"] = []
            state["is_completed"] = True
            state["is_truncated"] = False
            state["stop_condition"] = "has_error"
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000
            state["timing"]["generation_ms"] = elapsed_ms
            state["timing"]["total_ms"] = elapsed_ms
        return state


__all__ = ["CoreRLMEnv", "CoreRLMRubric", "trajectory_stats"]
