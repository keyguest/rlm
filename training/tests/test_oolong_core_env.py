from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import verifiers as vf
from datasets import Dataset
from verifiers.clients import Client
from verifiers.types import ClientConfig, Messages, Response, SamplingArgs, Tool

from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary

sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "oolong"))

from oolong import env as oolong_env  # noqa: E402
from oolong.core_env import CoreRLMEnv, trajectory_stats  # noqa: E402


class DummyClient(Client):
    def setup_client(self, config: ClientConfig) -> object:
        return object()

    async def to_native_tool(self, tool: Tool) -> object:
        return tool

    async def to_native_prompt(self, messages: Messages) -> tuple[Messages, dict]:
        return messages, {}

    async def get_native_response(
        self,
        prompt: Messages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[object] | None = None,
        **kwargs: Any,
    ) -> Response:
        raise AssertionError("The verifier client must not be called by CoreRLMEnv")

    async def raise_from_native_response(self, response: Response) -> None:
        return None

    async def from_native_response(self, response: Response) -> Response:
        return response

    async def close(self) -> None:
        return None


def nested_metadata() -> dict[str, Any]:
    return {
        "iterations": [
            {
                "code_blocks": [
                    {
                        "result": {
                            "rlm_calls": [
                                {
                                    "response": "child",
                                    "metadata": {
                                        "iterations": [
                                            {
                                                "code_blocks": [
                                                    {
                                                        "result": {
                                                            "rlm_calls": [
                                                                {
                                                                    "response": "leaf",
                                                                }
                                                            ]
                                                        }
                                                    }
                                                ]
                                            }
                                        ]
                                    },
                                },
                                {"response": "plain"},
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_trajectory_stats_walks_recursive_children() -> None:
    assert trajectory_stats(nested_metadata()) == {
        "rlm_total_iterations": 2,
        "rlm_total_repl_calls": 2,
        "rlm_sub_llm_calls": 3,
        "rlm_recursive_subcalls": 1,
        "rlm_max_depth_reached": 1,
    }


def test_core_env_validates_recursive_environment() -> None:
    dataset = Dataset.from_list([{"prompt": "q", "answer": "a", "info": {}}])
    with pytest.raises(ValueError, match="local, ipython, or docker"):
        CoreRLMEnv(
            dataset=dataset,
            rubric=vf.Rubric(),
            max_depth=2,
            rlm_environment="prime",
        )

    with pytest.raises(ValueError, match="max_depth >= 2"):
        CoreRLMEnv(
            dataset=dataset,
            rubric=vf.Rubric(),
            max_depth=1,
            require_recursive_subcall=True,
        )


def test_load_environment_selects_core_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset.from_list([{"example_id": 0, "prompt": "q", "answer": "a", "info": {}}])
    monkeypatch.setattr(oolong_env, "_build_dataset", lambda **kwargs: dataset)

    env = oolong_env.load_environment(
        engine="core",
        max_depth=3,
        max_iterations=7,
        max_concurrent_subcalls=2,
        require_recursive_subcall=True,
        sub_model="gpt-5-mini",
    )

    assert isinstance(env, CoreRLMEnv)
    assert env.max_depth == 3
    assert env.max_iterations == 7
    assert env.max_concurrent_subcalls == 2
    assert env.require_recursive_subcall is True
    assert env.sub_model == "gpt-5-mini"


def test_train_engine_rejects_recursive_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset.from_list([{"example_id": 0, "prompt": "q", "answer": "a", "info": {}}])
    monkeypatch.setattr(oolong_env, "_build_dataset", lambda **kwargs: dataset)

    with pytest.raises(ValueError, match="engine='core'"):
        oolong_env.load_environment(engine="train", max_depth=2)
    with pytest.raises(ValueError, match="sub_model"):
        oolong_env.load_environment(engine="train", sub_model="gpt-5-mini")


def test_core_env_builds_depth_limited_recursive_rlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_list([{"prompt": "q", "answer": "a", "info": {}}])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://relay.example/v1/")
    env = CoreRLMEnv(
        dataset=dataset,
        rubric=vf.Rubric(),
        max_depth=3,
        require_recursive_subcall=True,
        sub_model="gpt-5-mini",
    )

    rlm = env.build_rlm("test-model", {"max_tokens": 512, "n": 1})
    try:
        assert rlm.max_depth == 3
        assert rlm.backend_kwargs["api_key"] == "test-key"
        assert rlm.backend_kwargs["base_url"] == "https://relay.example/v1"
        assert rlm.backend_kwargs["sampling_args"]["max_tokens"] == 512
        assert rlm.sub_model == "gpt-5-mini"
        assert rlm.other_backend_kwargs is not None
        assert rlm.other_backend_kwargs[0]["model_name"] == "gpt-5-mini"
        assert "Recursive evaluation requirement" in rlm.system_prompt
    finally:
        rlm.close()


@pytest.mark.asyncio
async def test_core_env_rollout_records_nested_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = Dataset.from_list(
        [
            {
                "example_id": 0,
                "prompt": [{"role": "user", "content": "question"}],
                "answer": "['ok']",
                "info": {
                    "context": "long context",
                    "root_prompt": "aggregate question",
                    "answer": "['ok']",
                },
            }
        ]
    )
    env = CoreRLMEnv(dataset=dataset, rubric=vf.Rubric(), max_depth=3)
    result = RLMChatCompletion(
        root_model="test-model",
        prompt="long context",
        response="Answer: ok",
        usage_summary=UsageSummary(
            model_usage_summaries={
                "test-model": ModelUsageSummary(
                    total_calls=2,
                    total_input_tokens=100,
                    total_output_tokens=20,
                )
            }
        ),
        execution_time=0.1,
        metadata=nested_metadata(),
    )
    monkeypatch.setattr(env, "run_rlm", lambda **kwargs: result)

    state = await env.rollout(
        dataset[0],
        DummyClient(object()),
        "test-model",
        {"max_tokens": 256},
    )

    assert state["rlm_final_answer"] == "Answer: ok"
    assert state["completion"][0].content == "Answer: ok"
    assert state["rlm_iterations"] == 1
    assert state["rlm_repl_calls"] == 1
    assert state["rlm_sub_llm_calls"] == 3
    assert state["rlm_recursive_subcalls"] == 1
    assert state["rlm_max_depth_reached"] == 1
    assert state["is_completed"] is True
    assert state["stop_condition"] == "has_final_answer"
    assert env.get_state_usage(state) == {"input_tokens": 100.0, "output_tokens": 20.0}


@pytest.mark.asyncio
async def test_core_env_rejects_missing_recursive_subcall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = Dataset.from_list(
        [
            {
                "example_id": 0,
                "prompt": [{"role": "user", "content": "question"}],
                "answer": "['ok']",
                "info": {"context": "context", "root_prompt": "question"},
            }
        ]
    )
    env = CoreRLMEnv(
        dataset=dataset,
        rubric=vf.Rubric(),
        max_depth=2,
        require_recursive_subcall=True,
    )
    result = RLMChatCompletion(
        root_model="test-model",
        prompt="context",
        response="Answer: ok",
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.1,
        metadata={"iterations": []},
    )
    monkeypatch.setattr(env, "run_rlm", lambda **kwargs: result)

    state = await env.rollout(dataset[0], DummyClient(object()), "test-model")

    assert state["stop_condition"] == "has_error"
    assert isinstance(state["error"], vf.ModelError)
    assert "did not create a child RLM" in str(state["error"])
