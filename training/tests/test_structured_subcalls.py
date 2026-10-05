import asyncio
import json
import sys
from pathlib import Path

import pytest
from datasets import Dataset

from rlm_train.env import RLMTrainEnv
from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.worker import Worker

sys.path.insert(0, str(Path(__file__).parents[1] / "environments" / "oolong"))

from oolong.env import _score  # noqa: E402


def response_schema() -> dict:
    return {
        "title": "Classification result",
        "description": "A category and the number of matching rows.",
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "The exact category label.",
            },
            "count": {
                "type": "integer",
                "description": "The exact number of matching rows.",
            },
        },
        "required": ["label", "count"],
        "additionalProperties": False,
    }


class FakeBackend:
    async def start(self, **kwargs) -> None:
        pass

    async def load_context(self, payload, index=None) -> int:
        return 0

    async def bootstrap(self, code: str) -> None:
        pass

    async def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_setup_state_returns_state_for_verifiers_011_contract() -> None:
    backend = FakeBackend()
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "question"}], "answer": "answer"}]
    )
    env = RLMTrainEnv(dataset=dataset, backend_factory=lambda: backend)
    state = {
        "info": {"context": "context", "root_prompt": "question"},
        "client": object(),
        "model": "test-model",
    }
    try:
        initialized = await env.setup_state(state)
        assert initialized is state
        assert initialized["rlm_context_count"] == 1
    finally:
        await env.cleanup_rlm(state)
        await env.teardown_rlm()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_location", ["info", "input"])
async def test_oolong_score_resolves_metadata_from_state(metadata_location: str) -> None:
    metadata = {
        "answer": "['entity']",
        "answer_type": "ANSWER_TYPE.TEXT",
    }
    state = {"rlm_final_answer": "entity"}
    if metadata_location == "info":
        state["info"] = json.dumps(metadata)
    else:
        state["input"] = {"info": metadata}

    assert await _score(None, state) == 1.0


@pytest.mark.asyncio
async def test_oolong_score_reports_missing_metadata_clearly() -> None:
    with pytest.raises(ValueError, match="requires dataset metadata"):
        await _score(None, {"rlm_final_answer": "entity"})


@pytest.mark.asyncio
async def test_worker_receives_validated_structured_value_from_proxy() -> None:
    seen_prompts = []

    def fake_query(prompt, state):
        seen_prompts.append(prompt)
        return '{"label":"entity","count":3}'

    proxy = SubLLMProxy()
    await proxy.start()
    proxy.register(
        "rollout",
        ClientHandle(client=None, model="test", fake_query=fake_query),
    )
    worker = Worker(proxy.url, "rollout")
    try:
        result = await asyncio.to_thread(
            worker._rlm_query,
            "classify these rows",
            None,
            response_schema(),
        )
    finally:
        await proxy.stop()

    assert result == {"label": "entity", "count": 3}
    assert "The exact category label" in seen_prompts[0]
    assert "classify these rows" in seen_prompts[0]


@pytest.mark.asyncio
async def test_worker_structured_batch_preserves_order() -> None:
    def fake_query(prompt, state):
        task = prompt.rsplit("TASK\n", 1)[-1]
        return f'{{"label":"{task}","count":{len(task)}}}'

    proxy = SubLLMProxy()
    await proxy.start()
    proxy.register(
        "rollout",
        ClientHandle(client=None, model="test", fake_query=fake_query),
    )
    worker = Worker(proxy.url, "rollout")
    try:
        result = await asyncio.to_thread(
            worker._rlm_query_batched,
            ["a", "bbb", "cc"],
            None,
            response_schema(),
        )
    finally:
        await proxy.stop()

    assert result == [
        {"label": "a", "count": 1},
        {"label": "bbb", "count": 3},
        {"label": "cc", "count": 2},
    ]


@pytest.mark.asyncio
async def test_worker_structured_call_fails_loudly_on_invalid_output() -> None:
    proxy = SubLLMProxy()
    await proxy.start()
    proxy.register(
        "rollout",
        ClientHandle(client=None, model="test", fake_query=lambda prompt, state: "entity: 3"),
    )
    worker = Worker(proxy.url, "rollout")
    try:
        with pytest.raises(RuntimeError, match="not exactly one JSON value"):
            await asyncio.to_thread(
                worker._rlm_query,
                "classify",
                None,
                response_schema(),
            )
    finally:
        await proxy.stop()
