import asyncio

import pytest

from rlm_train.proxy import ClientHandle, SubLLMProxy
from rlm_train.worker import Worker


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
