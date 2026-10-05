from unittest.mock import Mock, patch

import pytest

import rlm.core.rlm as rlm_module
from rlm import RLM
from rlm.core.types import ModelUsageSummary, RLMChatCompletion, UsageSummary
from rlm.environments.local_repl import LocalREPL
from rlm.utils.exceptions import (
    StructuredOutputParseError,
    StructuredOutputValidationError,
    StructuredSchemaError,
)
from rlm.utils.structured_output import (
    build_structured_output_instruction,
    parse_and_validate_response,
    validate_response_schema,
)
from tests.mock_lm import MockLM


def response_schema() -> dict:
    return {
        "title": "Classification result",
        "description": "A machine-readable classification and its supporting count.",
        "type": "object",
        "properties": {
            "label": {
                "type": "string",
                "description": "The exact category assigned to the input.",
            },
            "count": {
                "type": "integer",
                "description": "The number of matching input records.",
                "minimum": 0,
            },
        },
        "required": ["label", "count"],
        "additionalProperties": False,
    }


def completion(parsed_response) -> RLMChatCompletion:
    return RLMChatCompletion(
        root_model="test-model",
        prompt="test",
        response='{"label":"entity","count":3}',
        usage_summary=UsageSummary(model_usage_summaries={}),
        execution_time=0.1,
        parsed_response=parsed_response,
    )


def test_schema_requires_semantic_descriptions() -> None:
    schema = response_schema()
    del schema["properties"]["label"]["description"]

    with pytest.raises(StructuredSchemaError, match="description"):
        validate_response_schema(schema)


def test_schema_requires_every_property() -> None:
    schema = response_schema()
    schema["required"] = ["label"]

    with pytest.raises(StructuredSchemaError, match="missing.*count"):
        validate_response_schema(schema)


def test_instruction_contains_schema_and_exact_json_contract() -> None:
    instruction = build_structured_output_instruction(response_schema())

    assert "exactly one JSON value" in instruction
    assert "The exact category assigned" in instruction
    assert '"additionalProperties": false' in instruction


def test_parse_and_validate_response_returns_python_value() -> None:
    value = parse_and_validate_response('{"label":"abbreviation","count":4}', response_schema())

    assert value == {"label": "abbreviation", "count": 4}


def test_parse_rejects_markdown_wrapped_json() -> None:
    with pytest.raises(StructuredOutputParseError, match="exactly one JSON value"):
        parse_and_validate_response('```json\n{"label":"entity","count":2}\n```', response_schema())


def test_validation_rejects_wrong_field_type() -> None:
    with pytest.raises(StructuredOutputValidationError, match=r"\$.count"):
        parse_and_validate_response('{"label":"entity","count":"three"}', response_schema())


def test_local_rlm_query_returns_validated_python_value() -> None:
    schema = response_schema()
    calls = []

    def structured_subcall(prompt, received_schema, model):
        calls.append((prompt, received_schema, model))
        return completion({"label": "entity", "count": 3})

    repl = LocalREPL(structured_subcall_fn=structured_subcall)
    try:
        result = repl.execute_code(
            f"schema = {schema!r}\n"
            "result = rlm_query('classify these rows', model='child', response_schema=schema)"
        )
        assert result.stderr == ""
        assert repl.locals["result"] == {"label": "entity", "count": 3}
        assert calls == [("classify these rows", schema, "child")]
        assert len(result.rlm_calls) == 1
    finally:
        repl.cleanup()


def test_parse_rejects_non_standard_json_numbers() -> None:
    with pytest.raises(StructuredOutputParseError, match="non-standard JSON value"):
        parse_and_validate_response('{"label":"entity","count":NaN}', response_schema())


def test_parse_rejects_duplicate_keys() -> None:
    with pytest.raises(StructuredOutputParseError, match="duplicate object key 'count'"):
        parse_and_validate_response('{"label":"entity","count":2,"count":3}', response_schema())


def test_local_structured_batch_preserves_order() -> None:
    schema = response_schema()

    def structured_subcall(prompt, received_schema, model):
        assert received_schema == schema
        assert model is None
        return completion({"label": prompt, "count": len(prompt)})

    repl = LocalREPL(structured_subcall_fn=structured_subcall)
    try:
        result = repl.execute_code(
            f"schema = {schema!r}\n"
            "results = rlm_query_batched(['a', 'bbb', 'cc'], response_schema=schema)"
        )
        assert result.stderr == ""
        assert repl.locals["results"] == [
            {"label": "a", "count": 1},
            {"label": "bbb", "count": 3},
            {"label": "cc", "count": 2},
        ]
        assert len(result.rlm_calls) == 3
    finally:
        repl.cleanup()


def test_local_structured_failure_is_not_converted_to_text() -> None:
    def structured_subcall(prompt, received_schema, model):
        raise StructuredOutputValidationError("bad child output")

    repl = LocalREPL(structured_subcall_fn=structured_subcall)
    try:
        result = repl.execute_code(
            f"schema = {response_schema()!r}\n"
            "result = rlm_query('classify', response_schema=schema)"
        )
        assert "StructuredOutputValidationError" in result.stderr
        assert "result" not in repl.locals
    finally:
        repl.cleanup()


def test_leaf_structured_subcall_instructs_and_validates_model_output() -> None:
    client = Mock()
    client.model_name = "child-model"
    client.completion.return_value = '{"label":"entity","count":3}'
    client.get_last_usage.return_value = ModelUsageSummary(1, 20, 10)
    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "child-model"},
        max_depth=1,
    )

    with patch.object(rlm_module, "get_client", return_value=client):
        result = parent._structured_subcall("classify", response_schema())

    assert result.parsed_response == {"label": "entity", "count": 3}
    sent_messages = client.completion.call_args.args[0]
    assert sent_messages[0]["role"] == "system"
    assert "The exact category assigned" in sent_messages[0]["content"]


def test_leaf_structured_subcall_raises_on_non_json_output() -> None:
    client = Mock()
    client.model_name = "child-model"
    client.completion.return_value = "entity: 3"
    parent = RLM(
        backend="openai",
        backend_kwargs={"model_name": "child-model"},
        max_depth=1,
    )

    with (
        patch.object(rlm_module, "get_client", return_value=client),
        pytest.raises(StructuredOutputParseError),
    ):
        parent._structured_subcall("classify", response_schema())


def test_rlm_retries_invalid_structured_final_answer() -> None:
    invalid = "```repl\nanswer['content'] = 'entity: 3'\nanswer['ready'] = True\n```"
    valid = (
        "```repl\n"
        'answer[\'content\'] = \'{"label":"entity","count":3}\'\n'
        "answer['ready'] = True\n"
        "```"
    )
    client = MockLM(responses=[invalid, valid])
    rlm = RLM(
        backend="openai",
        backend_kwargs={"model_name": "mock-model"},
        max_depth=1,
        max_iterations=2,
    )

    with patch.object(rlm_module, "get_client", return_value=client):
        result = rlm.completion(
            "classify",
            response_schema=response_schema(),
            structured_retries=1,
        )

    assert result.parsed_response == {"label": "entity", "count": 3}
    assert client._call_count == 2
