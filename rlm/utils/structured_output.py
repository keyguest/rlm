import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from rlm.core.types import JSONValue, ResponseSchema
from rlm.utils.exceptions import (
    StructuredOutputParseError,
    StructuredOutputValidationError,
    StructuredSchemaError,
)

MAX_SCHEMA_BYTES = 64_000
MAX_SCHEMA_DEPTH = 16
MAX_SCHEMA_PROPERTIES = 256


def validate_response_schema(schema: ResponseSchema) -> None:
    """Validate a documented, bounded JSON Schema response contract."""
    if not isinstance(schema, dict):
        raise StructuredSchemaError("response_schema must be a dictionary")

    try:
        encoded = json.dumps(schema, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise StructuredSchemaError(f"response_schema must be JSON-serializable: {e}") from e
    if len(encoded.encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise StructuredSchemaError(f"response_schema exceeds the {MAX_SCHEMA_BYTES:,}-byte limit")

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as e:
        raise StructuredSchemaError(f"invalid JSON Schema: {e.message}") from e

    if not _nonempty_string(schema.get("title")):
        raise StructuredSchemaError("response_schema must have a non-empty top-level 'title'")
    if not _nonempty_string(schema.get("description")):
        raise StructuredSchemaError("response_schema must have a non-empty top-level 'description'")

    property_count = 0

    def visit(node: Any, path: str, depth: int) -> None:
        nonlocal property_count
        if depth > MAX_SCHEMA_DEPTH:
            raise StructuredSchemaError(
                f"response_schema exceeds maximum depth {MAX_SCHEMA_DEPTH} at {path}"
            )
        if isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]", depth + 1)
            return
        if not isinstance(node, dict):
            return

        ref = node.get("$ref")
        if isinstance(ref, str) and not ref.startswith("#/"):
            raise StructuredSchemaError(f"remote $ref is not allowed at {path}: {ref!r}")

        properties = node.get("properties")
        if isinstance(properties, dict):
            property_count += len(properties)
            if property_count > MAX_SCHEMA_PROPERTIES:
                raise StructuredSchemaError(
                    f"response_schema exceeds {MAX_SCHEMA_PROPERTIES} total properties"
                )
            required = node.get("required")
            if not isinstance(required, list):
                raise StructuredSchemaError(f"object schema at {path} must declare 'required'")
            missing_required = sorted(set(properties) - set(required))
            if missing_required:
                raise StructuredSchemaError(
                    f"object schema at {path} must require every property; missing: "
                    f"{missing_required}"
                )
            if node.get("additionalProperties") is not False:
                raise StructuredSchemaError(
                    f"object schema at {path} must set 'additionalProperties' to false"
                )
            for name, child in properties.items():
                child_path = f"{path}.properties.{name}"
                if not isinstance(child, dict) or not _nonempty_string(child.get("description")):
                    raise StructuredSchemaError(
                        f"property {child_path} must have a non-empty 'description'"
                    )
                visit(child, child_path, depth + 1)

        for key, child in node.items():
            if key != "properties" and isinstance(child, (dict, list)):
                visit(child, f"{path}.{key}", depth + 1)

    visit(schema, "$", 0)


def build_structured_output_instruction(schema: ResponseSchema) -> str:
    validate_response_schema(schema)
    rendered = json.dumps(schema, indent=2, sort_keys=True, ensure_ascii=True)
    return (
        "\n\nSTRUCTURED OUTPUT CONTRACT\n"
        "Your final answer must be exactly one JSON value matching the JSON Schema below. "
        "Field descriptions define the semantic meaning of each value. Do not wrap the JSON "
        "in Markdown and do not add text outside it. If an RLM REPL answer dict is available, "
        "serialize the value with json.dumps(...) before assigning it to answer['content'].\n\n"
        f"{rendered}"
    )


def parse_and_validate_response(text: str, schema: ResponseSchema) -> JSONValue:
    validate_response_schema(schema)
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except StructuredOutputParseError:
        raise
    except json.JSONDecodeError as e:
        raise StructuredOutputParseError(
            f"response is not exactly one JSON value: {e.msg} at line {e.lineno} column {e.colno}"
        ) from e

    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as e:
        location = "$"
        for part in e.absolute_path:
            location += f"[{part}]" if isinstance(part, int) else f".{part}"
        raise StructuredOutputValidationError(
            f"response failed validation at {location}: {e.message}"
        ) from e
    return value


def structured_retry_message(error: Exception) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Your proposed final answer did not satisfy the structured output contract: "
            f"{error}. Correct it and submit a new final answer containing only valid JSON."
        ),
    }


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _reject_json_constant(value: str) -> None:
    raise StructuredOutputParseError(f"response contains non-standard JSON value {value!r}")


def _object_without_duplicate_keys(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    value: dict[str, JSONValue] = {}
    for key, item in pairs:
        if key in value:
            raise StructuredOutputParseError(f"response contains duplicate object key {key!r}")
        value[key] = item
    return value
