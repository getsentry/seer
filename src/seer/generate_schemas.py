import json
import os.path
from typing import Generator, Iterator, List, Optional, Set, Union

root = os.path.abspath(os.path.join(__file__, ".."))

from openapi_core import Spec
from openapi_pydantic import OpenAPI, Reference, Schema
from openapi_pydantic.util import PydanticSchema, construct_open_api_with_schema_class

import seer.app  # noqa
from seer.json_api import view_functions


class TypedDictsGenerator:
    spec: OpenAPI
    imports: Set[str]
    lines: List[str]

    def __init__(self, spec: OpenAPI):
        self.spec = spec
        self.imports = set()
        self.lines = []

    def typing(self, t: str) -> str:
        self.imports.add("typing")
        return f"typing.{t}"

    def generate(self) -> List[str]:
        assert self.spec.components
        assert self.spec.components.schemas
        for schema_ref, schema in self.spec.components.schemas.items():
            self.lines.extend([*self.generate_schema(schema_ref, schema)])
            self.lines.append("")

        return [*(f"import {module}" for module in self.imports), "\n", *self.lines]

    def generate_schema(self, schema_ref: str, schema: Schema) -> Iterator[str]:
        assert schema.properties
        self.imports.add("typing_extensions")
        yield f"{schema_ref} = typing_extensions.TypedDict({repr(schema_ref)}, {{"
        for prop_name, prop in schema.properties.items():
            annotation = yield from self.get_annotation(prop)
            yield f"  {repr(prop_name)}: {annotation},"
        yield "}, total=False)"

    def get_annotation(
        self, schema: Optional[Union[Reference, Schema, bool]]
    ) -> Generator[str, None, str]:
        if isinstance(schema, Schema):
            if schema.schema_format:
                yield f"  # format: {schema.schema_format}"
            if schema.default is not None:
                yield f"  # default: {repr(schema.default)}"

            if schema.const:
                return self.typing(f"Literal[{repr(schema.const)}]")

            if schema.anyOf:
                parts = []
                for part in schema.anyOf:
                    annotation = yield from self.get_annotation(part)
                    parts.append(annotation)
                return f"{self.typing('Union')}[{', '.join(parts)}]"
            if schema.type == "object":
                annotation = yield from self.get_annotation(schema.additionalProperties)
                return self.typing(f"Mapping[str, {annotation}]")
            elif schema.type == "array":
                if (
                    schema.prefixItems
                    and schema.minItems == schema.maxItems
                    and schema.minItems == len(schema.prefixItems)
                ):
                    annotations = []
                    for prefix in schema.prefixItems:
                        annotation = yield from self.get_annotation(prefix)
                        annotations.append(annotation)
                    return self.typing(f"Tuple[{', '.join(annotations)}]")

                items = yield from self.get_annotation(schema.items)
                return self.typing(f"List[{items}]")
            elif schema.type == "string":
                return "str"
            elif schema.type == "null":
                return "None"
            elif schema.type == "number":
                return "float"
            elif schema.type == "integer":
                return "int"
            elif schema.type == "boolean":
                return "bool"
        elif isinstance(schema, Reference):
            assert schema.ref.startswith("#/components/schemas/")
            return repr(schema.ref.split("/")[-1])
        return self.typing("Any")


if __name__ == "__main__":
    spec = OpenAPI.model_validate(
        dict(
            info=dict(
                title="Sentry Inference APIs",
                version="0.0.1",
            ),
            servers=[dict(url="http://seer")],
            paths={
                url_rule: {
                    "post": {
                        "tags": [],
                        "description": view_function.__doc__,
                        "operationId": view_function.__name__,
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": PydanticSchema(schema_class=request)
                                },
                            },
                            "required": True,
                        },
                        "responses": {
                            "200": {
                                "description": "Success",
                                "content": {
                                    "application/json": {
                                        "schema": PydanticSchema(schema_class=response)
                                    },
                                },
                            }
                        },
                    }
                }
                for url_rule, view_function, request, response in view_functions
            },
        )
    )

    spec = construct_open_api_with_schema_class(spec)
    spec_dict = spec.model_dump(by_alias=True, exclude_none=True)
    # For good measure, corroborate the pydantic modeling against openapi_core
    Spec.from_dict(spec_dict)  # type: ignore

    with open(os.path.join(root, "schemas", "seer_api.json"), "w") as file:
        file.write(json.dumps(spec_dict, indent=2))

    with open(os.path.join(root, "schemas", "seer.py"), "w") as file:
        file.write("\n".join(TypedDictsGenerator(spec).generate()))
