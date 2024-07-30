import dataclasses
from typing import Annotated, Any, Mapping

import pytest
from pydantic import BaseModel

from seer.dependency_injection import FactoryAnnotation, Labeled, Module, inject, injected, resolve


def test_FactoryAnnotation_from_annotation() -> None:
    assert FactoryAnnotation.from_annotation(int) == FactoryAnnotation(
        concrete_type=int,
        is_collection=False,
        label="",
        is_type=False,
    )
    assert FactoryAnnotation.from_annotation(
        Annotated[int, "a", Labeled("b"), Labeled("c")]
    ) == FactoryAnnotation(
        concrete_type=int,
        is_collection=False,
        label="b",
        is_type=False,
    )
    assert FactoryAnnotation.from_annotation(
        Annotated[list[int], Labeled("b")]
    ) == FactoryAnnotation(
        concrete_type=int,
        is_collection=True,
        label="b",
        is_type=False,
    )
    assert FactoryAnnotation.from_annotation(Annotated[list[int], "a"]) == FactoryAnnotation(
        concrete_type=int,
        is_collection=True,
        label="",
        is_type=False,
    )
    assert FactoryAnnotation.from_annotation(type[int]) == FactoryAnnotation(
        concrete_type=int,
        is_collection=False,
        label="",
        is_type=True,
    )
    assert FactoryAnnotation.from_annotation(list[type[int]]) == FactoryAnnotation(
        concrete_type=int,
        is_collection=True,
        label="",
        is_type=True,
    )
    assert FactoryAnnotation.from_annotation(
        Annotated[list[type[int]], Labeled("a")]
    ) == FactoryAnnotation(
        concrete_type=int,
        is_collection=True,
        label="a",
        is_type=True,
    )

    with pytest.raises(AssertionError):
        FactoryAnnotation.from_annotation(Mapping[str, int])

    with pytest.raises(AssertionError):
        FactoryAnnotation.from_annotation(list[list[int]])

    with pytest.raises(AssertionError):
        FactoryAnnotation.from_annotation(list)


def test_FactoryAnnotation_from_factory() -> None:
    def factory_without_rv():
        pass

    def factory_with_required_kwds(*, a: int) -> int:
        return 1

    def factory_with_required_args(a: int) -> int:
        return 1

    def factory_with_nothing_required(a: int = 2, *, c: int = 5) -> int:
        return 1

    def simple_factory() -> int:
        return 1

    with pytest.raises(AssertionError):
        FactoryAnnotation.from_factory(factory_without_rv)

    with pytest.raises(AssertionError):
        FactoryAnnotation.from_factory(factory_with_required_args)

    with pytest.raises(AssertionError):
        FactoryAnnotation.from_factory(factory_with_required_kwds)

    assert FactoryAnnotation.from_factory(
        factory_with_nothing_required
    ) == FactoryAnnotation.from_annotation(int)
    assert FactoryAnnotation.from_factory(simple_factory) == FactoryAnnotation.from_annotation(int)


class Configurations(dict[str, str]):
    pass


def test_injections():
    module = Module()
    magic_object: Any = object

    @module.provider
    @dataclasses.dataclass
    class ServiceA:
        config_a: Annotated[str, Labeled("a")] = injected

    @module.provider
    class ServiceB(BaseModel):
        service_a: ServiceA = injected
        config_b: Annotated[str, Labeled("b")] = injected

    @module.provider
    def a_configuration(inputs: list[Configurations] = injected) -> Annotated[str, Labeled("a")]:
        for c in inputs:
            if "a" in c:
                return c["a"]
        raise ValueError("No configuration found for a")

    @module.provider
    def b_configuration(inputs: list[Configurations] = injected) -> Annotated[str, Labeled("b")]:
        for c in inputs:
            if "b" in c:
                return c["b"]
        raise ValueError("No configuration found for b")

    @module.provider
    def configurations() -> list[Configurations]:
        return [Configurations(b="b-val"), Configurations(a="a-val")]

    @inject
    def main(ready: bool, service_b: ServiceB = injected) -> ServiceB:
        return service_b

    with module:
        existing = resolve(ServiceB)
        assert existing.service_a.config_a == "a-val"
        assert existing.config_b == "b-val"

        assert main(True) is existing
        assert main(True) is main(True)
        assert existing is resolve(ServiceB)
        assert type(existing) is ServiceB

        override = Module()
        override.constant(ServiceB, magic_object)

        with override as injector2:
            assert magic_object is resolve(ServiceB)
            assert resolve(ServiceA) is not existing.service_a
            assert resolve(ServiceA) == existing.service_a

            assert injector2.get(ServiceA) is resolve(ServiceA)

        assert existing is resolve(ServiceB)
