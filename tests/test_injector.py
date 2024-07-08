import dataclasses
from typing import Annotated, Any, Mapping

import pytest

from seer.injector import FactoryAnnotation, Injector, Labeled, inject, injected, resolve


def test_FactoryAnnotation_from_annotation() -> None:
    assert FactoryAnnotation.from_annotation(int) == FactoryAnnotation(
        concrete_type=int, is_collection=False, label=""
    )
    assert FactoryAnnotation.from_annotation(
        Annotated[int, "a", Labeled("b"), Labeled("c")]
    ) == FactoryAnnotation(concrete_type=int, is_collection=False, label="b")
    assert FactoryAnnotation.from_annotation(
        Annotated[list[int], Labeled("b")]
    ) == FactoryAnnotation(concrete_type=int, is_collection=True, label="b")
    assert FactoryAnnotation.from_annotation(Annotated[list[int], "a"]) == FactoryAnnotation(
        concrete_type=int, is_collection=True, label=""
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
    injector = Injector()
    magic_object: Any = object

    @injector.provider
    @dataclasses.dataclass
    class ServiceA:
        config_a: Annotated[str, Labeled("a")] = injected

    @injector.provider
    @dataclasses.dataclass
    class ServiceB:
        service_a: ServiceA = injected
        config_b: Annotated[str, Labeled("b")] = injected

    @injector.provider
    def a_configuration(inputs: list[Configurations] = injected) -> Annotated[str, Labeled("a")]:
        for c in inputs:
            if "a" in c:
                return c["a"]
        raise ValueError("No configuration found for a")

    @injector.provider
    def b_configuration(inputs: list[Configurations] = injected) -> Annotated[str, Labeled("b")]:
        for c in inputs:
            if "b" in c:
                return c["b"]
        raise ValueError("No configuration found for b")

    @injector.extension
    def configurations() -> list[Configurations]:
        return [Configurations(b="b-val")]

    @injector.extension
    def configurations_2() -> list[Configurations]:
        return [Configurations(a="a-val")]

    @inject
    def main(ready: bool, service_b: ServiceB = injected) -> ServiceB:
        return service_b

    with injector:
        existing = resolve(ServiceB)
        assert existing.service_a.config_a == "a-val"
        assert existing.config_b == "b-val"

        assert main(True) is existing
        assert main(True) is main(True)
        assert existing is resolve(ServiceB)
        assert type(existing) is ServiceB

        override = Injector()
        override.constant(ServiceB, magic_object)

        with override:
            assert magic_object is resolve(ServiceB)
            assert resolve(ServiceA) is existing.service_a

        assert existing is resolve(ServiceB)
