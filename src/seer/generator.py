import dataclasses
import enum
import inspect
import itertools
import math
import random
import string
import struct
import typing
from typing import Any, Iterator, TypeVar, get_type_hints

import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

_A = TypeVar("_A")
_B = TypeVar("_B")
_Pydantic = TypeVar("_Pydantic", bound=BaseModel)
_Tuple = TypeVar("_Tuple", bound=tuple)

if typing.TYPE_CHECKING:
    import pytest


@dataclasses.dataclass
class _RandomGenerator:
    state: tuple[Any, ...] = dataclasses.field(default_factory=lambda: random.Random().getstate())
    max_count: int = 10000

    def restart_at(self, seed: int) -> typing.Self:
        self.state = random.Random(seed).getstate()
        return self

    def __iter__(self) -> Iterator[random.Random]:
        r = random.Random()
        r.setstate(self.state)

        for _ in range(self.max_count):
            yield r
            self.state = r.getstate()
        raise ValueError("generator did not find a valid generation")

    @staticmethod
    def one_of(*options: typing.Iterable[_A]) -> Iterator[_A]:
        return (r.choice(next(zip(*options))) for r in gen)


gen = _RandomGenerator()
unsigned_ints = (r.getrandbits(2 ** r.randint(0, 6)) for r in gen)
negative_ints = (i * -1 for i in unsigned_ints)
ints = gen.one_of(unsigned_ints, negative_ints, (0,))
all_floats = (struct.unpack("d", struct.pack("Q", bits))[0] for bits in unsigned_ints)
floats = (v for v in all_floats if not math.isinf(v) and not math.isnan(v))
bools = gen.one_of([True, False])
objects = (object() for _ in gen)
ascii_words = ("".join(r.sample(string.printable, r.randint(0, 20))) for r in gen)


def _pydantic_has_default(field: FieldInfo) -> bool:
    return field.default is not PydanticUndefined or field.default_factory is not None


def _dataclass_has_default(field: dataclasses.Field) -> bool:
    return (
        field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING
    )


class Examples:
    """
    Annotate types with this item in order to get example inputs as part of test generation.
    """

    def __init__(self, *args: Any):
        self.args = args

    def __iter__(self):
        return iter(self.args)


def generate_dicts_for_annotations(
    annotations: typing.Mapping[str, Any], context: "GeneratorContext"
) -> Iterator[dict[str, Any]]:
    if not annotations:
        return ({} for _ in gen)

    generators: dict[str, Iterator[Any]] = {
        k: generate(context.step(v, k)) for k, v in annotations.items()
    }

    return (
        dict((k, v) for k, v in zip(generators.keys(), values))
        for values in zip(*generators.values())
    )


def generate_dicts_for_pydantic_model(
    context: "GeneratorContext",
) -> Iterator[dict[str, Any]]:
    hints = get_type_hints(context.source)
    return generate_dicts_for_annotations(
        {
            k: hints.get(k, Any)
            for k, field in context.source.model_fields.items()
            if context.include_defaults or not _pydantic_has_default(field)
        },
        context,
    )


def generate_dicts_for_dataclass_model(
    context: "GeneratorContext",
) -> Iterator[dict[str, Any]]:
    hints = get_type_hints(context.source)
    fields = {f.name: f for f in dataclasses.fields(context.source)}
    return generate_dicts_for_annotations(
        {
            k: hints.get(k, Any)
            for k, field in fields.items()
            if context.include_defaults or not _dataclass_has_default(field)
        },
        context,
    )


def generate_call_args_for_argspec(
    source: inspect.FullArgSpec, context: "GeneratorContext"
) -> typing.Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
    num_arg_defaults = len(source.defaults) if source.defaults is not None else 0

    arg_generators: list[Iterator[Any]] = []
    for i, argname in enumerate(source.args):
        if context.include_defaults or i < (len(source.args) - num_arg_defaults):
            arg_generators.append(
                generate(context.step(source.annotations.get(argname, Any), argname))
            )

    arg_generator: Iterator[tuple[Any, ...]]
    if arg_generators:
        arg_generator = zip(*arg_generators)
    else:
        arg_generator = itertools.repeat(tuple())

    if source.varargs and (context.include_defaults or num_arg_defaults == 0):
        star_args_generator = generate(
            context.step(tuple, source.varargs, (source.annotations.get(source.varargs, Any), ...))
        )
        arg_generator = (
            (*argset, *star_argset)
            for argset, star_argset in zip(arg_generator, star_args_generator)
        )

    kwarg_gens: list[Iterator[tuple[str, Any]]] = []
    for argname in source.kwonlyargs:
        kwarg_gens.append(
            (
                (argname, v)
                for v in generate(context.step(source.annotations.get(argname, Any), argname))
            )
        )

    kwargs_generator: Iterator[dict[str, Any]]

    if kwarg_gens:
        kwargs_generator = (dict(i for i in kwargset) for kwargset in zip(*kwarg_gens))
    else:
        kwargs_generator = itertools.repeat({})

    if source.varkw:
        kw_type = source.annotations.get(source.varkw, Any)
        origin = typing.get_origin(kw_type)
        args = typing.get_args(kw_type)

        varkw_generator: Iterator[dict[str, Any]] | None = None
        if origin is None or origin is not typing.Unpack:
            varkw_generator = generate_dicts(context.step(dict, source.varkw, (str, kw_type)))
        else:
            varkw_generator = generate(context.step((*args, Any)[0], source.varkw))

        if varkw_generator:
            kwargs_generator = (
                dict((k, v) for d in (args, kwargs) for k, v in d.items() if k)
                for args, kwargs in zip(kwargs_generator, varkw_generator)
            )

    return zip(arg_generator, kwargs_generator)


def generate_enums(context: "GeneratorContext") -> Iterator[Any] | None:
    if inspect.isclass(context.source) and (
        issubclass(context.source, enum.Enum) or issubclass(context.source, enum.IntEnum)
    ):
        return gen.one_of(context.source)
    return None


class Generator(typing.Protocol):
    def __call__(self, context: "GeneratorContext") -> Iterator[Any] | None:
        ...


@dataclasses.dataclass
class GeneratorContext:
    source: Any
    origin: Any | None
    args: tuple[Any, ...]
    context: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    include_defaults: bool = False

    generators: list[Generator] = dataclasses.field(
        default_factory=lambda: [
            generate_iterators,
            generate_pydantic_instances,
            generate_dataclass_instances,
            generate_dicts_from_typeddict,
            generate_dicts,
            generate_enums,
            generate_results_from_call,
            generate_tuples,
            generate_unions,
            generate_lists,
            generate_annotated,
            generate_not_required,
            generate_primitives,
            generate_any,
        ]
    )

    def step(
        self, source: Any, step: str | None = None, args: tuple[Any, ...] | None = None
    ) -> "GeneratorContext":
        result = GeneratorContext.from_source(source)
        if args is not None:
            result.origin = source
            result.args = args
        result.context = (*self.context, step) if step else self.context
        result.include_defaults = self.include_defaults
        result.generators = self.generators
        return result

    @classmethod
    def from_source(cls, source: Any) -> "GeneratorContext":
        return GeneratorContext(
            source=source,
            origin=typing.get_origin(source),
            args=typing.get_args(source) or (),
            context=(repr(source),),
        )


def generate_pydantic_instances(context: GeneratorContext) -> Iterator[BaseModel] | None:
    if isinstance(context.source, type) and issubclass(context.source, BaseModel):
        return (context.source(**d) for d in generate_dicts_for_pydantic_model(context))
    return None


def generate_dataclass_instances(context: GeneratorContext) -> Iterator[Any] | None:
    if dataclasses.is_dataclass(context.source):
        return (context.source(**d) for d in generate_dicts_for_dataclass_model(context))
    return None


def generate_dicts_from_typeddict(context: GeneratorContext) -> Iterator[Any] | None:
    if typing.is_typeddict(context.source):
        optional: list[str] = sorted(getattr(context.source, "__optional_keys__", frozenset()))
        hints = get_type_hints(context.source, include_extras=True)
        dicts = generate_dicts_for_annotations({k: v for k, v in hints.items()}, context)
        sampled_keys = (
            (r.sample(optional, r.randint(0, len(optional))) for r in gen)
            if context.include_defaults
            else ([] for _ in gen)
        )
        return (
            dict((k, v) for k, v in d.items() if k not in optional or k in optional_keys)
            for d, optional_keys in zip(dicts, sampled_keys)
        )

    return None


def generate_dicts(context: GeneratorContext) -> Iterator[dict[Any, Any]] | None:
    if (
        context.origin
        and inspect.isclass(context.origin)
        and issubclass(context.origin, typing.Mapping)
    ):
        key, value, *_ = (*context.args, object, object)
        key_generator = generate(context.step(key, "[Key]"))
        value_generator = generate(context.step(value, "[Value]"))
        return (
            dict((k, v) for k, v, _ in zip(key_generator, value_generator, range(length)))
            for length in (i % 10 for i in unsigned_ints)
        )

    return None


def generate_results_from_call(context: GeneratorContext) -> Iterator[Any] | None:
    if inspect.isfunction(context.source):
        argspec = inspect.getfullargspec(context.source)
        return (
            context.source(*a, **k) for a, k in generate_call_args_for_argspec(argspec, context)
        )

    return None


def generate_tuples(context: GeneratorContext) -> Iterator[Any] | None:
    if context.origin is tuple:
        has_ellipsis = not context.args or Ellipsis in context.args
        specified_parts = tuple(a for a in context.args if a is not Ellipsis)

        generators = [
            generate(context.step(arg, str(f"[{i}]"))) for i, arg in enumerate(specified_parts)
        ]

        specified_generator = zip(*generators)

        if has_ellipsis:
            extension_type = [Any, *specified_parts][-1]
            unspecified_generator = generate(context.step(list, "...", (extension_type,)))
            return (
                (*specified, *unspecified)
                for specified, unspecified in zip(specified_generator, unspecified_generator)
            )
        else:
            return specified_generator
    return None


def generate_unions(context: GeneratorContext) -> Iterator[Any] | None:
    if context.origin is typing.Union and context.args:
        return gen.one_of(generate(context.step(arg, f"|")) for arg in context.args)
    return None


def generate_lists(context: GeneratorContext) -> Iterator[Any] | None:
    if context.origin is list:
        arg = [*context.args, object][0]
        generator = generate(context.step(arg))
        return (
            [i for i, _ in zip(generator, range(length))]
            for length in (r.randint(0, 10) for r in gen)
        )
    return None


def generate_not_required(context: GeneratorContext) -> typing.Iterator[Any] | None:
    if context.origin is typing.NotRequired:
        annotated_inner = [*context.args, Any][0]
        return generate(context.step(annotated_inner))
    return None


def generate_annotated(context: GeneratorContext) -> typing.Iterator[Any] | None:
    if context.origin is typing.Annotated:
        annotated_inner = [*context.args, Any][0]
        examples = (v for ex in context.args[1:] if isinstance(ex, Examples) for v in ex)
        if examples:
            return gen.one_of(*(generate(v) for v in examples))

        return generate(context.step(annotated_inner))
    return None


def generate_iterators(context: GeneratorContext) -> typing.Iterator[Any] | None:
    if hasattr(context.source, "__next__"):
        return context.source
    return None


def generate_primitives(context: GeneratorContext) -> typing.Iterator[Any] | None:
    source = context.source
    if source is int:
        return ints
    if source is str:
        return ascii_words
    if source is bool:
        return bools
    if source is object:
        return objects
    if source is float:
        return floats
    return None


def generate_any(context: GeneratorContext) -> typing.Iterator[Any] | None:
    if context.source is typing.Any:
        return gen.one_of(ascii_words, ints, floats, bools)
    return None


def generate(context: GeneratorContext | Any) -> typing.Iterator[Any]:
    if not isinstance(context, GeneratorContext):
        gen_context = GeneratorContext.from_source(context)
    else:
        gen_context = context

    for generator in gen_context.generators:
        result = generator(gen_context)
        if result is not None:
            return result
    raise TypeError(f"Could not generate for {' '.join(gen_context.context)} {gen_context.source}")


def parameterize(
    seed: int, count: int = 10, include_defaults=False, **generators: Any
) -> typing.Callable[[typing.Callable[..., Any]], typing.Callable[..., Any]]:
    import pytest

    def decorator(func: typing.Callable[..., Any]) -> typing.Callable[..., Any]:
        argspec = inspect.getfullargspec(func)
        context = GeneratorContext.from_source(func)
        return pytest.mark.generate(
            seed=seed,
            count=count,
            include_defaults=include_defaults,
            generators=generators,
            argspec=argspec,
            context=context,
        )(func)

    return decorator


class GeneratorPlugin:
    def pytest_configure(self, config: pytest.Config):
        config.addinivalue_line("markers", "generate: runs test with generated inputs")

    def pytest_generate_tests(self, metafunc: pytest.Metafunc):
        generate = metafunc.definition.get_closest_marker("generate")
        if generate:
            seed = generate.kwargs["seed"]
            count = generate.kwargs["count"]
            include_defaults = generate.kwargs["include_defaults"]
            generators = generate.kwargs["generators"]
            argspec = generate.kwargs["argspec"]
            context = generate.kwargs["context"]

            gen.restart_at(seed)
            context.include_defaults = include_defaults
            call_args = generate_dicts_for_annotations(
                {k: generators.get(k, argspec.annotations.get(k, Any)) for k in argspec.args},
                context,
            )

            arg_groups = [d.values() for d, _ in zip(call_args, range(count))]
            assert arg_groups, "Could not generate test inputs for {}".format(
                context.source.__name__
            )
            metafunc.parametrize(argspec.args, arg_groups)
