import dataclasses
import datetime
import enum
import inspect
import itertools
import math
import random
import string
import struct
import typing
import uuid
import zlib
from typing import Any, Iterator, TypeVar, get_type_hints

import typing_extensions
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

_A = TypeVar("_A")
_B = TypeVar("_B")
_Pydantic = TypeVar("_Pydantic", bound=BaseModel)
_Tuple = TypeVar("_Tuple", bound=tuple)


@dataclasses.dataclass
class _RandomGenerator:
    r: random.Random = dataclasses.field(default_factory=lambda: random.Random())
    max_count: int = 10000000

    def restart_at(self, seed: int) -> typing.Self:
        self.r = random.Random(seed)
        self.max_count = 10000000
        return self

    def __next__(self) -> "random.Random":
        self.max_count -= 1
        if self.max_count <= 0:
            raise ValueError("generator did not find a valid generation")
        return self.r

    def __iter__(self) -> "Iterator[random.Random]":
        return self

    @staticmethod
    def one_of(*options: typing.Iterable[_A]) -> Iterator[_A]:
        composed_options: list[typing.Iterable[_A]] = [
            _RandomGenerator._normalize(i) for i in options
        ]
        return (next(iter(r.choice(composed_options))) for r in gen)

    @staticmethod
    def _normalize(i: typing.Iterable[_A]) -> typing.Iterator[_A]:
        if not hasattr(i, "__next__"):
            i = list(i)
            return (r.choice(i) for r in gen)
        return iter(i)


gen = _RandomGenerator()
unsigned_ints = (r.getrandbits(2 ** r.randint(0, 6)) for r in gen)
negative_ints = (i * -1 for i in unsigned_ints)
ints = gen.one_of(unsigned_ints, negative_ints, (0,))
all_floats = (struct.unpack("d", struct.pack("Q", bits))[0] for bits in unsigned_ints)
floats = (v for v in all_floats if not math.isinf(v) and not math.isnan(v))
bools = gen.one_of([True, False])
objects = (object() for _ in gen)
printable_strings = ("".join(r.sample(string.printable, r.randint(0, 20))) for r in gen)
colors = gen.one_of(
    [
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "cyan",
        "magenta",
        "magenta",
        "yellow",
        "gold",
        "silver",
        "black",
        "white",
    ]
)
names = gen.one_of(
    [
        "bob",
        "alice",
        "jennifer",
        "john",
        "mary",
        "jane",
        "sally",
        "fred",
        "dan",
        "alex",
        "margaret",
        "vincent",
        "timothy",
        "samuel",
    ]
)
things = gen.one_of(
    [
        "shirt",
        "sneaker",
        "shoe",
        "apple",
        "banana",
        "orange",
        "tea",
        "sandwich",
        "tennis",
        "football",
        "basketball",
        "fork",
        "table",
        "computer",
    ]
)
ascii_words = ("-".join(group) for group in zip(colors, names, things))
datetimes = (
    datetime.datetime(2013, 1, 1, 1)
    + datetime.timedelta(
        days=r.randint(0, 365 * 20),
        seconds=r.randint(0, 60 * 60 * 24),
        milliseconds=r.randint(0, 1000),
    )
    for r in gen
)
positive_timedeltas = (
    datetime.timedelta(seconds=r.randint(0, 59), hours=r.randint(0, 23)) for r in gen
)

uuids = (uuid.UUID(int=r.getrandbits(128), version=4) for r in gen)
uuid_hexes = (uid.hex for uid in uuids)

file_extensions = gen.one_of(
    (".jpg", ".png", ".gif", ".txt", ".py", ".ts", ".c", ".obj", ".ini", "")
)
path_segments = gen.one_of(
    (
        ".",
        "..",
        "tmp",
        "var",
        "usr",
        "Home",
        "data",
        "volumes",
        "etc",
        "tests",
        "src",
        "db",
        "conf",
        "events",
        "utils",
        "app",
        "versions",
        "models",
    ),
    uuid_hexes,
)
file_paths = (
    lead + "/".join(segment for _, segment in segments) + ext
    for lead, segments, ext in zip(
        gen.one_of(("", "/")),
        (zip(range(r.randint(1, 8)), path_segments) for r in gen),
        file_extensions,
    )
)
file_names = (segment + ext for segment, ext in zip(path_segments, file_extensions))


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
    annotations: typing.Mapping[str, Any],
    context: "GeneratorContext",
    optional_keys: list[str],
) -> Iterator[dict[str, Any]]:
    if not annotations:
        return ({} for _ in gen)

    generators: dict[str, Iterator[Any]] = {
        k: generate(context.step(v, k))
        for k, v in annotations.items()
        if context.include_defaults or k not in optional_keys
    }

    for k, v in generators.items():
        assert hasattr(v, "__next__")

    sampled_keys: Iterator[list[str]]
    if context.include_defaults == "holes":
        sampled_keys = (r.sample(optional_keys, r.randint(0, len(optional_keys))) for r in gen)
    elif context.include_defaults:
        sampled_keys = itertools.repeat(optional_keys)
    else:
        sampled_keys = itertools.repeat([])

    return (
        dict(
            (k, v)
            for k, v in zip(generators.keys(), values)
            if k not in optional_keys or k in included_keys
        )
        for values, included_keys in zip(zip(*generators.values()), sampled_keys)
    )


def generate_dicts_for_pydantic_model(
    context: "GeneratorContext",
) -> Iterator[dict[str, Any]]:
    hints = get_type_hints(context.source, include_extras=True)
    return generate_dicts_for_annotations(
        {k: hints.get(k, Any) for k, field in context.source.model_fields.items()},
        context,
        optional_keys=[
            k for k, field in context.source.model_fields.items() if _pydantic_has_default(field)
        ],
    )


def generate_dicts_for_dataclass_model(
    context: "GeneratorContext",
) -> Iterator[dict[str, Any]]:
    hints = get_type_hints(context.source, include_extras=True)
    fields = {f.name: f for f in dataclasses.fields(context.source)}
    return generate_dicts_for_annotations(
        {k: hints.get(k, Any) for k, field in fields.items()},
        context,
        optional_keys=[k for k, field in fields.items() if _dataclass_has_default(field)],
    )


def generate_call_args_for_argspec(
    source: inspect.FullArgSpec, context: "GeneratorContext"
) -> typing.Iterator[tuple[tuple[Any, ...], dict[str, Any]]]:
    num_arg_defaults = len(source.defaults) if source.defaults is not None else 0

    arg_generators: list[Iterator[Any]] = []
    for i, argname in enumerate(source.args):
        if i < (len(source.args) - num_arg_defaults) or context.include_defaults:
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


def generate_literals(context: "GeneratorContext") -> Iterator[Any] | None:
    if context.origin is typing.Literal:
        return gen.one_of(context.args)
    return None


@dataclasses.dataclass
class GeneratorContext:
    source: Any
    origin: Any | None
    args: tuple[Any, ...]
    context: tuple[str, ...] = dataclasses.field(default_factory=tuple)
    include_defaults: bool | typing.Literal["holes"] = False

    generators: list[Generator] = dataclasses.field(
        default_factory=lambda: [
            generate_literals,
            generate_iterators,
            generate_pydantic_instances,
            generate_dataclass_instances,
            generate_dicts_from_typeddict,
            generate_sqlalchemy_instance,
            generate_dicts,
            generate_enums,
            generate_results_from_call,
            generate_tuples,
            generate_unions,
            generate_lists,
            generate_annotated,
            generate_primitives,
            generate_any,
            generate_unexpected_annotation,
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
    if typing.is_typeddict(context.source) or typing_extensions.is_typeddict(context.source):
        optional: list[str] = sorted(getattr(context.source, "__optional_keys__", frozenset()))
        hints = get_type_hints(context.source, include_extras=True)
        return generate_dicts_for_annotations(
            {k: v for k, v in hints.items()}, context, optional_keys=list(optional)
        )

    return None


def generate_dicts(context: GeneratorContext) -> Iterator[dict[Any, Any]] | None:
    if any(
        source and inspect.isclass(source) and issubclass(source, typing.Mapping)
        for source in (context.source, context.origin)
    ):
        key, value, *_ = (*context.args, str, str)
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
        return gen.one_of(*(generate(context.step(arg, f"|")) for arg in context.args))
    return None


def generate_lists(context: GeneratorContext) -> Iterator[Any] | None:
    if context.origin is list:
        arg = [*context.args, object][0]
        generator = generate(context.step(arg))
        return ([i for i, _ in zip(generator, range(r.randint(0, 10)))] for r in gen)
    return None


def generate_unexpected_annotation(context: GeneratorContext) -> typing.Iterator[Any] | None:
    # Assume that the first argument is the actual type to generate
    if context.origin is not None and context.args is not None:
        annotated_inner = [*context.args, Any][0]
        return generate(context.step(annotated_inner))
    return None


def generate_annotated(context: GeneratorContext) -> typing.Iterator[Any] | None:
    if context.origin is typing.Annotated:
        annotated_inner = [*context.args, Any][0]
        examples = (v for ex in context.args[1:] if isinstance(ex, Examples) for v in ex)
        if examples:
            return gen.one_of(*examples)

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
    if source is uuid.UUID:
        return uuids
    if source is bool:
        return bools
    if source is object:
        return objects
    if source is float:
        return floats
    if source is datetime.datetime:
        return datetimes
    if source is datetime.timedelta:
        return positive_timedeltas
    if source is type(None):
        return itertools.repeat(None)
    return None


def generate_any(context: GeneratorContext) -> typing.Iterator[Any] | None:
    if context.source is typing.Any:
        return gen.one_of(ascii_words, ints, floats, bools)
    return None


def generate_sqlalchemy_instance(context: GeneratorContext) -> typing.Iterator[Any] | None:
    import sqlalchemy
    import sqlalchemy.orm

    if inspect.isclass(context.source) and issubclass(
        context.source, sqlalchemy.orm.DeclarativeBase
    ):
        hints = get_type_hints(context.source, include_extras=True)
        inspection = sqlalchemy.inspect(context.source)
        dict_generator = generate_dicts_for_annotations(
            {
                c.key: next(iter(typing.get_args(hint)), Any)
                if typing.get_origin(hint) is sqlalchemy.orm.Mapped
                else hint
                for c in inspection.c
                for hint in (hints.get(c.key, Any),)
            },
            context,
            optional_keys=[
                c.key
                for c in inspection.c
                if (
                    c.primary_key
                    or c.nullable
                    or c.default is not None
                    or c.server_default is not None
                )
            ],
        )

        return (context.source(**d) for d in dict_generator)
    return None


def generate(
    context: GeneratorContext | Any,
    include_defaults: bool | typing.Literal["holes"] = False,
) -> typing.Iterator[Any]:
    if not isinstance(context, GeneratorContext):
        gen_context = GeneratorContext.from_source(context)
        gen_context.include_defaults = include_defaults
    else:
        gen_context = context

    for generator in gen_context.generators:
        result = generator(gen_context)
        if result is not None:
            return result
    raise TypeError(f"Could not generate for {' '.join(gen_context.context)} {gen_context.source}")


def parameterize(
    _arg: Any = None,
    *,
    seed: int | None = None,
    count: int = 10,
    include_defaults: bool | typing.Literal["holes"] = False,
    arg_set: typing.Sequence[str] | None = None,
    generators: typing.Sequence[Generator] = (),
    **parameter_overrides: Any,
) -> typing.Callable:
    import pytest

    if _arg is not None:
        return parameterize()(_arg)

    def decorator(func: typing.Callable[..., Any]) -> Any:
        argspec = inspect.getfullargspec(func)
        injected_args: list[str] = sorted(arg_set if arg_set is not None else argspec.args)
        if invalid_arg := next((k not in injected_args for k in parameter_overrides.keys()), None):
            raise ValueError(
                f"Argument {invalid_arg} cannot be injected, check your arg_set and your kwd arguments to parameterize."
            )

        if seed is None:
            final_seed = zlib.crc32(func.__name__.encode("utf8")) & 0xFFFFFFFF
        else:
            final_seed = seed

        def generate() -> typing.Iterator[typing.Sequence[Any]]:
            gen.restart_at(final_seed)
            context = GeneratorContext.from_source(func)
            context.include_defaults = include_defaults
            context.generators = [*generators, *context.generators]
            call_args = generate_dicts_for_annotations(
                {
                    k: parameter_overrides.get(k, argspec.annotations.get(k, Any))
                    for k in injected_args
                },
                context,
                optional_keys=[],
            )

            i = 0
            for d, _ in zip(call_args, range(count)):
                yield list(d.values())
                i += 1
            if i < count:
                raise ValueError(
                    f"Failed to generator {count} test cases for {func.__name__} ({i}), check that constraint is not too strong."
                )

        return pytest.mark.parametrize(injected_args, list(generate()))(func)

    return decorator


class _Unset:
    pass


_unset = _Unset()


@dataclasses.dataclass
class change_watcher:
    cb: typing.Callable[[], Any]
    stack: list[Any] = dataclasses.field(default_factory=list)

    def __enter__(self):
        self.stack.append(ChangeResult(orig=self.cb()))
        return self.stack[-1]

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return None

        self.stack.pop().result = self.cb()


@dataclasses.dataclass
class NamedBool:
    message: str
    result: bool

    def __bool__(self):
        return self.result

    def __str__(self):
        return self.message

    __repr__ = __str__


@dataclasses.dataclass
class ChangeResult:
    orig: Any = _unset
    result: Any = _unset

    def from_value(self, value: Any):
        return NamedBool(f"{self} (expected from: {value!r})", bool(self and self.orig == value))

    def to_value(self, value: Any):
        return NamedBool(
            f"{self} (expected result: {value!r})", bool(self and self.result == value)
        )

    def __bool__(self) -> bool:
        assert (
            self.orig is not _unset
        ), "ChangeWatcher.__enter__ was not called, cannot compute result!"
        assert (
            self.result is not _unset
        ), "ChangeWatcher.__exit__ was not called, cannot compute result!"
        return self.result != self.orig

    def __str__(self) -> str:
        assert (
            self.orig is not _unset
        ), "ChangeWatcher.__enter__ was not called, cannot compute result!"
        assert (
            self.result is not _unset
        ), "ChangeWatcher.__exit__ was not called, cannot compute result!"
        return f"{self.orig!r} changed to {self.result!r}"

    __repr__ = __str__
