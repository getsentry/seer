import dataclasses
import functools
import inspect
import threading
from typing import Annotated, Any, Callable, TypeVar

from johen.generators.annotations import AnnotationProcessingContext

_A = TypeVar("_A")
_T = TypeVar("_T", bound=type)
_C = TypeVar("_C", bound=Callable)


@dataclasses.dataclass
class Labeled:
    """
    Used to 'label' a type so as to have a unique provider when the type itself is not unique.
    eg:

    @inject
    @dataclass
    class Config:
      host: Annotated[str, Labeled("host")]
      protocol: Annotated[str, Labeled("protocol")]
    """

    label: str


@dataclasses.dataclass(frozen=True)
class FactoryAnnotation:
    concrete_type: type
    is_collection: bool
    label: str

    @classmethod
    def from_annotation(cls, source: Any) -> "FactoryAnnotation":
        annotation = AnnotationProcessingContext.from_source(source)
        if annotation.origin is Annotated:
            label = next((arg.label for arg in annotation.args[1:] if isinstance(arg, Labeled)), "")
            inner = FactoryAnnotation.from_annotation(annotation.args[0])
            assert not inner.label, f"Cannot resolve {source}: Annotated has embedded Labeled"
            return dataclasses.replace(inner, label=label)
        elif annotation.concretely_implements(list):
            assert (
                len(annotation.args) == 1
            ), f"Cannot resolve {source}: list requires at least one argument"
            inner = FactoryAnnotation.from_annotation(annotation.args[0])
            assert not inner.label, f"Cannot resolve {source}: list has embedded Labeled"
            assert (
                not inner.is_collection
            ), f"Cannot resolve {source}: collections must be of concrete types, not other lists"
            return dataclasses.replace(inner, is_collection=True)

        assert (
            annotation.origin is None
        ), f"Cannot resolve {source}, only concrete types or lists of concrete types are supported"
        return FactoryAnnotation(concrete_type=annotation.source, is_collection=False, label="")

    @classmethod
    def from_factory(cls, c: _C) -> "FactoryAnnotation":
        argspec = inspect.getfullargspec(c)
        num_arg_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
        num_kwd_defaults = len(argspec.kwonlydefaults) if argspec.defaults is not None else 0
        assert num_arg_defaults >= len(
            argspec.args
        ), "Cannot decorate function with required positional args"
        assert num_kwd_defaults >= len(
            argspec.kwonlyargs
        ), "Cannot decorate function with required kwd args"
        rv = argspec.annotations.get("return", None)
        assert rv is not None, "Cannot decorate function without return annotation"
        return FactoryAnnotation.from_annotation(rv)


class FactoryNotFound(Exception):
    pass


@dataclasses.dataclass
class Injector:
    _registry: dict[FactoryAnnotation, Callable] = dataclasses.field(default_factory=dict)

    def extension(self, c: _C) -> _C:
        key = FactoryAnnotation.from_factory(c)
        c = inject(c)
        assert key.is_collection, f"{c} is compatible with provider method, not extension method"
        existing = self._registry.get(key, lambda: [])

        def extended():
            return [*existing(), *c()]

        self._registry[key] = extended
        return c

    def provider(self, c: _C) -> _C:
        key = FactoryAnnotation.from_factory(c)
        assert (
            not key.is_collection
        ), f"{c} is compatible with extension method, not provider method"
        assert (
            key not in self._registry
        ), f"{key.concrete_type} is already registered for this injector"
        c = inject(c)
        self._registry[key] = c
        return c

    def constant(self, annotation: type[_A], val: _A) -> _A:
        key = FactoryAnnotation.from_annotation(annotation)
        assert (
            not key.is_collection
        ), f"{annotation} is compatible with extension method, not constant method"
        self._registry[key] = lambda: val
        return val

    def __enter__(self):
        if _context.stack is None:
            _context.stack = []
        if _context.cache is None:
            _context.cache = []
        _context.stack.append(self)
        _context.cache.append({})

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        _context.stack.pop()
        _context.cache.pop()

    def resolve(self, source: type[_A]) -> _A:
        try:
            f = self._registry[FactoryAnnotation.from_annotation(source)]
        except KeyError:
            raise FactoryNotFound(
                f"Cannot resolve {source}, no factory available for this type of object."
            )

        return f()


class _Injected:
    pass


injected: Any = _Injected()


def inject(c: _A) -> _A:
    argspec = inspect.getfullargspec(c)

    @functools.wraps(c)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if argspec.defaults:
            new_args = list(args)
            offset = len(argspec.args) - len(argspec.defaults)
            for i, d in enumerate(argspec.defaults):
                if d is injected:
                    try:
                        new_args[offset + i] = resolve(
                            argspec.annotations[argspec.args[offset + i]]
                        )
                    except KeyError:
                        raise AssertionError(
                            f"Cannot inject argument {argspec.args[offset + i]} as it lacks annotations"
                        )
            args = tuple(new_args)
        if argspec.kwonlydefaults:
            new_kwds = {**kwargs}
            for k, v in argspec.kwonlydefaults.items():
                if v is injected:
                    try:
                        new_kwds[k] = resolve(argspec.annotations[k])
                    except KeyError:
                        raise AssertionError(f"Cannot inject argument {k} as it lacks annotations")
            kwds = new_kwds

        return c(*args, **kwds)

    return wrapper


def resolve(source: type[_A]) -> _A:
    if _context.stack is None:
        raise FactoryNotFound(
            "No Injector has been initiated, use `with injector:` to enable an injection context."
        )

    for injector in reversed(_context.stack):
        try:
            return injector.resolve(source)
        except FactoryNotFound:
            continue

    raise FactoryNotFound(
        f"Cannot resolve {source}, no factory registered for any active injector."
    )


class _Context(threading.local):
    stack: list[Injector] | None = None
    cache: list[dict[FactoryAnnotation, Any]] | None = None


_context = _Context()
