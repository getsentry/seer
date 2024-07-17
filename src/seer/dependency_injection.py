"""
Provides a basic dependency injection framework that uses callable annotations
to decide how and when to inject.

You can inject classes and values and lists of either with some basic constructs:

@module.provider
@dataclass
class MyService:
  other_service: OtherService = injected

MyService() # other_service will be instantiated and cached

Overrides/Stubs and tests can be provided via the `stub_module` or creating a `test_module` fixture
(see conftest.py).

You can also inject normal functions, like so:

@inject
def do_setup(a: int, b: MyService = injected):
   ...

do_setup(100) # b will be injected automatically.
"""

import dataclasses
import functools
import inspect
import threading
from typing import Annotated, Any, Callable, TypeVar

from johen.generators.annotations import AnnotationProcessingContext

_A = TypeVar("_A")
_C = TypeVar("_C", bound=Callable[[], Any])
_T = TypeVar("_T", bound=type)


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
    is_type: bool
    label: str

    @classmethod
    def from_annotation(cls, source: Any) -> "FactoryAnnotation":
        annotation = AnnotationProcessingContext.from_source(source)
        if annotation.origin is Annotated:
            label = next((arg.label for arg in annotation.args[1:] if isinstance(arg, Labeled)), "")
            inner = FactoryAnnotation.from_annotation(annotation.args[0])
            assert not inner.label, f"Cannot get_factory {source}: Annotated has embedded Labeled"
            return dataclasses.replace(inner, label=label)
        elif annotation.concretely_implements(list):
            assert (
                len(annotation.args) == 1
            ), f"Cannot get_factory {source}: list requires at least one argument"
            inner = FactoryAnnotation.from_annotation(annotation.args[0])
            assert not inner.label, f"Cannot get_factory {source}: list has embedded Labeled"
            assert (
                not inner.is_collection
            ), f"Cannot get_factory {source}: collections must be of concrete types, not other lists"
            return dataclasses.replace(inner, is_collection=True)
        elif annotation.origin is type:
            assert (
                len(annotation.args) == 1
            ), f"Cannot get_factory {source}: type requires at least one argument"
            inner = FactoryAnnotation.from_annotation(annotation.args[0])
            assert not inner.label, f"Cannot get_factory {source}: type has embedded Labeled"
            assert (
                not inner.is_collection and not inner.is_type
            ), f"Cannot get_factory {source}: type factories must be of concrete types, not lists or other types"
            return dataclasses.replace(inner, is_type=True)

        assert (
            annotation.origin is None
        ), f"Cannot get_factory {source}, only concrete types, type annotations, or lists of concrete types are supported"
        return FactoryAnnotation(
            concrete_type=annotation.source, is_collection=False, is_type=False, label=""
        )

    @classmethod
    def from_factory(cls, c: Callable) -> "FactoryAnnotation":
        argspec = inspect.getfullargspec(c)
        num_arg_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
        num_kwd_defaults = len(argspec.kwonlydefaults) if argspec.kwonlydefaults is not None else 0

        # Constructors have implicit self reference and return annotations -- themselves
        if inspect.isclass(c):
            num_arg_defaults += 1
            rv = c
        else:
            rv = argspec.annotations.get("return", None)
            assert rv is not None, "Cannot decorate function without return annotation"

        assert num_arg_defaults >= len(
            argspec.args
        ), "Cannot decorate function with required positional args"
        assert num_kwd_defaults >= len(
            argspec.kwonlyargs
        ), "Cannot decorate function with required kwd args"
        return FactoryAnnotation.from_annotation(rv)


class FactoryNotFound(Exception):
    pass


@dataclasses.dataclass
class Module:
    registry: dict[FactoryAnnotation, Callable] = dataclasses.field(default_factory=dict)

    def provider(self, c: _C) -> _C:
        c = inject(c)

        key = FactoryAnnotation.from_factory(c)
        assert (
            key not in self.registry
        ), f"{key.concrete_type} is already registered for this injector"
        self.registry[key] = c
        return c

    def constant(self, annotation: type[_A], val: _A) -> _A:
        key = FactoryAnnotation.from_annotation(annotation)
        self.registry[key] = lambda: val
        return val

    def enable(self):
        injector = Injector(self, _cur.injector)
        _cur.injector = injector
        return injector

    def __enter__(self):
        return self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _cur.injector, "Injector state was tampered with, or __exit__ invoked prematurely"
        assert (
            _cur.injector.module is self
        ), "Injector state was tampered with, or __exit__ invoked prematurely"
        _cur.injector = _cur.injector.parent


class _Injected:
    """
    Magical variable indicating that a parameter should be injected when constructed
    by an Injector object.  Invoking a method that uses an `injected` value directly
    will use the currently available injector instance to fill in the default value.
    """

    pass


# Marked as Any so it can be a stand in value for any annotation.
injected: Any = _Injected()


def inject(c: _A) -> _A:
    original_type = c
    if inspect.isclass(c):
        c = c.__init__

    argspec = inspect.getfullargspec(c)

    @functools.wraps(c)  # type: ignore
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        new_kwds = {**kwargs}

        if argspec.defaults:
            offset = len(argspec.args) - len(argspec.defaults)
            for i, d in enumerate(argspec.defaults):
                arg_idx = offset + i
                arg_name = argspec.args[arg_idx]

                if d is injected and len(args) <= arg_idx and arg_name not in new_kwds:
                    try:
                        resolved = resolve(argspec.annotations[arg_name])
                    except KeyError:
                        raise AssertionError(
                            f"Cannot inject argument {arg_name} as it lacks annotations"
                        )

                    new_kwds[arg_name] = resolved

        if argspec.kwonlydefaults:
            for k, v in argspec.kwonlydefaults.items():
                if v is injected and k not in new_kwds:
                    try:
                        new_kwds[k] = resolve(argspec.annotations[k])
                    except KeyError:
                        raise AssertionError(f"Cannot inject argument {k} as it lacks annotations")

        return c(*args, **new_kwds)  # type: ignore

    if inspect.isclass(original_type):
        return type(original_type.__name__, (original_type,), dict(__init__=wrapper))  # type: ignore

    return wrapper  # type: ignore


def resolve(source: type[_A]) -> _A:
    if _cur.injector is None:
        raise FactoryNotFound(f"Cannot resolve '{source}', no module injector is currently active.")

    key = FactoryAnnotation.from_annotation(source)

    if _cur.seen is None:
        _cur.seen = []

    try:
        if key in _cur.seen:
            raise FactoryNotFound(
                f"Circular dependency: {' -> '.join(str(k) for k in _cur.seen)} -> {key}"
            )
        _cur.seen.append(key)
        return _cur.injector.get(source)
    finally:
        _cur.seen.clear()


@dataclasses.dataclass
class Injector:
    module: Module
    parent: "Injector | None"
    _cache: dict[FactoryAnnotation, Any] = dataclasses.field(default_factory=dict)

    @property
    def cache(self) -> dict[FactoryAnnotation, Any]:
        if _cur.injector is not None:
            return _cur.injector._cache
        return self._cache

    def get(self, source: type[_A]) -> _A:
        key = FactoryAnnotation.from_annotation(source)
        if key in self.cache:
            return self.cache[key]

        try:
            f = self.module.registry[key]
        except KeyError:
            if self.parent is not None:
                return self.parent.get(source)
            raise FactoryNotFound(f"No registered factory for {source}")

        rv = self.cache[key] = f()
        return rv


class _Cur(threading.local):
    injector: Injector | None = None
    seen: list[FactoryAnnotation] | None = None


_cur = _Cur()
