import contextlib
import dataclasses
import functools
import inspect
import threading
from typing import Annotated, Any, Callable, ContextManager, Generator, Protocol, TypeVar

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

        assert (
            annotation.origin is None
        ), f"Cannot get_factory {source}, only concrete types or lists of concrete types are supported"
        return FactoryAnnotation(concrete_type=annotation.source, is_collection=False, label="")

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


class Destructor(Protocol):
    def __call__(self) -> None:
        pass


@dataclasses.dataclass
class Injector:
    registry: dict[FactoryAnnotation, Callable] = dataclasses.field(default_factory=dict)

    def initializer(self, c: "_CD") -> "_CD":
        c = inject(c)

        def initialize() -> Destructors:
            rv = c()
            if rv is None:
                return []
            if isinstance(rv, contextlib.AbstractContextManager):
                rv.__enter__()
                return [rv.__exit__]
            if inspect.isgenerator(rv):
                try:
                    next(rv)
                except StopIteration:
                    return []
                return [lambda: list(rv)]
            return []

        self.extension(initialize)

        return c

    def extension(self, c: _C) -> _C:
        assert inspect.isfunction(c), f"{c} is not compatible with extension, functions required"
        c = inject(c)

        key = FactoryAnnotation.from_factory(c)
        assert key.is_collection, f"{c} is compatible with provider method, not extension method"
        existing = self.registry.get(key, lambda: [])

        def extended():
            return [*existing(), *c()]

        self.registry[key] = extended
        return c

    def provider(self, c: _C) -> _C:
        c = inject(c)

        key = FactoryAnnotation.from_factory(c)
        assert (
            not key.is_collection
        ), f"{c} is compatible with extension method, not provider method"
        assert (
            key not in self.registry
        ), f"{key.concrete_type} is already registered for this injector"
        self.registry[key] = c
        return c

    def constant(self, annotation: type[_A], val: _A) -> _A:
        key = FactoryAnnotation.from_annotation(annotation)
        assert (
            not key.is_collection
        ), f"{annotation} is compatible with extension method, not constant method"
        self.registry[key] = lambda: val
        return val

    def enable(self):
        context = _Context()
        with self.install_into_context(context):
            if _cur.context:
                context.parent = _cur.context
            _cur.context = self
        return context

    @contextlib.contextmanager
    def install_into_context(self, context: "_Context"):
        context.injectors[self] = None

        try:
            dependencies: Callable[[], list[Injector]] = self.get_factory(
                FactoryAnnotation.from_annotation(Dependencies)
            )
        except FactoryNotFound:

            def dependencies():
                return []

        with contextlib.ExitStack() as stack:
            for dep in dependencies():
                if dep in context.injectors:
                    continue
                stack.enter_context(dep.install_into_context(context))

            yield

        try:
            destructors = self.get_factory(FactoryAnnotation.from_annotation(Destructors))
        except FactoryNotFound:

            def destructors():
                return []

        context.destructors.extend(destructors())

    def get_factory(self, annotation: FactoryAnnotation) -> Callable:
        try:
            return self.registry[annotation]
        except KeyError:
            raise FactoryNotFound(f"no factory available for {annotation!r}")


Destructors = Annotated[list[Destructor], Labeled("destructors")]
Dependencies = Annotated[list[Injector], Labeled("dependencies")]

_CD = TypeVar("_CD", bound=Callable[[], ContextManager | Generator | None])


class _Injected:
    pass


injected: Any = _Injected()


def inject(c: _A) -> _A:
    original_type = c
    if inspect.isclass(c):
        c = c.__init__

    argspec = inspect.getfullargspec(c)

    @functools.wraps(c)
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

        return c(*args, **new_kwds)

    if inspect.isclass(original_type):
        return type(original_type.__name__, (original_type,), dict(__init__=wrapper))

    return wrapper


def resolve(source: type[_A], key: FactoryAnnotation | None = None) -> _A:
    if _cur.context is None:
        raise FactoryNotFound(
            f"Cannot resolve '{source}', no factory registered for any active injector."
        )

    if key is None:
        key = FactoryAnnotation.from_annotation(source)
    assert key is not None

    if _cur.seen is None:
        _cur.seen = []

    try:
        if key in _cur.seen:
            raise FactoryNotFound(
                f"Circular dependency: {' -> '.join(str(k) for k in _cur.seen)} -> {key}"
            )
        _cur.seen.append(key)

        if key in _cur.context.cache:
            _cur.seen.clear()
            return _cur.context.cache[key]

        for injector in _cur.context.injectors.keys():
            try:
                factory = injector.get_factory(key)
                break
            except FactoryNotFound:
                continue
        else:
            orig = _cur.context
            _cur.context = _cur.context.parent
            try:
                return resolve(source, key)
            finally:
                _cur.context = orig

        rv = _cur.context.cache[key] = factory()
        return rv
    finally:
        _cur.seen.clear()


@dataclasses.dataclass
class _Context:
    injectors: dict[Injector, None] = dataclasses.field(default_factory=dict)
    cache: dict[FactoryAnnotation, Any] = dataclasses.field(default_factory=dict)
    destructors: list[Callable[[], None]] = dataclasses.field(default_factory=list)
    parent: "_Context | None" = dataclasses.field(default_factory=lambda: None)

    def __enter__(self):
        return self

    def disable(self):
        self.__exit__(None, None, None)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        assert _cur.context is self, "injector was prematurely disabled!"

        try:
            for destructor in self.destructors:
                destructor()
        finally:
            _cur.context = self.parent


class _Cur(threading.local):
    context: _Context | None = None
    seen: list[FactoryAnnotation] | None = None


_cur = _Cur()
