# Dependency Injection in Seer

[Dependency Injection](https://en.wikipedia.org/wiki/Dependency_injection) is a technique in which shared objects
and configuration used in a program are "injected" into functions on a need to know basis, separating the concerns
of instantiation and construction of objects, from defining the dependent components.  This is opposed to functions
or classes directly accessing those shared objects via global state.

This concept is actually very similar to [useContext](https://react.dev/reference/react/useContext) in React: consumers
of values are separated from providers rather than shared through global state.  The difference comes down to how
these abstraction relationships (consumers and providers) are eventually bound together.

# Quick Start

For shared configuration, mutable caches, or mockable clients, create a provider:

```python
module = Module()
module.enable()

@module.provider
def my_cache() -> Cache:
    return Cache()
```

Consume that provider downstream with `inject` and `injected`:

```python
@inject
def my_func(a: Cache = injected):
    a.set('key', 'value')
```

By using dependency injection, your shared objects will reset between tests, be available 'globally' in any method
that wants it via `injected`, and is replaceable in tests with mock replacements.

# Long walk explanation of `dependency_injection.py`

## inject and injected

Let's start with the most common example in `configuration.py`:

```python
class AppConfig(BaseModel):
    SEER_VERSION_SHA: str = ""

    SENTRY_DSN: str = ""
    SENTRY_ENVIRONMENT: str = "production"
```

`AppConfig` is a very useful broad set of application settings that we load from `os.environ` on bootup.  It is shared
and common to many different components.  How do we share access to the `AppConfig` instance?

Here's an example from `seer/automation/utils.py`:

```python
@inject
def check_genai_consent(
    org_id: int, client: RpcClient = injected, config: AppConfig = injected
) -> bool:
```

A method decorated with `inject` signifies that this method can receive "injected" instance values.  Those are denoted
by **any and all keyword arguments whose default value is `injected`**.  So what's going on?  Let's break it down by
reading `dependency_injection.py` some:

```python
def inject(c: _A) -> _A:
    ...
    argspec = inspect.getfullargspec(c)

    @functools.wraps(c)  # type: ignore
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ...
        new_kwds = {**kwargs}
        if argspec.kwonlydefaults:
            for k, v in argspec.kwonlydefaults.items():
                if v is injected and k not in new_kwds:
                    try:
                        new_kwds[k] = resolve(argspec.annotations[k])
                    except KeyError:
                        raise AssertionError(f"Cannot inject argument {k} as it lacks annotations")
        ...
        return c(*args, **new_kwds)  # type: ignore

    return wrapper  # type: ignore
```

Essentially `inject` "wraps" the given function or method such that it inspects **kwds** and replaces any
`kwonlydefaults` (that is, keywords that have defaults), with this logic:

```python
if v is injected and k not in new_kwds:
    try:
        new_kwds[k] = resolve(argspec.annotations[k])
```

`resolve` in this case is the function that "determines" what value to select for an `injected` value.  Note
that it is acting on the `annotation` of the keyword argument!  This is important: dependency injection depends
entirely on the annotation to do the binding, not the name of the argument iself.

But what is `injected`?  Well, it's rather simple:

```python
class _Injected:
    ...
    pass

# Marked as Any so it can be a stand in value for any annotation.
injected: Any = _Injected()
```

`injected` is literally a magical value that means nothing and will type match any annotation safely.  It's only
purpose is to signal to the decorator that **you didn't put some other value into the keyword argument!**  For instance,
consider these three cases:

```python
check_genai_consent(1)
check_genai_consent(1, my_rpc_client)
check_genai_consent(1, client=my_rpc_client)
```

In the first case, `client` will be `injected`, and thus have its value replaced with a call to `resolve`.  However,
in the other two cases, client will merely be `my_rpc_client` because that's what you provided!

In this way, our dependency injection has flexibility: it is only providing a default when no other is given.

## resolve

But how on earth does `resolve` come up with a "default value"?  Let's take a look in `dependency_injection.py` again:

```python
def resolve(source: type[_A]) -> _A:
    key = FactoryAnnotation.from_annotation(source)
    return resolve_annotation(key, source)
```

Not much here, but the key insight is that we're doing a transformation on, again, the `annotation`, or type structure,
of the input, in order to make a selection.  This transformed value is known as a `FactoryAnnotation` in this case
(implementation detail, other implementations may use different names).  Let's take a peak at `resolve_annotation`.

```python
def resolve_annotation(key: FactoryAnnotation, source: Any) -> Any:
    ...
    return _cur.injector.get(source, key=key)
```

Alot is happening in this function related to sanity checking failure conditions, but the crucial line involves
`_cur.injector`.  Again, it feels like we're just passing stuff around, so we need to understand the kernel here:

**What is _cur and what is its injector**?

## _cur and injector

`_cur` is defined as such:

```python
class _Cur(threading.local):
    injector: Injector | None = None
    seen: list[FactoryAnnotation] | None = None

_cur = _Cur()
```

Let's not worry too much about `seen` (it is related to sanity checking circular refences).  There are two things here
worth focusing on:

1.  The `injector` is an, well, `Injector`.  This is ultimately where values come from in dependency injection.  You can
think of an injector as a "context" the sits intermediate to `inject` and `injected`.
2.  The fact that `_cur: _Cur` is a subclass of `threading.local`.  What does that mean?  Well, essentially, each
mutation of `_cur` **only affects the thread the mutation is running in**.  In practice, this means that using this
library, you will need to initialize injectors independently between threads.  We'll discuss `Injector` objects here
shortly.

So all in all, `_cur` is some state that carries an `Injector`, cool.  Well, *what is an `Injector`????*

```python
@dataclasses.dataclass
class Injector:
    module: Module
    parent: "Injector | None"
    _cache: dict[FactoryAnnotation, Any] = dataclasses.field(default_factory=dict)
    ...
    def get(self, source: type[_A], key: FactoryAnnotation | None = None) -> _A:
        ...
        if key in self.cache:
            return self.cache[key]

        try:
            f = self.module.registry[key]
        except KeyError:
            if self.parent is not None:
                return self.parent.get(source, key=key)
            raise FactoryNotFound(f"No registered factory for {source}")

        rv = self.cache[key] = f()
        return rv
```

Taking a look at the members, there's some things to identify.

1.  `Injector`s have an optional `parent: Injector`, implying naturally that injectors *can be stacked*.
2.  `Injector`s have a reference to a `Module`, which we will need to explore a bit more.  This also implies given #1
that modules can be arranged via `Injectors` in a stacked fashion.
3.  `_cache: dict[FactoryAnnotation, Any]` seems to map those `FactoryAnnotation` values (the type annotations from
`resolve` invocations, as we saw earlier).

As it turns out, we'll see exactly what the `_cache`, `parent`, and `module` values are, and what they're doing
by breaking down the `get` method:

```python
if key in self.cache:
    return self.cache[key]
```

For a given `injector` instance, if the annotation **already has a computed value**, just return that.  Let's ignore
the fact that `_cache` is abstracted by another property `cache` for now and treat them the same.  The key insight
is that the `cache` acts as a buffer so that once a value is constructed for a type annotation, **we re-use and share
it** to all `injected` call sites.  That is, *so long as the injector instance remains*.  Since this state is tied
to a currently active injector, the implication is that the cache is, in fact, *scope sensitive*.  We'll revisit this
later.

## registry and cache

So how do values get into the `cache`?  Well,

```python
try:
    f = self.module.registry[key]
except KeyError:
    if self.parent is not None:
        return self.parent.get(source, key=key)
    raise FactoryNotFound(f"No registered factory for {source}")

rv = self.cache[key] = f()
```

Now we see `module` and `parent` come into play!  First, we ask if the `module` associated with this injector has a
value `f` for the type annotation.  If it does not (`KeyError`), we... ask the parent to `get` the value instead!

In essence, `parent: Injector | None`  helps us chain a stacked context, preferring the last in value that is available.
This means in fact that there can be multiple `registry` entries for a given `key`, where we prefer the injector at the
"top" of the stack.  It also means that if the "top" injector does not have a value for the annotation, a previous
injector can be used.  This sort of nested contextual scope mimics the way that
[symbolic values can be shadowed](https://en.wikipedia.org/wiki/Variable_shadowing), but in this case the scope is the
injector and the symbol is not a variable name, but a type annotation.

How does this relate to the `cache`, and what is an `f` value?  Well,

```python
rv = self.cache[key] = f()
return rv
```

`f` is often known as a "factory", but it can also be known as a "thunk", or in our case as we'll show later, we can
call it a `provider`.  Whatever you may call it, it is **deferred evaluation** since we care about caching not `f`
directly, but **the result of invoking f at the time that it is requested**.  We store it in the cache and make the
same instance value available everywhere this injector is used.

This is, by the way, *very similar to how @cached_property works*!  In the case of `cached_property` the scope of the
delayed evaluation is the instance, in our case, the scope is the `injector`.

We're making progress here, but we still have to tie two loose ends up.  How does a `Module` work, specifically, how
does its `registry` work?  And, how does `_cur.injector` and `injector.parent` get set?

## Modules and registry

Let's take a look at the top half of `Module`:

```python
@dataclasses.dataclass
class Module:
    registry: dict[FactoryAnnotation, Callable] = dataclasses.field(default_factory=dict)
```

First off, in essence, a `Module`, *is* a `registry` object, that's basically its entire state.  A registry,
*is* a mapping between `FactoryAnnotation` (type annotations) to a `Callable` (an `f` value from above).

Essentially... a registry tracks how you can produce a value for a type annotation by *calling* something!
That's it!  But how do these Callables, these factories, get assigned into a module?

```python
    def provider(self, c: _C) -> _C:
        ...

    def constant(self, annotation: type[_A], val: _A) -> _A:
        key = FactoryAnnotation.from_annotation(annotation)
        self.registry[key] = lambda: val
        return self
```

`provider` and `constant` are ways of providing an item to the `registry`, so that an injector can `resolve` a
type annotation.  Let's check some real life examples of usage:

In `gcs.py`:

```python
module = Module()
module.enable()

...

@module.provider
def gcs_client(config: EnvConfig = injected) -> GcsStorageClient:
    return storage.Client(project=config.GCS_PROJECT_ID).bucket(config.GCS_BUCKET_NAME)
```

We instantiate a `Module` object into `module`, and we decorate `gcs_client` with `module.provider`.

Checking out the definition of `provider` from above

```python
def provider(self, c: _C) -> _C:
    c = inject(c)
    key = FactoryAnnotation.from_factory(c)
    ...
    self.registry[key] = c
    return c
```

we see a couple of important things.  One, a function decorated with `@module.provider` (the `c` here) *also* is
decorated with `inject`.   That explains why even a `@module.provider` definition can also use `injected` to get other
values.

We also seem to pull the FactoryAnnotation from the function somehow... how? Long story short, `from_factory` pulls the annotation **from the return type** of the function definition.  Whatever
your function says it is returning, is the type annotation that can be used to retrieve it later
(with `injected` or `resolve`).

Then lastly, this wrapped `inject` wrapped `c` is put into the registry by the annotation.  Taking a look at
the `constant` method, we can see that `constant` is basically just a way to put a specific instance or value into
the registry without a function definition along the way.

So we've seen how to add definitions to a `Module`'s registry, and we've seen how the type annotation of a provider
function's return type can correspond with the type annotation of an `injected` parameter to an `inject` method, but
we're still unclear about this line in particular:

```python
module.enable()
```

So let's talk about how Injectors are created, what their relationship to `Module`s are, and how that defines "scope"
for a dependency injection.

## Module __enter__, enable, and _cur.injector

A module merely stores a registry, but to an `injected` values from the intermediate injector stack attached to `_cur`.
So at what point does the `Module` find itself inside of an `Injector` or the `_cur`?

The answer to that is defined on the `Module` class:

```python
    def enable(self):
        injector = Injector(self, _cur.injector)
        _cur.injector = injector
        return injector

    def __enter__(self):
        return self.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...
        _cur.injector = _cur.injector.parent
```

Firstly, it's worth remembering that `__enter__` and `__exit__` correspond with python
[context managers](https://docs.python.org/3/library/contextlib.html#module-contextlib), which is to say, they
correspond to entering and exiting `with` constructions.  So in essence, using a `Module` in a `with` statement
invokes the `__enter__` followed by `__exit__`.

Note, too, that `__enter__` *is* `enable()`.  In many ways, invoking `module.enable()` outside of a `with` is merely
saying, "this module is always enabled" since it won't implicitly invoke `__exit__`.

So that's actually happening in `enable`?

Well... the obvious thing!  We create an `Injector` from the `Module` object being invoked, passing in `_cur.injector`
as the second parameter (the `parent: Injector` attribute!), and we set the `_cur.injector` to this value.

And for `__exit__`?

Well.. it undoes that!  it pulls the `parent` of the `_cur.injector` and attaches that, to, `_cur.injector`.

If this all seems familiar, it should be -- this is 100% a linked list stack implementation.  The stack is the scope
context of `Injector` objects that join `Module` and `cache` values.  When you `__enter__` module, you get a fresh empty
`cache` (which implies future `injected` values are going to be reconstructed rather than shared), and potentially you
have new definitions from the `Module.registry`.

The whole construction provides the core parts of dependency injection we care about:

1.  The ability to override definitions by using `__enter__` or `enable` on a module.
2.  The ability to reconstruct all shared objects by using `__enter__` or `enable` on a module.
3.  The ability to share dependencies that are loosely coupled -- only the type annotations must be the same.

Item 3 here is probably the most obviously useful, since it allows us to carry `AppConfig` values deep into the program
(the most common use case), but why is 1 and 2 useful?

The answer most commonly has to do with tests.

## Motivation

For example, consider this classic pattern

```python
def my_method(a: int):
    if settings.DISABLE:
        return False
    return True
```

In this case, `my_method` has an explicit dependency on the `settings` instance object, and its DISABLE value.

This is relatively fine for simple cases, but it can start to get tedious in special cases.

* Q: What if you need to test with specialized settings objects or settings values?
* A: You could use mocks, but dynamic mocks can't have their type signatures validated, and in general are brittle in
the face of structural change (often creating false positives or just a lot of refactoring).

The need to contextually "replace" a part of a system with a variant is known as the [The Robot Leg Problem](https://github.com/google/guice/wiki/FrequentlyAskedQuestions#how-do-i-build-two-similar-but-slightly-different-trees-of-objects)
and can show up in numerous places, including tests and nested scopes.

* Q: What if `settings` object is stateful and mutated during the program's run?  How can I ensure that mutations
don't result in test polution (a situation where one previous setting updates a value and the next test only passes)
due to the previous test's mutation.
* A: You could manually reset the settings object between tests with a fixture, but that is only as good as your
ability to identify each part of your settings object that could be mutated or cached somewhere.

The need to 'reset' or 'clear' state between tests is not a small issue: it's the issue of lifecycles and is an
important part of ensuring that adding and changing new tests remains stable with respect to its shared state.

### Passing Dependencies as the Alternative

A wise solution to this problem is simply to make the settings object an explicit parameter rather than shared state
at all.  Consider this approach:

```python
def my_method(a: int, settings: Settings):
    if settings.DISABLE:
        return False
    return True
```

```python
def test_my_method():
   assert not my_method(10, Settings(DISABLE=False))
   assert my_method(10, Settings(DISABLE=True))
```

Nice!  This approach is actually great.  Explicit parameters are obvious, easy to use, and naturally deal with both
the Robot Leg problem and the Lifecycle problem directly.  **If you can get away with solving any problem with a simple
function, do it**.

However, there are some caveats here.  What if `my_method` isn't always called directly by you or your test?  For
instance, suppose that `my_method` is celery task that is invoked by your celery runner.  Suppose `my_method` is a
callback that doesn't provide any easy way to control its parameterization?   Or suppose that the mutated
state isn't explicit, but merely implicit due to some other process side effect?  You may have trouble finding a way
to thread your parameter in all the right places to maintain internal consistency of the instance:

```python
def test_my_method():
    settings = Settings(DISABLE=False)
    assert not my_method(10, settings)
    assert my_method(10, settings)
    with celery_runner():
        # How can I ensure the settings instance passed to the celery job is the same?
        assert settings.CACHE[key] == value
```


### Dependency Injection as the Alternative

With dependency injection, you build upon the option of passing in parameters directly with the concept of "injection",
the idea that a context can provide a default for a given value type.

```python
@inject
def my_method(a: int, settings: Settings = injected):
    if settings.DISABLE:
        return False
    return True
```

The `@inject` decorator is responsible for identifying any keyword argument whose default value is the special
`injected` value.  Any call to `my_method` that *does not specify the settings value directly* instead receive
a shared instance value to a `Settings` object.

As above, we can still control the value directly by passing in a Settings object, ignoring the need to use a mock.
But in cases where constructing a meaningful Settings object is hard, or where the invocation is indirect, you can
still bind a value to the invocation that uses the default.

```python
def unit_test_my_method():
    # You can provide a settings value directly for a simple unit test, to validate a special behavior
    assert not my_method(10, settings=Settings(DISABLE=True))
    # Or you can simply allow the inject decorator to provide the value directly.
    with module.constant(Settings, Settings(DISABLE=True)):
        assert my_method(10)
        with celery_runner():
            # Some implicitly run celery job will still receive the same configured Settings object above
            assert resolve(Settings).CACHE[key] == value
```

There is some additional complexity created here, but there is some gained flexibility in the ability to control
life cycles and the ability to inject and replace shared values.

## How does `inject` decide which value to select for an `injected` parameter?

As mentioned in the above example, we can use the concept of a `module` to define the context in which an `injected`
value can be replaced.  A `module` in dependency injection parlance is actually very similar to a typical python
module file, in the sense that it "stores" definitions that can be used.  The difference between a typical
python module file and a dependency injection `module` is how a definition is bound and referenced.

For a typical python module, upon loading it, any variable, function, or class that module references, is immediately
bound by its name.  A `module` in dependency injection has definitions, called `providers`, *but consumers do not hold
references to any particular definition or instantiation*.  Instead, all parameters whose default value is `injected`
resolves their final instantiation **via the type annotation itself**.

As an example, let us setup a module with some providers.

```python
module = Module()

@module.provider
def build_settings() -> Settings:
    return Settings(
        CACHE={},
        ENABLED=False
    )
```

The `module.provider` decorator is binding the **return type**  of the function (`Settings`, the type constructor), to
its returned value (`Settings(**kwds)` the value).  Instantiation does not happen immediately, but rather,
on demand, when `injected` needs resolution.

```python
@inject
def my_method(settings: Settings = injected):
    print(settings.CACHE)
```

A `module` in dependency injection controls both the definitions (the providers) as well as the scope in which values
are applied.  If you call `my_method` and you run into a `FactoryNotFound` exception, it means you have either
1. Not defined a `provider` for the type you are injecting or
2. Not "enabled" that module in your current context.

What does "enabling" a module mean?  Well, if you check seer source code files, you'll often see a line that looks like
`module.enable()`.  This essentially places all of that modules providers into scope for `injected` values of `inject`
methods.

```python
module.enable()
my_method() # now @inject will find `Settings` provider from above.
```

What a module.enable does is "enter" the context of that module, which has two effects:

1.  All previously instantiated values from providers will be ignored, future injection will create new instantiations
2.  All `provider` definitions are available to bind to future types.  Any conflicts between definitions for the same
type will be resolved with the "last" enabled module's providers.

Note that it is perfectly ok to enable a module before definitions are added, say at the top of a file,
but that enable and the provider definitions must occur **before invoking a method that depends upon them**.

Also note that `module.enable()` **only applies to the current thread!**  This means that if you run a multi-threaded
application, you may need to "enable" modules again when you enter new threads.  This is meant to ensure no unintended
cross thread state sharing.  If you must share state between threads, consider using Queues or other concurrency
primitives.

## Differentiating values on common types

In the easy case, dependency injection provides **one value** for any given type annotation, usually because that type
is essentially a Singleton.

For instance, there can only be one provider of type `int` available at a time, meaning if you need to provide
two `int` type bindings, only the most recent would actually work.

To avoid these conflicts, there is an "escape hatch" in which you can label a type in such a way that it does not
conflict with another type by the same name.

```python
LogLevel = Annotated[int, Labeled("log_level")]

@inject
def setup_logging(log_level: LogLevel = injected):
    ...
```

In this way, the underlying type is still `int` but it will be resolved using the labeled type alias `LogLevel`.
In general, I don't recommend using this approach except as an absolute last resort.  Better is often to simply
organize your injected types into unique domain objects bundling multiple values together.  For instance the
`AppConfig` type in seer, which includes many strings and ints organized together.

# Usage

## I want to override a value in a test

You can easily override an injected value in a test with 3 patterns
1.  an override module that includes a separate provider
2.  an adhoc module with a constant that replaces a provider in a test
3.  `resolve` to grab a shared reference and mutate directly in atest

```python
module = Module()
module.enable()

# We dont' enable this module directly, we save it for tests below
my_override_module = Module()

@dataclass
class Settings:
    a: int
    b: str

@module.provider
def default_settings() -> Settings:
    return Settings(a=2, b=3)

@my_override_module.provider
def override_settings() -> Settings:
    orig = default_settings()
    orig.b += 1
    return orig

@inject
def does_a_thing(settings: Settings = injected):
    ...

def test_does_a_thing():
    does_a_thing() # uses existing settings
    # .constant() is a utility provider that is a value instead of a function
    with Module().constant(Settings, Settings(a=10, b=20)):
      # The override only applies to this context manager's scope
        does_a_thing()

    # You can also setup a module
    with my_override_module:
        does_a_thing()

    # `resolve` directly grabs the current instance instead of overriding it,
    # so you can use this as a quick way to mutate a value in the current context
    resolve(Settings).a = 1
    does_a_thing()
```

## I want to implement an IO client that I can mock easily in tests

For instance, in development you may have an `LLMClient` that acts non-deterministically and requires secrets.  In order
to test the relationship of components without having to hard code mocks over and over, you can use modules as a means
of replacing implementations in a modular way.

```python
module = Module()
stub_module = Module()

module.enable()

class MyClient(abc.ABC):
    @abc.abstractmethod
    def invoke_llm(self, prompt: str) -> str:
        pass


class ProductionClient(MyClient):
    def invoke_llm(self, prompt: str) -> str:
        return self.openai.run(prompt)

@module.provider
def get_production_client() -> MyClient:
    return ProductionClient()

class StubClient(MyClient):
    def invoke_llm(self, prompt: str) -> str:
        return "this is a stub response"

@stub_module.provider
def get_stub_client() -> MyClient:
    return StubClient()
```

Notice how we only enable the module with the production client -- we're considering this the base case.  A test can
then use the `stub_module` to substitute on the fly:

```python
def my_test():
    with stub_module:
        do_logic()
```

Sometimes, you know that most if not all the tests would prefer the stub client anyways, in which cae you can easily
set this up using pytest fixtures:

```python
@pytest.fixture
def setup_stub_module():
    with stub_module:
        yield
```

In which case if a test wants the "real" client again, they do the inverse:

```python
def this_test_with_real_client():
    with module:
        ...
```

# Caveats

It is important to note that there are some "gotchas" and "caveats" to practical use of dependency injection,
especially in python.

## Type Generics

For one, seer's dependency injection *does not support generics or type parameters*!

For instance, it does not support `set[str]`, or `Iterable[int]`, or `MyClass[A]` sorts of types.  In general,
dependency injection works best on singleton, concrete types.

That said, seer's dependency injection does support 2 specific generics:

1.  `list` generics are supported, but the inner type must be concrete.  `list[list[str]]` does not work.
2.  `type` generics are supported, but again the inner type must be concrete. `type[MyClass]` is ok but not `type[list[str]]`.

## Module loading, circular dependencies, organization.

In an ideal world, python would separate symbolic loading from running a program, meaning there would be second order
resolution of symbols that can avoid circular references.  In reality, that isn't true.

Thanks to this, it is ideal in many cases for each python module to have its own `Module` object.  However, in some
cases it's still fine to just import the `seer.app.module` object and attach to that directly, if your import ordering
is safe.

Also note: you *need* your program to startup and load all `Module()` objects.  This is fine normally, as when you import
a type name, you tend to also import its python module, which should enable that module object.  But in some
theoritical lazy loading, circular, or "dynamic loading" situations, this can cause a problem.

Word of advice: the greatest extent possible, force explicit module loading in the order you need to.  And if you run
into a circular loading loop, you may have to put your loading inside of a function.  Until python changes the rules
for symbol loading, you may not have any other option :(

## Startup, loading

As mentioned before, dependency injection really depends heavily on being able to separate the symbolic linking
that happens in normal python module loading from the delayed, nominal type linking that happens via injection.

Unfortunately, this abstraction requires control of the startup order of an application, which means some utilities
that force startup and execution in inconvenient ways can really cause friction.

A good example is celery.  To fully initialize celery, one may want to have the `AppConfig` object which only
exists inside of a delayed module context.  But celery wants to "force" the celery object to immediately exist
while it is still loading all other modules, in order to bind tasks.  The same thing with the `Flask` app wanting
to exist before startup to bind endpoints.

In the two above cases, there is special handling so that, for instance, the `Celery` object does not finalize
initialization immediately, and instead we rely on callbacks to delay this finalization.  In the case of `Flask`, we
use `Blueprint` objects that merely abstract endpoint definitions away from `Flask` value initialization.

However, this is not always possible.  Consider `langfuse`, which uses decorators to attach metadata to methods
as they are defined, but which itself may depend on dependency injection before the load ordering has completed.
This circular dependency could be resolved if the `langfuse` decorator were designed lazily -- that is, delaying
the resolution of configuration until the wrapped method is invoked, instead of when it is defined, but alas,
we don't have control over the execution of a third party.
