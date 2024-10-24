# Dependency Injection in Seer

[Dependency Injection](https://en.wikipedia.org/wiki/Dependency_injection) is a technique in which shared objects
and configuration used in a program are "injected" into functions on a need to know basis, as opposed to, for instance,
functions accessing those objects or configuration via global functions or state.

This concept is actually very similar to [useContext](https://react.dev/reference/react/useContext) in React: consumers
of values are separated from providers rather than shared through global state.

### Motivation

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
references to any particular definition or instantiation*.  Instead, a parameter whose default value is `injected` resolves its final
instantiation **via the type annotation itself**.

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
but that enable and the provider definitions must occur **before invoking a method that depends upon them**.  Otherwise,
the ordering is loose and lazy.

Also note that `module.enable()` **only applies to the current thread!**.  This means that if you run a multi-threaded
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

In cases like this,

**do not force the square into the circular hole**.  You may need some configuration to exist outside the dependency
injection for it to work well.  That's conceding to (economic) reality.
