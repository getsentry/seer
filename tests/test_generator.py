import dataclasses
import enum
import typing
from typing import Any, Mapping, NotRequired, TypedDict

from pydantic import BaseModel

from seer.generator import Examples, ints, parameterize, unsigned_ints


@parameterize(seed=1000)
def test_generates_primitives(a: int, b: bool, c: float, d: object, e: str):
    assert (
        isinstance(a, int)
        and isinstance(b, bool)
        and isinstance(c, float)
        and isinstance(d, object)
        and isinstance(e, str)
    )
    assert (
        not isinstance(d, int)
        and not isinstance(d, bool)
        and not isinstance(d, float)
        and not isinstance(d, str)
    )


@parameterize(seed=1050, a=(i % 100 for i in ints))
def test_generates_from_iter(a: int):
    assert a < 100 and a >= 0


@parameterize(seed=3321)
def test_generates_tuples(a: tuple[int, tuple[int, float, int]], b: tuple[str, ...]):
    assert isinstance(a, tuple) and isinstance(b, tuple)
    for v in b:
        assert isinstance(v, str)
    assert isinstance(a[0], int) and isinstance(a[1], tuple)
    assert isinstance(a[1][0], int)
    assert isinstance(a[1][1], float)
    assert isinstance(a[1][2], int)


@parameterize(seed=400)
def test_generates_mappings_and_lists(a: dict[str, list[int]], b: Mapping[bool, str]):
    assert all(isinstance(k, str) for k in a.keys())
    assert all(isinstance(k, bool) for k in b.keys())
    assert all(isinstance(v, int) for l in a.values() for v in l)
    assert all(isinstance(v, str) for v in b.values())


def interesting_strings(a: int, b: list[int], *, c: str) -> str:
    return str(a) + str(b) + c


def interesting_strings_2(*c: int, **kwds: int) -> str:
    return ",".join(str(v) for v in c) + "." + ",".join(f"{k}:{v}" for k, v in kwds.items())


@parameterize(seed=400, a=interesting_strings, b=interesting_strings_2)
def test_generates_from_callables(a: str, b: str):
    assert isinstance(a, str) and isinstance(b, str)
    assert a
    assert b


@parameterize(seed=101)
def test_generates_from_examples(a: typing.Annotated[int, Examples(unsigned_ints)]):
    assert a >= 0


class BagWeight(enum.IntEnum):
    LIGHT = 1
    HEAVY = 2


class CandyType(enum.Enum):
    HARD = "hard"
    CHEWY = "chewy"


@dataclasses.dataclass
class Candy:
    type: CandyType
    num: int = 1


class CandyBox(BaseModel):
    candy: list[Candy]
    color: str = "red"


class CandyBag(TypedDict):
    boxes: list[CandyBox]
    weight: NotRequired[BagWeight]


@parameterize(seed=8)
def test_structured_kinds(bag: CandyBag):
    assert "weight" not in bag
    for box in bag["boxes"]:
        assert box.color == "red"
        for candy in box.candy:
            assert candy.num == 1
            assert candy.type in CandyType


@parameterize(seed=18, include_defaults=True)
def test_structured_kinds_with_defaults(bag: CandyBag):
    if "weight" in bag:
        assert bag["weight"] in BagWeight
    for box in bag["boxes"]:
        assert box.color != "red"
        for candy in box.candy:
            assert candy.type in CandyType


def f(a, b):
    return 1


@parameterize(seed=10, a=f)
def test_any(a, b: Any, c):
    assert a == 1
    assert b is not None
    assert c is not None
