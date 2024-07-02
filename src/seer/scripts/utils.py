import argparse
import dataclasses
import typing

_T = typing.TypeVar("_T")

_unset = object()


class DataclassArgumentParser(typing.Generic[_T]):
    def __init__(self, constructor: type[_T]) -> None:
        assert dataclasses.is_dataclass(constructor)
        self.constructor = constructor

    def parse_args(self, args: typing.Sequence[str]) -> _T:
        parser = argparse.ArgumentParser()

        for field in dataclasses.fields(self.constructor):
            defaults: dict[str, typing.Any] = (
                dict(default=_unset)
                if field.default is not dataclasses.MISSING
                or field.default_factory is not dataclasses.MISSING
                else {}
            )
            actions: dict[str, typing.Any] = (
                dict(action="store_true") if issubclass(field.type, bool) else dict(type=field.type)
            )
            parser.add_argument(
                f"--{field.name.replace('_', '-')}",
                required=not defaults,
                **defaults,
                **actions,
                **({str(k): v for k, v in field.metadata.items()}),
            )

        namespace = parser.parse_args(args)
        return self.constructor(**{k: v for k, v in vars(namespace).items() if v is not _unset})  # type: ignore
