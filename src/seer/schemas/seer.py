import typing

import typing_extensions

BreakpointEntry = typing_extensions.TypedDict(
    "BreakpointEntry",
    {
        "project": str,
        "transaction": str,
        "aggregate_range_1": float,
        "aggregate_range_2": float,
        "unweighted_t_value": float,
        "unweighted_p_value": float,
        "trend_percentage": float,
        "absolute_percentage_change": float,
        "trend_difference": float,
        "breakpoint": int,
        "request_start": int,
        "request_end": int,
        "data_start": int,
        "data_end": int,
        "change": typing.Union[typing.Literal["improvement"], typing.Literal["regression"]],
    },
    total=False,
)

BreakpointRequest = typing_extensions.TypedDict(
    "BreakpointRequest",
    {
        "data": typing.Mapping[str, "BreakpointTransaction"],
        # default: ''
        "sort": str,
        # default: '1'
        "allow_midpoint": str,
        # default: 0
        "validate_tail_hours": int,
        # default: 0.1
        "trend_percentage()": float,
        # default: 0.0
        "min_change()": float,
    },
    total=False,
)

BreakpointResponse = typing_extensions.TypedDict(
    "BreakpointResponse",
    {
        "data": typing.List["BreakpointEntry"],
    },
    total=False,
)

BreakpointTransaction = typing_extensions.TypedDict(
    "BreakpointTransaction",
    {
        "data": typing.List[typing.Tuple[int, typing.Tuple["SnubaMetadata"]]],
        "request_start": int,
        "request_end": int,
        "data_start": int,
        "data_end": int,
    },
    total=False,
)

SeverityRequest = typing_extensions.TypedDict(
    "SeverityRequest",
    {
        # default: ''
        "message": str,
        # default: 0
        "has_stacktrace": int,
        # default: False
        "handled": bool,
        "trigger_timeout": typing.Union[bool, None],
        "trigger_error": typing.Union[bool, None],
    },
    total=False,
)

SeverityResponse = typing_extensions.TypedDict(
    "SeverityResponse",
    {
        # default: 0.0
        "severity": float,
    },
    total=False,
)

SnubaMetadata = typing_extensions.TypedDict(
    "SnubaMetadata",
    {
        "count": int,
    },
    total=False,
)
