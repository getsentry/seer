import typing

import typing_extensions

AutofixEndpointResponse = typing_extensions.TypedDict(
    "AutofixEndpointResponse",
    {
        "started": bool,
    },
    total=False,
)

AutofixRequest = typing_extensions.TypedDict(
    "AutofixRequest",
    {
        "issue": "IssueDetails",
        "base_commit_sha": str,
        "additional_context": typing.Union[str, None],
    },
    total=False,
)

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

GroupingRequest = typing_extensions.TypedDict(
    "GroupingRequest",
    {
        "group_id": int,
        "project_id": int,
        "stacktrace": str,
        "message": str,
        # default: 1
        "k": int,
        # default: 0.99
        "threshold": float,
    },
    total=False,
)

GroupingResponse = typing_extensions.TypedDict(
    "GroupingResponse",
    {
        "parent_group_id": typing.Union[int, None],
        "stacktrace_similarity": float,
        "message_similarity": float,
        "should_group": bool,
    },
    total=False,
)

IssueDetails = typing_extensions.TypedDict(
    "IssueDetails",
    {
        "id": int,
        "title": str,
        "events": typing.List["SentryEvent"],
    },
    total=False,
)

SentryEvent = typing_extensions.TypedDict(
    "SentryEvent",
    {
        "entries": typing.List[typing.Mapping[str, typing.Any]],
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

SimilarityResponse = typing_extensions.TypedDict(
    "SimilarityResponse",
    {
        "responses": typing.List["GroupingResponse"],
    },
    total=False,
)

SnubaMetadata = typing_extensions.TypedDict(
    "SnubaMetadata",
    {
        "count": float,
    },
    total=False,
)
