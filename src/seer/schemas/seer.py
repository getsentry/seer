import typing

import typing_extensions

AutofixIdResponse = typing_extensions.TypedDict(
    "AutofixIdResponse",
    {
        "id": str,
        "state": str,
    },
    total=False,
)

AutofixOutput = typing_extensions.TypedDict(
    "AutofixOutput",
    {
        "title": str,
        "description": str,
        "plan": str,
        "usage": "Usage",
        "pr_url": str,
    },
    total=False,
)

AutofixRequest = typing_extensions.TypedDict(
    "AutofixRequest",
    {
        "issue": "IssueDetails",
        "base_commit_sha": typing.Union[str, None],
        "additional_context": typing.Union[str, None],
    },
    total=False,
)

AutofixResponse = typing_extensions.TypedDict(
    "AutofixResponse",
    {
        "fix": typing.Union["AutofixOutput", None],
    },
    total=False,
)

AutofixTaskResultResponse = typing_extensions.TypedDict(
    "AutofixTaskResultResponse",
    {
        "result": typing.Union["AutofixResponse", None],
        "status": str,
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

IssueDetails = typing_extensions.TypedDict(
    "IssueDetails",
    {
        "id": str,
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

SnubaMetadata = typing_extensions.TypedDict(
    "SnubaMetadata",
    {
        "count": float,
    },
    total=False,
)

TaskStatusRequest = typing_extensions.TypedDict(
    "TaskStatusRequest",
    {
        "task_id": str,
    },
    total=False,
)

Usage = typing_extensions.TypedDict(
    "Usage",
    {
        # default: 0
        "completion_tokens": int,
        # default: 0
        "prompt_tokens": int,
        # default: 0
        "total_tokens": int,
    },
    total=False,
)
