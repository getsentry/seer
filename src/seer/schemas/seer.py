import typing

AutofixEndpointResponse = typing.TypedDict(
    "AutofixEndpointResponse",
    {
        "started": bool,
    },
)

AutofixRequest = typing.TypedDict(
    "AutofixRequest",
    {
        "organization_id": int,
        "project_id": int,
        "repos": typing.List["RepoDefinition"],
        "issue": "IssueDetails",
        "invoking_user": typing.Union["AutofixUserDetails", None],
        "base_commit_sha": typing.Union[str, None],
        "additional_context": typing.Union[str, None],
        "timeout_secs": typing.Union[int, None],
        # format: date-time
        "last_updated": typing.Union[str, None],
    },
)

AutofixUserDetails = typing.TypedDict(
    "AutofixUserDetails",
    {
        "id": int,
        "display_name": str,
    },
)

BreakpointEntry = typing.TypedDict(
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
)

BreakpointRequest = typing.TypedDict(
    "BreakpointRequest",
    {
        "data": typing.Mapping[str, "BreakpointTransaction"],
        # default: ''
        "sort": typing.NotRequired[str],
        # default: '1'
        "allow_midpoint": typing.NotRequired[str],
        # default: 0
        "validate_tail_hours": typing.NotRequired[int],
        # default: 0.1
        "trend_percentage()": typing.NotRequired[float],
        # default: 0.0
        "min_change()": typing.NotRequired[float],
    },
)

BreakpointResponse = typing.TypedDict(
    "BreakpointResponse",
    {
        "data": typing.List["BreakpointEntry"],
    },
)

BreakpointTransaction = typing.TypedDict(
    "BreakpointTransaction",
    {
        "data": typing.List[typing.Tuple[int, typing.Tuple["SnubaMetadata"]]],
        "request_start": int,
        "request_end": int,
        "data_start": int,
        "data_end": int,
    },
)

GroupingRequest = typing.TypedDict(
    "GroupingRequest",
    {
        "group_id": int,
        "project_id": int,
        "stacktrace": str,
        "message": str,
        # default: 1
        "k": typing.NotRequired[int],
        # default: 0.01
        "threshold": typing.NotRequired[float],
    },
)

GroupingResponse = typing.TypedDict(
    "GroupingResponse",
    {
        "parent_group_id": typing.Union[int, None],
        "stacktrace_distance": float,
        "message_distance": float,
        "should_group": bool,
    },
)

IssueDetails = typing.TypedDict(
    "IssueDetails",
    {
        "id": int,
        "title": str,
        "short_id": typing.Union[str, None],
        "events": typing.List["SentryEvent"],
    },
)

RepoDefinition = typing.TypedDict(
    "RepoDefinition",
    {
        "provider": str,
        "owner": str,
        "name": str,
    },
)

SentryEvent = typing.TypedDict(
    "SentryEvent",
    {
        "entries": typing.List[typing.Mapping[str, typing.Any]],
    },
)

SeverityRequest = typing.TypedDict(
    "SeverityRequest",
    {
        # default: ''
        "message": typing.NotRequired[str],
        # default: 0
        "has_stacktrace": typing.NotRequired[int],
        "handled": typing.Union[bool, None],
        "trigger_timeout": typing.Union[bool, None],
        "trigger_error": typing.Union[bool, None],
    },
)

SeverityResponse = typing.TypedDict(
    "SeverityResponse",
    {
        # default: 0.0
        "severity": typing.NotRequired[float],
    },
)

SimilarityResponse = typing.TypedDict(
    "SimilarityResponse",
    {
        "responses": typing.List["GroupingResponse"],
    },
)

SnubaMetadata = typing.TypedDict(
    "SnubaMetadata",
    {
        "count": float,
    },
)
