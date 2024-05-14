import typing

import typing_extensions

AutofixCreatePrUpdatePayload = typing_extensions.TypedDict(
    "AutofixCreatePrUpdatePayload",
    {
        "type": typing.Literal["create_pr"],
        "repo_id": typing.Union[int, None],
    },
    total=False,
)

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
        "organization_id": int,
        "project_id": int,
        "repos": typing.List["RepoDefinition"],
        "issue": "IssueDetails",
        "invoking_user": typing.Union["AutofixUserDetails", None],
        "base_commit_sha": typing.Union[str, None],
        "additional_context": typing.Union[str, None],
    },
    total=False,
)

AutofixRootCauseUpdatePayload = typing_extensions.TypedDict(
    "AutofixRootCauseUpdatePayload",
    {
        "type": typing.Literal["select_root_cause"],
        "cause_id": typing.Union[int, None],
        "fix_id": typing.Union[int, None],
        "custom_root_cause": typing.Union[str, None],
    },
    total=False,
)

AutofixStateRequest = typing_extensions.TypedDict(
    "AutofixStateRequest",
    {
        "group_id": int,
    },
    total=False,
)

AutofixStateResponse = typing_extensions.TypedDict(
    "AutofixStateResponse",
    {
        "group_id": int,
        "state": typing.Union[typing.Mapping[str, typing.Any], None],
    },
    total=False,
)

AutofixUpdateRequest = typing_extensions.TypedDict(
    "AutofixUpdateRequest",
    {
        "run_id": int,
        "payload": typing.Any,
    },
    total=False,
)

AutofixUserDetails = typing_extensions.TypedDict(
    "AutofixUserDetails",
    {
        "id": int,
        "display_name": str,
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

BulkCreateGroupingRecordsResponse = typing_extensions.TypedDict(
    "BulkCreateGroupingRecordsResponse",
    {
        "success": bool,
    },
    total=False,
)

CodebaseStatusCheckRequest = typing_extensions.TypedDict(
    "CodebaseStatusCheckRequest",
    {
        "organization_id": int,
        "project_id": int,
        "repo": "RepoDefinition",
    },
    total=False,
)

CodebaseStatusCheckResponse = typing_extensions.TypedDict(
    "CodebaseStatusCheckResponse",
    {
        "status": str,
    },
    total=False,
)

CreateCodebaseRequest = typing_extensions.TypedDict(
    "CreateCodebaseRequest",
    {
        "organization_id": int,
        "project_id": int,
        "repo": "RepoDefinition",
    },
    total=False,
)

CreateGroupingRecordData = typing_extensions.TypedDict(
    "CreateGroupingRecordData",
    {
        "hash": str,
        "project_id": int,
        "message": str,
    },
    total=False,
)

CreateGroupingRecordsRequest = typing_extensions.TypedDict(
    "CreateGroupingRecordsRequest",
    {
        "data": typing.List["CreateGroupingRecordData"],
        "stacktrace_list": typing.List[str],
    },
    total=False,
)

GroupingRequest = typing_extensions.TypedDict(
    "GroupingRequest",
    {
        "project_id": int,
        "stacktrace": str,
        "message": str,
        "hash": str,
        "group_id": typing.Union[int, None],
        # default: 1
        "k": int,
        # default: 0.01
        "threshold": float,
    },
    total=False,
)

GroupingResponse = typing_extensions.TypedDict(
    "GroupingResponse",
    {
        "parent_group_id": typing.Union[int, None],
        "parent_hash": str,
        "stacktrace_distance": float,
        "message_distance": float,
        "should_group": bool,
    },
    total=False,
)

IssueDetails = typing_extensions.TypedDict(
    "IssueDetails",
    {
        "id": int,
        "title": str,
        "short_id": typing.Union[str, None],
        "events": typing.List["SentryEventData"],
    },
    total=False,
)

RepoAccessCheckRequest = typing_extensions.TypedDict(
    "RepoAccessCheckRequest",
    {
        "repo": "RepoDefinition",
    },
    total=False,
)

RepoAccessCheckResponse = typing_extensions.TypedDict(
    "RepoAccessCheckResponse",
    {
        "has_access": bool,
    },
    total=False,
)

RepoDefinition = typing_extensions.TypedDict(
    "RepoDefinition",
    {
        "provider": str,
        "owner": str,
        "name": str,
        "external_id": str,
    },
    total=False,
)

SentryEventData = typing_extensions.TypedDict(
    "SentryEventData",
    {
        "title": str,
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
        "handled": typing.Union[bool, None],
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

SimilarityBenchmarkResponse = typing_extensions.TypedDict(
    "SimilarityBenchmarkResponse",
    {
        "embedding": typing.List[float],
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
