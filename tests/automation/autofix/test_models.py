import json
import unittest

from johen.pytest import parametrize
from pydantic import ValidationError

from seer.automation.autofix.models import AutofixRequest, IssueDetails, RepoDefinition
from seer.automation.models import (
    EventDetails,
    SentryEventData,
    SentryEventEntryDataValue,
    SentryExceptionEntry,
    Stacktrace,
    StacktraceFrame,
)
from tests.generators import InvalidEventEntry, NoStacktraceExceptionEntry, SentryFrameDict


class TestStacktraceHelpers(unittest.TestCase):
    def test_stacktrace_to_str(self):
        frames = [
            StacktraceFrame(
                function="main",
                filename="app.py",
                abs_path="/path/to/app.py",
                line_no=10,
                col_no=20,
                context=[(10, "    main()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=True,
            ),
            StacktraceFrame(
                function="helper",
                filename="utils.py",
                abs_path="/path/to/utils.py",
                line_no=15,
                col_no=None,
                context=[(15, "    helper()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=False,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " helper in file utils.py in repo my_repo [Line 15] (Not in app)\n    helper()  <-- SUSPECT LINE\n------\n main in file app.py in repo my_repo [Line 10, column 20] (In app)\n    main()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(), expected_str)

    def test_stacktrace_to_str_cutoff(self):
        frames = [
            StacktraceFrame(
                function="main",
                filename="app.py",
                abs_path="/path/to/app.py",
                line_no=10,
                col_no=20,
                context=[(10, "    main()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=True,
            ),
            StacktraceFrame(
                function="helper",
                filename="utils.py",
                abs_path="/path/to/utils.py",
                line_no=15,
                col_no=None,
                context=[(15, "    helper()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=False,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " helper in file utils.py in repo my_repo [Line 15] (Not in app)\n    helper()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(max_frames=1), expected_str)


class TestRepoDefinition(unittest.TestCase):
    def test_repo_definition_creation(self):
        repo_def = RepoDefinition(provider="github", owner="seer", name="automation")
        self.assertEqual(repo_def.provider, "github")
        self.assertEqual(repo_def.owner, "seer")
        self.assertEqual(repo_def.name, "automation")

    def test_repo_definition_uniqueness(self):
        repo_def1 = RepoDefinition(provider="github", owner="seer", name="automation")
        repo_def2 = RepoDefinition(provider="github", owner="seer", name="automation")
        self.assertEqual(hash(repo_def1), hash(repo_def2))

    def test_multiple_repos(self):
        repo_def1 = RepoDefinition(provider="github", owner="seer", name="automation")
        repo_def2 = RepoDefinition(provider="github", owner="seer", name="automation-tools")
        self.assertNotEqual(hash(repo_def1), hash(repo_def2))

    def test_repo_with_provider_processing(self):
        repo_def = RepoDefinition(provider="integrations:github", owner="seer", name="automation")
        self.assertEqual(repo_def.provider, "github")
        self.assertEqual(repo_def.owner, "seer")
        self.assertEqual(repo_def.name, "automation")

    def test_repo_with_invalid_provider(self):
        with self.assertRaises(ValidationError):
            RepoDefinition(provider="invalid_provider", owner="seer", name="automation")

    def test_repo_with_none_provider(self):
        repo_dict = {"provider": None, "owner": "seer", "name": "automation"}
        with self.assertRaises(ValidationError):
            RepoDefinition(**repo_dict)


class TestAutofixRequest(unittest.TestCase):
    def test_autofix_request_handler(self):
        repo_def = RepoDefinition(provider="github", owner="seer", name="automation")
        issue_details = IssueDetails(
            id=789, title="Test Issue", events=[SentryEventData(title="yes", entries=[])]
        )
        autofix_request = AutofixRequest(
            organization_id=123,
            project_id=456,
            repos=[repo_def],
            issue=issue_details,
        )
        self.assertEqual(autofix_request.organization_id, 123)
        self.assertEqual(autofix_request.project_id, 456)
        self.assertEqual(len(autofix_request.repos), 1)
        self.assertEqual(autofix_request.issue.id, 789)
        self.assertEqual(autofix_request.issue.title, "Test Issue")

    def test_autofix_request_with_duplicate_repos(self):
        repo_def1 = RepoDefinition(provider="github", owner="seer", name="automation")
        repo_def2 = RepoDefinition(provider="github", owner="seer", name="automation")
        with self.assertRaises(ValidationError):
            AutofixRequest(
                organization_id=123,
                project_id=456,
                repos=[repo_def1, repo_def2],
                issue=IssueDetails(
                    id=789, title="Test Issue", events=[SentryEventData(title="yes", entries=[])]
                ),
            )

    def test_autofix_request_with_multiple_repos(self):
        repo_def1 = RepoDefinition(provider="github", owner="seer", name="automation")
        repo_def2 = RepoDefinition(provider="github", owner="seer", name="automation-tools")
        issue_details = IssueDetails(
            id=789, title="Test Issue", events=[SentryEventData(title="yes", entries=[])]
        )
        autofix_request = AutofixRequest(
            organization_id=123,
            project_id=456,
            repos=[repo_def1, repo_def2],
            issue=issue_details,
        )
        self.assertEqual(len(autofix_request.repos), 2)


@parametrize
def test_event_no_exception_events(event: SentryEventData, entry: InvalidEventEntry):
    event["entries"] = [entry]
    assert len(EventDetails.from_event(event).exceptions) is 0


@parametrize
def test_event_get_stacktrace_empty_frames(
    event: SentryEventData, entry: NoStacktraceExceptionEntry
):
    event["entries"] = [entry]
    event_details = EventDetails.from_event(event)
    assert len(event_details.exceptions) == 1
    assert len(event_details.exceptions[0].stacktrace.frames) == 0


@parametrize
def test_event_get_stacktrace_invalid_entry(
    event: SentryEventData,
    invalid: InvalidEventEntry,
    entry: SentryExceptionEntry,
    sentry_data_value: SentryEventEntryDataValue,
    valid_frame: SentryFrameDict,
):
    sentry_data_value["stacktrace"]["frames"].append(valid_frame)
    entry.data["values"] = [sentry_data_value]
    event["entries"] = [invalid, entry.model_dump(mode="json")]
    event_details = EventDetails.from_event(event)

    assert len(event_details.exceptions) == 1
    assert (
        StacktraceFrame.model_validate(valid_frame) in event_details.exceptions[0].stacktrace.frames
    )


@parametrize
def test_stacktrace_frame_vars_stringify(stacktrace: Stacktrace):
    stack_str = stacktrace.to_str()

    for frame in stacktrace.frames:
        if frame.vars:
            vars_str = json.dumps(frame.vars, indent=2)
            assert vars_str in stack_str
        else:
            assert "---\nvariables" not in stack_str
