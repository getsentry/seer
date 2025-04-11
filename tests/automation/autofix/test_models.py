import datetime
import json
import textwrap
import unittest
import uuid
from unittest.mock import Mock, patch

from johen.pytest import parametrize
from pydantic import ValidationError

from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisItem
from seer.automation.autofix.config import AUTOFIX_HARD_TIME_OUT_MINS, AUTOFIX_UPDATE_TIMEOUT_SECS
from seer.automation.autofix.models import (
    AutofixContinuation,
    AutofixRequest,
    AutofixStatus,
    BaseStep,
    CodeContextRootCauseSelection,
    CustomRootCauseSelection,
    DefaultStep,
    IssueDetails,
    RepoDefinition,
    RootCauseStep,
)
from seer.automation.models import (
    BreadcrumbsDetails,
    EventDetails,
    FilePatch,
    Hunk,
    Line,
    Profile,
    ProfileFrame,
    SentryEventData,
    SentryEventEntryDataValue,
    SentryExceptionEntry,
    Stacktrace,
    StacktraceFrame,
)
from seer.automation.utils import make_kill_signal
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
                in_app=False,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " helper in file utils.py in repo my_repo [Line 15] (Not in app)\n    helper()  <-- SUSPECT LINE\n------\n main in file app.py in repo my_repo [Line 10, column 20] (In app)\n    main()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(), expected_str)

    def test_stacktrace_no_frames(self):
        stacktrace = Stacktrace(frames=[])
        expected_str = ""
        self.assertEqual(stacktrace.to_str(), expected_str)

    def test_stacktrace_frame_without_context(self):
        frames = [
            StacktraceFrame(
                function="main",
                filename="app.py",
                abs_path="/path/to/app.py",
                line_no=10,
                col_no=20,
                context=[],
                repo_name="my_repo",
                in_app=True,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = (
            " main in file app.py in repo my_repo [Line 10, column 20] (In app)\n------\n"
        )
        self.assertEqual(stacktrace.to_str(), expected_str)

    def test_stacktrace_frame_no_function_no_filename(self):
        frames = [
            StacktraceFrame(
                function=None,
                filename=None,
                abs_path=None,
                line_no=10,
                col_no=20,
                context=[(10, "    unknown()")],
                repo_name=None,
                in_app=True,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " Unknown function in unknown file [Line 10, column 20] (In app)\n    unknown()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(), expected_str)

    def test_stacktrace_max_frames(self):
        frames = [
            StacktraceFrame(
                function=f"function_{i}",
                filename=f"file_{i}.py",
                abs_path=f"/path/to/file_{i}.py",
                line_no=i * 10,
                col_no=None,
                context=[(i * 10, f"    function_{i}()")],
                repo_name="my_repo",
                in_app=(i % 2 == 0),
            )
            for i in range(20)
        ]
        stacktrace = Stacktrace(frames=frames)
        result_str = stacktrace.to_str(max_frames=5)
        self.assertEqual(result_str.count("------"), 5)

    def test_event_no_entries(self):
        event = SentryEventData(title="title", entries=[])
        event_details = EventDetails.from_event(event)
        self.assertEqual(len(event_details.exceptions), 0)
        self.assertEqual(len(event_details.threads), 0)
        self.assertEqual(len(event_details.breadcrumbs), 0)

    def test_event_multiple_breadcrumbs(self):
        breadcrumbs = [
            BreadcrumbsDetails(
                type="log",
                category="category",
                level="info",
                message=f"Message {i}",
                data={},
                title="title",
            )
            for i in range(15)
        ]
        event = SentryEventData(
            title="title",
            entries=[
                {
                    "type": "breadcrumbs",
                    "data": {"values": [breadcrumb.model_dump() for breadcrumb in breadcrumbs]},
                }
            ],
        )
        event_details = EventDetails.from_event(event)
        formatted_breadcrumbs = event_details.format_breadcrumbs()

        # check that only the last 10 breadcrumbs are present in the formatted output
        for i in range(5, 15):
            self.assertIn(f"Message {i}\n", formatted_breadcrumbs)
        for i in range(5):
            self.assertNotIn(f"Message {i}\n", formatted_breadcrumbs)

    def test_event_invalid_thread_data(self):
        invalid_thread_data = {
            "id": 1,
            "name": "Invalid Thread",
            "state": "invalid_state",
            "current": True,
            "crashed": False,
            "main": False,
            "stacktrace": None,
        }
        event = SentryEventData(
            title="title", entries=[{"type": "threads", "data": {"values": [invalid_thread_data]}}]
        )
        event_details = EventDetails.from_event(event)
        self.assertEqual(len(event_details.threads), 0)

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
                in_app=False,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " helper in file utils.py in repo my_repo [Line 15] (Not in app)\n    helper()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(max_frames=1), expected_str)

    def test_trim_frames(self):
        frames = [
            StacktraceFrame(
                in_app=True, filename="test", abs_path="test", line_no=1, col_no=1, context=[]
            )
            for _ in range(10)
        ] + [
            StacktraceFrame(
                in_app=False, filename="test", abs_path="test", line_no=1, col_no=1, context=[]
            )
            for _ in range(10)
        ]
        result = Stacktrace._trim_frames(frames, frame_allowance=16)

        # assert the result has the correct length
        self.assertEqual(len(result), 16)
        # assert all app frames are kept
        self.assertEqual(len([f for f in result if f.in_app]), 10)
        # assert the remaining frames are system frames
        self.assertEqual(len([f for f in result if not f.in_app]), 6)
        # assert the first and last system frames are kept
        self.assertFalse(result[10].in_app)
        self.assertFalse(result[-1].in_app)

    def test_trim_vars(self):
        self.assertEqual(StacktraceFrame._trim_vars({}, "def foo():"), {})

        input_dict = {
            "bar": 1,
            "baz": 5,
            "foo": "ignored",
        }
        expected_output = {"bar": 1, "foo": "ignored"}
        self.assertEqual(StacktraceFrame._trim_vars(input_dict, "def foo(bar):"), expected_output)


class TestRepoDefinition(unittest.TestCase):
    def test_repo_definition_creation(self):
        repo_def = RepoDefinition(
            provider="integrations:github", owner="seer", name="automation", external_id="123"
        )
        self.assertEqual(repo_def.provider, "github")
        self.assertEqual(repo_def.owner, "seer")
        self.assertEqual(repo_def.name, "automation")
        self.assertEqual(repo_def.external_id, "123")

    def test_repo_definition_uniqueness(self):
        repo_def1 = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
        repo_def2 = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
        self.assertEqual(hash(repo_def1), hash(repo_def2))

    def test_multiple_repos(self):
        repo_def1 = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
        repo_def2 = RepoDefinition(
            provider="github", owner="seer", name="automation-tools", external_id="123"
        )
        self.assertNotEqual(hash(repo_def1), hash(repo_def2))

    def test_repo_with_provider_processing(self):
        repo_def = RepoDefinition(
            provider="integrations:github", owner="seer", name="automation", external_id="123"
        )
        self.assertEqual(repo_def.provider, "github")
        self.assertEqual(repo_def.provider_raw, "integrations:github")
        self.assertEqual(repo_def.owner, "seer")
        self.assertEqual(repo_def.name, "automation")

    def test_repo_with_none_provider(self):
        repo_dict = {"provider": None, "owner": "seer", "name": "automation"}
        with self.assertRaises(ValidationError):
            RepoDefinition(**repo_dict)


class TestAutofixRequest(unittest.TestCase):
    def test_autofix_request_handler(self):
        repo_def = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
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
        repo_def1 = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
        repo_def2 = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
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
        repo_def1 = RepoDefinition(
            provider="github", owner="seer", name="automation", external_id="123"
        )
        repo_def2 = RepoDefinition(
            provider="github", owner="seer", name="automation-tools", external_id="123"
        )
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

    def test_extract_relevant_functions(self):
        issue = IssueDetails(
            id=1,
            title="Test Issue",
            events=[
                {
                    "title": "Error",
                    "entries": [
                        {
                            "type": "exception",
                            "data": {
                                "values": [
                                    {
                                        "type": "Exception",
                                        "value": "Test error",
                                        "stacktrace": {
                                            "frames": [
                                                {
                                                    "function": "not_relevant",
                                                    "in_app": False,
                                                    "filename": "external.py",
                                                },
                                                {
                                                    "function": "relevant_func1",
                                                    "in_app": True,
                                                    "filename": "app.py",
                                                },
                                                {
                                                    "function": "relevant_func2",
                                                    "in_app": True,
                                                    "filename": "utils.py",
                                                },
                                                {
                                                    "function": None,  # Should be skipped
                                                    "in_app": True,
                                                    "filename": "app.py",
                                                },
                                            ]
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        )

        # Create a profile and request
        profile = Profile(
            profile_matches_issue=True,
            execution_tree=[
                ProfileFrame(
                    function="main",
                    module="app",
                    filename="app.py",
                    lineno=1,
                    in_app=True,
                    children=[
                        ProfileFrame(
                            function="relevant_func1",
                            module="app",
                            filename="app.py",
                            lineno=5,
                            in_app=True,
                        ),
                        ProfileFrame(
                            function="relevant_func2",
                            module="utils",
                            filename="utils.py",
                            lineno=10,
                            in_app=True,
                        ),
                    ],
                )
            ],
        )

        request = AutofixRequest(
            organization_id=1,
            project_id=1,
            repos=[
                RepoDefinition(
                    provider="github",
                    owner="test",
                    name="test",
                    external_id="test",
                )
            ],
            issue=issue,
            profile=profile,
        )

        self.assertEqual(request.profile.relevant_functions, {"relevant_func1", "relevant_func2"})
        self.assertNotIn("not_relevant", request.profile.relevant_functions)
        self.assertEqual(len(request.profile.relevant_functions), 2)

    def test_extract_relevant_functions_no_profile(self):
        issue = IssueDetails(
            id=1,
            title="Test Issue",
            events=[
                {
                    "title": "Error",
                    "entries": [
                        {
                            "type": "exception",
                            "data": {
                                "values": [
                                    {
                                        "type": "Exception",
                                        "value": "Test error",
                                        "stacktrace": {
                                            "frames": [
                                                {
                                                    "function": "relevant_func",
                                                    "in_app": True,
                                                    "filename": "app.py",
                                                }
                                            ]
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                }
            ],
        )

        request = AutofixRequest(
            organization_id=1,
            project_id=1,
            repos=[
                RepoDefinition(
                    provider="github",
                    owner="test",
                    name="test",
                    external_id="test",
                )
            ],
            issue=issue,
            profile=None,
        )

        self.assertIsNone(request.profile)


class TestAutofixStep(unittest.TestCase):
    def test_ensure_uuid_id(self):
        # Test with a non-UUID id
        step = DefaultStep(id="non-uuid-id", key="original-key", title="Test Step")
        step.ensure_uuid_id()
        self.assertTrue(BaseStep.is_valid_uuid(step.id))
        self.assertEqual(step.key, "non-uuid-id")

        # Test with a valid UUID id
        valid_uuid = str(uuid.uuid4())
        step = DefaultStep(id=valid_uuid, key="original-key", title="Test Step")
        original_id = step.id
        step.ensure_uuid_id()
        self.assertEqual(step.id, original_id)
        self.assertEqual(step.key, "original-key")

    def test_is_valid_uuid(self):
        # Test with a valid UUID
        valid_uuid = str(uuid.uuid4())
        self.assertTrue(BaseStep.is_valid_uuid(valid_uuid))

        # Test with an invalid UUID
        invalid_uuid = "not-a-uuid"
        self.assertFalse(BaseStep.is_valid_uuid(invalid_uuid))

        # Test with an empty string
        self.assertFalse(BaseStep.is_valid_uuid(""))

        # Test with a None value
        self.assertFalse(BaseStep.is_valid_uuid(None))  # type: ignore


class TestAutofixContinuation(unittest.TestCase):
    def setUp(self):
        self.request = Mock(spec=AutofixRequest)
        self.continuation = AutofixContinuation(request=self.request)

    def test_auto_generated_step_ids(self):
        # Create steps without specifying IDs
        step1 = DefaultStep(key="step1", title="Test Step 1")
        step2 = DefaultStep(key="step2", title="Test Step 2")

        # Add steps to the continuation
        added_step1 = self.continuation.add_step(step1)
        added_step2 = self.continuation.add_step(step2)

        # Check that IDs were auto-generated
        self.assertIsNotNone(added_step1.id)
        self.assertIsNotNone(added_step2.id)

        # Check that the IDs are strings (UUIDs)
        self.assertIsInstance(added_step1.id, str)
        self.assertIsInstance(added_step2.id, str)

        # Check that the IDs are unique
        self.assertNotEqual(added_step1.id, added_step2.id)

        # Verify that we can find steps by their auto-generated IDs
        self.assertEqual(self.continuation.find_step(id=added_step1.id), added_step1)
        self.assertEqual(self.continuation.find_step(id=added_step2.id), added_step2)

    def test_model_copy_with_new_id(self):
        original_step = DefaultStep(key="original", title="Original Step")
        self.continuation.add_step(original_step)

        copied_step = original_step.model_copy_with_new_id()

        # Check that a new ID was generated
        self.assertNotEqual(original_step.id, copied_step.id)

        # Check that other attributes remain the same
        self.assertEqual(original_step.key, copied_step.key)
        self.assertEqual(original_step.title, copied_step.title)

        # Add the copied step and verify it can be found by its new ID
        added_copied_step = self.continuation.add_step(copied_step)
        self.assertEqual(self.continuation.find_step(id=copied_step.id), added_copied_step)

    def test_find_step_by_key(self):
        step1 = DefaultStep(key="step1", title="test")
        step2 = DefaultStep(key="step2", title="test")
        self.continuation.steps = [step1, step2]

        self.assertEqual(self.continuation.find_step(key="step1"), step1)
        self.assertEqual(self.continuation.find_step(key="step2"), step2)
        self.assertIsNone(self.continuation.find_step(key="step3"))

    def test_find_step_by_id(self):
        step1 = DefaultStep(id="step1", title="test")
        step2 = DefaultStep(id="step2", title="test")
        self.continuation.steps = [step1, step2]

        self.assertEqual(self.continuation.find_step(id="step1"), step1)
        self.assertEqual(self.continuation.find_step(id="step2"), step2)
        self.assertIsNone(self.continuation.find_step(id="step3"))

    def test_find_step_by_index(self):
        step1 = DefaultStep(key="step1", title="test")
        step2 = DefaultStep(key="step2", title="test")
        self.continuation.steps = [step1, step2]

        self.assertEqual(self.continuation.find_step(index=0), step1)
        self.assertEqual(self.continuation.find_step(index=1), step2)
        self.assertIsNone(self.continuation.find_step(index=2))

    def test_add_step(self):
        step1 = DefaultStep(key="step1", title="test")
        step2 = DefaultStep(key="step2", title="test")

        # Add first step
        added_step1 = self.continuation.add_step(step1)
        self.assertEqual(added_step1.index, 0)
        self.assertEqual(len(self.continuation.steps), 1)
        self.assertEqual(self.continuation.steps[0], added_step1)

        # Add second step
        added_step2 = self.continuation.add_step(step2)
        self.assertEqual(added_step2.index, 1)
        self.assertEqual(len(self.continuation.steps), 2)
        self.assertEqual(self.continuation.steps[1], added_step2)

        # Verify both steps are in the continuation
        self.assertEqual(self.continuation.steps, [added_step1, added_step2])

    def test_find_or_add(self):
        step1 = DefaultStep(key="step1", title="test")
        self.continuation.steps = [step1]

        # Test finding existing step
        found_step = self.continuation.find_or_add(step1)
        self.assertEqual(found_step, step1)
        self.assertEqual(len(self.continuation.steps), 1)

        # Test adding new step
        step2 = DefaultStep(key="step2", title="test")
        added_step = self.continuation.find_or_add(step2)
        self.assertEqual(added_step.key, "step2")
        self.assertEqual(added_step.index, 1)
        self.assertEqual(len(self.continuation.steps), 2)

    def test_find_last_step_waiting_for_response(self):
        step1 = DefaultStep(key="step1", status=AutofixStatus.COMPLETED, title="test")
        step2 = DefaultStep(
            key="step2", status=AutofixStatus.WAITING_FOR_USER_RESPONSE, title="test"
        )
        step3 = DefaultStep(key="step3", status=AutofixStatus.PROCESSING, title="test")
        self.continuation.steps = [step1, step2, step3]

        self.assertEqual(self.continuation.find_last_step_waiting_for_response(), step2)

    def test_make_step_latest(self):
        step1 = DefaultStep(key="step1", title="test")
        step2 = DefaultStep(key="step2", title="test")
        step3 = DefaultStep(key="step3", title="test")
        self.continuation.steps = [step1, step2, step3]

        self.continuation.make_step_latest(step2)
        self.assertEqual(self.continuation.steps, [step1, step3, step2])

    def test_mark_all_running_steps_completed(self):
        step1 = DefaultStep(key="step1", status=AutofixStatus.PROCESSING, title="test")
        step2 = DefaultStep(key="step2", status=AutofixStatus.COMPLETED, title="test")
        step3 = DefaultStep(key="step3", status=AutofixStatus.ERROR, title="test")
        self.continuation.steps = [step1, step2, step3]

        self.continuation.mark_running_steps_completed()
        self.assertEqual(step1.status, AutofixStatus.COMPLETED)
        self.assertEqual(step2.status, AutofixStatus.COMPLETED)
        self.assertEqual(step3.status, AutofixStatus.ERROR)

    def test_mark_running_steps_errored(self):
        step1 = DefaultStep(key="step1", status=AutofixStatus.PROCESSING, title="test")
        step2 = DefaultStep(key="step2", status=AutofixStatus.PROCESSING, title="test")
        substep = DefaultStep(key="substep", status=AutofixStatus.PROCESSING, title="test")
        step1.progress = [substep]
        self.continuation.steps = [step1, step2]

        self.continuation.mark_running_steps_errored()
        self.assertEqual(step1.status, AutofixStatus.ERROR)
        self.assertEqual(step2.status, AutofixStatus.ERROR)
        self.assertEqual(substep.status, AutofixStatus.ERROR)

    def test_set_last_step_completed_message(self):
        step = DefaultStep(key="step1", title="test")
        self.continuation.steps = [step]

        self.continuation.set_last_step_completed_message("Test message")
        self.assertEqual(step.completedMessage, "Test message")

    def test_get_selected_root_cause_and_fix(self):
        root_cause_step = RootCauseStep(key="root_cause_analysis", title="test")
        cause = RootCauseAnalysisItem(
            id=1,
            title="test",
            description="test",
            reproduction="test",
            likelihood=0.5,
            actionability=0.5,
        )
        root_cause_step.causes = [cause]

        root_cause_step.selection = CodeContextRootCauseSelection(cause_id=1, instruction="test")
        self.continuation.steps = [root_cause_step]
        result, instruction = self.continuation.get_selected_root_cause()
        self.assertEqual(result, cause)
        self.assertEqual(instruction, "test")

        root_cause_step.selection = CustomRootCauseSelection(custom_root_cause="root cause")
        self.continuation.steps = [root_cause_step]
        result, instruction = self.continuation.get_selected_root_cause()
        self.assertEqual(result, "root cause")
        self.assertIsNone(instruction)

        root_cause_step.selection = None
        self.continuation.steps = [root_cause_step]
        result, instruction = self.continuation.get_selected_root_cause()
        self.assertEqual(result, None)
        self.assertIsNone(instruction)

    def test_mark_triggered(self):
        with patch("datetime.datetime") as mock_datetime:
            mock_now = datetime.datetime.now()
            mock_datetime.now.return_value = mock_now
            self.continuation.mark_triggered()
            self.assertEqual(self.continuation.last_triggered_at, mock_now)

    def test_mark_updated(self):
        with patch("datetime.datetime") as mock_datetime:
            mock_now = datetime.datetime.now()
            mock_datetime.now.return_value = mock_now
            self.continuation.mark_updated()
            self.assertEqual(self.continuation.updated_at, mock_now)

    def test_delete_steps_after(self):
        step1 = DefaultStep(key="step1", title="test")
        step2 = DefaultStep(key="step2", title="test")
        step3 = DefaultStep(key="step3", title="test")
        self.continuation.steps = [step1, step2, step3]

        self.continuation.delete_steps_after(step2)
        self.assertEqual(self.continuation.steps, [step1, step2])

    def test_delete_steps_after_including_self(self):
        step1 = DefaultStep(key="step1", title="test")
        step2 = DefaultStep(key="step2", title="test")
        step3 = DefaultStep(key="step3", title="test")
        self.continuation.steps = [step1, step2, step3]

        self.continuation.delete_steps_after(step2, include_current=True)
        self.assertEqual(self.continuation.steps, [step1])

    def test_clear_file_changes(self):
        codebase1 = Mock()
        codebase2 = Mock()
        self.continuation.codebases = {"repo1": codebase1, "repo2": codebase2}

        self.continuation.clear_file_changes()
        self.assertEqual(codebase1.file_changes, [])
        self.assertEqual(codebase2.file_changes, [])

    def test_is_running(self):
        self.continuation.status = AutofixStatus.PROCESSING
        self.assertTrue(self.continuation.is_running)

        self.continuation.status = AutofixStatus.PROCESSING
        self.assertTrue(self.continuation.is_running)

        self.continuation.status = AutofixStatus.COMPLETED
        self.assertFalse(self.continuation.is_running)

    def test_has_timed_out(self):
        now = datetime.datetime.now()
        self.continuation.status = AutofixStatus.PROCESSING
        self.continuation.last_triggered_at = now - datetime.timedelta(
            minutes=AUTOFIX_HARD_TIME_OUT_MINS + 1
        )
        self.assertTrue(self.continuation.has_timed_out)

        self.continuation.last_triggered_at = now - datetime.timedelta(
            minutes=AUTOFIX_HARD_TIME_OUT_MINS - 1
        )
        self.assertFalse(self.continuation.has_timed_out)

    def test_has_timed_out_not_running(self):
        now = datetime.datetime.now()
        self.continuation.status = AutofixStatus.COMPLETED
        self.continuation.last_triggered_at = now - datetime.timedelta(
            minutes=AUTOFIX_HARD_TIME_OUT_MINS + 1
        )
        self.assertFalse(self.continuation.has_timed_out)

    def test_has_timed_out_no_last_triggered(self):
        self.continuation.status = AutofixStatus.PROCESSING
        self.continuation.last_triggered_at = None
        self.assertFalse(self.continuation.has_timed_out)

    def test_has_timed_out_with_update(self):
        now = datetime.datetime.now()
        self.continuation.status = AutofixStatus.PROCESSING
        self.continuation.last_triggered_at = now - datetime.timedelta(
            minutes=AUTOFIX_HARD_TIME_OUT_MINS - 1
        )
        self.continuation.updated_at = now - datetime.timedelta(
            seconds=AUTOFIX_UPDATE_TIMEOUT_SECS - 1
        )
        self.assertFalse(self.continuation.has_timed_out)

        self.continuation.updated_at = now - datetime.timedelta(
            seconds=AUTOFIX_UPDATE_TIMEOUT_SECS + 1
        )
        self.assertTrue(self.continuation.has_timed_out)

    def test_has_timed_out_edge_cases(self):
        now = datetime.datetime.now()
        self.continuation.status = AutofixStatus.PROCESSING
        self.continuation.last_triggered_at = now - datetime.timedelta(
            minutes=AUTOFIX_HARD_TIME_OUT_MINS, seconds=1
        )
        self.assertTrue(self.continuation.has_timed_out)

        self.continuation.last_triggered_at = now - datetime.timedelta(
            minutes=AUTOFIX_HARD_TIME_OUT_MINS, seconds=-1
        )
        self.assertFalse(self.continuation.has_timed_out)

    def test_kill_all_processing_steps(self):
        step1 = DefaultStep(key="step1", status=AutofixStatus.PROCESSING, title="test")
        step2 = DefaultStep(key="step2", status=AutofixStatus.PROCESSING, title="test")
        self.continuation.steps = [step1, step2]

        self.continuation.kill_all_processing_steps()
        self.assertEqual(self.continuation.signals, [make_kill_signal(), make_kill_signal()])


@parametrize
def test_event_no_exception_events(event: SentryEventData, entry: InvalidEventEntry):
    event["entries"] = [entry]
    assert len(EventDetails.from_event(event).exceptions) == 0


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
            vars_str = json.dumps(StacktraceFrame._trim_vars(frame.vars), indent=2)
            assert vars_str in stack_str
        else:
            assert "---\nVariable" not in stack_str


class TestFilePatch:
    def test_to_hunks(self):
        patch = textwrap.dedent(
            """\
            @@ -1,3 +1,4 @@
             def hello():
                 print('hello')
            +    print('world')  # Line 3 is added
                 print('goodbye')
            __WHITESPACE__
            @@ -20,3 +21,4 @@ def __init__(self):
                 print('end')
            +    print('new end')  # Line 22 is added
                 return
            """
        ).replace("__WHITESPACE__", " ")
        hunks_expected = [
            Hunk(
                source_start=1,
                source_length=3,
                target_start=1,
                target_length=4,
                section_header="@@ -1,3 +1,4 @@",
                lines=[
                    Line(value=" def hello():", line_type=" "),
                    Line(value="     print('hello')", line_type=" "),
                    Line(value="+    print('world')  # Line 3 is added", line_type="+"),
                    Line(value="     print('goodbye')", line_type=" "),
                    Line(value=" ", line_type=" "),
                ],
            ),
            Hunk(
                source_start=20,
                source_length=3,
                target_start=21,
                target_length=4,
                section_header="@@ -20,3 +21,4 @@ def __init__(self):",
                lines=[
                    Line(value="     print('end')", line_type=" "),
                    Line(value="+    print('new end')  # Line 22 is added", line_type="+"),
                    Line(value="     return", line_type=" "),
                ],
            ),
        ]
        hunks = FilePatch.to_hunks(patch)
        assert len(hunks) == len(hunks_expected)
        assert hunks == hunks_expected
