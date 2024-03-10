import asyncio
import contextlib
import dataclasses
import datetime
import threading
import unittest
from queue import Queue
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
from celery import Celery
from celery.worker import WorkController
from flask.testing import FlaskClient
from pydantic import ValidationError

import seer.app
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix import Autofix
from seer.automation.autofix.models import (
    AutofixCompleteArgs,
    AutofixGroupState,
    AutofixRequest,
    AutofixStatus,
    AutofixStepUpdateArgs,
    IssueDetails,
    PlanningOutput,
    ProblemDiscoveryOutput,
    ProblemDiscoveryResult,
    RepoDefinition,
    SentryEvent,
    SentryEventEntryDataValue,
    SentryExceptionEntry,
)
from seer.automation.autofix.tasks import run_autofix
from seer.db import ProcessRequest
from seer.generator import change_watcher, parameterize
from seer.tasks import AsyncApp
from seer.utils import closing_queue
from tests.generators import GptClientMock, RpcClientMock, SentryFrameDict


@dataclasses.dataclass
class E2ETest:
    rpc_client_mock: RpcClientMock
    gpt_client_mock: GptClientMock
    request: AutofixRequest
    sentry_frames: tuple[SentryFrameDict, SentryFrameDict, SentryFrameDict]
    event: SentryEvent
    exception_entry: SentryExceptionEntry
    event_entry_data_value: SentryEventEntryDataValue
    autofix_group_state: AutofixGroupState
    client: FlaskClient = dataclasses.field(default_factory=seer.app.app.test_client)
    async_app: AsyncApp = dataclasses.field(default_factory=lambda: AsyncApp(fail_fast=True))

    def __post_init__(self):
        self.event_entry_data_value["stacktrace"]["frames"] = list(self.sentry_frames)
        self.exception_entry.data["values"] = [self.event_entry_data_value]
        self.event.entries = [self.exception_entry.model_dump(mode="json")]
        self.request.issue.events = [self.event]

    def send_autofix_request(self):
        response = self.client.post(
            "/v0/automation/autofix", json=self.request.model_dump(mode="json")
        )
        assert 200 <= response.status_code < 300

    def get_autofix_state(
        self, method_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | tuple[int, str]:
        if args.get("issue_id") == self.request.issue.id:
            return self.autofix_group_state.model_dump(mode="json")
        return 404, "Not Found"

    def on_autofix_complete(
        self, method_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | tuple[int, str]:
        if args.get("issue_id") != self.request.issue.id:
            return 404, "Not Found"
        try:
            args = AutofixCompleteArgs.model_validate(args)
        except ValidationError as e:
            return 400, str(e)

        self.autofix_group_state.model_copy(update=args.model_dump())
        self.autofix_group_state.completedAt = datetime.datetime.now()

        return {}

    def on_autofix_step_update(
        self, method_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | tuple[int, str]:
        if args.get("issue_id") != self.request.issue.id:
            return 404, "Not Found"
        try:
            args = AutofixStepUpdateArgs.model_validate(args)
        except ValidationError as e:
            return 400, str(e)

        self.autofix_group_state.model_copy(update=args.model_dump())

        return {}

    def make_problem_assessment(self, model: str, messages: list[Message], args: dict[str, Any]):
        if self.request.issue.title in messages[-1]:
            msg = (
                "<problem>This is a test</problem>"
                "<reasoning>Because we are testing</reasoning>"
                "<actionability_score>0.5</actionability_score>"
            )
            return Message(content=msg), Usage(
                completion_tokens=len(msg), prompt_tokens=sum(len(m.content) for m in messages)
            )
        return None

    @contextlib.contextmanager
    def enabled(self, celery_app: Celery, celery_worker: WorkController) -> Iterator[Queue]:
        q = Queue()
        kill_event = threading.Event()
        async_app = threading.Thread(target=lambda: asyncio.run(self.async_app.run(kill_event)))
        celery_app.task(run_autofix)
        celery_worker.reload()

        self.async_app.kill_event = threading.Event()
        self.async_app.completed_queue = q
        try:
            with self.rpc_client_mock.enabled(
                get_autofix_state=self.get_autofix_state,
                on_autofix_complete=self.on_autofix_complete,
                on_autofix_step_update=self.on_autofix_step_update,
            ), self.gpt_client_mock.enabled(self.make_problem_assessment), closing_queue(q):
                async_app.start()
                yield q
        finally:
            kill_event.set()
            async_app.join()


@pytest.mark.skip(reason="celery makes mocking impossible")
@parameterize(arg_set=("test_case",))
def test_non_actionable_response(
    test_case: E2ETest,
    celery_app: Celery,
    celery_worker: WorkController,
):
    steps_watcher = change_watcher(lambda: test_case.autofix_group_state.steps)
    status_watcher = change_watcher(lambda: test_case.autofix_group_state.status)

    with test_case.enabled(celery_app=celery_app, celery_worker=celery_worker) as processed_queue:
        test_case.send_autofix_request()
        with steps_watcher as steps_changed, status_watcher as status_changed:
            processed: ProcessRequest = processed_queue.get(timeout=5)

    assert processed.name == test_case.request.process_request_name
    assert processed.payload == test_case.request.model_dump(mode="json")

    assert steps_changed
    assert status_changed


class TestAutoFixRunFlow(unittest.TestCase):
    @patch("seer.automation.autofix.autofix.AutofixEventManager")
    @patch("seer.automation.autofix.autofix.AutofixContext")
    def test_problem_discovery_short_circuit(
        self,
        mock_context_manager,
        mock_event_manager,
    ):
        autofix = Autofix(
            request=AutofixRequest(
                additional_context="",
                organization_id=1,
                project_id=1,
                repos=[RepoDefinition(provider="github", owner="owner", name="name")],
                issue=IssueDetails(
                    id=1,
                    title="",
                    events=[
                        SentryEvent(
                            entries=[
                                {
                                    "type": "exception",
                                    "data": {
                                        "values": [
                                            {
                                                "stacktrace": {
                                                    "frames": [
                                                        {
                                                            "function": "test",
                                                            "filename": "file2.py",
                                                            "lineNo": 10,
                                                            "colNo": 0,
                                                            "context": [],
                                                            "inApp": True,
                                                            "absPath": "file2.py",
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    },
                                }
                            ]
                        )
                    ],
                ),
            ),
            event_manager=mock_event_manager,
        )

        autofix.run_problem_discovery_agent = MagicMock()
        autofix.run_problem_discovery_agent.return_value = ProblemDiscoveryOutput(
            actionability_score=0.5,
            description="",
            reasoning="",
        )

        autofix.run()

        autofix.run_problem_discovery_agent.assert_called_once()
        autofix.event_manager.send_problem_discovery_result.assert_called_once_with(
            ProblemDiscoveryResult(
                status="CANCELLED",
                description="",
                reasoning="",
            )
        )
        autofix.context.create_codebase_index.assert_not_called()

    @patch("seer.automation.autofix.autofix.AutofixEventManager")
    @patch("seer.automation.autofix.autofix.AutofixContext")
    def test_stacktrace_files_short_circuit(
        self,
        mock_autofix_context,
        mock_event_manager,
    ):
        autofix = Autofix(
            request=AutofixRequest(
                additional_context="",
                organization_id=1,
                project_id=1,
                repos=[RepoDefinition(provider="github", owner="owner", name="name")],
                base_commit_sha="",
                issue=IssueDetails(
                    id=1,
                    title="",
                    events=[
                        SentryEvent(
                            entries=[
                                {
                                    "type": "exception",
                                    "data": {
                                        "values": [
                                            {
                                                "stacktrace": {
                                                    "frames": [
                                                        {
                                                            "function": "test",
                                                            "filename": "file2.py",
                                                            "lineNo": 10,
                                                            "colNo": 0,
                                                            "context": [],
                                                            "inApp": True,
                                                            "absPath": "file2.py",
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    },
                                }
                            ]
                        )
                    ],
                ),
            ),
            event_manager=mock_event_manager,
        )

        autofix.context.diff_contains_stacktrace_files = MagicMock()
        autofix.context.diff_contains_stacktrace_files.return_value = False
        autofix.run_problem_discovery_agent = MagicMock()
        autofix.run_problem_discovery_agent.return_value = ProblemDiscoveryOutput(
            actionability_score=1.0,
            description="",
            reasoning="",
        )
        autofix.run_planning_agent = MagicMock()
        autofix.run_planning_agent.return_value = PlanningOutput(title="", description="", steps=[])
        autofix.run_execution_agent = MagicMock()

        mock_codebase_index = MagicMock()
        autofix.context.get_codebase = MagicMock()
        autofix.context.get_codebase.return_value = mock_codebase_index
        autofix.context.has_codebase_index.return_value = True

        autofix.run()

        autofix.run_problem_discovery_agent.assert_called_once()
        autofix.event_manager.send_problem_discovery_result.assert_called_once_with(
            ProblemDiscoveryResult(
                status="CONTINUE",
                description="",
                reasoning="",
            )
        )

        mock_codebase_index.update_codebase_index.assert_not_called()

        autofix.run_planning_agent.assert_called_once()
        autofix.run_execution_agent.assert_not_called()

    @patch("seer.automation.autofix.autofix.AutofixEventManager")
    @patch("seer.automation.autofix.autofix.AutofixContext")
    def test_stacktrace_not_short_circuit(
        self,
        mock_context_manager,
        mock_event_manager,
    ):
        autofix = Autofix(
            request=AutofixRequest(
                additional_context="",
                organization_id=1,
                project_id=1,
                repos=[RepoDefinition(provider="github", owner="owner", name="name")],
                base_commit_sha="",
                issue=IssueDetails(
                    id=1,
                    title="",
                    events=[
                        SentryEvent(
                            entries=[
                                {
                                    "type": "exception",
                                    "data": {
                                        "values": [
                                            {
                                                "stacktrace": {
                                                    "frames": [
                                                        {
                                                            "function": "test",
                                                            "filename": "file2.py",
                                                            "lineNo": 10,
                                                            "colNo": 0,
                                                            "context": [],
                                                            "inApp": True,
                                                            "absPath": "file2.py",
                                                        }
                                                    ]
                                                }
                                            }
                                        ]
                                    },
                                }
                            ]
                        )
                    ],
                ),
            ),
            event_manager=mock_event_manager,
        )

        autofix.context.diff_contains_stacktrace_files = MagicMock()
        autofix.context.diff_contains_stacktrace_files.return_value = True
        autofix.run_problem_discovery_agent = MagicMock()
        autofix.run_problem_discovery_agent.return_value = ProblemDiscoveryOutput(
            actionability_score=1.0,
            description="",
            reasoning="",
        )
        autofix.run_planning_agent = MagicMock()
        autofix.run_planning_agent.return_value = PlanningOutput(title="", description="", steps=[])
        autofix.run_execution_agent = MagicMock()

        mock_codebase_index = MagicMock()
        autofix.context.get_codebase = MagicMock()
        autofix.context.get_codebase.return_value = mock_codebase_index
        autofix.context.has_codebase_index.return_value = True

        autofix.run()

        autofix.run_problem_discovery_agent.assert_called_once()
        autofix.event_manager.send_problem_discovery_result.assert_called_once_with(
            ProblemDiscoveryResult(
                status="CONTINUE",
                description="",
                reasoning="",
            )
        )

        mock_codebase_index.update_codebase_index.assert_not_called()

        autofix.run_planning_agent.assert_called_once()
        autofix.run_execution_agent.assert_not_called()
