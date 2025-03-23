import unittest
from unittest.mock import MagicMock, patch

from seer.automation.agent.models import Message
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.models import RepoDefinition


class DummyRequest:
    def __init__(self, repo):
        self.repo = repo


class DummyContinuation:
    def __init__(self, run_id, request, file_changes=None, signals=None):
        self.run_id = run_id
        self.request = request
        self.file_changes = file_changes or []
        self.signals = signals or []


class DummyCodegenContinuationState:
    def __init__(self, run_id, request, file_changes=None, signals=None):
        self._state = DummyContinuation(run_id, request, file_changes, signals)

    def get(self):
        return self._state

    def update(self):
        class DummyContextManager:
            def __init__(self, state):
                self.state = state

            def __enter__(self):
                return self.state

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return DummyContextManager(self._state)


class DummyFileChange:
    def __init__(self, path, new_content):
        self.path = path
        self.new_content = new_content

    def apply(self, original_content):
        return self.new_content


class TestCodegenContext(unittest.TestCase):
    def setUp(self):
        self.repo = RepoDefinition(
            provider="github", owner="test_owner", name="test_repo", external_id="dummy_id"
        )
        self.request = DummyRequest(self.repo)
        self.state = DummyCodegenContinuationState(run_id=1, request=self.request)
        self.codegen_context = CodegenContext(self.state)

    def test_run_id(self):
        self.assertEqual(self.codegen_context.run_id, 1)

    def test_signals_getter_setter(self):
        self.codegen_context.signals = ["signal1", "signal2"]
        self.assertEqual(self.codegen_context.signals, ["signal1", "signal2"])

    def test_get_file_contents_no_local_changes(self):
        dummy_client = MagicMock()
        dummy_client.get_file_content.return_value = ("original content", None)
        with patch.object(self.codegen_context, "get_repo_client", return_value=dummy_client):
            content = self.codegen_context.get_file_contents(
                "dummy_path", ignore_local_changes=True
            )
            self.assertEqual(content, "original content")

    def test_get_file_contents_with_local_changes(self):
        file_change = DummyFileChange("dummy_path", "changed content")
        self.state.get().file_changes = [file_change]
        dummy_client = MagicMock()
        dummy_client.get_file_content.return_value = ("original content", None)
        with patch.object(self.codegen_context, "get_repo_client", return_value=dummy_client):
            content = self.codegen_context.get_file_contents("dummy_path")
            self.assertEqual(content, "changed content")

    @patch("seer.automation.codegen.codegen_context.Session")
    def test_store_and_get_memory(self, mock_session):
        fake_db = {}

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def query(self, model):
                class FakeQuery:
                    def __init__(self, db):
                        self.db = db

                    def where(self, condition):
                        return self

                    def one_or_none(self):
                        return self.db.get("memory")

                return FakeQuery(fake_db)

            def merge(self, obj):
                fake_db["memory"] = obj

            def commit(self):
                pass

        mock_session.return_value = FakeSession()
        key = "test_key"
        memory = [Message(role="user", content="Test message")]
        self.codegen_context.store_memory(key, memory)
        result = self.codegen_context.get_memory(key, past_run_id=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "user")
        self.assertEqual(result[0].content, "Test message")

    @patch("seer.automation.codegen.codegen_context.Session")
    def test_update_stored_memory(self, mock_session):
        fake_db = {}

        class FakeSession:
            def __init__(self):
                self.db = fake_db

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def query(self, model):
                class FakeQuery:
                    def __init__(self, db):
                        self.db = db

                    def where(self, condition):
                        return self

                    def one_or_none(self):
                        return self.db.get("memory")

                return FakeQuery(self.db)

            def merge(self, obj):
                self.db["memory"] = obj

            def commit(self):
                pass

        mock_session.return_value = FakeSession()
        key = "update_key"
        initial_memory = [Message(role="user", content="Old message")]
        self.codegen_context.store_memory(key, initial_memory)
        new_memory = [Message(role="user", content="New message")]
        self.codegen_context.update_stored_memory(key, new_memory, original_run_id=1)
        result = self.codegen_context.get_memory(key, past_run_id=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "New message")

    @patch("seer.automation.codegen.codegen_context.Session")
    def test_get_previous_run_context(self, mock_session):
        fake_context = MagicMock()

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def query(self, model):
                class FakeQuery:
                    def where(self, *args, **kwargs):
                        return self

                    def one_or_none(self):
                        return fake_context

                return FakeQuery()

        mock_session.return_value = FakeSession()
        result = self.codegen_context.get_previous_run_context("test_owner", "test_repo", 123)
        self.assertEqual(result, fake_context)
