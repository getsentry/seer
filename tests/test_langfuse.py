from unittest.mock import Mock, patch

import pytest
from langfuse import Langfuse

from seer.configuration import AppConfig, provide_test_defaults
from seer.dependency_injection import Module, resolve
from seer.langfuse import append_langfuse_trace_tags, provide_langfuse

langfuse_configuration_test_module = Module()


@langfuse_configuration_test_module.provider
def provide_config() -> AppConfig:
    test_config = provide_test_defaults()

    test_config.LANGFUSE_PUBLIC_KEY = "public_key"
    test_config.LANGFUSE_SECRET_KEY = "secret_key"
    test_config.LANGFUSE_HOST = "https://api.langfuse.com"

    return test_config


@langfuse_configuration_test_module.provider
def provide_langfuse_mock() -> Langfuse:
    return Mock(spec=Langfuse)


@pytest.fixture(autouse=True)
def setup_langfuse_config():
    with langfuse_configuration_test_module:
        yield


class TestProvideLangfuse:
    @patch("seer.langfuse.Langfuse")
    def test_provide_langfuse(self, mock_langfuse):
        provide_langfuse()

        mock_langfuse.assert_called_once_with(
            public_key="public_key",
            secret_key="secret_key",
            host="https://api.langfuse.com",
            enabled=True,
        )

    @patch("seer.langfuse.Langfuse")
    def test_provide_langfuse_disabled(self, mock_langfuse):
        resolve(AppConfig).LANGFUSE_HOST = ""

        provide_langfuse()

        mock_langfuse.assert_called_once_with(
            public_key="public_key",
            secret_key="secret_key",
            host="",
            enabled=False,
        )


class TestAppendLangfuseTraceTags:
    @patch("seer.langfuse.langfuse_context")
    def test_append_langfuse_trace_tags(self, mock_langfuse_context):
        langfuse = resolve(Langfuse)

        mock_langfuse_context.get_current_trace_id.return_value = "trace_id"
        mock_trace = Mock()
        mock_trace.tags = ["existing_tag"]
        langfuse.get_trace.return_value = mock_trace

        new_tags = ["new_tag1", "new_tag2"]
        append_langfuse_trace_tags(new_tags)

        langfuse.get_trace.assert_called_once_with("trace_id")
        mock_langfuse_context.update_current_trace.assert_called_once_with(
            tags=["existing_tag", "new_tag1", "new_tag2"]
        )

    @patch("seer.langfuse.langfuse_context")
    def test_append_langfuse_trace_tags_no_existing_tags(self, mock_langfuse_context):
        langfuse = resolve(Langfuse)

        mock_langfuse_context.get_current_trace_id.return_value = "trace_id"
        mock_trace = Mock()
        mock_trace.tags = None
        langfuse.get_trace.return_value = mock_trace

        new_tags = ["new_tag1", "new_tag2"]
        append_langfuse_trace_tags(new_tags)

        langfuse.get_trace.assert_called_once_with("trace_id")
        mock_langfuse_context.update_current_trace.assert_called_once_with(
            tags=["new_tag1", "new_tag2"]
        )

    @patch("seer.langfuse.langfuse_context")
    def test_append_langfuse_trace_tags_no_trace_id(self, mock_langfuse_context):
        langfuse = resolve(Langfuse)

        mock_langfuse_context.get_current_trace_id.return_value = None

        new_tags = ["new_tag1", "new_tag2"]
        append_langfuse_trace_tags(new_tags)

        langfuse.get_trace.assert_not_called()
        mock_langfuse_context.update_current_trace.assert_not_called()

    @patch("seer.langfuse.langfuse_context")
    @patch("seer.langfuse.logger")
    def test_append_langfuse_trace_tags_exception(self, mock_logger, mock_langfuse_context):
        mock_langfuse_context.get_current_trace_id.side_effect = Exception("Test exception")

        new_tags = ["new_tag1", "new_tag2"]
        append_langfuse_trace_tags(new_tags)

        mock_logger.exception.assert_called_once()
