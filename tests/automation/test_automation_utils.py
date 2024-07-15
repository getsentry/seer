import os
from unittest.mock import patch
from xml.etree.ElementTree import Element

import pytest

from seer.automation.utils import (
    ConsentError,
    check_genai_consent,
    escape_multi_xml,
    escape_xml,
    escape_xml_chars,
    extract_text_inside_tags,
    extract_xml_element_text,
    raise_if_no_genai_consent,
)


class TestCheckGenAiConsent:
    @pytest.fixture
    def mock_rpc_client(self):
        with patch("seer.automation.utils.SentryRpcClient") as mock:
            yield mock

    def test_check_passing(self, mock_rpc_client):
        os.environ["NO_SENTRY_INTEGRATION"] = ""
        mock_rpc_client.return_value.call.return_value = {"consent": True}
        assert check_genai_consent(1) is True

    def test_check_failing(self, mock_rpc_client):
        os.environ["NO_SENTRY_INTEGRATION"] = ""
        mock_rpc_client.return_value.call.return_value = {"consent": False}
        assert check_genai_consent(1) is False

    def test_check_failing_none(self, mock_rpc_client):
        os.environ["NO_SENTRY_INTEGRATION"] = ""
        mock_rpc_client.return_value.call.return_value = None
        assert check_genai_consent(1) is False

    def test_check_passing_without_integration(self, mock_rpc_client):
        os.environ["NO_SENTRY_INTEGRATION"] = "1"
        mock_rpc_client.return_value.call.return_value = None

        assert check_genai_consent(1) is True
        mock_rpc_client.return_value.call.assert_not_called()


class TestCheckGenAiConsentRaise:
    @pytest.fixture
    def mock_check_genai_consent(self):
        with patch("seer.automation.utils.check_genai_consent", return_value=False) as mock:
            yield mock

    def test_check_failing_raise(self, mock_check_genai_consent):
        with pytest.raises(ConsentError):
            raise_if_no_genai_consent(1)


class TestXmlUtils:
    def test_escape_xml_chars(self):
        input_str = 'Test & "quote" < > \''
        expected = "Test &amp; &quot;quote&quot; &lt; &gt; &apos;"
        assert escape_xml_chars(input_str) == expected

    def test_escape_xml(self):
        input_str = '<tag>Test & "quote" < > \'</tag>'
        expected = "<tag>Test &amp; &quot;quote&quot; &lt; &gt; &apos;</tag>"
        assert escape_xml(input_str, "tag") == expected

    def test_escape_multi_xml(self):
        input_str = '<tag1>Test & "quote"</tag1><tag2>< > \'</tag2>'
        expected = "<tag1>Test &amp; &quot;quote&quot;</tag1><tag2>&lt; &gt; &apos;</tag2>"
        assert escape_multi_xml(input_str, ["tag1", "tag2"]) == expected

    def test_extract_text_inside_tags(self):
        input_str = "<tag>\n  Test content  </tag>"
        assert extract_text_inside_tags(input_str, "tag", strip_newlines=True) == "  Test content  "

    def test_extract_text_inside_tags_with_newlines(self):
        input_str = "<tag>\n  Test content\n  </tag>"
        assert (
            extract_text_inside_tags(input_str, "tag", strip_newlines=False)
            == "\n  Test content\n  "
        )

    def test_extract_xml_element_text(self):
        element = Element("root")
        child = Element("child")
        child.text = "  Test content  "
        element.append(child)
        assert extract_xml_element_text(element, "child") == "Test content"

    def test_extract_xml_element_text_missing(self):
        element = Element("root")
        assert extract_xml_element_text(element, "child") is None
