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
    remove_cdata,
)
from seer.configuration import AppConfig
from seer.dependency_injection import resolve
from seer.rpc import DummyRpcClient


class TestCheckGenAiConsent:
    def test_check_with_should_consent(self):
        resolve(AppConfig).NO_SENTRY_INTEGRATION = False
        resolve(DummyRpcClient).should_consent = True
        assert check_genai_consent(1) is True
        assert ("get_organization_autofix_consent", {"org_id": 1}) in resolve(
            DummyRpcClient
        ).invocations

    def test_check_without_should_consent(self):
        resolve(AppConfig).NO_SENTRY_INTEGRATION = False
        resolve(DummyRpcClient).should_consent = False
        assert check_genai_consent(1) is False
        assert ("get_organization_autofix_consent", {"org_id": 1}) in resolve(
            DummyRpcClient
        ).invocations

        with pytest.raises(ConsentError):
            raise_if_no_genai_consent(1)

    def test_check_with_no_sentry_integration(self):
        resolve(AppConfig).NO_SENTRY_INTEGRATION = True
        assert check_genai_consent(1) is True
        assert not resolve(DummyRpcClient).invocations


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

    def test_remove_cdata(self):
        assert remove_cdata("<![CDATA[Hello World]]>") == "Hello World"
        assert remove_cdata("No CDATA here") == "No CDATA here"
        assert remove_cdata("<![CDATA[Line1\nLine2]]>") == "Line1\nLine2"
