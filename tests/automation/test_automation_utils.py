from typing import cast
from xml.etree.ElementTree import Element

import pytest
from pydantic import BaseModel

from seer.automation.utils import (
    ConsentError,
    batch_texts_by_token_count,
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


class _BatchTextsByTokensTestCase(BaseModel):
    # Function parameters:
    texts: list[str]
    max_tokens: int
    avg_num_chars_per_token: float
    # Test case attributes:
    batches_expected: list[list[str]]
    id: str


@pytest.mark.parametrize(
    "test_case",
    [  # A test case generator like hypothesis would be better here.
        _BatchTextsByTokensTestCase(
            texts=[],
            max_tokens=100,
            avg_num_chars_per_token=4.0,
            batches_expected=[],
            id="empty_list_produces_no_batches",
        ),
        _BatchTextsByTokensTestCase(
            texts=["Hello world"],
            max_tokens=10,
            avg_num_chars_per_token=4.0,
            batches_expected=[["Hello world"]],
            id="single_text_under_limit_stays_in_one_batch",
        ),
        _BatchTextsByTokensTestCase(
            texts=["This is a very long text that exceeds the token limit"],
            max_tokens=5,
            avg_num_chars_per_token=4.0,
            batches_expected=[["This is a very long text that exceeds the token limit"]],
            id="single_text_over_limit_gets_own_batch",
        ),
        _BatchTextsByTokensTestCase(
            texts=["Hi", "world", "this", "is", "a", "test"],
            max_tokens=3,
            avg_num_chars_per_token=4.0,
            batches_expected=[["Hi", "world"], ["this", "is", "a"], ["test"]],
            id="multiple_texts_are_batched_by_token_limit",
        ),
        _BatchTextsByTokensTestCase(
            texts=["Hello", "world", "test"],
            max_tokens=3,
            avg_num_chars_per_token=2.0,
            batches_expected=[["Hello"], ["world"], ["test"]],
            id="small_chars_per_token_creates_more_batches",
        ),
        _BatchTextsByTokensTestCase(
            texts=["Hello", "world", "test"],
            max_tokens=3,
            avg_num_chars_per_token=10.0,
            batches_expected=[["Hello", "world", "test"]],
            id="large_chars_per_token_allows_more_texts_per_batch",
        ),
        _BatchTextsByTokensTestCase(
            texts=["First", "Second", "ThisIsAVeryLongText", "Third", "Fourth"],
            max_tokens=5,
            avg_num_chars_per_token=4.0,
            batches_expected=[["First", "Second"], ["ThisIsAVeryLongText"], ["Third", "Fourth"]],
            id="long_text_in_middle_creates_separate_batch",
        ),
        _BatchTextsByTokensTestCase(
            texts=["1234", "5678", "90"],
            max_tokens=2,
            avg_num_chars_per_token=4.0,
            batches_expected=[["1234", "5678"], ["90"]],
            id="texts_exactly_hitting_token_limit_boundary",
        ),
        _BatchTextsByTokensTestCase(
            texts=[
                "short",
                "a relatively much longer text here",
                "medium text",
                "tiny",
            ],
            max_tokens=10,
            avg_num_chars_per_token=4.0,
            batches_expected=[
                ["short"],
                ["a relatively much longer text here"],
                ["medium text", "tiny"],
            ],
            id="mixed_length_texts_are_batched_appropriately",
        ),
    ],
    ids=lambda test_case: cast(_BatchTextsByTokensTestCase, test_case).id,
)
def test_batch_texts_by_token_count(test_case: _BatchTextsByTokensTestCase):
    text_generator = (text for text in test_case.texts)
    batches = batch_texts_by_token_count(
        text_generator,
        max_tokens=test_case.max_tokens,
        avg_num_chars_per_token=test_case.avg_num_chars_per_token,
    )
    assert list(batches) == test_case.batches_expected
