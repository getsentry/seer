import pytest
from lxml import etree
from seer.automation.autofix.components.root_cause.component import RootCauseAnalysisComponent
from seer.automation.autofix.components.root_cause.models import RootCauseAnalysisRequest, RootCauseAnalysisOutputPromptXml
from seer.automation.autofix.utils import escape_multi_xml
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.models import EventDetails, ExceptionDetails, Stacktrace, StacktraceFrame

class MockAutofixContext(AutofixContext):
    def __init__(self):
        self.state = MockState()

class MockState:
    def get(self):
        return MockRequestState()

    def update(self):
        return self

class MockRequestState:
    def __init__(self):
        self.request = MockRequest()

class MockRequest:
    def __init__(self):
        self.issue = MockIssue()

class MockIssue:
    def __init__(self):
        self.events = [MockEvent()]

class MockEvent:
    def __init__(self):
        self.title = \"TypeError: Cannot read properties of undefined (reading 'dropped')\"
        self.exceptions = [ExceptionDetails(type='TypeError', value=\"Cannot read properties of undefined (reading 'dropped')\", stacktrace=Stacktrace(frames=[StacktraceFrame(function='R', filename='../node_modules/scheduler/cjs/scheduler.production.min.js', abs_path='webpack:///../node_modules/scheduler/cjs/scheduler.production.min.js', line_no=14, col_no=128, context=[(9, ' */'), (10, '{snip} ar d=c-1&gt;&gt;&gt;1,e=a[d];if(0&lt;g(e,b))a[d]=b,a[c]=e,c=d;else break a}}function h(a){return 0===a.length?null:a[0]}function k(a){if(0===a.length?re {snip}'), (11, '{snip} id}if(\\\"object\\\"===typeof performance&amp;&amp;\\\"function\\\"===typeof performance.now){var l=performance;exports.unstable_now=function(){return l.now()}} {snip}'), (12, '{snip} 0!==navigator.scheduling.isInputPending&amp;&amp;navigator.scheduling.isInputPending.bind(navigator.scheduling);function G(a){for(var b=h(t);null!== {snip}'), (13, '{snip} =h(r);null!==v&amp;&amp;(!(v.expirationTime&gt;b)||a&amp;&amp;!M());){var d=v....\")]))]

@pytest.fixture
def context():
    return MockAutofixContext()

def test_valid_xml(context):
    component = RootCauseAnalysisComponent(context)
    request = RootCauseAnalysisRequest(event_details=context.state.get().request.issue.events[0])
    valid_xml = \"&amp;lt;root&amp;gt;&amp;lt;thoughts&amp;gt;Valid thoughts&amp;lt;/thoughts&amp;gt;&amp;lt;/root&amp;gt;\"
    sanitized_response = escape_multi_xml(valid_xml, ['thoughts', 'snippet', 'title', 'description'])
    xml_content = f\"&amp;lt;root&amp;gt;{sanitized_response}&amp;lt;/root&amp;gt;\"

    try:
        etree.fromstring(xml_content)
    except etree.XMLSyntaxError:
        pytest.fail(\"etree.XMLSyntaxError raised unexpectedly for valid XML\")

def test_malformed_xml(context):
    component = RootCauseAnalysisComponent(context)
    request = RootCauseAnalysisRequest(event_details=context.state.get().request.issue.events[0])
    malformed_xml = \"&amp;lt;root&amp;gt;&amp;lt;thoughts&amp;gt;Malformed thoughts&amp;lt;/root&amp;gt;\"  # Missing closing tag for thoughts
    sanitized_response = escape_multi_xml(malformed_xml, ['thoughts', 'snippet', 'title', 'description'])
    xml_content = f\"&amp;lt;root&amp;gt;{sanitized_response}&amp;lt;/root&amp;gt;\"

    with pytest.raises(ValueError, match=\"Malformed XML content\"):
        etree.fromstring(xml_content)

def test_empty_xml(context):
    component = RootCauseAnalysisComponent(context)
    request = RootCauseAnalysisRequest(event_details=context.state.get().request.issue.events[0])
    empty_xml = \"&amp;lt;root&amp;gt;&amp;lt;/root&amp;gt;\"
    sanitized_response = escape_multi_xml(empty_xml, ['thoughts', 'snippet', 'title', 'description'])
    xml_content = f\"&amp;lt;root&amp;gt;{sanitized_response}&amp;lt;/root&amp;gt;\"

    try:
        etree.fromstring(xml_content)
    except etree.XMLSyntaxError:
        pytest.fail(\"etree.XMLSyntaxError raised unexpectedly for empty XML\")