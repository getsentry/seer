import textwrap

from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.models import EventDetails
from seer.automation.resources.models import FindIssueResourcesRequest, FindIssueResourcesResponse
from seer.dependency_injection import inject, injected


@observe(name="Find Issue Resources")
@inject
def find_issue_resources(
    request: FindIssueResourcesRequest, llm_client: LlmClient = injected
) -> FindIssueResourcesResponse:
    event_details = EventDetails.from_event(request.issue.events[0])
    event_content = event_details.format_event()

    prompt = textwrap.dedent(
        f"""I've seen and understood the following issue in my code. Are there any online resources to help me understand and debug the issue, such as forums, tutorials, or documentation?
        If there are no relevant resources, return nothing.
        Your final answer should highlight the key insights from the most relevant resources you find online, all in under 30 words.

        {event_content}
        """
    )

    text, resources = llm_client.generate_text_from_web_search(
        prompt=prompt, model=GeminiProvider.model("gemini-2.0-flash-exp")
    )
    return FindIssueResourcesResponse(group_id=request.group_id, text=text, resources=resources)


def run_find_issue_resources(request: FindIssueResourcesRequest):
    langfuse_tags = []
    if request.organization_slug:
        langfuse_tags.append(f"org:{request.organization_slug}")
    if request.project_id:
        langfuse_tags.append(f"project:{request.project_id}")
    if request.group_id:
        langfuse_tags.append(f"group:{request.group_id}")

    resources = find_issue_resources(request=request)
    return resources
