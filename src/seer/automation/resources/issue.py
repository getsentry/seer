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
        f"""I've seen and understood the following issue in my code. Are there any online resources to help me understand and debug the issue, such as forums, tutorials, and documentation?
        If so, please find the most relevant ones and share them with me. Do not summarize the issue itself, but just the key applicable info you find online.
        It is possible however that the issue below is unique to my code and online resources you find don't apply. In that case, return nothing.
        But if you do find useful resources relevant to the issue, please make your answer a very concise paragraph (max 25 words).

        {event_content}
        """
    )

    text, resources = llm_client.generate_text_from_web_search(
        prompt=prompt, model=GeminiProvider.model("gemini-2.0-flash-exp")
    )
    print(text)
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
