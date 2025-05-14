import textwrap

import sentry_sdk
from langfuse.decorators import observe

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.models import Message, Usage
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.components.insight_sharing.models import (
    BreadcrumbsSource,
    CodeSource,
    ConnectedErrorSource,
    DiffSource,
    HttpRequestSource,
    InsightSharingOutput,
    InsightSources,
    JustificationOutput,
    ProfileSource,
    StacktraceSource,
    TraceEventSource,
)
from seer.automation.autofix.utils import find_original_snippet
from seer.automation.models import TraceTree
from seer.dependency_injection import inject, injected


class InsightSharingPrompts:
    @staticmethod
    def format_step_one(
        *,
        latest_thought: str,
        past_insights: list[str],
        step_type: str,
    ):
        if step_type == "root_cause_analysis_processing":
            template = """\
            We have a new set of notes and thoughts:
            <NOTES>
            {latest_thought}
            </NOTES>

            We are in the process of writing a small paragraph describing the in-depth root cause of an issue in our codebase. Here it is so far:
            <PARAGRAPH_SO_FAR>
            {insights}
            </PARAGRAPH_SO_FAR>

            If there is something new and useful here to extend the root cause analysis, what is the NEXT sentence in the paragraph (write it so it flows nicely; max 15 words; write nothing but the sentence itself)? {no_insight_instruction}
            """
        elif step_type == "plan":
            template = """\
            We have a new set of notes and thoughts about the issue:
            <NOTES>
            {latest_thought}
            </NOTES>

            We are in the process of writing a small paragraph describing the code changes we need to make in our codebase. Here it is so far:
            <PARAGRAPH_SO_FAR>
            {insights}
            </PARAGRAPH_SO_FAR>

            If there is something new and useful here to extend the description on the code changes, what is the NEXT sentence in the paragraph (write it so it flows nicely; max 15 words; write nothing but the sentence itself)? {no_insight_instruction}
            """
        elif step_type == "solution_processing":
            template = """\
            We have a new set of notes and thoughts about the issue:
            <NOTES>
            {latest_thought}
            </NOTES>

            We are in the process of writing a small paragraph describing a solution to an issue in our codebase. Here it is so far:
            <PARAGRAPH_SO_FAR>
            {insights}
            </PARAGRAPH_SO_FAR>

            If there is something new and useful here to extend the solution proposal, what is the NEXT sentence in the paragraph (write it so it flows nicely; max 15 words; write nothing but the sentence itself)? {no_insight_instruction}
            """
        else:
            raise NotImplementedError(f"Insight sharing not implemented for step key: {step_type}")

        return textwrap.dedent(template).format(
            latest_thought=latest_thought,
            insights=" ".join(past_insights) if past_insights else "[paragraph is empty]",
            no_insight_instruction="If there is nothing SUPER IMPORTANT AND INSIGHTFUL to add, just return <NO_INSIGHT/>. If there is no new conclusion, but just a plan of what to search for, return <NO_INSIGHT/>. If this repeats something already in the paragraph, return <NO_INSIGHT/>.",
        )

    @staticmethod
    def format_step_two(*, insight: str, latest_thought: str, past_insights: list[str]):
        return textwrap.dedent(
            """\
            We have already proven these past insights:
            {past_insights}

            We had this thought:
            {latest_thought}

            And we concluded this:
            {insight}

            Now write the evidence. One sentence of evidence backing the conclusion. Then the snippets of evidence, extremely brief and concise: add any code, stacktraces, variable values, logs, or other evidence in Markdown code blocks using ```triple backticks``` when relevant to the conclusion and explain how they support the conclusion. This should all be in the "evidence" field.
            Afterwards, mark which sources you DIRECTLY used, specifically for this thought/conclusion. All context is helpful in general, so only include the sources very specific to THIS insight, assuming all the past insights are already proven. e.g. a stacktrace may generally be useful, but if it's not specific to this insight, don't include it. This should all be in the "sources" field."""
        ).format(
            insight=insight,
            latest_thought=latest_thought,
            past_insights="\n".join(past_insights),
        )


def process_sources(sources: list, context: AutofixContext, trace_tree: TraceTree | None):
    """
    Process the list of sources and consolidate them into the InsightSources structure.

    Args:
        sources: List of source objects from the LLM response
        context: The AutofixContext object
        trace_tree: The trace tree from the context state

    Returns:
        InsightSources object with all relevant sources populated
    """
    final_sources = InsightSources(
        stacktrace_used=False,
        breadcrumbs_used=False,
        http_request_used=False,
        trace_event_ids_used=[],
        connected_error_ids_used=[],
        diff_urls=[],
        profile_ids_used=[],
        code_used_urls=[],
        thoughts="",
        event_trace_id=trace_tree.trace_id if trace_tree else None,
    )

    for source in sources:
        if isinstance(source, StacktraceSource) and source.stacktrace_used:
            final_sources.stacktrace_used = True
        elif isinstance(source, BreadcrumbsSource) and source.breadcrumbs_used:
            final_sources.breadcrumbs_used = True
        elif isinstance(source, HttpRequestSource) and source.http_request_used:
            final_sources.http_request_used = True
        elif isinstance(source, TraceEventSource) and trace_tree:
            short_event_id = source.trace_event_id
            full_event = trace_tree.get_event_by_id(short_event_id)
            if full_event and full_event.event_id:
                final_sources.trace_event_ids_used.append(full_event.event_id)
        elif isinstance(source, ConnectedErrorSource) and trace_tree:
            short_event_id = source.connected_error_id
            full_event = trace_tree.get_event_by_id(short_event_id)
            if full_event and full_event.event_id:
                final_sources.connected_error_ids_used.append(full_event.event_id)
        elif isinstance(source, ProfileSource) and trace_tree:
            short_event_id = source.trace_event_id
            full_event = trace_tree.get_event_by_id(short_event_id)
            if full_event and full_event.profile_id:
                final_sources.profile_ids_used.append(
                    f"{full_event.project_slug}/{full_event.profile_id}"
                )
        elif isinstance(source, CodeSource):
            repo_name = context.autocorrect_repo_name(source.repo_name)
            if not repo_name:
                continue
            repo_client = context.get_repo_client(repo_name)
            file_name = source.file_name
            corrected_file_name = context.autocorrect_file_path(path=file_name, repo_name=repo_name)
            if not corrected_file_name:
                continue
            file_name = corrected_file_name
            snippet_to_find = source.code_snippet

            file_content = context.get_file_contents(file_name, repo_name)
            result = (
                find_original_snippet(
                    snippet_to_find, file_content, initial_line_threshold=0.8, threshold=0.8
                )
                if file_content
                else None
            )
            if result:
                start_line = result[1]
                end_line = result[2]
                code_url = repo_client.get_file_url(file_name, start_line, end_line)
                if code_url not in final_sources.code_used_urls:
                    final_sources.code_used_urls.append(code_url)
            else:
                code_url = repo_client.get_file_url(file_name)
                if code_url not in final_sources.code_used_urls:
                    final_sources.code_used_urls.append(code_url)
        elif isinstance(source, DiffSource):
            repo_name = context.autocorrect_repo_name(source.repo_name)
            if not repo_name:
                continue
            repo_client = context.get_repo_client(repo_name)
            diff_url = repo_client.get_commit_url(source.commit_sha)
            if diff_url not in final_sources.diff_urls:
                final_sources.diff_urls.append(diff_url)

    return final_sources


@observe(name="Sharing Insights")
@sentry_sdk.trace
@inject
def create_insight_output(
    *,
    latest_thought: str,
    past_insights: list[str],
    step_type: str,
    memory: list[Message],
    generated_at_memory_index: int = -1,
    context: AutofixContext,
    llm_client: LlmClient = injected,
) -> tuple[InsightSharingOutput | None, Usage]:
    usage = Usage()

    if not latest_thought or len(latest_thought.strip()) < 10:
        return None, usage

    prompt_one = InsightSharingPrompts.format_step_one(
        step_type=step_type,
        latest_thought=latest_thought,
        past_insights=past_insights,
    )

    completion = llm_client.generate_text(
        model=GeminiProvider.model("gemini-2.0-flash-001"),
        prompt=prompt_one,
        temperature=0.0,
    )

    insight = completion.message.content
    if (
        not insight
        or "<NO_INSIGHT/>" in insight
        or "</NO_INSIGHT/>" in insight
        or "</NO_INSIGHT>" in insight
    ):
        return None, usage

    prompt_two = InsightSharingPrompts.format_step_two(
        insight=insight,
        latest_thought=latest_thought,
        past_insights=past_insights,
    )

    memory = [msg for msg in memory if msg.role != "system"]

    completion = llm_client.generate_structured(
        messages=memory,
        prompt=prompt_two,
        model=GeminiProvider.model("gemini-2.0-flash-001"),
        temperature=0.0,
        max_tokens=4096,
        response_format=JustificationOutput,
    )
    usage += completion.metadata.usage
    justification = completion.parsed

    answer = justification and justification.evidence or ""
    markdown_snippets = justification and justification.markdown_snippets or ""
    sources = justification and justification.sources or []

    trace_tree = context.state.get().request.trace_tree
    final_sources = process_sources(sources=sources, context=context, trace_tree=trace_tree)
    final_sources.thoughts = latest_thought

    response = InsightSharingOutput(
        insight=insight,
        justification=answer,
        markdown_snippets=markdown_snippets,
        sources=final_sources,
        generated_at_memory_index=generated_at_memory_index,
    )
    return response, usage
