import textwrap
from venv import logger

from google.genai.errors import ClientError
from langfuse.decorators import observe
from pydantic import BaseModel

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.summarize.models import (
    SpanInsight,
    SummarizeTraceRequest,
    SummarizeTraceResponse,
)
from seer.dependency_injection import inject, injected


class TraceSummaryForLlmToGenerate(BaseModel):
    summary: str
    anomalous_spans: list[SpanInsight]
    key_observations: str
    performance_characteristics: str
    suggested_investigations: list[SpanInsight]


@observe(name="Summarize Trace")
@inject
def summarize_trace(
    request: SummarizeTraceRequest,
    llm_client: LlmClient = injected,
) -> SummarizeTraceResponse:
    """
    Summarizes a single trace in the EAP Trace Waterfall format.

    params:
        request: SummarizeTraceRequest

    returns:
        SummarizeTraceResponse
    """
    logger.info(f"Summarizing trace: {request.trace_id}")
    trace, only_transactions = request.trace, request.only_transactions
    trace_str = trace.get_and_format_trace(only_transactions)

    prompt = _get_prompt(trace_str, only_transactions)

    try:
        completion = llm_client.generate_structured(
            model=GeminiProvider.model(
                "gemini-2.5-flash-preview-04-17",
            ),
            prompt=prompt,
            response_format=TraceSummaryForLlmToGenerate,
            temperature=0.0,
            max_tokens=4096,
        )
    except ClientError as e:
        if "token count" in str(e) and "exceeds the maximum number of tokens allowed" in str(e):
            logger.warning(f"Trace too large to summarize: {e}")
            raise
        else:
            logger.error(f"ClientError when summarizing trace: {e}")
            raise
    except Exception as e:
        logger.error(f"Error summarizing trace: {e}")
        raise

    trace_summary = completion.parsed

    return SummarizeTraceResponse(
        trace_id=request.trace_id,
        summary=trace_summary.summary,
        key_observations=trace_summary.key_observations,
        performance_characteristics=trace_summary.performance_characteristics,
        suggested_investigations=trace_summary.suggested_investigations,
    )


def _get_prompt(trace_str: str, only_transactions: bool) -> str:

    prompt = ""
    if only_transactions:
        prompt = textwrap.dedent(
            f"""
            You are a principal performance engineer who is excellent at explaining concepts simply to engineers of all levels. Our traces have a lot of dense information that is hard to understand quickly. Please provide key insights about the trace below so our engineers can immediately understand what's going on.
            Please note that the engineers have access to the same information as you do, so please do not state any obvious high level information about the trace and its spans.

            Here are some key concepts:
            - Trace:
              - A trace represents a single transaction or request through your system. This includes things like user browser sessions, HTTP requests, DB queries, middleware, caches and more.
              - It captures a series of operations (spans) that show how different parts of your application interacted during that transaction.
            - Span
              - A span represents an individual operation within a trace. This could be a database query, HTTP request, or UI rendering task.
              - Each span has:
                - Attributes: Key-value pairs like http.method, db.query, span.description, or custom attributes like cart.value, provide additional context that can be useful for debugging and investigating patterns. These are either numbers or strings. Note: numeric span attributes can be used to calculate span metrics, shown below.
                  - Duration (span.duration): The time the operation took, used to measure performance.

            NOTE: In this trace, we are only providing you the transaction spans. These are high level spans that do not have the most granular information about the trace.

            Your #1 goal is to help our engineers immediately understand what's going on in the trace.

            Provide conscise insights about the trace and its spans in the following sections:

            1. **Summary**:
            - Provide a 1-3 sentence high-level summary of what is going on in the trace in the spans that make it up. Make sure to only include the most important information such that a developer can understand the trace at a glance.
            - DO NOT EXCEED MORE THAN 3 SENTENCES. THIS IS NOT OPTIONAL.

            2. **Key Observations**:
            - Summarize the key observations about the trace in a bulleted list with 3 bullets MAX. DO NOT include any information about the trace structure, just the key observations.
            - When making observations, comment on groupings of spans -- not just individual spans.
            - Are there any seemingly uninteresting transactions, spans, or other events that provide context about the trace? Some issues in a span may be due to adjacent spans.
            - If you make a statement, be extremely specific about why you made that statement. DO NOT STATE THE OBVIOUS.

            3. **Performance Characteristics**:
            - Summarize the performance characteristics of the trace in 1-3 bullet points.
            - Are there any slow transactions, slow spans, bottlenecks, or any other performance characteristics that stand out?
            - Explain why the spans may be slow.
            - Do not comment on the the overall trace duration as this is already provided to the user.
            - Do not just say "the trace is slow". Explain why extremely concisely.
            - Avoid commenting on fast operations since they are not actionable for the user.

            4. **Suggested Investigations**:
            - Identify up to 3 spans that are anomalous or stand out from the rest of the trace that the user should investigate. If there are none, you must return an empty list.
            - These can be spans that are slow, have an odd grouping of spans around it, missing instrumentation, bottlenecks, or anything else that very clearly sticks out. These should be things that a developer can immediately act on.
            - Do not call out spans just because the are slow or have a high duration. Be specific about why you are calling out the span and have a good reason for it.
            - You will respond with a list of objects with the following fields:
              - "Explanation": Provide a brief 1 sentence explanation for why a user should investigate this span AND a suggested action to investigate the span. The sentence must be extremely concise with no more than 20 words.
              - "Span Id": span id of the anomalous span
              - "Span Op": op of the anomalous span

            Please use markdown formatting for the trace to make it easier to read. You should bold important insights or key words and use other markdown formatting to make these insights easy to skim.

            IMPORTANT: Do not repeat the same information in the summary, key observations, or performance characteristics. Each section should be unique and contain distinct information.

            Here is the trace:

            <trace>
            {trace_str}
            </trace>
            """
        )
    else:
        prompt = textwrap.dedent(
            f"""
            You are a principal performance engineer who is excellent at explaining concepts simply to engineers of all levels. Our traces have a lot of dense information that is hard to understand quickly. Please provide key insights about the trace below so our engineers can immediately understand what's going on.
            Please note that the engineers have access to the same information as you do, so please do not state any obvious high level information about the trace and its spans.

            Here are some key concepts:
            - Trace:
              - A trace represents a single transaction or request through your system. This includes things like user browser sessions, HTTP requests, DB queries, middleware, caches and more.
              - It captures a series of operations (spans) that show how different parts of your application interacted during that transaction.
            - Span
              - A span represents an individual operation within a trace. This could be a database query, HTTP request, or UI rendering task.
              - Each span has:
                - Attributes: Key-value pairs like http.method, db.query, span.description, or custom attributes like cart.value, provide additional context that can be useful for debugging and investigating patterns. These are either numbers or strings. Note: numeric span attributes can be used to calculate span metrics, shown below.
                  - Duration (span.duration): The time the operation took, used to measure performance.

            Your #1 goal is to help our engineers immediately understand what's going on in the trace.

            Provide conscise insights about the trace and its spans in the following sections:

            1. **Summary**:
            - Provide a 1-3 sentence high-level summary of what is going on in the trace in the spans that make it up. Make sure to only include the most important information such that a developer can understand the trace at a glance.
            - DO NOT EXCEED MORE THAN 3 SENTENCES. THIS IS NOT OPTIONAL.

            2. **Key Observations**:
            - Summarize the key observations about the trace in a bulleted list with 3 bullets MAX. DO NOT include any information about the trace structure, just the key observations.
            - When making observations, comment on groupings of spans -- not just individual spans.
            - Are there any seemingly uninteresting transactions, spans, or other events that provide context about the trace? Some issues in a span may be due to adjacent spans.
            - If you make a statement, be extremely specific about why you made that statement. DO NOT STATE THE OBVIOUS.

            3. **Performance Characteristics**:
            - Summarize the performance characteristics of the trace in 1-3 bullet points.
            - Are there any slow transactions, slow spans, bottlenecks, or any other performance characteristics that stand out?
            - Explain why the spans may be slow.
            - Do not comment on the the overall trace duration as this is already provided to the user.
            - Do not just say "the trace is slow". Explain why extremely concisely.
            - Avoid commenting on fast operations since they are not actionable for the user.

            4. **Suggested Investigations**:
            - Identify up to 3 spans that are anomalous or stand out from the rest of the trace that the user should investigate. If there are none, you must return an empty list.
            - These can be spans that are slow, have an odd grouping of spans around it, missing instrumentation, bottlenecks, or anything else that very clearly sticks out. These should be things that a developer can immediately act on.
            - Do not call out spans just because the are slow or have a high duration. Be specific about why you are calling out the span and have a good reason for it.
            - You will respond with a list of objects with the following fields:
              - "Explanation": Provide a brief 1 sentence explanation for why a user should investigate this span AND a suggested action to investigate the span. The sentence must be extremely concise with no more than 20 words.
              - "Span Id": span id of the anomalous span
              - "Span Op": op of the anomalous span

            Please use markdown formatting for the trace to make it easier to read. You should bold important insights or key words and use other markdown formatting to make these insights easy to skim.

            IMPORTANT: Do not repeat the same information in the summary, key observations, or performance characteristics. Each section should be unique and contain distinct information.

            Here is the trace:

            <trace>
            {trace_str}
            </trace>
            """
        )

    return prompt
