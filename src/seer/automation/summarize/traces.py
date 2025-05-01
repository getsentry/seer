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
        anomalous_spans=trace_summary.anomalous_spans,
        key_observations=trace_summary.key_observations,
        performance_characteristics=trace_summary.performance_characteristics,
        suggested_investigations=trace_summary.suggested_investigations,
    )


def _get_prompt(trace_str: str, only_transactions: bool) -> str:

    prompt = ""
    if only_transactions:
        prompt = textwrap.dedent(
            f"""
            You are a principal performance engineer who is excellent at explaining concepts simply to engineers of all levels. Our traces have a lot of dense information that is hard to understand quickly. Please summarize the trace below so our engineers can immediately understand what's going on.

            This trace tree is made up only spans (<span>) that are transactions (<txn>). Each transaction represents a single instance of a service being called, and the trace is made up of all the transactions in a tree like structure.
            Here is the trace:

            <trace>
            {trace_str}
            </trace>

            Your #1 goal is to help our engineers immediately understand what's going on in the trace.

            Write a concise summary that includes the flow of events in the trace, key transactions, and the performance characteristics of the trace:

            1. **Overall Summary**: A 1 sentence high-level summary of what this trace represents (e.g., "This trace represents a user login flow that authenticates with multiple services").

            2. **Flow Overview**: Summarize the entire trace flow in 3-5 bullet points, focusing on the main sequence of operations from start to finish.

            3. **Performance Characteristics**:
            - Identify critical path operations that contribute most to the total duration
            - Highlight relationships between spans where one span is blocking another
            - Point out potential parallelization opportunities
            - Focus only on optimizations that would make a significant impact (e.g., >5% of total trace time)

            4. **Technical Insights**:
            - Distinguish between blocking and non-blocking operations
            - Identify any unusual patterns or anomalies (only if they impact performance significantly)
            - Consider whether operations are sequential by necessity or could be reorganized
            """
        )
    else:
        prompt = textwrap.dedent(
            f"""
            You are a principal performance engineer who is excellent at explaining concepts simply to engineers of all levels. Our traces have a lot of dense information that is hard to understand quickly. Please provide key insights about the trace below so our engineers can immediately understand what's going on.
            Please not that the engineers have access to the same information as you do, so please do not state any obvious high level information about the trace and its spans. The trace is made up of nested spans which have more granular information and represents the overall hierarchy of the trace.

            Here is the trace:

            <trace>
            {trace_str}
            </trace>

            Your #1 goal is to help our engineers immediately understand what's going on in the trace.

            Provide conscise insights about the trace and its spans in the following sections:

            1. **Summary**:
            - Provide a 1-3 sentence high-level summary of what is going on in the trace in the spans that make it up. Make sure to only include the most important information such that a developer can understand the trace at a glance.
            - DO NOT EXCEED MORE THAN 3 SENTENCES. THIS IS NOT OPTIONAL.

            2. **Anomalous Spans**:
            - Identify up to 3 spans that are anomalous or stand out from the rest of the trace.
            - These can be spans that are slow, have an odd grouping of spans above or below it, missing instrumentation, or anything else that very clearly sticks out. These should be things that a developer can immediately act on.
            - You will respond with a list of objects with the following fields:
              - "Explanation": Provide a brief 1 sentence explanation for why these spans are anomalous. The sentence must be extremely concise with no more than 15 words.
              - "Span Id": span id of the anomalous span
              - "Span Op": op of the anomalous span
            - If there are no anomalous spans, you must return an empty list.

            3. **Key Observations**: Key observations about the trace.
            - Summarize the key observations about the trace in a bulleted list with 3 bullets MAX. DO NOT include any information about the trace structure, just the key observations.
            - When making observations, comment on groupings of spans -- not just individual spans.
            - Are there any seemingly uninteresting transactions, spans, or other events that provide context about the trace? Some issues in a span may be due to adjacent spans.
            - If you make a statement, be extremely specific about why you made that statement. DO NOT STATE THE OBVIOUS.

            3. **Performance Characteristics**: The performance characteristics of the trace.
            - Summarize the performance characteristics of the trace in a bulleted list.
            - Are there any slow transactions, slow spans, or any other performance characteristics that stand out?
            - Explain why the spans may be slow.
            - Do not comment on the the overall trace duration as this is already provided to the user.
            - Do not just say "the trace is slow" or "the trace is fast". Explain why extremely concisely.

            4. **Suggested Investigations**: Suggested investigations to improve the performance of the trace. BE SPECIFIC WHEN SUGGESTING THESE INVESTIGATIONS.
            - Suggest specific span event ids to investigate. Also include the span's op and description. THIS IS NOT OPTIONAL.
            - Are there any bottlenecks in the trace?
            - Are there any areas for improvement in the trace?
            - You will respond with a list of objects with the following fields:
              - "Explanation": Provide a brief 1 sentence explanation for why you should investigate this span. The sentence must be extremely concise with no more than 15 words.
              - "Span Id": span id of the anomalous span
              - "Span Op": op of the anomalous span
            - DO NOT GIVE ANY GENERIC INVESTIGATIONS. YOU MUST BE SPECIFIC.

            Please use markdown formatting for the trace to make it easier to read. You should bold important insights or key words and use other markdown formatting to make these insights easy to skim.

            IMPORTANT: Do not repeat the same information in the summary, key observations, or performance characteristics. Each section should be unique and contain distinct information.
            """
        )

    return prompt
