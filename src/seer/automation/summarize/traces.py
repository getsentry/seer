import textwrap
from venv import logger

from google.genai.errors import ClientError
from langfuse.decorators import observe
from pydantic import BaseModel

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.summarize.models import SummarizeTraceRequest, SummarizeTraceResponse
from seer.dependency_injection import inject, injected


class TraceSummaryForLlmToGenerate(BaseModel):
    summary: str
    key_observations: str
    performance_characteristics: str
    suggested_investigations: str


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
                "gemini-2.0-flash-001",
            ),
            prompt=prompt,
            response_format=TraceSummaryForLlmToGenerate,
            temperature=0.0,
            max_tokens=1024,
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
            You are a principal performance engineer who is excellent at explaining concepts simply to engineers of all levels. Our traces have a lot of dense information that is hard to understand quickly. Please summarize the trace below so our engineers can immediately understand what's going on.
            Please not that the engineers have access to the same information as you do, so please do not state any obvious high level information about the trace and its spans. The trace includes nested transactions and spans which have more granular information and represents the overall hierarchy of the spans.
            Here is the trace:

            <trace>
            {trace_str}
            </trace>

            Your #1 goal is to help our engineers immediately understand what's going on in the trace.

            Write a concise summary that includes the flow of events in the trace, key spans and transactions, and the performance characteristics of the trace in the following 3 sections:

            1. **Summary**: A high-level summary of the trace.
            - Summarize the flow of events in the trace in a list in chronological order. ONLY INCLUDE THE MOST IMPORTANT INFORMATION.
            - Do this in as few bullet points as possible

            2. **Key Observations**: Key observations about the trace.
            - Summarize the key observations about the trace in a bulleted list with 3 bullets MAX. DO NOT include any information about the trace structure, just the key observations.
            - When making observations, comment on groupings of spans -- not just individual spans.
            - Are there any seemingly uninteresting transactions, spans, or other events that provide context about the trace? Some issues in a span may be due to adjacent spans.
            - If you make a statement, be extremely specific about why you made that statement. DO NOT STATE THE OBVIOUS.

            3. **Performance Characteristics**: The performance characteristics of the trace.
            - Summarize the performance characteristics of the trace in a bulleted list.
            - Are there any slow transactions, slow spans, or any other performance characteristics that stand out?
            - Explain why the spans may be slow.
            - Do not just say "the trace is slow" or "the trace is fast". Explain briefly why.

            4. **Suggested Investigations**: Suggested investigations to improve the performance of the trace. BE SPECIFIC WHEN SUGGESTING THESE INVESTIGATIONS.
            - Suggest specific span event ids to investigate. Also include the span's op and description. THIS IS NOT OPTIONAL.
            - Are there any bottlenecks in the trace?
            - Are there any areas for improvement in the trace?
            - DO NOT GIVE ANY GENERIC INVESTIGATIONS. YOU MUST BE SPECIFIC.

            You must limit your summary to 100 words maximum while cramming as much detail as possible, to make the summary super easy to skim. Please bold important insights and unique terms by surrounding them with double asterisks (**like this**).

            You must also include a title for the trace.

            IMPORTANT: Do not repeat the same information in the summary, key observations, or performance characteristics. Each section should be unique and contain distinct information.
            """
        )

    return prompt
