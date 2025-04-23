import textwrap
from datetime import datetime

from seer.automation.agent.agent import AgentConfig, LlmAgent, RunConfig
from seer.automation.agent.client import GeminiProvider
from seer.automation.autofix.autofix_context import AutofixContext
from seer.automation.autofix.tools.tools import SuspectCommitTools  # Import the new tools
from seer.automation.models import EventDetails


def search_commit_history(query: str, repo_name: str, context: AutofixContext) -> str:
    """
    Uses an LLM agent to search the commit history of a repository to find one that is suspicious.
    """
    state = context.state.get()
    issue_first_seen = state.request.issue.first_seen
    event_seen = None
    if state.request.issue.events:
        event_details = EventDetails.from_event(state.request.issue.events[0])
        event_seen = event_details.datetime

    def _parse_iso_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str)
        except ValueError:
            return None

    dt_issue_first_seen = _parse_iso_date(issue_first_seen)
    dt_event_seen = _parse_iso_date(event_seen)
    valid_dates = [d for d in [dt_issue_first_seen, dt_event_seen] if d is not None]
    earliest_dt: datetime | None = min(valid_dates) if valid_dates else None

    tools = SuspectCommitTools(context=context, earliest_dt=earliest_dt)

    agent_config = AgentConfig(interactive=False)
    agent = LlmAgent(
        config=agent_config,
        tools=tools.get_tools(can_access_repos=True),
        name="SuspectCommitAgent",
    )

    whole_repo_initial_history = tools.view_commit_history_for_file(
        file_path=None,
        repo_name=repo_name,
        skip_first_n_commits=0,
    )

    system_prompt = textwrap.dedent(
        """\
        You are an exceptional assistant specialized in finding the most relevant commits in git history.
        Your goal is to identify the most relevant commit(s), if any, to the user's query.

        Use the available tools:
        - `view_commit_history_for_file`: To view commit history for relevant files. You can specify `skip_first_n_commits` to paginate through older commits if needed.
        - `view_diff`: To examine the changes introduced by a specific commit SHA.

        You have the following repo to work with: {repo_str}

        Steps:
        1. Use `view_commit_history_for_file` to get the recent commit history for relevant files. If no commit is immediately relevant, try increasing `skip_first_n_commits` to see older commits.
        2. If a commit looks relevant, use `view_diff` with its SHA to see the exact changes.
        3. Continue exploring history (using `skip_first_n_commits`) and diffs until you find a commit that seems highly likely to be the cause, or you have exhausted the relevant history.

        In your final response, list the exact commit SHAs, descriptions, and diffs that you suspect, and explain why you suspect each one. If no relevant commit is found, explain why.

        To start you off, here are the most recent commits for the entire repository (path "/"):
        {whole_repo_initial_history}
    """
    ).format(
        repo_str=repo_name,
        whole_repo_initial_history=whole_repo_initial_history,
    )

    run_config = RunConfig(
        system_prompt=system_prompt,
        prompt=f"Look for a commit about: {query}",
        model=GeminiProvider(model_name="gemini-2.5-flash-preview-04-17"),
        temperature=0.0,
        max_iterations=32,
    )

    result = agent.run(run_config)
    return result
