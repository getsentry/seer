import textwrap

from seer.automation.models import Logs, Profile, RepoDefinition, TraceTree
from seer.automation.summarize.issue import IssueSummary


def format_instruction(instruction: str | None):
    return (
        "\n"
        + textwrap.dedent(  # The newlines are intentional
            f"""\
            Instructions have been provided. Please ensure that they are reflected in your work:
            \"{instruction}\"
            """
        )
        + "\n"
        if instruction
        else ""
    )


def format_repo_instructions(repo: RepoDefinition):
    instructions = repo.instructions.rstrip().replace("\n", " ") if repo.instructions else None
    return f": {instructions}\n" if instructions else ""


def format_repo_prompt(
    readable_repos: list[RepoDefinition],
    unreadable_repos: list[RepoDefinition] = [],
    is_using_claude_tools: bool = False,
):
    if not readable_repos:
        return "You can't access repositories or look up code, but you're still amazing at solving the problem regardless. Do so without looking up code."

    multi_repo_suffix = ""
    if len(readable_repos) > 1 and is_using_claude_tools:
        multi_repo_suffix = "\n\nYou can access multiple repositories, when passing in a path to the `str_replace_editor` tool, you will need to use the format `repo_name:path` to access a specific file or directory in a specific repository, such as `owner/repo:src/foo/bar.py`."

    readable_str = textwrap.dedent(
        """\
        You have the following repositories to work with:
        {names_list_str}
        You may see references to other code that you cannot access, such as other repositories, third-party libraries, or frames marked "Not In App". You may consider them in your analysis, but do not attempt to search for their source code."""
    ).format(
        names_list_str="\n".join(
            [f"- {repo.full_name}{format_repo_instructions(repo)}" for repo in readable_repos]
        )
    )

    if unreadable_repos:
        unreadable_str = textwrap.dedent(
            """\
            The follow repositories may show up, but you don't have access to them:
            {names_list_str}"""
        ).format(names_list_str="\n".join([f"- {repo.full_name}" for repo in unreadable_repos]))

        return readable_str + "\n\n" + unreadable_str + multi_repo_suffix

    return readable_str + multi_repo_suffix


def format_summary(summary: IssueSummary | None) -> str:
    if not summary:
        return ""

    return textwrap.dedent(
        """\
        {whats_wrong}
        {trace}
        {possible_cause}
        """
    ).format(
        whats_wrong=summary.whats_wrong,
        trace=summary.session_related_issues,
        possible_cause=summary.possible_cause,
    )


def format_code_map(code_map: Profile | None):
    if not code_map:
        return ""
    return f"\nHere's a partial map of the code {('at the time of the issue' if code_map.profile_matches_issue else 'that may help')}: \n{code_map.format_profile()}\n"


def format_trace_tree(trace_tree: TraceTree | None):
    if not trace_tree:
        return ""
    return f"\nHere's a high-level trace to give you context on the whole system (note it may be incomplete or irrelevant to the issue): \n{trace_tree.format_trace_tree()}\n"


def format_logs(logs: Logs | None):
    if not logs:
        return ""
    return f"\nHere are some logs from the system as it was running: \n{logs.format_logs()}\n"
