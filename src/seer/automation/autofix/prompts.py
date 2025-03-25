import textwrap

from seer.automation.models import Profile, RepoDefinition, TraceTree
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
    return (
        f"\n<repo_instructions>\n{repo.instructions.rstrip()}\n</repo_instructions>"
        if hasattr(repo, "instructions") and repo.instructions
        else ""
    )


def format_repo_prompt(
    readable_repos: list[RepoDefinition], unreadable_repos: list[RepoDefinition] = []
):
    if not readable_repos:
        return "You can't access repositories or look up code, but you're still amazing at solving the problem regardless. Do so without looking up code."

    readable_str = textwrap.dedent(
        """\
        You have the following repositories to work with:
        {names_list_str}"""
    ).format(
        names_list_str="\n".join(
            [
                f"<repo>\n{repo.full_name}{format_repo_instructions(repo)}\n</repo>"
                for repo in readable_repos
            ]
        )
    )

    if unreadable_repos:
        unreadable_str = textwrap.dedent(
            """\
            The follow repositories may show up, but you don't have access to them:
            {names_list_str}"""
        ).format(names_list_str="\n".join([f"- {repo.full_name}" for repo in unreadable_repos]))

        return readable_str + "\n\n" + unreadable_str

    return readable_str


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
