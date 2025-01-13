import textwrap

from seer.automation.models import Profile
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


def format_repo_names(repo_names: list[str]):
    return textwrap.dedent(
        """\
        You have the following repositories to work with:
        {names_list_str}
        """
    ).format(names_list_str="\n".join([f"- {repo_name}" for repo_name in repo_names]))


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
    return f"Here's a partial map of the code{('at the time of the issue' if code_map.profile_matches_issue else 'that may help')}: \n{code_map.format_profile()}"
