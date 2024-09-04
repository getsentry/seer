import textwrap

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
        {details}
        {analysis}
        """
    ).format(
        details=summary.summary_of_the_issue_based_on_your_step_by_step_reasoning,
        analysis="\n".join(
            [f"- {step.reasoning} {step.justification}" for step in summary.reason_step_by_step]
        ),
    )
