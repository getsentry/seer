import textwrap


def format_instruction(instruction: str | None):
    return textwrap.dedent(  # The leading newline is intentional
        f"""\
        \
        Instructions have been provided. Please ensure that they are reflected in your work:
        \"{instruction}\"
        \
        """
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
