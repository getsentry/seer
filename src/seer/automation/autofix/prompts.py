import textwrap

from seer.automation.models import ExceptionDetails, ThreadDetails


def format_instruction(instruction: str | None):
    return textwrap.dedent(  # The leading newline is intentional
        f"""\

        Instructions have been provided. Please ensure that they are reflected in your work:
        <instruction>
        {instruction}
        </instruction>"""
        if instruction
        else ""
    )
