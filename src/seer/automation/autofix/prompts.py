import textwrap

from seer.automation.autofix.models import ExceptionDetails


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


def format_exceptions(exceptions: list[ExceptionDetails]):
    return "\n".join(
        textwrap.dedent(
            """\
                <exception_{i}>
                <exception_type>
                {exception_type}
                </exception_type>
                <exception_message>
                {exception_message}
                </exception_message>
                <stacktrace>
                {stacktrace}
                </stacktrace>
                </exception_{i}>"""
        ).format(
            i=i,
            exception_type=exception.type,
            exception_message=exception.value,
            stacktrace=exception.stacktrace.to_str() if exception.stacktrace else "",
        )
        for i, exception in enumerate(exceptions)
    )
