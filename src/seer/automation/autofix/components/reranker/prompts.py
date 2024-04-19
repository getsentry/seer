import textwrap


class RerankerPrompts:
    @staticmethod
    def format_system_msg():
        return textwrap.dedent(
            """\
            You are exceptional at gathering information.
            - Given a task, you must return the list of code snippet ids that are relevant to the task.

            Code snippets will be given to you in the format:
            <chunk id="3232" path="file/path.ext" repo_name="repo/name">
            def foo():
                return "bar"
            </chunk>

            You must output the relevant ids to the task as a list of code snippet ids in JSON array format inside a <code_snippet_ids> tag:
            <code_snippet_ids>
            ["3232", "3233", "3234"]
            </code_snippet_ids>"""
        )

    @staticmethod
    def format_default_msg(task_instruction: str, code_dump: str):
        return textwrap.dedent(
            """\
            <code_snippets>
            {code_dump}
            </code_snippets>

            Given the task:
            <task>
            {task_instruction}
            </task>
            you must return all relevant code snippet ids in order of relevance.

            Think out loud step-by-step in a <thoughts> block then output all relevant code snippet IDs in the JSON inside a <code_snippet_ids> block."""
        ).format(
            code_dump=code_dump,
            task_instruction=task_instruction,
        )
