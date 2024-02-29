import unittest

from seer.automation.autofix.models import Stacktrace, StacktraceFrame


class TestStacktraceHelpers(unittest.TestCase):
    def test_stacktrace_to_str(self):
        frames = [
            StacktraceFrame(
                function="main",
                filename="app.py",
                abs_path="/path/to/app.py",
                line_no=10,
                col_no=20,
                context=[(10, "    main()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=True,
            ),
            StacktraceFrame(
                function="helper",
                filename="utils.py",
                abs_path="/path/to/utils.py",
                line_no=15,
                col_no=None,
                context=[(15, "    helper()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=False,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " helper in file utils.py in repo my_repo [Line 15] (Not in app)\n    helper()  <-- SUSPECT LINE\n------\n main in file app.py in repo my_repo [Line 10:20] (In app)\n    main()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(), expected_str)

    def test_stacktrace_to_str_cutoff(self):
        frames = [
            StacktraceFrame(
                function="main",
                filename="app.py",
                abs_path="/path/to/app.py",
                line_no=10,
                col_no=20,
                context=[(10, "    main()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=True,
            ),
            StacktraceFrame(
                function="helper",
                filename="utils.py",
                abs_path="/path/to/utils.py",
                line_no=15,
                col_no=None,
                context=[(15, "    helper()")],
                repo_name="my_repo",
                repo_id=1,
                in_app=False,
            ),
        ]
        stacktrace = Stacktrace(frames=frames)
        expected_str = " helper in file utils.py in repo my_repo [Line 15] (Not in app)\n    helper()  <-- SUSPECT LINE\n------\n"
        self.assertEqual(stacktrace.to_str(max_frames=1), expected_str)

    def test_stacktrace_frame_str(self):
        frame = StacktraceFrame(
            function="main",
            filename="app.py",
            abs_path="/path/to/app.py",
            line_no=10,
            col_no=20,
            context=[(10, "    main()")],
            repo_name="my_repo",
            repo_id=1,
            in_app=True,
        )
        expected_str = " main in file app.py in repo my_repo [Line 10:20] (In app)\n    main()  <-- SUSPECT LINE\n"
        stack_str = ""
        col_no_str = f":{frame.col_no}" if frame.col_no is not None else ""
        repo_str = f" in repo {frame.repo_name}" if frame.repo_name else ""
        stack_str += f" {frame.function} in file {frame.filename}{repo_str} [Line {frame.line_no}{col_no_str}] ({'In app' if frame.in_app else 'Not in app'})\n"
        for ctx in frame.context:
            is_suspect_line = ctx[0] == frame.line_no
            stack_str += f"{ctx[1]}{'  <-- SUSPECT LINE' if is_suspect_line else ''}\n"
        self.assertEqual(stack_str, expected_str)
