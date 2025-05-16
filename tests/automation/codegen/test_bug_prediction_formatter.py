import json
import os
from unittest.mock import MagicMock

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.codegen.bug_prediction_component import BugPredictorFormatterComponent
from seer.automation.codegen.models import BugPredictorFormatterInput, BugPredictorFormatterOutput


class TestBugPredictionFormatterComponent:
    def setup_method(self):
        self.component = BugPredictorFormatterComponent(context=MagicMock())
        self.mock_llm_client = MagicMock()
        self.use_real_llm = os.environ.get("USE_REAL_LLM", "0") == "1"
        self.severity_variance = 0.3

    # def test_invoke_with_valid_prediction(self):
    #     if not self.use_real_llm:
    #         self.mock_llm_client.generate_structured.return_value = MagicMock(
    #             parsed={
    #                 "conclusion": "There is a potential bug in the error handling logic",
    #                 "description": "The code doesn't properly handle null values",
    #                 "affected_files": ["src/main.py"],
    #                 "suggested_fix": "The code should check for null values before processing",
    #                 "severity": "medium",
    #                 "confidence": "high",
    #                 "is_valid": True,
    #                 "code_locations": ["src/main.py:42"],
    #             }
    #         )
    #         request = BugPredictorFormatterInput(followups=["Test prediction"])
    #         result = self.component.invoke(request, llm_client=self.mock_llm_client)

    #         assert isinstance(result, BugPredictorFormatterOutput)
    #         assert len(result.bug_predictions) == 1
    #         prediction = result.bug_predictions[0]
    #         assert prediction.is_fatal_bug is True
    #     else:
    #         request = BugPredictorFormatterInput(followups=["Test prediction"])
    #         result = self.component.invoke(request)
    #         assert isinstance(result, BugPredictorFormatterOutput)

    # def test_invoke_with_fixture_predictions(self):
    #     if not self.use_real_llm:
    #         pass

    #     fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures", "bug_prediction")
    #     fixture_ids = [
    #         d for d in os.listdir(fixture_dir) if os.path.isdir(os.path.join(fixture_dir, d))
    #     ]

    #     # Create outputs directory if it doesn't exist
    #     outputs_dir = os.path.join(fixture_dir, "outputs")
    #     os.makedirs(outputs_dir, exist_ok=True)

    #     for fixture_id in fixture_ids:
    #         fixture_path = os.path.join(fixture_dir, fixture_id)
    #         fixture_files = sorted([f for f in os.listdir(fixture_path) if f.endswith(".txt")])

    #         # Dictionary to store all results for this fixture
    #         fixture_results = []

    #         for fixture_file in fixture_files:
    #             file_path = os.path.join(fixture_path, fixture_file)
    #             with open(file_path, "r") as f:
    #                 raw_prediction = f.read()

    #             expected_results = {
    #                 "1870": [
    #                     {"is_valid": False, "expected_severity": 0.7},
    #                     {"is_valid": False, "expected_severity": 0},
    #                     {"is_valid": False, "expected_severity": 0},
    #                     {"is_valid": True, "expected_severity": 0.7},
    #                 ],
    #                 "2081": [
    #                     {"is_valid": True, "expected_severity": 0.7},
    #                     {"is_valid": True, "expected_severity": 0.7},
    #                     {"is_valid": False, "expected_severity": 0},
    #                     {"is_valid": True, "expected_severity": 0.7},
    #                     {"is_valid": False, "expected_severity": 0},
    #                 ],
    #                 "2128": [
    #                     {"is_valid": True, "expected_severity": 0.7},
    #                     {"is_valid": True, "expected_severity": 0.7},
    #                 ],
    #             }

    #             if fixture_id in expected_results:
    #                 file_index = int(os.path.splitext(fixture_file)[0])
    #                 expected_list = expected_results[fixture_id]

    #                 if file_index >= len(expected_list):
    #                     continue

    #                 expected = expected_list[file_index]

    #                 request = BugPredictorFormatterInput(followups=[raw_prediction])
    #                 result = self.component.invoke(request)

    #                 assert isinstance(result, BugPredictorFormatterOutput)
    #                 assert (
    #                     len(result.bug_predictions) == 1
    #                 ), f"Fixture {fixture_id}/{fixture_file}: expected 1 prediction"

    #                 prediction = result.bug_predictions[0]
    #                 assert (
    #                     prediction.is_fatal_bug == expected["is_valid"]
    #                 ), f"Fixture {fixture_id}/{fixture_file}: is_valid mismatch"

    #                 assert isinstance(prediction.severity, float)
    #                 assert (
    #                     abs(prediction.severity - expected["expected_severity"])
    #                     <= self.severity_variance
    #                 ), f"Fixture {fixture_id}/{fixture_file}: severity {prediction.severity} not close enough to expected {expected['expected_severity']} (±{self.severity_variance})"

    #                 assert prediction.title is not None
    #                 assert prediction.short_description is not None
    #                 assert prediction.suggested_fix is not None
    #                 assert isinstance(prediction.confidence, float)
    #                 assert 0 <= prediction.confidence <= 1

    #                 # TODO - finding the file is not working (arrays are empty)
    #                 """
    #                 if prediction.is_valid:
    #                     assert (
    #                         len(prediction.affected_files) > 0
    #                     ), f"Fixture {fixture_id}/{fixture_file}: affected_files should not be empty for valid predictions"
    #                     assert (
    #                         len(prediction.code_locations) > 0
    #                     ), f"Fixture {fixture_id}/{fixture_file}: code_locations should not be empty for valid predictions"
    #                 """

    #                 fixture_results.append(prediction.__dict__)

    #         if fixture_results:
    #             output_file = os.path.join(outputs_dir, f"{fixture_id}.json")
    #             with open(output_file, "w") as f:
    #                 json.dump(fixture_results, f, indent=2, default=str)


def test_write_outputs():
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures", "bug_prediction")
    fixture_ids = [
        d for d in os.listdir(fixture_dir) if os.path.isdir(os.path.join(fixture_dir, d))
    ]

    # Process each fixture directory
    for fixture_id in fixture_ids:
        print(f"Processing fixture: {fixture_id}")

        # Get followups from the fixture directory
        followups = []
        fixture_path = os.path.join(fixture_dir, fixture_id)
        for file_name in sorted([f for f in os.listdir(fixture_path) if f.endswith(".txt")]):
            with open(os.path.join(fixture_path, file_name), "r") as f:
                followups.append(f.read())
        if followups:
            # Create the formatter component
            component = BugPredictorFormatterComponent(context=MagicMock())
            llm_client = LlmClient()

            # Create the request with all followups
            request = BugPredictorFormatterInput(followups=followups)

            # Process the request
            result = component.invoke(request, llm_client=llm_client)

            # Output the results
            print(f"Found {len(result.bug_predictions)} bug predictions for {fixture_id}")

            # Save results to output directory
            outputs_dir = os.path.join(
                os.path.dirname(__file__), "fixtures", "bug_prediction", "outputs"
            )
            os.makedirs(outputs_dir, exist_ok=True)

            # Get expected output if available
            expected_output = None
            fixture_id_int = int(fixture_id)
            if fixture_id_int in pr_number_to_expected_output:
                expected_output = pr_number_to_expected_output[fixture_id_int]

            # Prepare output with predictions and expected output
            output_data = {
                "expected": (
                    {
                        "description": expected_output.description,
                        "encoded_location": expected_output.encoded_location,
                        "repos": expected_output.repos,
                    }
                    if expected_output
                    else None
                ),
                "actual": [pred.__dict__ for pred in result.bug_predictions],
            }

            output_file = os.path.join(outputs_dir, f"{fixture_id}.json")
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)


class ExpectedOutput:
    def __init__(self, description: str, encoded_location: str, repos: list[str]):
        self.repos = repos
        self.encoded_location = encoded_location
        self.description = description


pr_number_to_expected_output = {
    1870: ExpectedOutput(
        description="IndexError in root_cause_task due to out-of-range list access. The root_cause_output.causes list may be empty. Occurs when accessing root_cause_output.causes[0].id.",
        encoded_location="src/seer/automation/autofix/steps/root_cause_step.py:117",
        repos=["getsentry/seer"],
    ),
    2175: ExpectedOutput(
        description="IndexError: list index out of range in github/PaginatedList.py and seer/automation/codebase/repo_client.py during PR creation. The error occurs in commit_changes_task. The pulls list, returned by repo.create_pull, may be empty, causing the IndexError when accessing pulls[0]. The GitHub API returns a totalCount greater than 0, indicating that a PR exists, but the actual pulls list is empty due to the malformed head parameter.",
        encoded_location="src/seer/automation/codebase/repo_client.py:696",
        repos=["getsentry/seer"],
    ),
    2081: ExpectedOutput(
        description="The resolve_comment_thread function retrieves the AutofixContinuation state and attempts to access the active_comment_thread for a specific step. The active_comment_thread for the given step_index is None, but the code attempts to access its id attribute without a null check, resulting in an AttributeError.",
        encoded_location="src/seer/automation/autofix/tasks.py:881",
        repos=["getsentry/seer"],
    ),
    2128: ExpectedOutput(
        description="The _fetch_issues_for_pr_file function makes an RPC call to Sentry's get_issues_related_to_file_patches endpoint. The Sentry backend's get_issues_related_to_file_patches endpoint fails to find any projects associated with the given file. Because no projects were found, the Sentry backend omits the file from the response dictionary, deviating from the expected API contract. The _fetch_issues_for_pr_file function receives the RPC response and encounters an AssertionError. The assertion assert list(pr_filename_to_issues.keys()) == [pr_file.filename] fails because the key pr_file.filename is not present in the pr_filename_to_issues dictionary.",
        encoded_location="src/seer/automation/codegen/relevant_warnings_component.py:141",
        repos=["getsentry/seer"],
    ),
    2198: ExpectedOutput(
        description="The process_event_paths method is called to annotate and correct file paths within the EventDetails object, but its return value is incorrectly assigned. error_event_details = self.context.process_event_paths(error_event_details). process_event_paths modifies the error_event_details object in-place but implicitly returns None, which is then assigned back to error_event_details. The code attempts to call format_event_without_breadcrumbs on the now-None error_event_details object, resulting in an AttributeError.",
        encoded_location="src/seer/automation/autofix/tools.py:488",
        repos=["getsentry/seer"],
    ),
    2344: ExpectedOutput(
        description="ValueError: unexpected '{' in field name during error event parsing. The error is due to incorrect string formatting within the format_event_without_breadcrumbs function, specifically with the use of curly braces. The code attempts to use both f-string interpolation and `.format()` on the same string.",
        encoded_location="src/seer/automation/models.py:449",
        repos=["getsentry/seer"],
    ),
    2393: ExpectedOutput(
        description="The raw_lines_to_lines function encounters a line starting with '\\', indicating a 'No newline at end of file' marker, and raises a ValueError because it doesn't handle this line type. The raw_lines_to_lines function needs to handle lines starting with a backslash.",
        encoded_location="src/seer/automation/models.py:1002",
        repos=["getsentry/seer"],
    ),
    2327: ExpectedOutput(
        description="The tmp_dir variable, used for repo download, can unexpectedly be None, causing the TypeError when attempting to update it. The issue was that: BaseTools.cleanup() incorrectly set tmp_dir to None, causing a TypeError when the background thread tried to update it.",
        encoded_location="src/seer/automation/autofix/tools.py:85",
        repos=["getsentry/seer"],
    ),
    2306: ExpectedOutput(
        description="A TypeError is raised when attempting to dump the request object after failing Pydantic validation. The error occurs because the Flask request object is not JSON serializable.",
        encoded_location="src/seer/json_api.py:137",
        repos=["getsentry/seer"],
    ),
    2245: ExpectedOutput(
        description="In _ensure_repos_downloaded, the ThreadPoolExecutor here needs initializer=copy_modules_initializer() to be passed. Without it, there will be a FactoryNotFound error: Cannot resolve '<class 'seer.configuration.AppConfig'>', no module injector is currently active.",
        encoded_location="src/seer/automation/autofix/tools.py:457",
        repos=["getsentry/seer"],
    ),
    2508: ExpectedOutput(
        description="The Dockerfile doesn't install the ripgrep package. This is necessary for the new run_ripgrep_in_repo tool. The ripgrep package is installed in the Lightweight.Dockerfile, but this is only used for local development. The production environment uses Dockerfile.",
        encoded_location="src/seer/automation/autofix/tools/ripgrep_search.py:15",
        repos=["getsentry/seer"],
    ),
}

prs_with_no_expected_output = [2329, 2408, 2504, 2506]
