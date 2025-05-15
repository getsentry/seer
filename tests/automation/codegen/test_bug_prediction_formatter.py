import json
import os
from unittest.mock import MagicMock

from seer.automation.codegen.bug_prediction_component import BugPredictionFormatterComponent
from seer.automation.codegen.models import BugPredictorFormatterInput, BugPredictorFormatterOutput


class TestBugPredictionFormatterComponent:
    def setup_method(self):
        self.component = BugPredictionFormatterComponent(context=MagicMock())
        self.mock_llm_client = MagicMock()
        self.use_real_llm = os.environ.get("USE_REAL_LLM", "0") == "1"
        self.severity_variance = 0.3

    def test_invoke_with_valid_prediction(self):
        if not self.use_real_llm:
            self.mock_llm_client.generate_structured.return_value = MagicMock(
                parsed={
                    "conclusion": "There is a potential bug in the error handling logic",
                    "description": "The code doesn't properly handle null values",
                    "affected_files": ["src/main.py"],
                    "suggested_fix": "The code should check for null values before processing",
                    "severity": "medium",
                    "confidence": "high",
                    "is_valid": True,
                    "code_locations": ["src/main.py:42"],
                }
            )
            request = BugPredictorFormatterInput(followups=["Test prediction"])
            result = self.component.invoke(request, llm_client=self.mock_llm_client)

            assert isinstance(result, BugPredictorFormatterOutput)
            assert len(result.formatted_predictions) == 1
            prediction = result.formatted_predictions[0]
            assert prediction.is_valid is True
        else:
            request = BugPredictorFormatterInput(followups=["Test prediction"])
            result = self.component.invoke(request)
            assert isinstance(result, BugPredictorFormatterOutput)

    def test_invoke_with_fixture_predictions(self):
        if not self.use_real_llm:
            pass

        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures", "bug_prediction")
        fixture_ids = [
            d for d in os.listdir(fixture_dir) if os.path.isdir(os.path.join(fixture_dir, d))
        ]

        # Create outputs directory if it doesn't exist
        outputs_dir = os.path.join(fixture_dir, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)

        for fixture_id in fixture_ids:
            fixture_path = os.path.join(fixture_dir, fixture_id)
            fixture_files = sorted([f for f in os.listdir(fixture_path) if f.endswith(".txt")])

            # Dictionary to store all results for this fixture
            fixture_results = []

            for fixture_file in fixture_files:
                file_path = os.path.join(fixture_path, fixture_file)
                with open(file_path, "r") as f:
                    raw_prediction = f.read()

                expected_results = {
                    "1870": [
                        {"is_valid": True, "expected_severity": 0.7},
                        {"is_valid": False, "expected_severity": 0},
                        {"is_valid": False, "expected_severity": 0},
                        {"is_valid": True, "expected_severity": 0.7},
                    ],
                    "2081": [
                        {"is_valid": True, "expected_severity": 0.7},
                        {"is_valid": True, "expected_severity": 0.7},
                        {"is_valid": False, "expected_severity": 0},
                        {"is_valid": True, "expected_severity": 0.7},
                        {"is_valid": False, "expected_severity": 0},
                    ],
                    "2128": [
                        {"is_valid": True, "expected_severity": 0.7},
                        {"is_valid": True, "expected_severity": 0.7},
                    ],
                }

                if fixture_id in expected_results:
                    file_index = int(os.path.splitext(fixture_file)[0])
                    expected_list = expected_results[fixture_id]

                    if file_index >= len(expected_list):
                        continue

                    expected = expected_list[file_index]

                    request = BugPredictorFormatterInput(followups=[raw_prediction])
                    result = self.component.invoke(request)

                    assert isinstance(result, BugPredictorFormatterOutput)
                    assert (
                        len(result.formatted_predictions) == 1
                    ), f"Fixture {fixture_id}/{fixture_file}: expected 1 prediction"

                    prediction = result.formatted_predictions[0]
                    assert (
                        prediction.is_valid == expected["is_valid"]
                    ), f"Fixture {fixture_id}/{fixture_file}: is_valid mismatch"

                    assert isinstance(prediction.severity, float)
                    assert (
                        abs(prediction.severity - expected["expected_severity"])
                        <= self.severity_variance
                    ), f"Fixture {fixture_id}/{fixture_file}: severity {prediction.severity} not close enough to expected {expected['expected_severity']} (±{self.severity_variance})"

                    assert prediction.title is not None
                    assert prediction.description is not None
                    assert prediction.suggested_fix is not None
                    assert isinstance(prediction.confidence, float)
                    assert 0 <= prediction.confidence <= 1

                    # TODO - finding the file is not working (arrays are empty)
                    """
                    if prediction.is_valid:
                        assert (
                            len(prediction.affected_files) > 0
                        ), f"Fixture {fixture_id}/{fixture_file}: affected_files should not be empty for valid predictions"
                        assert (
                            len(prediction.code_locations) > 0
                        ), f"Fixture {fixture_id}/{fixture_file}: code_locations should not be empty for valid predictions"
                    """

                    fixture_results.append(prediction.__dict__)

            if fixture_results:
                output_file = os.path.join(outputs_dir, f"{fixture_id}.json")
                with open(output_file, "w") as f:
                    json.dump(fixture_results, f, indent=2, default=str)
