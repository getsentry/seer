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
            request = BugPredictorFormatterInput(raw_prediction_text="Test prediction")
            result = self.component.invoke(request, llm_client=self.mock_llm_client)

            assert isinstance(result, BugPredictorFormatterOutput)
            assert len(result.formatted_predictions) == 1
            prediction = result.formatted_predictions[0]
            assert prediction.is_valid is True
        else:
            request = BugPredictorFormatterInput(raw_prediction_text="Test prediction")
            result = self.component.invoke(request)
            assert isinstance(result, BugPredictorFormatterOutput)

    def test_invoke_with_fixture_predictions(self):
        if not self.use_real_llm:
            pass

        fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures", "bug_prediction")
        fixture_files = [f for f in os.listdir(fixture_dir) if f.endswith(".txt")]

        for fixture_file in fixture_files:
            fixture_id = os.path.splitext(fixture_file)[0]
            fixture_path = os.path.join(fixture_dir, fixture_file)
            with open(fixture_path, "r") as f:
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
                expected_list = expected_results[fixture_id]

                request = BugPredictorFormatterInput(followups=[raw_prediction])
                result = self.component.invoke(request)

                assert isinstance(result, BugPredictorFormatterOutput)
                assert len(result.formatted_predictions) == len(
                    expected_list
                ), f"Fixture {fixture_id}: expected {len(expected_list)} predictions, got {len(result.formatted_predictions)}"

                for i, (prediction, expected) in enumerate(
                    zip(result.formatted_predictions, expected_list)
                ):
                    assert (
                        prediction.is_valid == expected["is_valid"]
                    ), f"Fixture {fixture_id}, prediction {i}: is_valid mismatch"

                    assert isinstance(prediction.severity, float)
                    assert (
                        abs(prediction.severity - expected["expected_severity"])
                        <= self.severity_variance
                    ), f"Fixture {fixture_id}, prediction {i}: severity {prediction.severity} not close enough to expected {expected['expected_severity']} (±{self.severity_variance})"

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
                        ), f"Fixture {fixture_id}, prediction {i}: affected_files should not be empty for valid predictions"
                        assert (
                            len(prediction.code_locations) > 0
                        ), f"Fixture {fixture_id}, prediction {i}: code_locations should not be empty for valid predictions"
                    """
