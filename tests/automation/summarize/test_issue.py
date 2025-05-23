from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from johen import generate

from seer.automation.agent.client import LlmGenerateStructuredResponse, LlmResponseMetadata
from seer.automation.agent.models import LlmProviderType, Usage
from seer.automation.autofixability import AutofixabilityModel
from seer.automation.models import IssueDetails
from seer.automation.summarize.issue import (
    IssueSummaryForLlmToGenerate,
    IssueSummaryWithScores,
    evaluate_autofixability,
    run_fixability_score,
    run_summarize_issue,
    summarize_issue,
)
from seer.automation.summarize.models import (
    GetFixabilityScoreRequest,
    SummarizeIssueRequest,
    SummarizeIssueResponse,
    SummarizeIssueScores,
)
from seer.db import DbIssueSummary
from seer.stubs import can_use_model_stubs


class TestSummarizeIssue:
    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def sample_request(self):
        issues_dir = Path(__file__).parent / "fixtures" / "issues"
        issues: list[IssueDetails] = []
        for path in issues_dir.glob("issue_to_summarize*.json"):
            with path.open() as f:
                issues.append(IssueDetails.model_validate_json(f.read()))
        assert len(issues) >= 2, "Need at least 2 issues so that there's a connected issue"
        return SummarizeIssueRequest(
            group_id=123,
            issue=issues[0],
            connected_issues=issues[1:],
            organization_id=456,
            organization_slug="test-org",
            project_id=789,
        )

    @pytest.mark.vcr()
    def test_summarize_issue_success(self, sample_request, score_num_decimal_places: int = 10):
        result = summarize_issue(sample_request)

        # Debug: Print out the actual result to understand what fields might be different
        print(f"\nActual result: {result}")
        print(f"Actual result scores: {result.scores}")

        expected_result = IssueSummaryWithScores(
            title="red-timothy-sandwich: Data Corruption?",
            whats_wrong="**red-timothy-sandwich** is the main issue.",
            session_related_issues="Other issues: **cyan-vincent-banana**, **green-fred-tennis**.",
            possible_cause="Perhaps a **data corruption** issue is the root cause.",
            scores=SummarizeIssueScores(
                possible_cause_confidence=0.4,
                possible_cause_novelty=0.8,
                fixability_score=None,
                fixability_score_version=None,
                is_fixable=None,
            ),
        )

        # Round for some tolerance during equality comparison.
        result.scores.possible_cause_confidence = round(
            result.scores.possible_cause_confidence, score_num_decimal_places
        )
        result.scores.possible_cause_novelty = round(
            result.scores.possible_cause_novelty, score_num_decimal_places
        )

        # Ensure all fixability-related fields are set to None for comparison
        result.scores.fixability_score = None
        result.scores.fixability_score_version = None
        result.scores.is_fixable = None

        # Compare individual fields for better debug output
        assert (
            result.title == expected_result.title
        ), f"Title mismatch: {result.title} != {expected_result.title}"
        assert result.whats_wrong == expected_result.whats_wrong, "whats_wrong mismatch"
        assert (
            result.session_related_issues == expected_result.session_related_issues
        ), "session_related_issues mismatch"
        assert result.possible_cause == expected_result.possible_cause, "possible_cause mismatch"
        assert (
            result.scores.possible_cause_confidence
            == expected_result.scores.possible_cause_confidence
        ), "possible_cause_confidence mismatch"
        assert (
            result.scores.possible_cause_novelty == expected_result.scores.possible_cause_novelty
        ), "possible_cause_novelty mismatch"

        # Compare the dictionaries instead of direct equality to avoid unexpected pydantic differences
        assert result.model_dump() == expected_result.model_dump()

        response = result.to_summarize_issue_response(123)
        assert isinstance(response, SummarizeIssueResponse)
        assert response.group_id == 123
        assert response.headline == expected_result.title
        assert response.whats_wrong == expected_result.whats_wrong
        assert response.trace == expected_result.session_related_issues
        assert response.possible_cause == expected_result.possible_cause
        assert response.scores is not None
        assert (
            response.scores.possible_cause_confidence
            == expected_result.scores.possible_cause_confidence
        )
        assert (
            response.scores.possible_cause_novelty == expected_result.scores.possible_cause_novelty
        )
        assert response.scores.fixability_score is None
        assert response.scores.fixability_score_version is None
        assert response.scores.is_fixable is None

    @patch("seer.automation.summarize.issue.EventDetails.from_event")
    def test_summarize_issue_event_details(self, mock_from_event, mock_llm_client, sample_request):
        mock_event_details = Mock()
        mock_event_details.format_event.side_effect = ["foo details", "bar details", "baz details"]
        mock_from_event.return_value = mock_event_details

        mock_llm_client.generate_structured.return_value = LlmGenerateStructuredResponse(
            parsed=IssueSummaryForLlmToGenerate(
                whats_wrong="Test what's wrong",
                session_related_issues="Test session related issues",
                possible_cause="Test possible cause",
                possible_cause_novelty_score=0.5,
                possible_cause_confidence_score=0.5,
                title="Test headline",
            ),
            metadata=LlmResponseMetadata(
                model="test-model",
                provider_name=LlmProviderType.GEMINI,
                usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            ),
        )

        summarize_issue(sample_request, llm_client=mock_llm_client)

        mock_from_event.assert_any_call(sample_request.issue.events[0])
        mock_from_event.assert_any_call(sample_request.connected_issues[0].events[0])
        mock_from_event.assert_any_call(sample_request.connected_issues[1].events[0])
        assert mock_event_details.format_event.call_count == 3
        assert "foo details" in mock_llm_client.generate_structured.call_args[1]["prompt"]
        assert "bar details" in mock_llm_client.generate_structured.call_args[1]["prompt"]
        assert "baz details" in mock_llm_client.generate_structured.call_args[1]["prompt"]


class TestRunSummarizeIssue:
    @patch("seer.automation.summarize.issue.summarize_issue")
    @patch("seer.automation.summarize.issue.Session")
    def test_run_summarize_issue_langfuse_metadata(self, mock_session, mock_summarize_issue):
        mock_db_session = Mock()
        mock_session.return_value.__enter__.return_value = mock_db_session

        issue_summary = IssueSummaryWithScores(
            title="Test headline",
            whats_wrong="Test what's wrong",
            session_related_issues="Test session related issues",
            possible_cause="Test possible cause",
            scores=SummarizeIssueScores(
                possible_cause_confidence=0.5,
                possible_cause_novelty=0.5,
            ),
        )
        mock_summarize_issue.return_value = issue_summary

        # Create a sample request
        request = SummarizeIssueRequest(
            group_id=123,
            issue=next(generate(IssueDetails)),
            organization_id=456,
            organization_slug="test-org",
            project_id=789,
        )

        # Call the function
        run_summarize_issue(request)

        # Assert that summarize_issue was called with the correct langfuse metadata
        mock_summarize_issue.assert_called_once()
        call_kwargs = mock_summarize_issue.call_args[1]

        assert call_kwargs["langfuse_tags"] == ["org:test-org", "project:789", "group:123"]
        assert call_kwargs["langfuse_session_id"] == "group:123"
        assert call_kwargs["langfuse_user_id"] == "org:test-org"

        # Verify the DB session was used correctly
        mock_db_session.merge.assert_called_once()
        mock_db_session.commit.assert_called_once()

    @patch("seer.automation.summarize.issue.summarize_issue")
    @patch("seer.automation.summarize.issue.Session")
    def test_run_summarize_issue_langfuse_metadata_no_org_slug(
        self, mock_session, mock_summarize_issue
    ):
        mock_db_session = Mock()
        mock_session.return_value.__enter__.return_value = mock_db_session

        issue_summary = IssueSummaryWithScores(
            title="Test headline",
            whats_wrong="Test what's wrong",
            session_related_issues="Test session related issues",
            possible_cause="Test possible cause",
            scores=SummarizeIssueScores(
                possible_cause_confidence=0.5,
                possible_cause_novelty=0.5,
            ),
        )
        mock_summarize_issue.return_value = issue_summary

        # Create a sample request without organization_slug
        request = SummarizeIssueRequest(
            group_id=123,
            issue=next(generate(IssueDetails)),
            organization_id=456,
            project_id=789,
        )

        # Call the function
        run_summarize_issue(request)

        # Assert that summarize_issue was called with the correct langfuse metadata
        mock_summarize_issue.assert_called_once()
        call_kwargs = mock_summarize_issue.call_args[1]

        assert call_kwargs["langfuse_tags"] == ["project:789", "group:123"]
        assert call_kwargs["langfuse_session_id"] == "group:123"
        assert call_kwargs["langfuse_user_id"] is None


class TestFixabilityScore:
    @pytest.fixture
    def sample_issue_summary(self):
        return IssueSummaryWithScores(
            title="Test Error in API Response",
            whats_wrong="The API is returning a 500 error when trying to access user data",
            session_related_issues="Several related errors seen in session",
            possible_cause="Missing validation on null user input",
            scores=SummarizeIssueScores(
                possible_cause_confidence=0.8,
                possible_cause_novelty=0.5,
            ),
        )

    @pytest.fixture
    def autofixability_model(self):
        return AutofixabilityModel("models/autofixability_v3/embeddings")

    @patch("seer.automation.summarize.issue.evaluate_autofixability")
    @patch("seer.automation.summarize.issue.Session")
    def test_run_fixability_score(
        self, mock_session, mock_evaluate_autofixability, autofixability_model
    ):
        # Setup mocks
        mock_db_session = Mock()
        mock_session.return_value.__enter__.return_value = mock_db_session

        scores_current = {
            "possible_cause_confidence": 0.8,
            "possible_cause_novelty": 0.6,
        }

        # Create a proper mock of DbIssueSummary with all required attributes
        mock_db_summary = Mock(spec=DbIssueSummary)
        mock_db_summary.summary = {
            "title": "Test Error",
            "whats_wrong": "Something is broken",
            "session_related_issues": "No related issues",
            "possible_cause": "Bad code",
            "scores": scores_current,
        }
        # Add the new fixability-related fields
        mock_db_summary.fixability_score = None
        mock_db_summary.is_fixable = None
        mock_db_summary.fixability_score_version = None

        mock_db_session.get.return_value = mock_db_summary

        # Set the return value for evaluate_autofixability
        mock_evaluate_autofixability.return_value = (0.75, True)

        # Create the request
        request = GetFixabilityScoreRequest(group_id=123)

        # Call the function
        result = run_fixability_score(request, autofixability_model)

        # Assertions
        mock_db_session.get.assert_called_once_with(DbIssueSummary, 123)
        mock_evaluate_autofixability.assert_called_once()
        mock_db_session.merge.assert_called_once()
        mock_db_session.commit.assert_called_once()

        assert result.group_id == 123
        assert result.scores is not None
        assert result.scores.fixability_score == 0.75
        assert result.scores.fixability_score_version == 3
        assert result.scores.is_fixable is True
        for score_name, score_value in scores_current.items():
            assert getattr(result.scores, score_name) == score_value

    @patch("seer.automation.summarize.issue.Session")
    def test_run_fixability_score_no_summary(self, mock_session, autofixability_model):
        # Setup mocks
        mock_db_session = Mock()
        mock_session.return_value.__enter__.return_value = mock_db_session
        mock_db_session.get.return_value = None

        # Create the request
        request = GetFixabilityScoreRequest(group_id=123)

        # Call the function and expect exception
        with pytest.raises(ValueError, match="No issue summary found for group_id: 123"):
            run_fixability_score(request, autofixability_model)

    def test_evaluate_autofixability(self, autofixability_model: AutofixabilityModel):
        issue_summary_fixable = IssueSummaryWithScores(
            title="KeyError: Overwriting 'message' in LogRecord during logging of similar issues embeddings",
            whats_wrong="**KeyError** in logging: Attempt to overwrite 'message'. Occurs when logging **extra** data.  Happens in `group_similar_issues_embeddings.py`.",
            session_related_issues="",
            possible_cause="The `extra` parameter in `logger.info` contains a key named 'message', which conflicts with the LogRecord's internal 'message' attribute.  This is a **logging configuration issue**.",
            scores=SummarizeIssueScores(
                possible_cause_confidence=0.95,
                possible_cause_novelty=0.85,
            ),
        )
        score, is_fixable = evaluate_autofixability(issue_summary_fixable, autofixability_model)
        assert isinstance(score, float)
        assert 0 < score < 1
        assert isinstance(is_fixable, bool)
        if not can_use_model_stubs():
            assert is_fixable
            assert score == pytest.approx(0.7751516, abs=1e-5)

    def test_issue_summary_db_conversions(self, sample_issue_summary):
        # Test to_db_state
        db_state = sample_issue_summary.to_db_state(456)
        assert db_state.group_id == 456
        assert isinstance(db_state.summary, dict)
        assert db_state.summary["title"] == sample_issue_summary.title
        assert db_state.fixability_score is None
        assert db_state.is_fixable is None
        assert db_state.fixability_score_version is None

        # Update with fixability scores
        sample_issue_summary.scores.fixability_score = 0.85
        sample_issue_summary.scores.is_fixable = True
        sample_issue_summary.scores.fixability_score_version = 3

        db_state = sample_issue_summary.to_db_state(456)
        assert db_state.fixability_score == 0.85
        assert db_state.is_fixable is True
        assert db_state.fixability_score_version == 3

        # Test from_db_state
        db_summary = DbIssueSummary(
            group_id=789,
            summary=sample_issue_summary.model_dump(mode="json"),
            fixability_score=0.65,
            is_fixable=False,
            fixability_score_version=3,
        )

        loaded_summary = IssueSummaryWithScores.from_db_state(db_summary)
        assert loaded_summary.title == sample_issue_summary.title
        assert loaded_summary.whats_wrong == sample_issue_summary.whats_wrong
        assert loaded_summary.scores.fixability_score == 0.65
        assert loaded_summary.scores.is_fixable is False
        assert loaded_summary.scores.fixability_score_version == 3
