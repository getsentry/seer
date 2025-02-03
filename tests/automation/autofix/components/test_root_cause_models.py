import pytest
from pydantic import ValidationError

from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisItemPrompt,
    RootCauseAnalysisRelevantContext,
    RootCauseRelevantCodeSnippet,
    RootCauseRelevantContext,
)


class TestRootCauseModels:
    def test_basic_model_validation(self):
        """Test that basic model validation works with minimal required fields."""
        item = RootCauseAnalysisItem(
            title="Test Title",
            description="Test Description",
        )
        assert item.title == "Test Title"
        assert item.description == "Test Description"
        assert item.code_context is None
        assert item.id == -1  # Default value

    def test_full_model_validation(self):
        """Test that model validation works with all fields provided."""
        code_context = [
            RootCauseRelevantContext(
                id=1,
                title="Context Title",
                description="Context Description",
                snippet=RootCauseRelevantCodeSnippet(
                    file_path="test.py",
                    snippet="def test(): pass",
                ),
            )
        ]

        item = RootCauseAnalysisItem(
            id=0,
            title="Test Title",
            description="Test Description",
            code_context=code_context,
        )

        assert item.id == 0
        assert item.title == "Test Title"
        assert item.description == "Test Description"
        assert len(item.code_context) == 1
        assert item.code_context[0].id == 1
        assert item.code_context[0].title == "Context Title"

    def test_model_transformation(self):
        """Test that transformation between Prompt and Item models works correctly."""
        # Create a prompt model
        relevant_code = RootCauseAnalysisRelevantContext(
            snippets=[
                RootCauseRelevantContext(
                    id=1,
                    title="Context Title",
                    description="Context Description",
                    snippet=RootCauseRelevantCodeSnippet(
                        file_path="test.py",
                        snippet="def test(): pass",
                    ),
                )
            ]
        )

        prompt = RootCauseAnalysisItemPrompt(
            title="Test Title",
            description="Test Description",
            relevant_code=relevant_code,
        )

        # Transform to item model
        item = prompt.to_model()

        assert item.title == "Test Title"
        assert item.description == "Test Description"
        assert len(item.code_context) == 1
        assert item.code_context[0].id == 1
        assert item.code_context[0].title == "Context Title"

    def test_model_missing_required_fields(self):
        """Test that model validation fails when required fields are missing."""
        with pytest.raises(ValidationError) as exc_info:
            RootCauseAnalysisItem(title="Test Title")
        assert "description" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            RootCauseAnalysisItem(description="Test Description")
        assert "title" in str(exc_info.value)

    def test_to_markdown_string(self):
        """Test that markdown string generation works correctly."""
        code_context = [
            RootCauseRelevantContext(
                id=1,
                title="Context Title",
                description="Context Description",
                snippet=RootCauseRelevantCodeSnippet(
                    file_path="test.py",
                    snippet="def test(): pass",
                    repo_name="test/repo",
                ),
            )
        ]

        item = RootCauseAnalysisItem(
            id=0,
            title="Test Title",
            description="Test Description",
            code_context=code_context,
        )

        markdown = item.to_markdown_string()

        assert "# Test Title" in markdown
        assert "## Description" in markdown
        assert "Test Description" in markdown
        assert "## Relevant Code Context" in markdown
        assert "### Context Title" in markdown
        assert "Context Description" in markdown
        assert "**File:** test.py" in markdown
        assert "**Repository:** test/repo" in markdown
        assert "def test(): pass" in markdown