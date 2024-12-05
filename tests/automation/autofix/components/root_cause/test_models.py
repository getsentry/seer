import unittest

from seer.automation.autofix.components.root_cause.models import (
    RootCauseAnalysisItem,
    RootCauseAnalysisItemPrompt,
    RootCauseAnalysisRelevantContext,
    RootCauseRelevantCodeSnippet,
    RootCauseRelevantContext,
)


class TestRootCauseAnalysisItemPrompt(unittest.TestCase):
    def setUp(self):
        self.title = "Test Root Cause"
        self.description = "Test Description"
        self.code_snippet = RootCauseRelevantCodeSnippet(
            file_path="test/file.py",
            snippet="def test():\n    pass",
        )
        self.relevant_context = RootCauseRelevantContext(
            id=1,
            title="Test Context",
            description="Context Description",
            snippet=self.code_snippet,
        )

    def test_to_model_with_minimal_fields(self):
        """Test that to_model() works with just required fields"""
        prompt = RootCauseAnalysisItemPrompt(
            title=self.title,
            description=self.description,
        )

        model = prompt.to_model()

        self.assertIsInstance(model, RootCauseAnalysisItem)
        self.assertEqual(model.title, self.title)
        self.assertEqual(model.description, self.description)
        self.assertIsNone(model.code_context)

    def test_to_model_with_code_context(self):
        """Test that to_model() properly transforms code context"""
        prompt = RootCauseAnalysisItemPrompt(
            title=self.title,
            description=self.description,
            relevant_code=RootCauseAnalysisRelevantContext(
                snippets=[self.relevant_context]
            ),
        )

        model = prompt.to_model()

        self.assertIsInstance(model, RootCauseAnalysisItem)
        self.assertEqual(len(model.code_context), 1)
        
        context = model.code_context[0]
        self.assertEqual(context.id, 1)
        self.assertEqual(context.title, "Test Context")
        self.assertEqual(context.description, "Context Description")
        
        snippet = context.snippet
        self.assertEqual(snippet.file_path, "test/file.py")
        self.assertEqual(snippet.snippet, "def test():\n    pass")

    def test_roundtrip_conversion(self):
        """Test converting from model to prompt and back"""
        # Create an original model
        original = RootCauseAnalysisItem(
            id=1,
            title=self.title,
            description=self.description,
            code_context=[self.relevant_context],
        )

        # Convert to prompt
        prompt = RootCauseAnalysisItemPrompt.from_model(original)

        # Convert back to model
        final = prompt.to_model()

        # Verify fields match
        self.assertEqual(final.title, original.title)
        self.assertEqual(final.description, original.description)
        
        self.assertEqual(len(final.code_context), len(original.code_context))
        self.assertEqual(
            final.code_context[0].snippet.file_path,
            original.code_context[0].snippet.file_path
        )

    def test_to_model_preserves_required_fields(self):
        """Test that to_model() preserves required fields during transformation"""
        prompt = RootCauseAnalysisItemPrompt(
            title="Required Title",
            description="Required Description",
            relevant_code=None,
        )

        # This should not raise a validation error
        try:
            model = prompt.to_model()
        except Exception as e:
            self.fail(f"to_model() raised an exception unexpectedly: {e}")

        self.assertEqual(model.title, "Required Title")
        self.assertEqual(model.description, "Required Description")