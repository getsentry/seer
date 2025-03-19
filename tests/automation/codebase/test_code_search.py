import os
import tempfile
import unittest
from unittest.mock import patch

from seer.automation.codebase.code_search import CodeSearcher
from seer.automation.codebase.models import SearchResult


class TestCodeSearcher(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.supported_extensions = {".py", ".js", ".ts"}
        self.code_searcher = CodeSearcher(
            directory=self.temp_dir,
            supported_extensions=self.supported_extensions,
            max_results=5,
        )

        # Create a test file
        self.test_file_path = os.path.join(self.temp_dir, "test_file.py")
        with open(self.test_file_path, "w") as f:
            f.write("def test_function():\n    print('Hello, World!')\n")

    def tearDown(self):
        # Clean up temporary files
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        os.rmdir(self.temp_dir)

    def test_search_file_exists(self):
        # Test searching for a keyword in an existing file
        result = self.code_searcher.search_file(self.test_file_path, "Hello")
        self.assertIsInstance(result, SearchResult)
        self.assertEqual(result.relative_path, "test_file.py")
        self.assertEqual(len(result.matches), 1)

    def test_search_file_does_not_exist(self):
        # Test searching for a keyword in a non-existent file
        non_existent_file = os.path.join(self.temp_dir, "non_existent.py")
        result = self.code_searcher.search_file(non_existent_file, "keyword")
        self.assertIsNone(result)

    def test_search_with_multiple_files(self):
        # Create another test file
        test_file2_path = os.path.join(self.temp_dir, "test_file2.py")
        with open(test_file2_path, "w") as f:
            f.write(
                "# This is another test file\ndef another_function():\n    return 'Another Test'"
            )

        try:
            # Test searching for a keyword across multiple files
            results = self.code_searcher.search("function")
            self.assertEqual(len(results), 2)  # Should find matches in both files
        finally:
            # Clean up
            if os.path.exists(test_file2_path):
                os.remove(test_file2_path)

    @patch("os.path.exists")
    @patch("os.path.getsize")
    def test_missing_file_handling(self, mock_getsize, mock_exists):
        # Simulate a file that doesn't exist
        mock_exists.return_value = False

        # This should not raise an error even though getsize would raise FileNotFoundError
        result = self.code_searcher.search_file("/path/to/nonexistent/file.py", "keyword")

        # Verify the result is None
        self.assertIsNone(result)

        # Verify getsize was never called
        mock_getsize.assert_not_called()
