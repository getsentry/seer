import os
import shutil
import tempfile
import unittest

from seer.automation.codebase.code_search import CodeSearcher, SearchResult


class TestCodeSearcher(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = tempfile.mkdtemp()
        self.create_test_files()
        self.code_searcher = CodeSearcher(
            self.test_dir, {".txt"}, start_path=os.path.join(self.test_dir, "folder1")
        )

    def tearDown(self):
        # Remove the temporary directory after tests
        shutil.rmtree(self.test_dir)

    def create_test_files(self):
        # Create a simple directory structure with test files
        os.makedirs(os.path.join(self.test_dir, "folder1"))
        os.makedirs(os.path.join(self.test_dir, "folder2"))
        os.makedirs(os.path.join(self.test_dir, "folder1", "subfolder"))

        with open(os.path.join(self.test_dir, "folder1", "file1.txt"), "w") as f:
            f.write("This is file1 with keyword")

        with open(os.path.join(self.test_dir, "folder2", "file2.txt"), "w") as f:
            f.write("This is file2 with keyword")

        with open(os.path.join(self.test_dir, "root_file.txt"), "w") as f:
            f.write("This is root_file with keyword")

        with open(os.path.join(self.test_dir, "folder1", "subfolder", "file3.txt"), "w") as f:
            f.write("This is file3 with keyword")

        with open(os.path.join(self.test_dir, "folder1", "subfolder", "file4.txt"), "w") as f:
            f.write("This is file4 without")

    def test_calculate_proximity_score(self):
        score1 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "folder1", "file1.txt")
        )
        score2 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "folder1", "subfolder", "file3.txt")
        )
        score3 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "folder2", "file2.txt")
        )
        score4 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "root_file.txt")
        )

        assert score1 >= score2 >= score3 >= score4, "Scores should be in descending order"

    def test_calculate_proximity_score_for_file_proximity(self):
        self.code_searcher.start_path = os.path.join(self.test_dir, "folder1", "file1.txt")
        score1 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "folder1", "file1.txt")
        )
        score2 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "root_file.txt")
        )
        score3 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "folder1", "subfolder", "file3.txt")
        )
        score4 = self.code_searcher.calculate_proximity_score(
            os.path.join(self.test_dir, "folder2", "file2.txt")
        )

        assert score1 >= score2 >= score3 >= score4, "Scores should be in descending order"

    def test_search_file(self):
        result = self.code_searcher.search_file(
            os.path.join(self.test_dir, "folder1", "file1.txt"), "keyword"
        )
        assert isinstance(result, SearchResult), "Should return a SearchResult object"
        assert result.relative_path == os.path.join(
            "folder1", "file1.txt"
        ), "Relative path should be correct"
        assert len(result.matches) == 1, "Should find one match"
        assert result.matches[0].line_number == 1, "Match should be on the first line"
        assert "keyword" in result.matches[0].context, "Context should contain the keyword"

    def test_search_file_max_size(self):
        self.code_searcher.max_file_size_bytes = 1
        result = self.code_searcher.search_file(
            os.path.join(self.test_dir, "folder1", "file1.txt"), "keyword"
        )
        assert result is None, "Should return None for files exceeding max size"

    def test_search(self):
        results = self.code_searcher.search("keyword")
        assert len(results) == 4, "Should find keyword in four files but not the fifth"
        assert (
            results[0].score >= results[1].score >= results[2].score
        ), "Results should be sorted by score"

    def test_max_results(self):
        code_searcher = CodeSearcher(
            self.test_dir,
            {".txt"},
            max_results=2,
            start_path=os.path.join(self.test_dir, "folder1"),
        )
        results = code_searcher.search("keyword")
        assert len(results) == 2, "Should return only max_results number of results"

    def test_supported_extensions(self):
        # Create a file with unsupported extension
        with open(os.path.join(self.test_dir, "unsupported.dat"), "w") as f:
            f.write("This file has keyword but unsupported extension")

        results = self.code_searcher.search("keyword")
        unsupported_found = any("unsupported.dat" in result.relative_path for result in results)
        assert not unsupported_found, "Should not find keyword in unsupported file types"

    def test_no_start_path(self):
        code_searcher_no_start = CodeSearcher(self.test_dir, {".txt"})
        results = code_searcher_no_start.search("keyword")
        assert all(
            result.score == 1.0 for result in results
        ), "All scores should be 1.0 when no start_path is provided"

    def test_keyword_not_found(self):
        results = self.code_searcher.search("nonexistent")
        assert len(results) == 0, "Should return empty list when keyword is not found"

    def test_read_file_with_encoding(self):
        # Test UTF-8 file
        with open(os.path.join(self.test_dir, "utf8.txt"), "w", encoding="utf-8") as f:
            f.write("Hello in UTF-8 🌍")

        # Test different encoding
        with open(os.path.join(self.test_dir, "latin1.txt"), "wb") as f:
            f.write("Hello in Latin-1 é".encode("latin-1"))

        # Test file with different encoding than default
        result1 = self.code_searcher.search_file(os.path.join(self.test_dir, "utf8.txt"), "Hello")
        result2 = self.code_searcher.search_file(os.path.join(self.test_dir, "latin1.txt"), "Hello")

        assert result1 is not None, "Should read UTF-8 file"
        assert result2 is not None, "Should read Latin-1 file"
        assert "Hello in UTF-8" in result1.matches[0].context
        assert "Hello in Latin-1" in result2.matches[0].context

    def test_read_file_with_invalid_encoding(self):
        # Create a binary file that's not valid in any text encoding
        with open(os.path.join(self.test_dir, "binary.txt"), "wb") as f:
            f.write(bytes([0xFF, 0xFE, 0x00, 0x00]))  # Invalid UTF-8

        result = self.code_searcher.search_file(
            os.path.join(self.test_dir, "binary.txt"), "keyword"
        )
        assert result is None, "Should return None for unreadable files"

    def test_custom_default_encoding(self):
        # Create a file with specific encoding
        test_text = "Test with special char é"
        with open(os.path.join(self.test_dir, "special.txt"), "w", encoding="latin-1") as f:
            f.write(test_text)

        # Create searcher with latin-1 as default encoding
        latin1_searcher = CodeSearcher(self.test_dir, {".txt"}, default_encoding="latin-1")

        result = latin1_searcher.search_file(os.path.join(self.test_dir, "special.txt"), "special")

        assert result is not None, "Should read file with custom default encoding"
        assert "special char" in result.matches[0].context

    def test_empty_file(self):
        # Test handling of empty files
        empty_file = os.path.join(self.test_dir, "empty.txt")
        with open(empty_file, "w"):
            pass

        result = self.code_searcher.search_file(empty_file, "keyword")
        assert result is None, "Should handle empty files gracefully"

    def test_max_context_characters(self):
        # Create a file with a long content
        long_content = "prefix " * 100 + "keyword" + " suffix" * 100
        test_file = os.path.join(self.test_dir, "long.txt")
        with open(test_file, "w") as f:
            f.write(long_content)

        # Create searcher with limited context
        limited_searcher = CodeSearcher(self.test_dir, {".txt"}, max_context_characters=20)

        result = limited_searcher.search_file(test_file, "keyword")
        assert result is not None, "Should find the keyword"
        assert len(result.matches) == 1, "Should have one match"
        assert (
            len(result.matches[0].context) <= 23
        ), "Context should be limited to max_context_characters + '...'"
        assert result.matches[0].context.endswith("..."), "Truncated context should end with ..."

        # Test with larger limit
        larger_searcher = CodeSearcher(self.test_dir, {".txt"}, max_context_characters=1000)
        result = larger_searcher.search_file(test_file, "keyword")
        assert result is not None, "Should find the keyword"
        assert (
            len(result.matches[0].context) <= 1003
        ), "Context should be limited to max_context_characters + '...'"
        assert "keyword" in result.matches[0].context, "Context should contain the keyword"
