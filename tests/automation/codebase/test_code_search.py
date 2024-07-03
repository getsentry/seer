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
