import os
from unittest.mock import mock_open, patch

import pytest

from seer.automation.autofix.tools.read_file_contents import read_file_contents


class TestReadFileContents:
    def test_successful_read(self, tmp_path):
        # Create a temporary file with content
        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)

        # Test reading the file
        content, error = read_file_contents(str(tmp_path), "test.txt")
        assert content == test_content
        assert error is None

    def test_file_does_not_exist(self, tmp_path):
        # Test reading a non-existent file
        content, error = read_file_contents(str(tmp_path), "nonexistent.txt")
        assert content is None
        assert "does not exist" in error

    def test_path_is_directory(self, tmp_path):
        # Create a directory instead of a file
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        # Test reading a directory
        content, error = read_file_contents(str(tmp_path), "test_dir")
        assert content is None
        assert "exists but is not a file" in error

    @patch("builtins.open", side_effect=PermissionError("Permission denied"))
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_permission_error(self, mock_isfile, mock_exists, mock_open, tmp_path):
        # Test permission error when reading file
        content, error = read_file_contents(str(tmp_path), "test.txt")
        assert content is None
        assert "Permission denied" in error

    @patch("builtins.open", side_effect=UnicodeDecodeError("utf-8", b"test", 0, 1, "invalid utf-8"))
    @patch("os.path.exists", return_value=True)
    @patch("os.path.isfile", return_value=True)
    def test_unicode_decode_error(self, mock_isfile, mock_exists, mock_open, tmp_path):
        # Test unicode decode error when reading file
        content, error = read_file_contents(str(tmp_path), "test.txt")
        assert content is None
        assert "invalid utf-8" in error

    def test_empty_file(self, tmp_path):
        # Create an empty file
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")

        # Test reading an empty file
        content, error = read_file_contents(str(tmp_path), "empty.txt")
        assert content == ""
        assert error is None

    def test_file_with_special_characters(self, tmp_path):
        # Create a file with special characters
        test_file = tmp_path / "special.txt"
        test_content = "Hello\n世界\n🌍"
        test_file.write_text(test_content)

        # Test reading file with special characters
        content, error = read_file_contents(str(tmp_path), "special.txt")
        assert content == test_content
        assert error is None
