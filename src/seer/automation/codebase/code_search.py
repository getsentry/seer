import logging
import os
from typing import List, Optional

from seer.automation.codebase.models import Match, SearchResult
from seer.automation.utils import detect_encoding

logger = logging.getLogger(__name__)


class CodeSearcher:
    def __init__(
        self,
        directory: str,
        supported_extensions: set,
        max_results: int = 16,
        max_file_size_bytes: int = 1_000_000,  # 1 MB by default
        max_context_characters: int = 2000,
        start_path: Optional[str] = None,
        default_encoding: str = "utf-8",
    ):
        self.directory = directory
        self.supported_extensions = supported_extensions
        self.max_results = max_results
        self.start_path = start_path
        self.max_file_size_bytes = max_file_size_bytes
        self.max_context_characters = max_context_characters
        self.default_encoding = default_encoding

    def calculate_proximity_score(self, file_path: str) -> float:
        if not self.start_path:
            return 1.0  # No proximity score if no start_path is provided

        # Convert both paths to be relative to the base directory
        rel_start_path = os.path.relpath(self.start_path, self.directory)
        rel_file_path = os.path.relpath(file_path, self.directory)

        # If the start_path is a directory, remove the filename from the file_path for comparison
        if os.path.isdir(self.start_path):
            rel_file_path = os.path.dirname(rel_file_path)

        start_parts = rel_start_path.split(os.sep)
        file_parts = rel_file_path.split(os.sep)

        # Calculate the common path length
        common_length = sum(1 for a, b in zip(start_parts, file_parts) if a == b)

        # Calculate the distance from the common path to each file
        start_distance = len(start_parts) - common_length
        file_distance = len(file_parts) - common_length

        # Calculate proximity score based on distances
        total_distance = start_distance + file_distance
        return 1 / (total_distance + 1)

    def _read_file_with_encoding(self, file_path: str) -> Optional[List[str]]:
        """
        Smart file reader that attempts to detect and handle different file encodings.
        Returns list of lines if successful, None if file cannot be read.
        """
        # First try: Read a sample to detect encoding
        try:
            # Read only first 256KB to detect encoding for large files
            with open(file_path, "rb") as f:
                # theoretically, this could cause an incorrectly detected encoding if there are some special characters
                # in the file after the first 256KB of data
                # Hopefully it is a corner case, especially since files larger than 1MB are ignored from search
                raw_data = f.read(262144)
            if not raw_data:
                return []

            encoding = detect_encoding(raw_data, fallback_encoding=self.default_encoding)

            # Attempt to read with detected encoding
            with open(file_path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            # If detection failed, try common fallback encodings
            fallback_encodings = ["latin-1", "iso-8859-1", "cp1252", "windows-1251"]
            for enc in fallback_encodings:
                try:
                    with open(file_path, "r", encoding=enc) as f:
                        return f.readlines()
                except UnicodeDecodeError:
                    continue

            logger.warning(
                f"Failed to read {file_path} with all attempted encodings: "
                f"detected={encoding}, fallbacks={fallback_encodings}"
            )
            return None
        except Exception as e:
            logger.exception(f"Unexpected error reading {file_path}: {str(e)}")
            return None

    def search_file(self, file_path: str, keyword: str) -> Optional[SearchResult]:
        relative_path = os.path.relpath(file_path, self.directory)
        matches: List[Match] = []

        try:
            if not os.path.exists(file_path):
                logger.debug(f"Skipping {file_path} as it does not exist.")
                return None
                
            if os.path.getsize(file_path) > self.max_file_size_bytes:
                logger.debug(f"Skipping {file_path} as it exceeds the maximum file size limit.")
                return None

            lines = self._read_file_with_encoding(file_path)
            if lines is None:
                return None

            for i, line in enumerate(lines):
                if keyword.lower() in line.lower():
                    start = max(0, i - 8)
                    end = min(len(lines), i + 9)
                    context = "".join(lines[start:end])
                    if len(context) > self.max_context_characters:
                        context = context[: self.max_context_characters] + "..."
                    matches.append(Match(line_number=i + 1, context=context))
                    break  # Stop after finding the first match

            if matches:
                score = self.calculate_proximity_score(file_path)
                return SearchResult(relative_path=relative_path, matches=matches, score=score)
        except Exception as e:
            logger.exception(f"Error processing {file_path}: {str(e)}")
        return None

    def search(self, keyword: str) -> List[SearchResult]:
        results = []
        for root, _, filenames in os.walk(self.directory):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in self.supported_extensions):
                    file_path = os.path.join(root, filename)
                    result = self.search_file(file_path, keyword)
                    if result:
                        results.append(result)

        # Sort by score (proximity) and limit to max_results
        ranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        return ranked_results[: self.max_results]
