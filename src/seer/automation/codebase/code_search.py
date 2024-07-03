import os
from typing import List, Optional

from seer.automation.autofix.utils import autofix_logger
from seer.automation.codebase.models import Match, SearchResult


class CodeSearcher:
    def __init__(
        self,
        directory: str,
        supported_extensions: set,
        max_results: int = 16,
        max_file_size_bytes: int = 1_000_000,  # 1 MB by default
        start_path: Optional[str] = None,
    ):
        self.directory = directory
        self.supported_extensions = supported_extensions
        self.max_results = max_results
        self.start_path = start_path
        self.max_file_size_bytes = max_file_size_bytes

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
        return 1 / (
            total_distance + 1
        )  # +1 to avoid division by zero and to ensure score is never 0

    def search_file(self, file_path: str, keyword: str) -> Optional[SearchResult]:
        relative_path = os.path.relpath(file_path, self.directory)
        matches = []

        try:
            if os.path.getsize(file_path) > self.max_file_size_bytes:
                autofix_logger.debug(
                    f"Skipping {file_path} as it exceeds the maximum file size limit."
                )
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if keyword.lower() in line.lower():
                        start = max(0, i - 8)
                        end = min(len(lines), i + 9)
                        context = "".join(lines[start:end])
                        matches.append(Match(line_number=i + 1, context=context))
                        break  # Stop after finding the first match

            if matches:
                score = self.calculate_proximity_score(file_path)
                return SearchResult(relative_path=relative_path, matches=matches, score=score)
        except UnicodeDecodeError:
            autofix_logger.error(f"Unable to read {file_path}")
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
