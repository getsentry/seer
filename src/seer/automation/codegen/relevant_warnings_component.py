import bisect
import logging
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from cachetools import LRUCache, cached  # type: ignore[import-untyped]
from cachetools.keys import hashkey  # type: ignore[import-untyped]
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.codebase.models import Location, PrFile, StaticAnalysisWarning
from seer.automation.codegen.codegen_context import CodegenContext
from seer.automation.codegen.models import (
    AssociateWarningsWithIssuesOutput,
    AssociateWarningsWithIssuesRequest,
    CodeAreIssuesFixableOutput,
    CodeAreIssuesFixableRequest,
    CodeFetchIssuesOutput,
    CodeFetchIssuesRequest,
    CodePredictRelevantWarningsOutput,
    CodePredictRelevantWarningsRequest,
    FilterWarningsOutput,
    FilterWarningsRequest,
    RelevantWarningResult,
    WarningAndPrFile,
)
from seer.automation.codegen.prompts import IsFixableIssuePrompts, ReleventWarningsPrompts
from seer.automation.component import BaseComponent
from seer.automation.models import EventDetails, IssueDetails
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient


class FilterWarningsComponent(BaseComponent[FilterWarningsRequest, FilterWarningsOutput]):
    """
    Filter out warnings that aren't on the PR diff lines.
    """

    context: CodegenContext

    @staticmethod
    def _left_truncated_paths(path: Path, max_num_paths: int = 2) -> list[str]:
        """
        Example::

            path = Path("src/seer/automation/agent/client.py")
            paths = FilterWarningsComponent._left_truncated_paths(path, 2)
            assert paths == [
                "seer/automation/agent/client.py",
                "automation/agent/client.py",
            ]
        """
        parts = list(path.parts)
        num_dirs = len(parts) - 1  # -1 for the filename
        num_paths = min(max_num_paths, num_dirs)

        result = []
        for _ in range(num_paths):
            parts.pop(0)
            result.append(Path(*parts).as_posix())
        return result

    def _build_filepath_mapping(self, pr_files: list[PrFile]) -> dict[str, PrFile]:
        """Build mapping of possible filepaths to PR files, including truncated variations."""
        filepath_to_pr_file: dict[str, PrFile] = {}
        for pr_file in pr_files:
            pr_path = Path(pr_file.filename)
            filepath_to_pr_file[pr_path.as_posix()] = pr_file
            for truncated in self._left_truncated_paths(pr_path, max_num_paths=1):
                filepath_to_pr_file[truncated] = pr_file
        return filepath_to_pr_file

    def _find_matching_pr_file(
        self,
        warning: StaticAnalysisWarning,
        filepath_to_pr_file: dict[str, PrFile],
    ) -> tuple[PrFile, list[int]] | None:
        matching_pr_files = self._get_matching_pr_files(warning, filepath_to_pr_file)
        warning_location = Location.from_encoded(warning.encoded_location)
        for pr_file in matching_pr_files:
            hunk_ranges = self._get_sorted_hunk_ranges(pr_file)
            overlapping_hunk_idxs = self._overlapping_hunk_idxs(
                (int(warning_location.start_line), int(warning_location.end_line)),
                hunk_ranges,
            )
            if overlapping_hunk_idxs:
                return pr_file, overlapping_hunk_idxs

        return None

    def _get_sorted_hunk_ranges(self, pr_file: PrFile) -> list[tuple[int, int]]:
        """Returns sorted tuples of 1-indexed line numbers (start_inclusive, end_exclusive) in the updated pr file.

        Determined by parsing git diff hunk headers of the form:
        @@ -n,m +p,q @@ where:
        - n: start line in original file
        - m: number of lines from original file
        - p: start line in modified file
        - q: number of lines in modified file

        For example, given this diff:
        @@ -1,3 +1,4 @@
        def hello():
            print("hello")
        +    print("world")  # Line 3 is added
        print("goodbye")

        @@ -20,3 +21,4 @@
            print("end")
        +    print("new end")  # Line 22 is added
            return

        This would return [(1,5), (21,25)] representing the modified file's hunk ranges.

        Args:
            pr_file: PrFile object containing the patch/diff (sorted by line number)

        Returns:
            List of sorted tuples containing 1-indexed line numbers (start_inclusive, end_exclusive) in the updated file
        """
        return [
            (hunk.target_start, hunk.target_start + hunk.target_length) for hunk in pr_file.hunks
        ]

    def _get_matching_pr_files(
        self, warning: StaticAnalysisWarning, filepath_to_pr_file: dict[str, PrFile]
    ) -> list[PrFile]:
        """
        Find PR files that may match a warning's location.
        This handles cases where the warning location and PR file paths may be specified differently:
        - With different numbers of parent directories
        - With or without a repo prefix
        - With relative vs absolute paths
        """
        filename = warning.encoded_location.split(":")[0]
        path = Path(filename)
        # If the path is relative, it shouldn't contain intermediate `..`s.
        first_idx_non_dots = next((idx for idx, part in enumerate(path.parts) if part != ".."))
        path = Path(*path.parts[first_idx_non_dots:])
        if ".." in path.parts:
            raise ValueError(
                f"Found `..` in the middle of path. Encoded location: {warning.encoded_location}"
            )
        warning_filepath_variations = {
            path.as_posix(),
            *self._left_truncated_paths(path, max_num_paths=2),
        }
        return [
            filepath_to_pr_file[filepath]
            for filepath in warning_filepath_variations & set(filepath_to_pr_file)
        ]

    def _overlapping_hunk_idxs(
        self, warning_range: tuple[int, int], sorted_hunk_ranges: list[tuple[int, int]]
    ) -> list[int]:
        # TODO(kddubey): to be correct (grab all hunks overlapping w/ the warning), need to bisect
        # and then add until no overlap.
        if not sorted_hunk_ranges or not warning_range:
            return []

        warning_start, warning_end = warning_range
        # Handle special case of single line warning by making end inclusive
        if warning_start == warning_end:
            warning_end += 1
        index = bisect.bisect_left(sorted_hunk_ranges, (warning_start,))

        overlapping_hunk_idxs = []

        does_overlap_with_previous_hunk = (
            index > 0 and warning_start < sorted_hunk_ranges[index - 1][1]
        )
        if does_overlap_with_previous_hunk:
            overlapping_hunk_idxs.append(index - 1)

        does_overlap_with_next_hunk = (
            index < len(sorted_hunk_ranges) and sorted_hunk_ranges[index][0] < warning_end
        )
        if does_overlap_with_next_hunk:
            overlapping_hunk_idxs.append(index)

        return overlapping_hunk_idxs

    @observe(name="Codegen - Relevant Warnings - Filter Warnings Component")
    @ai_track(description="Codegen - Relevant Warnings - Filter Warnings Component")
    def invoke(self, request: FilterWarningsRequest) -> FilterWarningsOutput:
        filepath_to_pr_file = self._build_filepath_mapping(request.pr_files)
        warning_and_pr_files: list[WarningAndPrFile] = []
        for warning in request.warnings:
            try:
                match = self._find_matching_pr_file(warning, filepath_to_pr_file)
            except Exception as e:
                self.logger.warning(
                    f"Failed to match warning. Skipping: {warning.id} ({warning.encoded_location})",
                    exc_info=e,
                )
            else:
                if match is not None:
                    pr_file, overlapping_hunk_idxs = match
                    warning_and_pr_files.append(
                        WarningAndPrFile(
                            warning=warning,
                            pr_file=pr_file,
                            overlapping_hunk_idxs=overlapping_hunk_idxs,
                        )
                    )

        return FilterWarningsOutput(warning_and_pr_files=warning_and_pr_files)


def _fetch_issues_for_pr_file_cache_key(
    organization_id: int, provider: str, external_id: str, pr_file: PrFile, *args
) -> tuple[str]:
    return hashkey(organization_id, provider, external_id, pr_file.filename, pr_file.sha)


@cached(cache=LRUCache(maxsize=128), key=_fetch_issues_for_pr_file_cache_key)
@inject
def _fetch_issues_for_pr_file(
    organization_id: int,
    provider: str,
    external_id: str,
    pr_file: PrFile,
    run_id: int,
    logger: logging.Logger,
    client: RpcClient = injected,
) -> list[dict[str, Any]]:
    pr_filename_to_issues = client.call(
        "get_issues_related_to_file_patches",
        organization_id=organization_id,
        provider=provider,
        external_id=external_id,
        pr_files=[pr_file.model_dump()],
        run_id=run_id,
    )
    if pr_filename_to_issues is None:
        logger.exception(
            "Something went wrong with the issue-fetching RPC call",
            extra={"file": pr_file.filename},
        )
        return []
    if not pr_filename_to_issues:
        return []
    assert list(pr_filename_to_issues.keys()) == [pr_file.filename]
    return list(pr_filename_to_issues.values())[0]


class FetchIssuesComponent(BaseComponent[CodeFetchIssuesRequest, CodeFetchIssuesOutput]):
    """
    Fetch issues related to the files in a PR by analyzing stacktrace frames in the issue.
    """

    context: CodegenContext

    def _fetch_issues(
        self,
        organization_id: int,
        provider: str,
        external_id: str,
        pr_files: list[PrFile],
        max_files_analyzed: int = 7,
        max_lines_analyzed: int = 500,
    ) -> dict[str, list[IssueDetails]]:
        """
        Returns a dict mapping a subset of file names in the PR to issues related to the file.
        They're related if the functions and filenames in the issue's stacktrace overlap with those
        modified in the PR.

        The `max_files_analyzed` and `max_lines_analyzed` checks ensure that the payload we send to
        seer_rpc doesn't get too large.
        They're roughly like the qualification checks in [Open PR Comments](https://sentry.engineering/blog/how-open-pr-comments-work#qualification-checks).
        """
        pr_files_eligible = [
            pr_file
            for pr_file in pr_files
            if pr_file.status == "modified" and pr_file.changes <= max_lines_analyzed
        ]
        if not pr_files_eligible:
            self.logger.info("No eligible files in PR.")
            return {}

        self.logger.info(f"Repo query: {organization_id=}, {provider=}, {external_id=}")
        filename_to_issues = {
            pr_file.filename: _fetch_issues_for_pr_file(
                organization_id, provider, external_id, pr_file, self.context.run_id, self.logger
            )
            for pr_file in pr_files_eligible[:max_files_analyzed]
        }
        return {
            filename: [IssueDetails.model_validate(issue) for issue in issues]
            for filename, issues in filename_to_issues.items()
        }

    @observe(name="Codegen - Relevant Warnings - Fetch Issues Component")
    @ai_track(description="Codegen - Relevant Warnings - Fetch Issues Component")
    def invoke(self, request: CodeFetchIssuesRequest) -> CodeFetchIssuesOutput:
        if self.context.repo.provider_raw is None:
            raise TypeError(
                f"provider_raw is not set for repo: {self.context.repo}. "
                "Something went wrong during initialization of the RepoDefinition."
            )
        filename_to_issues = self._fetch_issues(
            organization_id=request.organization_id,
            provider=self.context.repo.provider_raw,
            external_id=self.context.repo.external_id,
            pr_files=request.pr_files,
        )
        return CodeFetchIssuesOutput(filename_to_issues=filename_to_issues)


class AssociateWarningsWithIssuesComponent(
    BaseComponent[AssociateWarningsWithIssuesRequest, AssociateWarningsWithIssuesOutput]
):
    """
    Given a list of warnings and a list of issues, return warning-issue pairs which should be
    analyzed by an LLM.

    The purpose of this step is to reduce LLM calls. If we have n warnings and m issues,
    we can reduce the number of pairs to consider from n * m to the top k, which is configurable.
    """

    context: CodegenContext

    @staticmethod
    def _format_issue_with_related_filename(issue: IssueDetails, related_filename: str) -> str:
        event_details = EventDetails.from_event(issue.events[0])
        return textwrap.dedent(
            f"""\
            {event_details.format_event_without_breadcrumbs(include_context=False, include_var_values=False)}
            ----------
            This file, in particular, contained function(s) that overlapped with the exceptions: {related_filename}
            """
        )

    @staticmethod
    def _top_k_indices(distances: np.ndarray, k: int) -> list[tuple[int, ...]]:
        flat_indices_sorted_by_distance = distances.argsort(axis=None)
        top_k_indices = np.unravel_index(flat_indices_sorted_by_distance[:k], distances.shape)
        return list(zip(*top_k_indices))

    @observe(name="Codegen - Relevant Warnings - Associate Warnings With Issues Component")
    @ai_track(description="Codegen - Relevant Warnings - Associate Warnings With Issues Component")
    def invoke(
        self, request: AssociateWarningsWithIssuesRequest
    ) -> AssociateWarningsWithIssuesOutput:

        warnings_formatted = [
            warning_and_pr_file.warning.format_warning()
            for warning_and_pr_file in request.warning_and_pr_files
        ]
        issue_id_to_issue_with_pr_filename = {
            issue.id: (issue, filename)
            for filename, issues in request.filename_to_issues.items()
            for issue in issues
        }
        # De-duplicate in case the same issue is present across multiple files. That's possible when
        # the issue's stacktrace matches multiple files modified in the PR.
        # This should be ok b/c the issue should contain enough information that the downstream LLM
        # calls can match any relevant warnings to it. The filename is not the strongest signal.

        if not request.warning_and_pr_files:
            self.logger.info("No warnings to associate with issues.")
            return AssociateWarningsWithIssuesOutput(candidate_associations=[])
        if not issue_id_to_issue_with_pr_filename:
            self.logger.info("No issues to associate with warnings.")
            return AssociateWarningsWithIssuesOutput(candidate_associations=[])

        issues_with_pr_filename = list(issue_id_to_issue_with_pr_filename.values())
        issues_formatted = [
            self._format_issue_with_related_filename(issue, pr_filename)
            for issue, pr_filename in issues_with_pr_filename
        ]

        model = GoogleProviderEmbeddings.model(
            "text-embedding-005", task_type="CODE_RETRIEVAL_QUERY"
        )
        embeddings_warnings = model.encode(warnings_formatted)
        embeddings_issues = model.encode(issues_formatted)
        warning_issue_cosine_similarities = embeddings_warnings @ embeddings_issues.T
        warning_issue_cosine_distances = 1 - warning_issue_cosine_similarities
        warning_issue_indices = self._top_k_indices(
            warning_issue_cosine_distances, request.max_num_associations
        )
        candidate_associations = [
            (request.warning_and_pr_files[warning_idx], issues_with_pr_filename[issue_idx][0])
            for warning_idx, issue_idx in warning_issue_indices
        ]
        return AssociateWarningsWithIssuesOutput(candidate_associations=candidate_associations)


def _is_issue_fixable_cache_key(issue: IssueDetails) -> tuple[str]:
    return hashkey(issue.id)


@cached(cache=LRUCache(maxsize=4096), key=_is_issue_fixable_cache_key)
@inject
def _is_issue_fixable(issue: IssueDetails, llm_client: LlmClient = injected) -> bool:
    # LRU-cached by the issue id. The same issue could be analyzed many times if, e.g.,
    # a repo has a set of files which are frequently used to handle and raise exceptions.
    completion = llm_client.generate_structured(
        model=GeminiProvider.model("gemini-2.0-flash-lite"),
        system_prompt=IsFixableIssuePrompts.format_system_msg(),
        prompt=IsFixableIssuePrompts.format_prompt(
            formatted_error=EventDetails.from_event(
                issue.events[0]
            ).format_event_without_breadcrumbs(),
        ),
        response_format=IsFixableIssuePrompts.IsIssueFixable,
        temperature=0.0,
        max_tokens=64,
    )
    if completion.parsed is None:
        raise ValueError("No structured output from LLM.")
    return completion.parsed.is_fixable


class AreIssuesFixableComponent(
    BaseComponent[CodeAreIssuesFixableRequest, CodeAreIssuesFixableOutput]
):
    """
    Given a list of issues, predict whether each is fixable.
    """

    context: CodegenContext

    @observe(name="Codegen - Relevant Warnings - Are Issues Fixable Component")
    @ai_track(description="Codegen - Relevant Warnings - Are Issues Fixable Component")
    def invoke(self, request: CodeAreIssuesFixableRequest) -> CodeAreIssuesFixableOutput:
        """
        It's fine if there are duplicate issues in the request. That can happen if issues were
        passed in from a list of warning-issue associations.
        """
        issue_id_to_issue = {issue.id: issue for issue in request.candidate_issues}
        issue_ids = list(issue_id_to_issue.keys())[: request.max_num_issues_analyzed]
        issue_id_to_is_fixable = {}
        for issue_id in issue_ids:
            try:
                is_fixable = _is_issue_fixable(issue_id_to_issue[issue_id])
            except Exception:
                # It's not critical that this component makes an actual prediction.
                # Assume it's fixable b/c the next (predict relevancy) step handles it.
                self.logger.exception("Error predicting fixability of issue")
                is_fixable = True
            issue_id_to_is_fixable[issue_id] = is_fixable
        return CodeAreIssuesFixableOutput(
            are_fixable=[issue_id_to_is_fixable.get(issue.id) for issue in request.candidate_issues]
        )


class PredictRelevantWarningsComponent(
    BaseComponent[CodePredictRelevantWarningsRequest, CodePredictRelevantWarningsOutput]
):
    """
    Given a list of warning-issue associations, predict whether each is relevant.
    A warning is relevant to an issue if fixing the warning would fix the issue (according to an
    LLM).
    """

    context: CodegenContext

    @cached(cache=LRUCache(maxsize=5))  # TODO(kddubey): test this
    def _file_contents(self, path: str, commit_sha: str) -> str | None:
        try:
            return self.context.get_repo_client().get_file_content(path, sha=commit_sha)
        except Exception:
            self.logger.exception(
                "Error getting file contents",
                extra={"path": path, "repo": self.context.repo.full_name},
            )
            return None

    def _right_justified(min_num: int, max_num: int) -> list[str]:
        max_digits = len(str(max_num))
        return [f"{number:>{max_digits}}" for number in range(min_num, max_num)]

    def _code_snippet_around_warning(
        self, warning_and_pr_file: WarningAndPrFile, commit_sha: str, window_size: int = 5
    ) -> str | None:
        file_contents = self._file_contents(warning_and_pr_file.pr_file.filename, commit_sha)
        if file_contents is None:
            return None

        warning_location = Location.from_encoded(warning_and_pr_file.warning.encoded_location)
        # -1 b/c the line numbers are 1-indexed.
        warning_start_line = int(warning_location.start_line) - 1
        warning_end_line = int(warning_location.end_line) - 1
        start_window = max(0, warning_start_line - window_size)
        end_window = warning_end_line + window_size
        lines = file_contents.split("\n")
        lines_snippet = lines[start_window:end_window]
        line_idxs = self._right_justified(start_window, end_window)
        lines_snippet = [
            f"{line_idx + 1}| {line}" for line_idx, line in zip(line_idxs, lines_snippet)
        ]
        snippet = "\n".join(lines_snippet)
        return snippet

    @observe(name="Codegen - Relevant Warnings - Predict Relevant Warnings Component")
    @ai_track(description="Codegen - Relevant Warnings - Predict Relevant Warnings Component")
    @inject
    def invoke(
        self, request: CodePredictRelevantWarningsRequest, llm_client: LlmClient = injected
    ) -> CodePredictRelevantWarningsOutput:
        # TODO(kddubey): instead of looking at every association, probably faster and cheaper to input one
        # warning and prompt for which of its associated issues are relevant. May not work as well.
        relevant_warning_results: list[RelevantWarningResult] = []
        for warning_and_pr_file, issue in request.candidate_associations:
            self.logger.info(
                f"Predicting relevance of warning {warning_and_pr_file.warning.id} and issue {issue.id}"
            )
            completion = llm_client.generate_structured(
                model=GeminiProvider.model("gemini-2.0-flash-001"),
                system_prompt=ReleventWarningsPrompts.format_system_msg(),
                prompt=ReleventWarningsPrompts.format_prompt(
                    formatted_warning=warning_and_pr_file.warning.format_warning(),
                    formatted_error=EventDetails.from_event(
                        issue.events[0]
                    ).format_event_without_breadcrumbs(),
                ),
                response_format=ReleventWarningsPrompts.DoesFixingWarningFixIssue,
                temperature=0.0,
                max_tokens=2048,
                timeout=15.0,
            )
            if completion.parsed is None:  # Gemini quirk
                self.logger.warning(
                    f"No response from LLM for warning {warning_and_pr_file.warning.id} and issue {issue.id}"
                )
                continue
            relevant_warning_results.append(
                RelevantWarningResult(
                    warning_id=warning_and_pr_file.warning.id,
                    issue_id=issue.id,
                    does_fixing_warning_fix_issue=completion.parsed.does_fixing_warning_fix_issue,
                    relevance_probability=completion.parsed.relevance_probability,
                    reasoning=completion.parsed.analysis,
                    short_description=completion.parsed.short_description or "",
                    short_justification=completion.parsed.short_justification or "",
                    encoded_location=warning_and_pr_file.warning.encoded_location,
                )
            )
        num_relevant_warnings = sum(
            result.does_fixing_warning_fix_issue for result in relevant_warning_results
        )
        self.logger.info(
            f"Found {num_relevant_warnings} relevant warnings out of "
            f"{len(relevant_warning_results)} pairs."
        )
        return CodePredictRelevantWarningsOutput(relevant_warning_results=relevant_warning_results)
