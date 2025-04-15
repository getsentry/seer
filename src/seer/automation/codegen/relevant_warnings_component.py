import logging
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from cachetools import LRUCache, cached  # type: ignore[import-untyped]
from cachetools.keys import hashkey  # type: ignore[import-untyped]
from langfuse.decorators import observe
from sentry_sdk.ai.monitoring import ai_track

from seer.automation.agent.client import GeminiProvider, LlmClient
from seer.automation.agent.embeddings import GoogleProviderEmbeddings
from seer.automation.codebase.models import PrFile, StaticAnalysisWarning
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codebase.utils import code_snippet, left_truncated_paths
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
    CodePredictStaticAnalysisSuggestionsOutput,
    CodePredictStaticAnalysisSuggestionsRequest,
    FilterWarningsOutput,
    FilterWarningsRequest,
    RelevantWarningResult,
    WarningAndPrFile,
)
from seer.automation.codegen.prompts import (
    IsFixableIssuePrompts,
    ReleventWarningsPrompts,
    StaticAnalysisSuggestionsPrompts,
)
from seer.automation.component import BaseComponent
from seer.automation.models import EventDetails, FilePatch, IssueDetails, annotate_hunks
from seer.dependency_injection import inject, injected
from seer.rpc import RpcClient

MAX_FILES_ANALYZED = 7
MAX_LINES_ANALYZED = 500


class FilterWarningsComponent(BaseComponent[FilterWarningsRequest, FilterWarningsOutput]):
    """
    Filter out warnings that aren't on the PR diff lines.
    """

    context: CodegenContext

    def _build_filepath_mapping(self, pr_files: list[PrFile]) -> dict[str, PrFile]:
        """
        Build mapping of possible filepaths to PR files, including truncated variations.
        """
        filepath_to_pr_file: dict[str, PrFile] = {}
        for pr_file in pr_files:
            path = Path(pr_file.filename)
            filepath_to_pr_file[path.as_posix()] = pr_file
            for truncated in left_truncated_paths(path, max_num_paths=1):
                filepath_to_pr_file[truncated] = pr_file
        return filepath_to_pr_file

    def _matching_pr_files(
        self, warning: StaticAnalysisWarning, filepath_to_pr_file: dict[str, PrFile]
    ) -> list[PrFile]:
        """
        Find PR files that may match a warning's location.
        This handles cases where the warning location and PR file paths may be specified differently:
        - With different numbers of parent directories
        - With or without a repo prefix
        - With relative vs absolute paths
        """
        warning_filename = warning.encoded_location.split(":")[0]
        warning_path = Path(warning_filename)

        # If the path is relative, it shouldn't contain intermediate `..`s.
        first_idx_non_dots = next(
            (idx for idx, part in enumerate(warning_path.parts) if part != "..")
        )
        warning_path = Path(*warning_path.parts[first_idx_non_dots:])
        if ".." in warning_path.parts:
            raise ValueError(
                f"Found `..` in the middle of the warning's path. Encoded location: {warning.encoded_location}"
            )

        warning_filepath_variations = {
            warning_path.as_posix(),
            *left_truncated_paths(warning_path, max_num_paths=2),
        }
        return [
            filepath_to_pr_file[filepath]
            for filepath in warning_filepath_variations & set(filepath_to_pr_file)
        ]

    def _find_matching_pr_file(
        self,
        warning: StaticAnalysisWarning,
        filepath_to_pr_file: dict[str, PrFile],
    ) -> WarningAndPrFile | None:
        matching_pr_files = self._matching_pr_files(warning, filepath_to_pr_file)
        for pr_file in matching_pr_files:
            warning_and_pr_file = WarningAndPrFile(warning=warning, pr_file=pr_file)
            if warning_and_pr_file.overlapping_hunk_idxs:  # the warning is roughly in the patch
                return warning_and_pr_file
        return None

    @observe(name="Codegen - Relevant Warnings - Filter Warnings Component")
    @ai_track(description="Codegen - Relevant Warnings - Filter Warnings Component")
    def invoke(self, request: FilterWarningsRequest) -> FilterWarningsOutput:
        filepath_to_pr_file = self._build_filepath_mapping(request.pr_files)
        warning_and_pr_files: list[WarningAndPrFile] = []
        for warning in request.warnings:
            try:
                warning_and_pr_file = self._find_matching_pr_file(warning, filepath_to_pr_file)
            except Exception:
                self.logger.exception(
                    "Failed to match warning. Skipping.", extra={"warning_id": warning.id}
                )
            else:
                if warning_and_pr_file is not None:
                    warning_and_pr_files.append(warning_and_pr_file)
        return FilterWarningsOutput(warning_and_pr_files=warning_and_pr_files)


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
    assert list(pr_filename_to_issues.keys()) == [
        pr_file.filename
    ], f"expected {pr_file.filename} but got {list(pr_filename_to_issues.keys())}"
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
        max_files_analyzed: int = MAX_FILES_ANALYZED,
        max_lines_analyzed: int = MAX_LINES_ANALYZED,
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
        for filename, issues in filename_to_issues.items():
            self.logger.info(
                f"Found {len(issues)} issues for file {filename}",
                extra={"issue_ids": [issue.id for issue in issues]},
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


def _format_patch_with_warnings(
    pr_file: PrFile,
    warnings: list[StaticAnalysisWarning],
    include_warnings_after_patch: bool = False,
) -> str:
    target_line_to_warnings: dict[int, list[StaticAnalysisWarning]] = defaultdict(list)
    for warning in warnings:
        target_line_to_warnings[warning.start_line].append(warning)

    target_line_to_warning_annotation = {
        target_line: "  <-- STATIC ANALYSIS WARNINGS: "
        + " || ".join(
            warning.format_warning_id_and_message().replace("\n", "\\n") for warning in warnings
        )
        for target_line, warnings in target_line_to_warnings.items()
    }

    hunks = FilePatch.to_hunks(
        pr_file.patch, target_line_to_extra=target_line_to_warning_annotation
    )
    formatted_hunks = "\n\n".join(annotate_hunks(hunks))

    if include_warnings_after_patch:
        if not warnings:
            formatted_warnings = "No warnings were found in this file."
        else:
            formatted_warnings = "\n\n".join(
                warning.format_warning(filename=pr_file.filename) for warning in warnings
            )  # override the filename to reduce the chance of a hallucinated path
            formatted_warnings = "\n\n".join(
                (
                    f"Here's more information about the static analysis warnings in {pr_file.filename}:",
                    f"<warnings>\n\n{formatted_warnings}\n\n</warnings>",
                )
            )
    else:
        formatted_warnings = ""

    tag_start = f"<file><filename>{pr_file.filename}</filename>"
    tag_end = "</file>"
    title = f"Here are the changes made to file {pr_file.filename}:"
    return "\n\n".join((tag_start, title, formatted_hunks, formatted_warnings, tag_end))


def format_diff(
    warning_and_pr_files: list[WarningAndPrFile],
    pr_files: list[PrFile],
    patch_delim: str = "\n\n#################\n\n",
    include_warnings_after_patch: bool = True,
) -> str:
    filename_to_warnings: dict[str, list[StaticAnalysisWarning]] = defaultdict(list)
    for warning_and_pr_file in warning_and_pr_files:
        filename_to_warnings[warning_and_pr_file.pr_file.filename].append(
            warning_and_pr_file.warning
        )
    body = patch_delim.join(
        _format_patch_with_warnings(
            pr_file, filename_to_warnings[pr_file.filename], include_warnings_after_patch
        )
        for pr_file in pr_files
    )
    return f"<diff>\n\n{body}\n\n</diff>"


@cached(cache=LRUCache(maxsize=MAX_FILES_ANALYZED))
def _cached_file_contents(repo_client: RepoClient, path: str, commit_sha: str) -> str:
    file_contents, _ = repo_client.get_file_content(path, sha=commit_sha)
    if file_contents is None:
        raise ValueError("Failed to get file contents")  # raise => don't cache
    return file_contents


class PredictRelevantWarningsComponent(
    BaseComponent[CodePredictRelevantWarningsRequest, CodePredictRelevantWarningsOutput]
):
    """
    Given a list of warning-issue associations, predict whether each is relevant.
    A warning is relevant to an issue if fixing the warning would fix the issue (according to an
    LLM).
    """

    context: CodegenContext

    def _code_snippet_around_warning(
        self, warning_and_pr_file: WarningAndPrFile, commit_sha: str, padding_size: int = 10
    ) -> list[str] | None:
        try:
            file_contents = _cached_file_contents(
                self.context.get_repo_client(), warning_and_pr_file.pr_file.filename, commit_sha
            )
        except Exception:
            self.logger.exception("Error getting file contents")
            return None

        lines = file_contents.split("\n")
        if warning_and_pr_file.warning.end_line > len(lines):
            self.logger.error(
                "The warning's end line is greater than the number of lines in the file. "
                "Warning-file matching in FilterWarningsComponent was wrong or out of date.",
            )
            return None

        return code_snippet(
            lines,
            warning_and_pr_file.warning.start_line,
            warning_and_pr_file.warning.end_line,
            padding_size=padding_size,
        )

    def _format_code_snippet_around_warning(
        self, warning_and_pr_file: WarningAndPrFile, commit_sha: str, padding_size: int = 10
    ) -> str:
        code_snippet = self._code_snippet_around_warning(
            warning_and_pr_file, commit_sha, padding_size
        )
        if code_snippet is None:
            return "< Could not extract the code snippet containing the warning >"
        return "\n".join(code_snippet)

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


class StaticAnalysisSuggestionsComponent(
    BaseComponent[
        CodePredictStaticAnalysisSuggestionsRequest, CodePredictStaticAnalysisSuggestionsOutput
    ]
):
    """
    Given a diff, a list of warnings around the diff, and a list of fixable issues,
    surface potential issues in the diff (according to an LLM)
    """

    def _format_issue(self, issue: IssueDetails) -> str:
        # EventDetails are not formatted with the ID, so we add it manually.
        # Also the formatting is a weird half-XML, so we complete the XML tags.
        event_details = EventDetails.from_event(issue.events[0]).format_event_without_breadcrumbs()
        title, other_lines = event_details.split("\n", 1)
        return (
            f"<sentry_issue><issue_id>{issue.id}</issue_id>\n"
            + f"<title>{title}</title>\n"
            + other_lines
            + "</sentry_issue>"
        )

    @observe(name="Codegen - Relevant Warnings - Static-Analysis-Suggestions-Based Component")
    @ai_track(
        description="Codegen - Relevant Warnings - Static-Analysis-Suggestions-Based Component"
    )
    @inject
    def invoke(
        self, request: CodePredictStaticAnalysisSuggestionsRequest, llm_client: LlmClient = injected
    ) -> CodePredictStaticAnalysisSuggestionsOutput | None:
        # Current open questions on trading-off context for suggestions:
        # Limit diff size?
        # Limit number of warnings?
        # Limit number of fixable issues? or issue size?
        # Better, more concise way to encode the information for the LLM in the prompt?
        diff_with_warnings = format_diff(
            request.warning_and_pr_files, request.pr_files, include_warnings_after_patch=True
        )
        formatted_issues = (
            "<sentry_issues>\n"
            + "\n".join([self._format_issue(issue) for issue in request.fixable_issues])
            + "</sentry_issues>"
        )
        completion = llm_client.generate_structured(
            model=GeminiProvider.model("gemini-2.0-flash-001"),
            system_prompt=StaticAnalysisSuggestionsPrompts.format_system_msg(),
            prompt=StaticAnalysisSuggestionsPrompts.format_prompt(
                diff_with_warnings=diff_with_warnings,
                formatted_issues=formatted_issues,
            ),
            response_format=StaticAnalysisSuggestionsPrompts.AnalysisAndSuggestions,
            temperature=0.0,
            max_tokens=8192,
        )
        if completion.parsed is None:
            return None
        return CodePredictStaticAnalysisSuggestionsOutput(suggestions=completion.parsed.suggestions)
