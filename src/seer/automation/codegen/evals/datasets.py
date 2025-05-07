"""
This module contains functions for creating and getting datasets from Langfuse.
⚠️ It is created to be executed manually.

Running this file:
1. Collect the dataset details in a JSON file.
    Check `DatasetDetails` object below for the expected format.
    You can create a `.datasets` folder under this directory and it won't be tracked by git.
    TODO: add more guidance on the dataset optional fields.
2. Move the dataset details to `.artifacts` folder (root of seer, accessible from the container).
3. Run the script.
```
make shell
python src/seer/automation/codegen/evals/datasets.py create-dataset --details-file .artifacts/dataset_details.json
```

A new dataset will be created in Langfuse.
"""

import os
from pathlib import Path
from typing import NotRequired, TypedDict

import click
import httpx
import tqdm
from click import option
from langfuse import Langfuse
from pydantic.main import BaseModel

from seer.automation.codebase.models import PrFile, StaticAnalysisWarning
from seer.automation.codebase.repo_client import RepoClient, RepoClientType
from seer.automation.codegen.evals.models import EvalItemInput, EvalItemOutput, RepoInfo
from seer.automation.models import IssueDetails, RepoDefinition


# INPUT STRUCTURES ------------------------------------------------------------
class OrgDetails(BaseModel):
    organization_id: int
    organization_name: str
    repo_definitions: dict[str, RepoDefinition]


class GitHubInfo(BaseModel):
    org_details: OrgDetails
    repo_definition: RepoDefinition
    pr_id: int
    commit_sha: str


class DatasetDetails(BaseModel):
    dataset_name: str
    metadata: dict
    dataset_items: list[tuple[EvalItemInput, EvalItemOutput | list[EvalItemOutput]]]
    org_definitions: dict[str, OrgDetails]


class ItemRawData(TypedDict):
    github_info: GitHubInfo
    warnings: list[StaticAnalysisWarning]
    issues: list[IssueDetails] | None


# TODO: auto-generate expected suggestions based on a commit that fixes some other PR with issues.
# This will require an LLM to summarize the changes in the fix-commit as a list of suggestions.

# TODO: add function to collect warnings from a repo in a given commit.

# CREATE DATASET FUNCTIONS -----------------------------------------------------


@click.group()
def main():
    pass


@main.command()
@option(
    "--details-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="The path to the file containing the dataset details.",
)
@option(
    "--collect-diff",
    is_flag=True,
    help="Whether to collect the diffs of the PRs.",
    default=True,
)
def create_dataset(details_file: Path, collect_diff: bool):
    dataset_details = DatasetDetails.model_validate_json(details_file.read_text())
    items_raw_data: list[tuple[EvalItemInput, EvalItemOutput | list[EvalItemOutput]]] = []

    for item_input, item_output in tqdm.tqdm(
        dataset_details.dataset_items,
        total=len(dataset_details.dataset_items),
        unit="item",
        desc="Filling in item details",
    ):
        org_definition = dataset_details.org_definitions[item_input.repo.owner]
        repo_definition = org_definition.repo_definitions[item_input.repo.name]
        github_info = GitHubInfo(
            org_details=org_definition,
            repo_definition=repo_definition,
            pr_id=item_input.pr_id,
            commit_sha=item_input.commit_sha,
        )
        raw_item_data = ItemRawData(
            github_info=github_info,
            warnings=item_input.warnings,
            issues=item_input.issues,
        )
        items_raw_data.append(
            (enrich_item(**raw_item_data, should_collect_diff=collect_diff), item_output)
        )

    # Create the dataset
    dataset = create_langfuse_dataset(
        dataset_name=dataset_details.dataset_name,
        metadata=dataset_details.metadata,
        dataset_items=items_raw_data,
    )

    click.echo(f"✓ Dataset created: {dataset.id}")


def create_langfuse_dataset(
    dataset_name: str,
    metadata: dict,
    dataset_items: list[tuple[EvalItemInput, EvalItemOutput | list[EvalItemOutput]]],
):
    """
    Create a new dataset in Langfuse.
    """
    langfuse = Langfuse()
    dataset = langfuse.create_dataset(
        name=dataset_name,
        metadata=metadata,
    )

    for item_input, item_output in dataset_items:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input=item_input.model_dump(),
            expected_output=item_output,
        )

    return dataset


def collect_diff(github_info: GitHubInfo) -> list[PrFile]:
    """
    Collect the diffs of the PRs.
    """
    repo_client = RepoClient.from_repo_definition(
        github_info.repo_definition, type=RepoClientType.READ
    )
    pr_files = repo_client.repo.get_pull(github_info.pr_id).get_files()
    return [
        PrFile(
            filename=file.filename,
            patch=file.patch,
            status=file.status,
            changes=file.changes,
            sha=file.sha,
        )
        for file in pr_files
        if file.patch
    ]


def enrich_item(
    github_info: GitHubInfo,
    warnings: list[StaticAnalysisWarning],
    issues: list[IssueDetails] | None = None,
    should_collect_diff: bool = False,
) -> EvalItemInput:
    """
    Creates an EvalItem from the given GitHub info, warnings, expected suggestions, and (optional) issues.

    GitHub info is used to fetch the PR files (aka diff).
        This is done when creating the EvalItem so we don't have to do it at evaluation time,
        and to avoid access issues (depending on the GH token you use to create the EvalItem).

    If issues are not provided, it will fetch them from Sentry (public API).
        This means you can exactly control the issues that are used for the evaluation, and thus test
        how impactful to the suggestion having the issue context is.
    """

    # Get the PR files from the PR definition
    if should_collect_diff:
        pr_files = collect_diff(github_info)
    else:
        pr_files = None

    if pr_files and issues is None:
        issues = []
        sentry_token = os.getenv("SENTRY_TOKEN")
        assert sentry_token, "SENTRY_TOKEN is not set"
        for file in pr_files:
            issues_in_file = list_issues(github_info.repo_definition, file.filename, sentry_token)
            for issue in issues_in_file:
                events_for_issue = get_issue_events(issue.id, sentry_token)
                issue.events = [events_for_issue]
            issues.extend(issues_in_file)

    return EvalItemInput(
        repo=RepoInfo(
            provider=github_info.repo_definition.provider,
            owner=github_info.repo_definition.owner,
            name=github_info.repo_definition.name,
            external_id=github_info.repo_definition.external_id,
        ),
        pr_id=github_info.pr_id,
        organization_id=github_info.org_details.organization_id,
        commit_sha=github_info.commit_sha,
        warnings=warnings,
        pr_files=pr_files,
        issues=issues,
    )


# HELPER FUNCTIONS ------------------------------------------------------------


class SentryEventData(TypedDict):
    title: str
    entries: list[dict]
    tags: NotRequired[list[dict[str, str | None]]]


def list_issues(
    repo_definition: RepoDefinition, file_to_search: str, sentry_token: str
) -> list[IssueDetails]:
    """
    Uses the Sentry public API to list issues for a given file.
    It requires that `repo_owner/repo_name` maps exactly to the sentry org and sentry project name.
    """
    url = f"https://sentry.io/api/0/projects/{repo_definition.owner}/{repo_definition.name}/issues/"
    headers = {
        "Authorization": f"Bearer {sentry_token}",
    }
    query = f"is:unresolved {file_to_search}"
    response = httpx.get(url, headers=headers, params={"query": query})
    response.raise_for_status()
    issues = response.json()
    return [IssueDetails.model_validate({**issue, "events": []}) for issue in issues]


def get_issue_events(issue_id: int, sentry_token: str) -> SentryEventData:
    url = f"https://sentry.io/api/0/issues/{issue_id}/events/"
    headers = {
        "Authorization": f"Bearer {sentry_token}",
    }
    response = httpx.get(url, headers=headers)
    response.raise_for_status()
    entries = response.json()
    return SentryEventData(
        title=entries[0]["title"],
        entries=entries,
        tags=entries[0]["tags"],
    )


if __name__ == "__main__":
    main()
