"""Codebase changes tied to step

Revision ID: cefa0023a864
Revises: 86ba1a6c0cc3
Create Date: 2025-01-13 11:26:37.335196

"""

import json
from typing import Literal

import sqlalchemy as sa  # noqa
from alembic import op
from pydantic import BaseModel
from sqlalchemy.sql import text

# revision identifiers, used by Alembic.
revision = "cefa0023a864"
down_revision = "86ba1a6c0cc3"
branch_labels = None
depends_on = None


class Line(BaseModel):
    source_line_no: int | None = None
    target_line_no: int | None = None
    diff_line_no: int | None = None
    value: str
    line_type: Literal[" ", "+", "-"]


class Hunk(BaseModel):
    source_start: int
    source_length: int
    target_start: int
    target_length: int
    section_header: str
    lines: list[Line]


class FilePatch(BaseModel):
    type: Literal["A", "M", "D"]
    path: str
    added: int
    removed: int
    source_file: str
    target_file: str
    hunks: list[Hunk]


class ChangeDetails(BaseModel):
    title: str
    description: str
    diff: list[FilePatch] = []
    diff_str: str | None = None


class CommittedPullRequestDetails(BaseModel):
    pr_number: int
    pr_url: str
    pr_id: int | None = None


class FileChange(BaseModel):
    change_type: Literal["create", "edit", "delete"]
    path: str
    reference_snippet: str | None = None
    new_snippet: str | None = None
    description: str | None = None
    commit_message: str | None = None


class CodebaseChange(BaseModel):
    repo_external_id: str
    repo_name: str
    details: ChangeDetails | None = None
    branch_name: str | None = None
    pull_request: CommittedPullRequestDetails | None = None
    file_changes: list[FileChange] = []


class DeprecatedCodebaseChange(BaseModel):
    repo_id: int | None = None
    repo_external_id: str | None = None
    repo_name: str
    title: str
    description: str
    diff: list[FilePatch] = []
    diff_str: str | None = None
    branch_name: str | None = None
    pull_request: CommittedPullRequestDetails | None = None


class CodebaseState(BaseModel):
    repo_id: int | None = None
    repo_external_id: str | None = None

    # @deprecated, use the changes in the coding step
    file_changes: list[FileChange] = []


def get_changes_step(data: dict) -> dict | None:
    for step in data["steps"]:
        if step["key"] == "changes":
            return step
    return None


def migrate_old_codebase_changes_to_new(codebase_states: dict, changes: list[dict]) -> dict:
    codebase_changes = {}
    for external_id, codebase_state_dict in codebase_states.items():
        old_codebase_state = CodebaseState.model_validate(codebase_state_dict)
        codebase_change = next((c for c in changes if c["repo_external_id"] == external_id), None)

        if not codebase_change:
            continue

        old_codebase_change = DeprecatedCodebaseChange.model_validate(codebase_change)

        new_codebase_change = CodebaseChange(
            repo_external_id=old_codebase_change.repo_external_id or "<unknown>",
            repo_name=old_codebase_change.repo_name,
            details=ChangeDetails(
                title=old_codebase_change.title,
                description=old_codebase_change.description,
                diff=old_codebase_change.diff,
                diff_str=old_codebase_change.diff_str,
            ),
            branch_name=old_codebase_change.branch_name,
            pull_request=old_codebase_change.pull_request,
            file_changes=old_codebase_state.file_changes,
        )

        codebase_changes[external_id] = new_codebase_change.model_dump(mode="json")

    return codebase_changes


def upgrade():
    # Get database connection
    connection = op.get_bind()

    # Fetch all run_state records
    results = connection.execute(text("SELECT id, value FROM run_state")).fetchall()

    for id, value in results:
        if not value:
            continue

        try:
            data = value if isinstance(value, dict) else json.loads(value)

            # Skip if no codebases or steps
            if "codebases" not in data or not data["codebases"]:
                continue

            if "steps" not in data or not data["steps"]:
                continue

            # Get the last step and add the codebase info
            changes_step = get_changes_step(data)
            if changes_step is None:
                continue

            changes_step["codebase_changes"] = migrate_old_codebase_changes_to_new(
                data["codebases"], changes_step["changes"]
            )

            # TODO: Follow up with this on a subsequent migration if this one proves stable.
            # del data["codebases"]

            # Update the record
            connection.execute(
                text("UPDATE run_state SET value = :value WHERE id = :id"),
                {"id": id, "value": json.dumps(data)},
            )

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            # Log error but continue processing other records
            print(f"Error processing run_state id {id}: {str(e)}")
            continue


def downgrade():
    # Get database connection
    connection = op.get_bind()

    # Fetch all run_state records
    results = connection.execute(text("SELECT id, value FROM run_state")).fetchall()

    for id, value in results:
        if not value:
            continue

        try:
            data = value if isinstance(value, dict) else json.loads(value)

            # Skip if no steps
            if "steps" not in data or not data["steps"]:
                continue

            # Find the changes step and migrate data back
            changes_step = get_changes_step(data)
            changes = []
            if changes_step is not None and "codebase_changes" in changes_step:
                # Restore codebases data before removing codebase_changes
                data["codebases"] = {}
                for repo_id, codebase_change in changes_step["codebase_changes"].items():
                    data["codebases"][repo_id] = {
                        "repo_external_id": repo_id,
                        "file_changes": codebase_change.get("file_changes", []),
                    }

                    new_codebase_change = CodebaseChange.model_validate(codebase_change)

                    changes.append(
                        DeprecatedCodebaseChange(
                            repo_name=new_codebase_change.repo_name,
                            title=(
                                new_codebase_change.details.title
                                if new_codebase_change.details
                                else ""
                            ),
                            description=(
                                new_codebase_change.details.description
                                if new_codebase_change.details
                                else ""
                            ),
                            diff=(
                                new_codebase_change.details.diff
                                if new_codebase_change.details
                                else []
                            ),
                            diff_str=(
                                new_codebase_change.details.diff_str
                                if new_codebase_change.details
                                else None
                            ),
                            branch_name=new_codebase_change.branch_name,
                            pull_request=new_codebase_change.pull_request,
                        ).model_dump(mode="json")
                    )

                del changes_step["codebase_changes"]
                changes_step["changes"] = changes

            # Update the record
            connection.execute(
                text("UPDATE run_state SET value = :value WHERE id = :id"),
                {"id": id, "value": json.dumps(data)},
            )

        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            # Log error but continue processing other records
            print(f"Error processing run_state id {id}: {str(e)}")
            continue
