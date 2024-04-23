"""Drop and recreate the repositories table.

Revision ID: 5398765db926
Revises: d8b8874b594d
Create Date: 2024-04-22 20:15:18.429995

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "5398765db926"
down_revision = "d8b8874b594d"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("codebase_namespaces")
    op.drop_table("repositories")
    op.create_table(
        "repositories",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("organization", sa.BigInteger(), nullable=False),
        sa.Column("project", sa.BigInteger(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("external_slug", sa.String(), nullable=False),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("default_namespace", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization", "project", "provider", "external_id"),
    )
    with op.batch_alter_table("repositories", schema=None) as batch_op:
        batch_op.create_index(
            "ix_repository_organization_project_provider_slug",
            ["organization", "project", "provider", "external_id"],
            unique=False,
        )
    op.create_table(
        "codebase_namespaces",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("repo_id", sa.Integer(), nullable=False),
        sa.Column("sha", sa.String(length=40), nullable=False),
        sa.Column("tracking_branch", sa.String(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("accessed_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["repo_id"],
            ["repositories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("repo_id", "sha"),
        sa.UniqueConstraint("repo_id", "tracking_branch"),
    )
    with op.batch_alter_table("codebase_namespaces", schema=None) as batch_op:
        batch_op.create_index("ix_codebase_namespace_repo_id_sha", ["repo_id", "sha"], unique=False)
        batch_op.create_index(
            "ix_codebase_namespace_repo_id_tracking_branch",
            ["repo_id", "tracking_branch"],
            unique=False,
        )


def downgrade():
    op.drop_table("codebase_namespaces")
    op.drop_table("repositories")
    op.create_table(
        "repositories",
        sa.Column("id", sa.INTEGER(), autoincrement=True, nullable=False),
        sa.Column("organization", sa.INTEGER(), nullable=False),
        sa.Column("project", sa.INTEGER(), nullable=False),
        sa.Column("provider", sa.VARCHAR(), nullable=False),
        sa.Column("external_slug", sa.VARCHAR(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization", "project", "provider", "external_slug"),
    )
    op.create_table(
        "codebase_namespaces",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("repo_id", sa.Integer(), nullable=False),
        sa.Column("sha", sa.String(length=40), nullable=False),
        sa.Column("tracking_branch", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["repo_id"],
            ["repositories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("repo_id", "sha"),
        sa.UniqueConstraint("repo_id", "tracking_branch"),
    )
    with op.batch_alter_table("repositories", schema=None) as batch_op:
        batch_op.create_index(
            "ix_repository_organization_project_provider_slug",
            ["organization", "project", "provider", "external_slug"],
            unique=False,
        )
    with op.batch_alter_table("codebase_namespaces", schema=None) as batch_op:
        batch_op.create_index("ix_codebase_namespace_repo_id_sha", ["repo_id", "sha"], unique=False)
        batch_op.create_index(
            "ix_codebase_namespace_repo_id_tracking_branch",
            ["repo_id", "tracking_branch"],
            unique=False,
        )
