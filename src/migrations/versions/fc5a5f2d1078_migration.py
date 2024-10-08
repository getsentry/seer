"""Migration

Revision ID: fc5a5f2d1078
Revises: 52d3b519f90c
Create Date: 2024-02-29 01:20:29.126890

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "fc5a5f2d1078"
down_revision = "52d3b519f90c"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "document_chunk_tombstones",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("repo_id", sa.Integer(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("namespace", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(
            ["repo_id"],
            ["repositories.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("document_chunk_tombstones", schema=None) as batch_op:
        batch_op.create_index(
            "idx_repo_namespace_path", ["repo_id", "namespace", "path"], unique=True
        )

    with op.batch_alter_table("document_chunks", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False)
        )
        batch_op.create_index(
            "idx_repo_id_namespace_path",
            ["repo_id", "namespace", "path"],
            unique=True,
            postgresql_where=sa.text("namespace IS NOT NULL"),
        )
        batch_op.create_index(
            "idx_repo_path",
            ["repo_id", "path"],
            unique=True,
            postgresql_where=sa.text("namespace IS NULL"),
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("document_chunks", schema=None) as batch_op:
        batch_op.drop_index("idx_repo_path", postgresql_where=sa.text("namespace IS NULL"))
        batch_op.drop_index(
            "idx_repo_id_namespace_path", postgresql_where=sa.text("namespace IS NOT NULL")
        )
        batch_op.drop_column("created_at")

    with op.batch_alter_table("document_chunk_tombstones", schema=None) as batch_op:
        batch_op.drop_index("idx_repo_namespace_path")

    op.drop_table("document_chunk_tombstones")
    # ### end Alembic commands ###
