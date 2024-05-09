"""delete s4s group records

Revision ID: 704569328ec7
Revises: b6cca7c6d99c
Create Date: 2024-05-09 17:12:28.920636

"""
import os

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector  # type: ignore

# revision identifiers, used by Alembic.
revision = "704569328ec7"
down_revision = "b6cca7c6d99c"
branch_labels = None
depends_on = None


def upgrade():
    if os.getenv("SENTRY_REGION", None) == "s4s":
        with op.batch_alter_table("grouping_records", schema=None) as batch_op:
            batch_op.drop_index(
                "ix_grouping_records_stacktrace_embedding_hnsw",
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
            )
            batch_op.drop_index("ix_grouping_records_project_id")

        op.drop_table("grouping_records")
        print("DROPPED!")
        op.create_table(
            "grouping_records",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("group_id", sa.BigInteger(), nullable=True),
            sa.Column("project_id", sa.BigInteger(), nullable=False),
            sa.Column("message", sa.String(), nullable=False),
            sa.Column("stacktrace_embedding", Vector(dim=768), nullable=False),
            sa.Column(
                "hash",
                sa.String(length=32),
                server_default="00000000000000000000000000000000",
                nullable=False,
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("grouping_records", schema=None) as batch_op:
            batch_op.create_index("ix_grouping_records_project_id", ["project_id"], unique=False)
            batch_op.create_index(
                "ix_grouping_records_stacktrace_embedding_hnsw",
                ["stacktrace_embedding"],
                unique=False,
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
            )


def downgrade():
    pass
