"""Update HNSW parameters for grouping_records

Revision ID: d87a6410efe4
Revises: a0d00121d118
Create Date: 2024-07-09 22:28:26.035785

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "d87a6410efe4"
down_revision = "a0d00121d118"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("grouping_records", schema=None) as batch_op:
        batch_op.drop_index(
            "ix_grouping_records_stacktrace_embedding_hnsw",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
        )
        batch_op.create_index(
            "ix_grouping_records_stacktrace_embedding_hnsw",
            ["stacktrace_embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
        )


def downgrade():
    with op.batch_alter_table("grouping_records", schema=None) as batch_op:
        batch_op.drop_index(
            "ix_grouping_records_stacktrace_embedding_hnsw",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 200},
            postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
        )
        batch_op.create_index(
            "ix_grouping_records_stacktrace_embedding_hnsw",
            ["stacktrace_embedding"],
            unique=False,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
        )
