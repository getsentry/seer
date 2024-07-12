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
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_grouping_records_stacktrace_embedding_hnsw_new"
        )

        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            ix_grouping_records_stacktrace_embedding_hnsw_new
            ON grouping_records USING hnsw (stacktrace_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 200)
            """
        )

        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_grouping_records_stacktrace_embedding_hnsw"
        )

        op.execute(
            """
            ALTER INDEX ix_grouping_records_stacktrace_embedding_hnsw_new
            RENAME TO ix_grouping_records_stacktrace_embedding_hnsw
            """
        )


def downgrade():
    with op.get_context().autocommit_block():
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_grouping_records_stacktrace_embedding_hnsw_old"
        )

        op.execute(
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS
            ix_grouping_records_stacktrace_embedding_hnsw_old
            ON grouping_records USING hnsw (stacktrace_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )

        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS ix_grouping_records_stacktrace_embedding_hnsw"
        )

        op.execute(
            """
            ALTER INDEX ix_grouping_records_stacktrace_embedding_hnsw_old
            RENAME TO ix_grouping_records_stacktrace_embedding_hnsw
            """
        )
