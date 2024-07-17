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
    op.execute("DROP TABLE IF EXISTS grouping_records_new CASCADE;")

    op.execute(
        """
        CREATE TABLE grouping_records_new (
            id INTEGER NOT NULL,
            project_id BIGINT NOT NULL,
            hash VARCHAR(32) NOT NULL,
            message VARCHAR NOT NULL,
            error_type VARCHAR,
            stacktrace_embedding VECTOR(768) NOT NULL,
            PRIMARY KEY (id, project_id)
        ) PARTITION BY HASH (project_id);
        """
    )

    for i in range(100):
        op.execute(
            f"""
            CREATE TABLE grouping_records_new_p{i} PARTITION OF grouping_records_new
            FOR VALUES WITH (MODULUS 100, REMAINDER {i});
            """
        )

    op.execute(
        """
        INSERT INTO grouping_records_new (id, project_id, message, error_type, stacktrace_embedding, hash)
        SELECT id, project_id, message, error_type, stacktrace_embedding, hash
        FROM grouping_records;
        """
    )

    op.execute(
        """
        CREATE INDEX ix_grouping_records_new_stacktrace_embedding_hnsw
        ON grouping_records_new USING hnsw (stacktrace_embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 200);
        """
    )

    op.execute(
        "CREATE INDEX ix_grouping_records_new_project_id ON grouping_records_new (project_id);"
    )

    with op.batch_alter_table("grouping_records_new", schema=None) as batch_op:
        batch_op.create_unique_constraint("u_project_id_hash_composite", ["project_id", "hash"])


def downgrade():
    op.execute("DROP TABLE IF EXISTS grouping_records_new CASCADE;")
