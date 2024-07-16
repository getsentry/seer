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
    # Create a temporary table with the new structure
    op.execute(
        """
        CREATE TABLE grouping_records_tmp (LIKE grouping_records INCLUDING ALL)
        PARTITION BY HASH (project_id);
    """
    )

    # Create partitions
    for i in range(100):
        op.execute(
            f"""
            CREATE TABLE grouping_records_tmp_{i}
            PARTITION OF grouping_records_tmp
            FOR VALUES WITH (modulus 100, remainder {i});
        """
        )

    # Copy data to the temporary table
    op.execute("INSERT INTO grouping_records_tmp SELECT * FROM grouping_records;")

    # Rename tables
    op.execute("ALTER TABLE grouping_records RENAME TO grouping_records_old;")
    op.execute("ALTER TABLE grouping_records_tmp RENAME TO grouping_records;")

    # Recreate the index with new parameters
    op.create_index(
        "ix_grouping_records_stacktrace_embedding_hnsw",
        "grouping_records",
        ["stacktrace_embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 200},
        postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
    )

    # Drop the old table
    op.execute("DROP TABLE grouping_records_old;")


def downgrade():
    # Create a temporary table without partitioning
    op.execute("CREATE TABLE grouping_records_tmp (LIKE grouping_records INCLUDING ALL);")

    # Copy data to the temporary table
    op.execute("INSERT INTO grouping_records_tmp SELECT * FROM grouping_records;")

    # Rename tables
    op.execute("DROP TABLE grouping_records;")
    op.execute("ALTER TABLE grouping_records_tmp RENAME TO grouping_records;")

    # Recreate the index with old parameters
    op.create_index(
        "ix_grouping_records_stacktrace_embedding_hnsw",
        "grouping_records",
        ["stacktrace_embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"stacktrace_embedding": "vector_cosine_ops"},
    )
