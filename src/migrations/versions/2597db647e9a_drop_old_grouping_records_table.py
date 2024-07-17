"""drop old grouping_records table

Revision ID: 2597db647e9a
Revises: a0d00121d118
Create Date: 2024-07-17 03:16:52.924194

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "2597db647e9a"
down_revision = "a0d00121d118"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE IF EXISTS grouping_records RENAME to grouping_records_old;")

    op.execute("ALTER TABLE grouping_records_new RENAME TO grouping_records;")
    for i in range(100):
        op.execute(f"ALTER TABLE grouping_records_new_p{i} RENAME TO grouping_records_p{i};")

    op.execute("DROP TABLE IF EXISTS grouping_records_old")


def downgrade():
    op.execute("ALTER TABLE IF EXISTS grouping_records RENAME TO grouping_records_new;")

    for i in range(100):
        op.execute(f"ALTER TABLE grouping_records_p{i} RENAME TO grouping_records_new_p{i};")

    op.execute("ALTER TABLE grouping_records_old RENAME TO grouping_records;")

    op.execute("DROP TABLE IF EXISTS grouping_records_new")
