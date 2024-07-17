"""Migration

Revision ID: 95b4ba4f731d
Revises: 2597db647e9a
Create Date: 2024-07-17 23:35:18.871569

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "95b4ba4f731d"
down_revision = "2597db647e9a"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TABLE grouping_records ALTER COLUMN id TYPE BIGINT")
    op.execute("CREATE SEQUENCE IF NOT EXISTS grouping_records_id_seq")
    op.execute(
        "ALTER TABLE grouping_records ALTER COLUMN id SET DEFAULT nextval('grouping_records_id_seq')"
    )
    op.execute("ALTER SEQUENCE grouping_records_id_seq OWNED BY grouping_records.id")
    op.execute(
        "SELECT setval('grouping_records_id_seq', COALESCE((SELECT MAX(id) FROM grouping_records), 1), true)"
    )


def downgrade():
    op.execute("ALTER TABLE grouping_records ALTER COLUMN id DROP DEFAULT")
    op.execute("DROP SEQUENCE grouping_records_id_seq")
    op.execute("ALTER TABLE grouping_records ALTER COLUMN id TYPE INTEGER")
