"""delete s4s group records

Revision ID: 704569328ec7
Revises: b6cca7c6d99c
Create Date: 2024-05-09 17:12:28.920636

"""
import os

from alembic import op

# revision identifiers, used by Alembic.
revision = "704569328ec7"
down_revision = "b6cca7c6d99c"
branch_labels = None
depends_on = None


def upgrade():
    if os.getenv("SENTRY_REGION", None) != "s4s":
        return
    op.execute("TRUNCATE TABLE grouping_records")


def downgrade():
    pass
