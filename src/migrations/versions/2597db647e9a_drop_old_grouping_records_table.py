"""drop old grouping_records table

Revision ID: 2597db647e9a
Revises: a0d00121d118
Create Date: 2024-07-17 03:16:52.924194

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "2597db647e9a"
down_revision = "a0d00121d118"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_table("grouping_records_old")


def downgrade():
    op.create_table(
        "grouping_records_old",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("group_id", sa.Integer(), nullable=False),
        sa.Column("record_id", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["groups.id"],
        ),
        sa.ForeignKeyConstraint(
            ["record_id"],
            ["records.id"],
        ),
    )
