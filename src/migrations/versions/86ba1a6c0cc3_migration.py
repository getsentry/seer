"""Migration

Revision ID: 86ba1a6c0cc3
Revises: 5a7ff418f726
Create Date: 2025-01-03 19:33:47.489760

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "86ba1a6c0cc3"
down_revision = "5a7ff418f726"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("dynamic_alert_time_series_history", schema=None) as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.INTEGER(),
            type_=sa.BigInteger(),
            existing_nullable=False,
            autoincrement=True,
        )
        batch_op.create_index(
            "ix_dynamic_alert_time_series_history_timestamp", ["timestamp"], unique=False
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("dynamic_alert_time_series_history", schema=None) as batch_op:
        batch_op.drop_index("ix_dynamic_alert_time_series_history_timestamp")
        batch_op.alter_column(
            "id",
            existing_type=sa.BigInteger(),
            type_=sa.INTEGER(),
            existing_nullable=False,
            autoincrement=True,
        )

    # ### end Alembic commands ###
