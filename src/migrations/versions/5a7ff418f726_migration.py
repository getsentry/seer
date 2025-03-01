"""Migration

Revision ID: 5a7ff418f726
Revises: 3914e6cdb818
Create Date: 2024-12-30 23:36:51.280449

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "5a7ff418f726"
down_revision = "3914e6cdb818"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "dynamic_alert_time_series_history",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("alert_id", sa.BigInteger(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("anomaly_type", sa.String(), nullable=False),
        sa.Column("saved_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id", "alert_id"),
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table("dynamic_alert_time_series_history")
    # ### end Alembic commands ###
