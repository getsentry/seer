"""Migration

Revision ID: e0fcdc14251c
Revises: 8548b504edfe
Create Date: 2025-02-25 23:53:14.920479

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e0fcdc14251c"
down_revision = "8548b504edfe"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "prophet_alert_time_series_history",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("alert_id", sa.BigInteger(), nullable=False),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("yhat", sa.Float(), nullable=False),
        sa.Column("yhat_lower", sa.Float(), nullable=False),
        sa.Column("yhat_upper", sa.Float(), nullable=False),
        sa.Column("saved_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("prophet_alert_time_series_history", schema=None) as batch_op:
        batch_op.create_index(
            "ix_prophet_alert_time_series_history_timestamp", ["timestamp"], unique=False
        )

    op.create_table(
        "prophet_alert_time_series",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dynamic_alert_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("yhat", sa.Float(), nullable=False),
        sa.Column("yhat_lower", sa.Float(), nullable=False),
        sa.Column("yhat_upper", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dynamic_alert_id"], ["dynamic_alerts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dynamic_alert_id", "timestamp"),
    )
    with op.batch_alter_table("prophet_alert_time_series", schema=None) as batch_op:
        batch_op.create_index(
            "ix_prophet_alert_time_series_alert_id_timestamp",
            ["dynamic_alert_id", "timestamp"],
            unique=False,
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("prophet_alert_time_series", schema=None) as batch_op:
        batch_op.drop_index("ix_prophet_alert_time_series_alert_id_timestamp")

    op.drop_table("prophet_alert_time_series")
    with op.batch_alter_table("prophet_alert_time_series_history", schema=None) as batch_op:
        batch_op.drop_index("ix_prophet_alert_time_series_history_timestamp")

    op.drop_table("prophet_alert_time_series_history")
    # ### end Alembic commands ###
