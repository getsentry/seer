"""Migration

Revision ID: 1cfb0e2cc1f5
Revises: 95b4ba4f731d
Create Date: 2024-07-25 20:16:44.797997

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "1cfb0e2cc1f5"
down_revision = "95b4ba4f731d"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table(
        "dynamic_alerts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("organization_id", sa.BigInteger(), nullable=False),
        sa.Column("project_id", sa.BigInteger(), nullable=False),
        sa.Column("external_alert_id", sa.BigInteger(), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_alert_id"),
    )
    with op.batch_alter_table("dynamic_alerts", schema=None) as batch_op:
        batch_op.create_index(
            "ix_dynamic_alert_external_alert_id", ["external_alert_id"], unique=False
        )

    op.create_table(
        "dynamic_alert_time_series",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dynamic_alert_id", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.TIMESTAMP(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dynamic_alert_id"], ["dynamic_alerts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dynamic_alert_id", "timestamp"),
    )
    with op.batch_alter_table("dynamic_alert_time_series", schema=None) as batch_op:
        batch_op.create_index(
            "ix_dynamic_alert_time_series_alert_id_timestamp",
            ["dynamic_alert_id", "timestamp"],
            unique=False,
        )

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("dynamic_alert_time_series", schema=None) as batch_op:
        batch_op.drop_index("ix_dynamic_alert_time_series_alert_id_timestamp")

    op.drop_table("dynamic_alert_time_series")
    with op.batch_alter_table("dynamic_alerts", schema=None) as batch_op:
        batch_op.drop_index("ix_dynamic_alert_external_alert_id")

    op.drop_table("dynamic_alerts")
    # ### end Alembic commands ###