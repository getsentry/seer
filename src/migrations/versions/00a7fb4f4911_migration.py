"""Migration

Revision ID: 00a7fb4f4911
Revises: 96e56e375579
Create Date: 2024-08-27 21:59:23.740336

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "00a7fb4f4911"
down_revision = "96e56e375579"
branch_labels = None
depends_on = None


def upgrade():
    # Clear out existing data
    connection = op.get_bind()
    connection.execute(sa.text("TRUNCATE TABLE dynamic_alert_time_series RESTART IDENTITY"))
    connection.execute(sa.text("TRUNCATE TABLE dynamic_alerts RESTART IDENTITY CASCADE"))

    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("dynamic_alert_time_series", schema=None) as batch_op:
        batch_op.add_column(sa.Column("anomaly_type", sa.String(), nullable=False))
        batch_op.add_column(sa.Column("anomaly_score", sa.Float(), nullable=False))
        batch_op.add_column(sa.Column("anomaly_algo_data", sa.JSON(), nullable=True))

    with op.batch_alter_table("dynamic_alerts", schema=None) as batch_op:
        batch_op.add_column(sa.Column("anomaly_algo_data", sa.JSON(), nullable=False))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("dynamic_alerts", schema=None) as batch_op:
        batch_op.drop_column("anomaly_algo_data")

    with op.batch_alter_table("dynamic_alert_time_series", schema=None) as batch_op:
        batch_op.drop_column("anomaly_algo_data")
        batch_op.drop_column("anomaly_score")
        batch_op.drop_column("anomaly_type")

    # ### end Alembic commands ###
