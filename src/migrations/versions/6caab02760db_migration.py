"""Migration

Revision ID: 6caab02760db
Revises: 9b8704bd8c4a
Create Date: 2024-08-12 22:12:45.057917

"""

import sqlalchemy as sa  # noqa: F401
from alembic import op

# revision identifiers, used by Alembic.
revision = "6caab02760db"
down_revision = "9b8704bd8c4a"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("run_state", schema=None) as batch_op:
        batch_op.create_index("ix_run_state_group_id", ["group_id"], unique=False)
        batch_op.create_index("ix_run_state_last_triggered_at", ["last_triggered_at"], unique=False)
        batch_op.create_index("ix_run_state_updated_at", ["updated_at"], unique=False)

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("run_state", schema=None) as batch_op:
        batch_op.drop_index("ix_run_state_updated_at")
        batch_op.drop_index("ix_run_state_last_triggered_at")
        batch_op.drop_index("ix_run_state_group_id")

    # ### end Alembic commands ###