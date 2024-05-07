"""Migration

Revision ID: 34eef02b2555
Revises: 1939547a1261
Create Date: 2024-04-25 22:20:11.942620

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "34eef02b2555"
down_revision = "1939547a1261"
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("codebase_namespaces", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("status", sa.String(), nullable=False, server_default="created")
        )
    # Set default value for existing rows
    op.alter_column("codebase_namespaces", "status", server_default=None)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table("codebase_namespaces", schema=None) as batch_op:
        batch_op.drop_column("status")

    # ### end Alembic commands ###