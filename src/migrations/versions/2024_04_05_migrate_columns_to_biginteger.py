\"\"\"\nMigration to alter organization and project columns to BigInteger\n\nRevision ID: 2024_04_05_migrate_columns_to_biginteger\nRevises: fc5a5f2d1078\nCreate Date: 2024-04-05 10:00:00.000000\n\n\"\"\"
import sqlalchemy as sa
from alembic import op

def upgrade():
    op.alter_column('repositories', 'organization',
                    existing_type=sa.Integer(),
                    type_=sa.BigInteger(),
                    existing_nullable=False)
    op.alter_column('repositories', 'project',
                    existing_type=sa.Integer(),
                    type_=sa.BigInteger(),
                    existing_nullable=False)

def downgrade():
    op.alter_column('repositories', 'organization',
                    existing_type=sa.BigInteger(),
                    type_=sa.Integer(),
                    existing_nullable=False)
    op.alter_column('repositories', 'project',
                    existing_type=sa.BigInteger(),
                    type_=sa.Integer(),
                    existing_nullable=False)
