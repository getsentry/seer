"""Insert missing namespace_id into codebase_namespaces table.

Revision ID: insert_missing_namespace_id
Revises: 1939547a1261
Create Date: 2024-06-27 15:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'insert_missing_namespace_id'
down_revision = '1939547a1261'
branch_labels = None
depends_on = None

def upgrade():
    # Insert the missing namespace_id into the codebase_namespaces table
    op.execute(
        """
        INSERT INTO codebase_namespaces (id, repo_id, sha, updated_at, accessed_at)
        VALUES (88, 1, 'dummy_sha', NOW(), NOW())
        ON CONFLICT (id) DO NOTHING;
        """
    )

def downgrade():
    # Remove the inserted namespace_id if necessary
    op.execute(
        """
        DELETE FROM codebase_namespaces WHERE id = 88;
        """
    )