"""chore(anomaly_detection): update the source ids for existing alerts

Revision ID: 1c4ab1bd7b98
Revises: 21051ea8b612
Create Date: 2025-06-02 13:04:00.883401

"""

import csv

from alembic import op

# revision identifiers, used by Alembic.
revision = "1c4ab1bd7b98"
down_revision = "21051ea8b612"
branch_labels = None
depends_on = None


def upgrade():
    with open("/app/src/migrations/ad_alert_ids_to_source_ids_2025_06_02.csv", "r") as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i == 0:
                # Skip the header row
                continue
            external_alert_id = int(row[0])
            external_source_id = int(row[1])
            external_source_type = 1
            op.execute(
                f"UPDATE dynamic_alerts SET external_alert_source_id = {external_source_id}, external_alert_source_type = {external_source_type} WHERE external_alert_id = {external_alert_id}"
            )
            op.execute(
                f"UPDATE dynamic_alert_time_series_history SET external_alert_source_id = {external_source_id}, external_alert_source_type = {external_source_type} WHERE alert_id = {external_alert_id}"
            )
            op.execute(
                f"UPDATE prophet_alert_time_series_history SET external_alert_source_id = {external_source_id}, external_alert_source_type = {external_source_type} WHERE alert_id = {external_alert_id}"
            )


def downgrade():
    # This is a one-way migration. No need for a downgrade as the ids never change.
    pass
