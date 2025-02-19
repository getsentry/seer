from enum import StrEnum

from seer.db import DbSeerEvent, Session


class SeerEventNames(StrEnum):
    AUTOFIX_RUN_ERROR = (
        "autofix_run_error"  # An error occurred and was logged to the user while running an autofix
    )


def log_seer_event(name: SeerEventNames, event_metadata: dict | None = None):
    """Log a Seer event to the database.

    Args:
        name: The name/type of the event being logged
        event_metadata: A dictionary containing additional event data/context

    The event will be persisted to the seer_metric_events table with:
    - A unique auto-incrementing ID
    - The provided name and event_metadata
    - An automatically set created_at timestamp
    """
    with Session() as session:
        event = DbSeerEvent(name=name, event_metadata=event_metadata)
        session.add(event)
        session.commit()
