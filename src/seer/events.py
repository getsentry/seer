from enum import StrEnum

from seer.db import DbSeerEvent, Session


class SeerEventNames(StrEnum):
    AUTOFIX_RUN_ERROR = (
        "autofix_run_error"  # An error occurred and was logged to the user while running an autofix
    )
    AUTOFIX_ASKED_USER_QUESTION = "autofix_asked_user_question"  # Autofix asked the user a question
    AUTOFIX_USER_QUESTION_RESPONSE_RECEIVED = (
        "autofix_user_question_response_received"  # Autofix received a response from the user
    )
    AUTOFIX_RESTARTED_FROM_POINT = "autofix_restarted_from_point"  # Autofix restarted from a point
    AUTOFIX_ROOT_CAUSE_STARTED = (
        "autofix_root_cause_started"  # Autofix started to find the root cause of the issue
    )
    AUTOFIX_ROOT_CAUSE_COMPLETED = (
        "autofix_root_cause_completed"  # Autofix found the root cause of the issue
    )
    AUTOFIX_SOLUTION_STARTED = "autofix_solution_started"  # Autofix started to find a solution
    AUTOFIX_SOLUTION_COMPLETED = "autofix_solution_completed"  # Autofix found a solution
    AUTOFIX_CODING_STARTED = "autofix_coding_started"  # Autofix started to code the solution
    AUTOFIX_CODING_COMPLETED = "autofix_coding_completed"  # Autofix completed coding the solution
    AUTOFIX_COMPLETED = "autofix_completed"  # Autofix full run completed successfully

    COMPARATIVE_WORKFLOWS_STARTED = "comparative_workflows_started"  # Comparative workflows started


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
