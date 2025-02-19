import datetime

from seer.db import DbSeerEvent, Session
from seer.events import SeerEventNames, log_seer_event


def test_log_seer_event_basic():
    """Test basic event logging functionality."""
    # Test data
    event_name = SeerEventNames.AUTOFIX_RUN_ERROR
    event_metadata = {"error": "test error", "context": "test context"}

    # Log the event
    log_seer_event(event_name, event_metadata)

    # Verify the event was logged correctly
    with Session() as session:
        event = session.query(DbSeerEvent).order_by(DbSeerEvent.id.desc()).first()
        assert event is not None
        assert event.name == event_name
        assert event.event_metadata == event_metadata
        assert isinstance(event.created_at, datetime.datetime)


def test_log_seer_event_null_metadata():
    """Test event logging with null metadata."""
    event_name = SeerEventNames.AUTOFIX_RUN_ERROR
    event_metadata = None

    # Log the event
    log_seer_event(event_name, event_metadata)

    # Verify the event was logged correctly
    with Session() as session:
        event = session.query(DbSeerEvent).order_by(DbSeerEvent.id.desc()).first()
        assert event is not None
        assert event.name == event_name
        assert event.event_metadata is None
        assert isinstance(event.created_at, datetime.datetime)


def test_log_seer_event_multiple():
    """Test logging multiple events."""
    events_data = [
        (SeerEventNames.AUTOFIX_RUN_ERROR, {"error": "error 1"}),
        (SeerEventNames.AUTOFIX_RUN_ERROR, {"error": "error 2"}),
        (SeerEventNames.AUTOFIX_RUN_ERROR, {"error": "error 3"}),
    ]

    # Log multiple events
    for name, metadata in events_data:
        log_seer_event(name, metadata)

    # Verify all events were logged correctly
    with Session() as session:
        events = session.query(DbSeerEvent).order_by(DbSeerEvent.id.desc()).limit(3).all()
        events.reverse()  # Reverse to match input order

        assert len(events) == 3
        for i, event in enumerate(events):
            assert event.name == events_data[i][0]
            assert event.event_metadata == events_data[i][1]
            assert isinstance(event.created_at, datetime.datetime)


def test_log_seer_event_complex_metadata():
    """Test event logging with complex metadata structure."""
    event_name = SeerEventNames.AUTOFIX_RUN_ERROR
    event_metadata = {
        "error": {
            "type": "RuntimeError",
            "message": "Something went wrong",
            "stack_trace": ["line 1", "line 2", "line 3"],
        },
        "context": {
            "user_id": 123,
            "timestamp": "2024-03-20T10:00:00Z",
            "environment": "test",
            "tags": ["tag1", "tag2"],
        },
        "metrics": {
            "duration_ms": 1500,
            "memory_usage_mb": 256,
            "cpu_percent": 45.6,
        },
    }

    # Log the event
    log_seer_event(event_name, event_metadata)

    # Verify the event was logged correctly
    with Session() as session:
        event = session.query(DbSeerEvent).order_by(DbSeerEvent.id.desc()).first()
        assert event is not None
        assert event.name == event_name
        assert event.event_metadata == event_metadata
        assert isinstance(event.created_at, datetime.datetime)

        # Verify nested structure is preserved
        assert event.event_metadata["error"]["type"] == "RuntimeError"
        assert len(event.event_metadata["error"]["stack_trace"]) == 3
        assert event.event_metadata["context"]["user_id"] == 123
        assert event.event_metadata["metrics"]["duration_ms"] == 1500
