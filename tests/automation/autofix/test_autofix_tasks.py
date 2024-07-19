import pytest
from johen import generate

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.autofix.tasks import get_autofix_state, get_autofix_state_from_pr_id
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session


class TestGetAutofixState:
    def test_get_state_by_group_id(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=100, value=state.model_dump(mode="json")))
            session.commit()

        retrieved_state = get_autofix_state(group_id=100)
        assert retrieved_state is not None
        if retrieved_state is not None:
            assert retrieved_state.get() == state

    def test_get_state_by_run_id(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=2, group_id=200, value=state.model_dump(mode="json")))
            session.commit()

        retrieved_state = get_autofix_state(run_id=2)
        assert retrieved_state is not None
        if retrieved_state is not None:
            assert retrieved_state.get() == state

    def test_get_state_no_matching_group_id(self):
        retrieved_state = get_autofix_state(group_id=999)
        assert retrieved_state is None

    def test_get_state_no_matching_run_id(self):
        retrieved_state = get_autofix_state(run_id=999)
        assert retrieved_state is None

    def test_get_state_multiple_runs_for_group(self):
        states = [next(generate(AutofixContinuation)) for _ in range(3)]
        with Session() as session:
            for i, state in enumerate(states, start=1):
                session.add(DbRunState(id=i, group_id=300, value=state.model_dump(mode="json")))
            session.commit()

        retrieved_state = get_autofix_state(group_id=300)
        assert retrieved_state is not None
        if retrieved_state is not None:
            # Should return the most recent state (highest id)
            assert retrieved_state.get() == states[-1]

    def test_get_state_no_parameters(self):
        with pytest.raises(ValueError, match="Either group_id or run_id must be provided"):
            get_autofix_state()

    def test_get_state_both_parameters(self):
        with pytest.raises(
            ValueError, match="Either group_id or run_id must be provided, not both"
        ):
            get_autofix_state(group_id=1, run_id=1)


class TestGetStateFromPr:
    def test_successful_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 1)
        assert retrieved_state is not None
        if retrieved_state is not None:
            assert retrieved_state.get() == state

    def test_no_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 2)
        assert retrieved_state is None
