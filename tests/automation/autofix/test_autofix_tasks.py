import unittest

from johen import generate

from seer.automation.autofix.models import AutofixContinuation
from seer.automation.autofix.tasks import get_autofix_state_from_pr_id
from seer.db import DbPrIdToAutofixRunIdMapping, DbRunState, Session


class TestGetStateFromPr(unittest.TestCase):
    def test_successful_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 1)
        self.assertIsNotNone(retrieved_state)
        if retrieved_state is not None:
            self.assertEqual(retrieved_state.get(), state)

    def test_no_state_mapping(self):
        state = next(generate(AutofixContinuation))
        with Session() as session:
            session.add(DbRunState(id=1, group_id=1, value=state.model_dump(mode="json")))
            session.flush()
            session.add(DbPrIdToAutofixRunIdMapping(provider="test", pr_id=1, run_id=1))
            session.commit()

        retrieved_state = get_autofix_state_from_pr_id("test", 2)
        self.assertIsNone(retrieved_state)
