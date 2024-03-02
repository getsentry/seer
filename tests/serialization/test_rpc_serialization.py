import json
import unittest
from unittest.mock import patch
from seer.rpc import SentryRpcClient
from seer.utils import SeerJSONEncoder, json_dumps
from seer.automation.autofix.event_manager import AutofixStatus


class TestRpcSerialization(unittest.TestCase):
    def test_json_encoder(self):
        """
        Test that SeerJSONEncoder serializes AutofixStatus enum correctly.
        """
        test_status = AutofixStatus.PROCESSING

        serialized = json.dumps(test_status, cls=SeerJSONEncoder)

        self.assertEqual(serialized, json.dumps(test_status.value))
