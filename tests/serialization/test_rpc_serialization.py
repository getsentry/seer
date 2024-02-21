import json
import unittest
from unittest.mock import patch

from seer.rpc import SentryRpcClient
from seer.utils import SeerJSONEncoder, json_dumps
from seer.automation.autofix.event_manager import AutofixStatus


class TestRpcSerialization(unittest.TestCase):

    def test_rpc_method_call_serialization_with_custom_encoder(self):
        """
        Test that RpcClient.call serializes AutofixStatus enum correctly using SeerJSONEncoder.
        """
        test_status = AutofixStatus.PROCESSING  # Assumes AutofixStatus enum is accessible
        test_issue_id = 123

        with patch('src.seer.rpc.json_dumps', side_effect=json.dumps) as patched_json_dumps:
            # Patching json_dumps to validate its call with custom encoder
            client = SentryRpcClient(base_url='https://dummy.url')
            client.call('on_autofix_step_update', issue_id=test_issue_id, status=test_status)

            patched_json_dumps.assert_called_once_with(
                {'args': {'issue_id': test_issue_id, 'status': test_status}},
                separators=(',', ':'),
                default=lambda o: o.value if hasattr(o, 'value') else json.JSONEncoder().default(o)
            )

