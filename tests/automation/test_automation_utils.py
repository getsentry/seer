import os
import unittest
from unittest.mock import patch

import pytest

from seer.automation.utils import ConsentError, check_genai_consent, raise_if_no_genai_consent


class TestCheckGenAiConsent(unittest.TestCase):
    @patch.dict(os.environ, {"NO_SENTRY_INTEGRATION": ""})
    @patch("seer.automation.utils.SentryRpcClient")
    def test_check_passing(self, mock_RpcClient):
        mock_RpcClient.return_value.call.return_value = {"consent": True}
        assert check_genai_consent(1) is True

    @patch.dict(os.environ, {"NO_SENTRY_INTEGRATION": ""})
    @patch("seer.automation.utils.SentryRpcClient")
    def test_check_failing(self, mock_RpcClient):
        mock_RpcClient.return_value.call.return_value = {"consent": False}
        assert check_genai_consent(1) is False

    @patch.dict(os.environ, {"NO_SENTRY_INTEGRATION": ""})
    @patch("seer.automation.utils.SentryRpcClient")
    def test_check_failing_none(self, mock_RpcClient):
        mock_RpcClient.return_value.call.return_value = None
        assert check_genai_consent(1) is False

    @patch.dict(os.environ, {"NO_SENTRY_INTEGRATION": "1"})
    @patch("seer.automation.utils.SentryRpcClient")
    def test_check_passing_without_integration(self, mock_RpcClient):
        mock_RpcClient.return_value.call.return_value = None

        assert check_genai_consent(1) is True
        mock_RpcClient.return_value.call.assert_not_called()


class TestCheckGenAiConsentRaise(unittest.TestCase):
    @patch("seer.automation.utils.check_genai_consent", return_value=False)
    def test_check_failing_raise(self, mock_check_genai_consent):
        with pytest.raises(ConsentError):
            raise_if_no_genai_consent(1)
