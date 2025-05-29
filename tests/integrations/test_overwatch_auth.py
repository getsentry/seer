import hashlib
import hmac
import json
import unittest
from unittest.mock import MagicMock, patch

from integrations.overwatch.overwatch_auth import OverwatchAuthentication, get_overwatch_auth_header


class DummyConfigOutgoing:
    OVERWATCH_OUTGOING_SIGNATURE_SECRET = "secret_outgoing"


class TestOverwatchAuth(unittest.TestCase):
    def test_get_overwatch_auth_header(self):
        request_data = json.dumps({"test": "data"}).encode("utf-8")
        signature_header = "X-Test-Signature"
        signature_secret = "test_secret"

        # Calculate expected signature
        expected_hmac = hmac.new(
            signature_secret.encode("utf-8"), request_data, hashlib.sha256
        ).hexdigest()
        expected_signature = f"sha256={expected_hmac}"

        # Get actual header
        header = get_overwatch_auth_header(request_data, signature_header, signature_secret)

        # Verify header contents
        self.assertEqual(header["Content-Type"], "application/json")
        self.assertEqual(header[signature_header], expected_signature)

    @patch("integrations.overwatch.overwatch_auth.requests.post")
    def test_authenticate_overwatch_app_install_valid(self, mock_post):
        external_owner_id = "owner"
        data = {"external_owner_id": external_owner_id}
        request_data = json.dumps(data).encode("utf-8")
        key = DummyConfigOutgoing.OVERWATCH_OUTGOING_SIGNATURE_SECRET
        expected_signature = (
            "sha256=" + hmac.new(key.encode("utf-8"), request_data, hashlib.sha256).hexdigest()
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_valid": True}
        mock_post.return_value = mock_resp

        valid = OverwatchAuthentication.authenticate_overwatch_app_install(
            external_owner_id, config=DummyConfigOutgoing()
        )
        self.assertTrue(valid)
        args, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        self.assertEqual(headers.get("X-GEN-AI-AUTH-SIGNATURE"), expected_signature)

    @patch("integrations.overwatch.overwatch_auth.requests.post")
    def test_authenticate_overwatch_app_install_non_200(self, mock_post):
        external_owner_id = "owner"
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_post.return_value = mock_resp

        valid = OverwatchAuthentication.authenticate_overwatch_app_install(
            external_owner_id, config=DummyConfigOutgoing()
        )
        self.assertFalse(valid)

    @patch("integrations.overwatch.overwatch_auth.requests.post")
    def test_authenticate_overwatch_app_install_invalid_response(self, mock_post):
        external_owner_id = "owner"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_valid": False}
        mock_post.return_value = mock_resp

        valid = OverwatchAuthentication.authenticate_overwatch_app_install(
            external_owner_id, config=DummyConfigOutgoing()
        )
        self.assertFalse(valid)
