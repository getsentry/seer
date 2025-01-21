import unittest, hashlib, hmac, json, requests
from unittest.mock import MagicMock, patch
from werkzeug.exceptions import Unauthorized
from integrations.codecov.codecov_auth import CodecovAuthentication


class DummyRequestData:
    def __init__(self, json_str):
        self.json_str = json_str

    def model_dump_json(self):
        return self.json_str


class DummyTaskRequest:
    def __init__(self, json_str):
        self.data = DummyRequestData(json_str)


class DummyConfigOutgoing:
    CODECOV_OUTGOING_SIGNATURE_SECRET = "secret_outgoing"


class TestCodecovAuth(unittest.TestCase):
    @patch("integrations.codecov.codecov_auth.requests.post")
    def test_authenticate_codecov_app_install_valid(self, mock_post):
        external_owner_id = "owner"
        repo_service_id = "repo"
        data = {"external_owner_id": external_owner_id, "repo_service_id": repo_service_id}
        request_data = json.dumps(data).encode("utf-8")
        key = DummyConfigOutgoing.CODECOV_OUTGOING_SIGNATURE_SECRET
        expected_signature = (
            "sha256=" + hmac.new(key.encode("utf-8"), request_data, hashlib.sha256).hexdigest()
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"is_valid": True}
        mock_post.return_value = mock_resp

        valid = CodecovAuthentication.authenticate_codecov_app_install(
            external_owner_id, repo_service_id, config=DummyConfigOutgoing()
        )
        self.assertTrue(valid)
        args, kwargs = mock_post.call_args
        headers = kwargs.get("headers", {})
        self.assertEqual(headers.get("HTTP-X-GEN-AI-AUTH-SIGNATURE"), expected_signature)

    @patch("integrations.codecov.codecov_auth.requests.post")
    def test_authenticate_codecov_app_install_non_200(self, mock_post):
        external_owner_id = "owner"
        repo_service_id = "repo"
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_post.return_value = mock_resp

        valid = CodecovAuthentication.authenticate_codecov_app_install(
            external_owner_id, repo_service_id, config=DummyConfigOutgoing()
        )
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
