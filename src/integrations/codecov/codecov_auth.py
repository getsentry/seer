import hashlib
import hmac
import json
import requests
from seer.automation.codegen.models import CodecovTaskRequest
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected
from werkzeug.exceptions import Unauthorized

INCOMING_REQUEST_SIGNATURE_HEADER = "X-HMAC-CODECOV-SIGNATURE"
OUTGOING_REQUEST_SIGNATURE_HEADER = "HTTP-X-GEN-AI-AUTH-SIGNATURE"


class CodecovAuthentication:
    @staticmethod
    @inject
    def authenticate_incoming_request(
        signature: str, data: CodecovTaskRequest, config: AppConfig = injected
    ):
        key = config.CODECOV_INCOMING_SIGNATURE_SECRET

        if not key:
            raise Unauthorized("Cannot sign requests without CODECOV_INCOMING_SIGNATURE_SECRET")

        computed_sig = (
            "sha256="
            + hmac.new(
                key.encode("utf-8"),
                data.data.model_dump_json().encode("utf-8"),
                digestmod=hashlib.sha256,
            ).hexdigest()
        )

        if (
            computed_sig is None
            or signature is None
            or len(computed_sig) != len(signature)
            or computed_sig != signature
        ):
            raise Unauthorized("Invalid signature")

    @staticmethod
    @inject
    def authenticate_codecov_app_install(
        external_owner_id: str, repo_service_id: str, config: AppConfig = injected
    ) -> bool:
        data = {"external_owner_id": external_owner_id, "repo_service_id": repo_service_id}
        request_data = json.dumps(data).encode("utf-8")
        key = config.CODECOV_OUTGOING_SIGNATURE_SECRET
        signature = (
            "sha256=" + hmac.new(key.encode("utf-8"), request_data, hashlib.sha256).hexdigest()
        )

        headers = {
            "Content-Type": "application/json",
            OUTGOING_REQUEST_SIGNATURE_HEADER: signature,
        }
        response = requests.post(
            "http://api.codecov.io/gen_ai/auth/", headers=headers, data=request_data
        )
        return response.json().get("is_valid", False) if response.status_code == 200 else False
