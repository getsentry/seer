import hashlib
import hmac
import requests
from seer.automation.codegen.models import CodecovTaskRequest
from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected
from rest_framework.exceptions import AuthenticationFailed

INCOMING_REQUEST_SIGNATURE_HEADER = "X_HMAC_CODECOV_SIGNATURE"
OUTGOING_REQUEST_SIGNATURE_HEADER = "HTTP_X_GEN_AI_AUTH_SIGNATURE"


class CodecovAuthentication:
    @staticmethod
    @inject
    def authenticate_incoming_request(request: CodecovTaskRequest, config: AppConfig = injected):
        signature = request.headers.get(INCOMING_REQUEST_SIGNATURE_HEADER)
        key = config.CODECOV_INCOMING_SIGNATURE_SECRET

        if not key:
            raise AuthenticationFailed(
                "Cannot sign requests without CODECOV_OUTGOING_SIGNATURE_SECRET"
            )

        computed_sig = (
            "sha256="
            + hmac.new(key.encode("utf-8"), request.data, digestmod=hashlib.sha256).hexdigest()
        )

        if (
            computed_sig is None
            or signature is None
            or len(computed_sig) != len(signature)
            or computed_sig != signature
        ):
            raise AuthenticationFailed("Invalid signature")

    @staticmethod
    @inject
    def generate_codecov_request_signature(
        external_owner_id: str, repo_service_id: str, config: AppConfig = injected
    ) -> str:
        key = config.CODECOV_INCOMING_SIGNATURE_SECRET

        request_data = {
            "external_owner_id": external_owner_id,
            "repo_service_id": repo_service_id,
        }.encode("utf-8")

        if not key:
            raise AuthenticationFailed(
                "Cannot sign requests without CODECOV_OUTGOING_SIGNATURE_SECRET"
            )

        signature = (
            "sha256=" + hmac.new(key.encode("utf-8"), request_data, hashlib.sha256).hexdigest()
        )

        return signature

    @staticmethod
    def authenticate_codecov_app_install(external_owner_id: str, repo_service_id: str):
        generated_signature = CodecovAuthentication.generate_codecov_request_signature(
            external_owner_id, repo_service_id
        )
        url = "https://api.codecov.io/gen_ai/auth/"
        headers = {
            "Accept": "application/json",
            OUTGOING_REQUEST_SIGNATURE_HEADER: generated_signature,
        }
        payload = {
            "external_owner_id": external_owner_id,
            "repo_service_id": repo_service_id,
        }
        response = requests.post(url, headers=headers, json=payload)
        return response.json()["is_valid"] if response.status_code == 200 else False
