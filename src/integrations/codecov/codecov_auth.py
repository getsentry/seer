import hashlib
import hmac
import json

import requests

from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


def get_codecov_auth_header(
    request_data: bytes, signature_header: str, signature_secret: str
) -> dict[str, str]:
    """
    `request_data` is expected to be utf-8 encoded.
    """
    hmac_ = hmac.new(signature_secret.encode("utf-8"), request_data, hashlib.sha256).hexdigest()
    signature = f"sha256={hmac_}"
    return {"Content-Type": "application/json", signature_header: signature}


class CodecovAuthentication:
    @staticmethod
    @inject
    def authenticate_codecov_app_install(
        external_owner_id: str, repo_service_id: str, config: AppConfig = injected
    ) -> bool:
        request = {"external_owner_id": external_owner_id, "repo_service_id": repo_service_id}
        request_data = json.dumps(request).encode("utf-8")
        headers = get_codecov_auth_header(
            request_data,
            signature_header="HTTP-X-GEN-AI-AUTH-SIGNATURE",
            signature_secret=config.CODECOV_OUTGOING_SIGNATURE_SECRET,
        )
        response = requests.post(
            url="https://api.codecov.io/gen_ai/auth/", headers=headers, data=request_data
        )
        return response.json().get("is_valid", False) if response.status_code == 200 else False
