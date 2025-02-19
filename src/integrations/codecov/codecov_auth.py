import hashlib
import hmac
import json
from typing import Any

import requests

from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected

OUTGOING_REQUEST_SIGNATURE_HEADER = "HTTP-X-GEN-AI-AUTH-SIGNATURE"


@inject
def get_codecov_auth_header(
    request: dict[str, Any], config: AppConfig = injected
) -> dict[str, str]:
    request_data = json.dumps(request).encode("utf-8")
    key = config.CODECOV_OUTGOING_SIGNATURE_SECRET
    hmac_ = hmac.new(key.encode("utf-8"), request_data, hashlib.sha256).hexdigest()
    signature = f"sha256={hmac_}"
    return {
        "Content-Type": "application/json",
        OUTGOING_REQUEST_SIGNATURE_HEADER: signature,
    }


class CodecovAuthentication:
    @staticmethod
    @inject
    def authenticate_codecov_app_install(
        external_owner_id: str, repo_service_id: str, config: AppConfig = injected
    ) -> bool:
        request = {"external_owner_id": external_owner_id, "repo_service_id": repo_service_id}
        headers = get_codecov_auth_header(request, config=config)
        response = requests.post(
            url="https://api.codecov.io/gen_ai/auth/", headers=headers, json=request
        )
        return response.json().get("is_valid", False) if response.status_code == 200 else False
