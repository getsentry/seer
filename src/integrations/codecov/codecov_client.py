import requests

from seer.configuration import AppConfig
from seer.dependency_injection import inject, injected


class CodecovClient:
    @staticmethod
    @inject
    def fetch_coverage(owner_username, repo_name, pullid, config: AppConfig = injected):
        token = config.CODECOV_SUPER_TOKEN
        url = f"https://api.codecov.io/api/v2/github/{owner_username}/repos/{repo_name}/pulls/{pullid}"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text
        else:
            return None

    @staticmethod
    @inject
    def fetch_test_results_for_commit(
        owner_username, repo_name, latest_commit_sha, config: AppConfig = injected
    ):
        token = config.CODECOV_SUPER_TOKEN
        url = f"https://api.codecov.io/api/v2/github/{owner_username}/repos/{repo_name}/test-results?commit_id={latest_commit_sha}&outcome=failure"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            if data["count"] == 0:
                return None
            return [
                {"name": r["name"], "failure_message": r["failure_message"]}
                for r in data["results"]
            ]
        return None
