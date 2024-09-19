import requests


CODECOV_TOKEN = 'FETCH FROM ENV'
class CodecovClient:
    @staticmethod
    def fetch_coverage(owner_username, repo_name, pullid, token=CODECOV_TOKEN):
        url = f"https://api.codecov.io/api/v2/github/{owner_username}/repos/{repo_name}/pulls/{pullid}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json"
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    @staticmethod
    def fetch_test_results_for_commit(owner_username, repo_name, latest_commit_sha, token=CODECOV_TOKEN):
        url = f"https://api.codecov.io/api/v2/github/{owner_username}/repos/{repo_name}/test-results"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            latest_commit_sha: latest_commit_sha,
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()


    @staticmethod
    def ping():
        return "pong"

