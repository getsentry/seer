import requests

class CodecovClient:
    @staticmethod
    def fetch_coverage(owner_username, repo_name, pullid, token=''):
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
    def ping():
        return "pong"

