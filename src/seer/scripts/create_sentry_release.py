import dataclasses
import json
import os
import sys
import urllib
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlencode

from urllib3 import HTTPResponse

from seer.scripts.utils import DataclassArgumentParser


@dataclasses.dataclass
class ReleaseConfig:
    sentry_org: str = dataclasses.field(
        metadata=dict(help="Sentry org name to create the release in")
    )
    sentry_project: str = dataclasses.field(
        metadata=dict(help="Sentry project name to create the release in")
    )
    sha: str

    environment: str = "production"
    api_domain: str = dataclasses.field(
        metadata=dict(help="Regional domain for the sentry API"), default="sentry.io"
    )
    dry_run: bool = False


@dataclasses.dataclass
class ReleaseContext:
    version: str
    ref: dict


class SentryApiClient:
    def get_project_id(self, config: ReleaseConfig) -> int:
        return 1

    def get_latest_sha(self, config: ReleaseConfig, project_id: int) -> None:
        return None

    def create_and_deploy_release(self, config: ReleaseConfig, context: "ReleaseContext"):
        pass


@dataclasses.dataclass
class SentryApiClientImpl(SentryApiClient):
    auth: str

    def get_project_id(self, config: ReleaseConfig) -> int:
        resp = self._make_request(
            "GET",
            f"https://{config.api_domain}/api/0/projects/{config.sentry_org}/{config.sentry_project}/",
        )
        return int(resp["id"])

    def _make_request(self, method: str, url: str, body: dict | None = None) -> Any:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8") if body else None,
            method=method,
            headers={
                "Authorization": f"Bearer {self.auth}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
        )
        try:
            resp: HTTPResponse = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            print(f"Sentry Api Error for {method} {url}:\n{e.read().decode()}")
            raise

        assert (
            200 <= resp.status < 300
        ), f"Sentry Api HTTP Error code: {resp.status} for {method} {url}:\n{resp.read()}"
        result = json.loads(resp.read())
        return result

    def get_latest_sha(self, config: ReleaseConfig, project_id: int) -> str | None:
        query = urlencode({"project": project_id, "environment": config.environment})

        resp = self._make_request(
            "GET",
            f"https://{config.api_domain}/api/0/organizations/{config.sentry_org}/releases/?{query}",
        )

        if len(resp) == 0:
            return None

        resp.sort(
            key=lambda r: (r.get("lastDeploy", {}).get("dateFinished", "z"), r["dateCreated"])
        )
        return resp[0]["versionInfo"]["buildHash"]

    def create_and_deploy_release(self, config: ReleaseConfig, context: "ReleaseContext"):
        self._make_request(
            "POST",
            f"https://{config.api_domain}/api/0/organizations/{config.sentry_org}/releases/",
            {
                "version": context.version,
                "refs": [context.ref],
                "projects": [config.sentry_project],
            },
        )

        if not config.dry_run:
            self._make_request(
                "POST",
                f"https://{config.api_domain}/api/0/organizations/{config.sentry_org}/releases/{context.version}/deploys/",
                {
                    "environment": config.environment,
                    "name": f"{config.sha} to {config.environment}"[:64],
                },
            )


def prepare_release_context(config: ReleaseConfig, client: SentryApiClient) -> ReleaseContext:
    project_id = client.get_project_id(config)
    print(f"Determined project id: {project_id}")
    prev_deploy = client.get_latest_sha(config, project_id)
    print(f"Found previous deploy sha: {prev_deploy}")

    return ReleaseContext(
        version=f"seer@{config.sha}",
        ref={
            "repository": "getsentry/seer",
            "commit": config.sha,
            "previousCommit": prev_deploy,
        },
    )


def main(config: ReleaseConfig, client: SentryApiClient | None = None) -> int:
    if client is None:
        client = SentryApiClientImpl(os.environ["SENTRY_AUTH_TOKEN"])
    assert client
    context = prepare_release_context(config, client)
    client.create_and_deploy_release(config, context)


if __name__ == "__main__":
    main(DataclassArgumentParser(ReleaseConfig).parse_args(sys.argv[1:]))
