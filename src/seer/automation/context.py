from abc import ABC, abstractmethod
from typing import Any

from seer.automation.agent.models import Message
from seer.automation.codebase.repo_client import (
    RepoClient,
    RepoClientType,
    autocorrect_repo_name,
    get_repo_client,
)
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineContext


class BasePipelineContext(PipelineContext, ABC):
    state: Any  # Child class specific state
    event_manager: Any
    repos: list[RepoDefinition]

    @property
    def run_id(self) -> int:
        return self.state.get().run_id

    @property
    def signals(self) -> list[str]:
        return self.state.get().signals

    @signals.setter
    def signals(self, value: list[str]):
        with self.state.update() as state:
            state.signals = value

    def get_repo_client(
        self,
        repo_name: str | None = None,
        repo_external_id: str | None = None,
        type: RepoClientType = RepoClientType.READ,
    ) -> RepoClient:
        return get_repo_client(
            repos=self.repos, repo_name=repo_name, repo_external_id=repo_external_id, type=type
        )

    def autocorrect_repo_name(self, repo_name: str) -> str | None:
        return autocorrect_repo_name(
            readable_repos=self.state.get().readable_repos, repo_name=repo_name
        )

    @abstractmethod
    def get_file_contents(
        self, path: str, repo_name: str | None = None, ignore_local_changes: bool = False
    ) -> str | None:
        pass

    def does_file_exist(
        self, path: str, repo_name: str | None = None, ignore_local_changes: bool = False
    ) -> bool:
        if len(self.repos) > 1:
            if not repo_name:
                raise ValueError("Repo name is required when there are multiple repos.")

            if repo_name not in [repo.full_name for repo in self.repos]:
                raise ValueError(f"Repo '{repo_name}' not found in the list of repos.")

        repo_client = self.get_repo_client(repo_name)
        does_exist_on_remote = repo_client.does_file_exist(path)
        if does_exist_on_remote:
            return True

        if not ignore_local_changes:
            cur_state = self.state.get()
            repo_file_changes = cur_state.codebases[repo_client.repo_external_id].file_changes
            current_file_changes = list(
                filter(lambda x: x.path == path and x.change_type == "create", repo_file_changes)
            )
            if current_file_changes:
                return True

        return False

    @abstractmethod
    def store_memory(self, key: str, memory: list[Message]):
        pass
