import logging

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.codegen.models import CodegenContinuation
from seer.automation.codegen.state import CodegenContinuationState
from seer.automation.models import RepoDefinition
from seer.automation.pipeline import PipelineContext

logger = logging.getLogger(__name__)

RepoExternalId = str
RepoInternalId = int
RepoKey = RepoExternalId | RepoInternalId
RepoIdentifiers = tuple[RepoExternalId, RepoInternalId]


class CodegenContext(PipelineContext):
    state: CodegenContinuationState
    repo: RepoDefinition

    def __init__(
        self,
        state: CodegenContinuationState,
    ):
        request = state.get().request

        self.repo = request.repo
        self.state = state

        logger.info(f"CodegenContext initialized with run_id {self.run_id}")

    @classmethod
    def from_run_id(cls, run_id: int):
        state = CodegenContinuationState.from_id(run_id, model=CodegenContinuation)
        with state.update() as cur:
            cur.mark_triggered()

        return cls(state)

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

    def get_repo_client(self):
        """
        Gets a repo client for the current single repo or for a given repo name.
        If there are more than 1 repos, a repo name must be provided.
        """
        return RepoClient.from_repo_definition(self.repo, "read")
