import logging

from seer.automation.agent.models import Message
from seer.automation.codegen.codegen_event_manager import CodegenEventManager
from seer.automation.codegen.models import CodegenContinuation, UnitTestRunMemory
from seer.automation.codegen.state import CodegenContinuationState
from seer.automation.context import BasePipelineContext
from seer.automation.models import RepoDefinition
from seer.automation.state import DbStateRunTypes
from seer.db import DbPrContextToUnitTestGenerationRunIdMapping, DbRunMemory, Session

logger = logging.getLogger(__name__)

RepoExternalId = str
RepoInternalId = int
RepoKey = RepoExternalId | RepoInternalId
RepoIdentifiers = tuple[RepoExternalId, RepoInternalId]


class CodegenContext(BasePipelineContext):
    state: CodegenContinuationState
    event_manager: CodegenEventManager
    repo: RepoDefinition

    def __init__(
        self,
        repo: RepoDefinition,
        state: CodegenContinuationState,
    ):
        request = state.get().request

        self.repo = repo
        self.state = state
        self.event_manager = CodegenEventManager(state)

        logger.info(f"CodegenContext initialized with run_id {self.run_id}")

    @property
    def repos(self) -> list[RepoDefinition]:
        return [self.repo]

    @classmethod
    def from_run_id(cls, run_id: int, type: DbStateRunTypes = DbStateRunTypes.UNIT_TEST):
        state = CodegenContinuationState(run_id, model=CodegenContinuation, type=type)
        with state.update() as cur:
            cur.mark_triggered()

        return cls(state)

    def get_file_contents(
        self, path: str, repo_name: str | None = None, ignore_local_changes: bool = False
    ) -> str | None:
        repo_client = self.get_repo_client(repo_name=repo_name)

        file_contents, _ = repo_client.get_file_content(path)

        if not ignore_local_changes:
            cur_state = self.state.get()
            current_file_changes = list(filter(lambda x: x.path == path, cur_state.file_changes))
            for file_change in current_file_changes:
                file_contents = file_change.apply(file_contents)

        return file_contents

    def store_memory(self, key: str, memory: list[Message]):
        with Session() as session:
            memory_record = (
                session.query(DbRunMemory).where(DbRunMemory.run_id == self.run_id).one_or_none()
            )

            if not memory_record:
                memory_model = UnitTestRunMemory(run_id=self.run_id)
            else:
                memory_model = UnitTestRunMemory.from_db_model(memory_record)

            memory_model.memory[key] = memory
            memory_record = memory_model.to_db_model()

            session.merge(memory_record)
            session.commit()

    def update_stored_memory(self, key: str, memory: list[Message], original_run_id: int):
        with Session() as session:
            memory_record = (
                session.query(DbRunMemory)
                .where(DbRunMemory.run_id == original_run_id)
                .one_or_none()
            )

            if not memory_record:
                raise RuntimeError(
                    f"No memory record found for run_id {original_run_id}. Cannot update stored memory."
                )
            else:
                memory_model = UnitTestRunMemory.from_db_model(memory_record)

            memory_model.memory[key] = memory
            memory_record = memory_model.to_db_model()

            session.merge(memory_record)
            session.commit()

    def get_memory(self, key: str, past_run_id: int) -> list[Message]:
        with Session() as session:
            memory_record = (
                session.query(DbRunMemory).where(DbRunMemory.run_id == past_run_id).one_or_none()
            )

            if not memory_record:
                return []

            return UnitTestRunMemory.from_db_model(memory_record).memory.get(key, [])

    def get_previous_run_context(
        self, owner: str, repo: str, pr_id: int
    ) -> DbPrContextToUnitTestGenerationRunIdMapping | None:
        with Session() as session:
            previous_context = (
                session.query(DbPrContextToUnitTestGenerationRunIdMapping)
                .where(DbPrContextToUnitTestGenerationRunIdMapping.owner == owner)
                .where(DbPrContextToUnitTestGenerationRunIdMapping.repo == repo)
                .where(DbPrContextToUnitTestGenerationRunIdMapping.pr_id == pr_id)
                .one_or_none()
            )

            return previous_context
