import logging
import time
from github.PullRequest import PullRequest
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import FileChange
from seer.db import DbPrContextToUnitTestGenerationRunIdMapping, Session

logger = logging.getLogger(__name__)


class RetryUnitTestGithubPrCreator:
    def __init__(
        self,
        file_changes_payload: list[FileChange],
        pr: PullRequest,
        repo_client: RepoClient,
        unit_test_run_id: int,
        previous_context: DbPrContextToUnitTestGenerationRunIdMapping,
    ):
        self.file_changes_payload = file_changes_payload
        self.pr = pr
        self.repo_client = repo_client
        self.unit_test_run_id = unit_test_run_id
        self.previous_context = previous_context

    def update_github_pull_request(self):
        self.repo_client.base_commit_sha = self.pr.head.sha
        commit_messages = [f"- {change.commit_message}" for change in self.file_changes_payload]
        commit_msg = f"Update tests for PR#{self.pr.number} at {int(time.time())}"
        new_commit = self.repo_client.push_new_commit_to_pr(
            pr=self.pr,
            commit_message=commit_msg,
            file_changes=self.file_changes_payload,
        )
        if not new_commit:
            logger.warning("Failed to push new commit to PR")
            return
        new_description = (
            f"This PR has been updated to add tests for #{self.pr.number}\n\n"
            "### Commits:\n" + "\n".join(commit_messages)
        )
        self.repo_client.push_new_commit_to_pr(
            pr=self.pr,
            commit_message=new_description,
            file_changes=self.file_changes_payload,
        )
        self.update_stored_pr_context(self.previous_context)

    def update_stored_pr_context(
        self, previous_context: DbPrContextToUnitTestGenerationRunIdMapping
    ):
        with Session() as session:
            previous_context.iterations += 1
            session.commit()
