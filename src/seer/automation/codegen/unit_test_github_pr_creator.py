import logging
import time

from github.PullRequest import PullRequest

from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import FileChange
from seer.db import DbPrContextToUnitTestGenerationRunIdMapping, Session

logger = logging.getLogger(__name__)


class GeneratedTestsPullRequestCreator:
    def __init__(
        self,
        file_changes_payload: list[FileChange],
        pr: PullRequest,
        repo_client: RepoClient,
        unit_test_run_id: int,
    ):
        self.file_changes_payload = file_changes_payload
        self.pr = pr
        self.repo_client = repo_client
        self.unit_test_run_id = unit_test_run_id

    def create_github_pull_request(self):
        self.repo_client.base_commit_sha = self.pr.head.sha

        branch_name = f"ai_tests_for_pr{self.pr.number}_{int(time.time())}"
        pr_title = f"Add Tests for PR#{self.pr.number}"

        commit_messages = []
        for change in self.file_changes_payload:
            commit_messages.append(f"- {change.commit_message}")
        branch_ref = self.repo_client.create_branch_from_changes(
            pr_title=pr_title,
            file_changes=self.file_changes_payload,
            branch_name=branch_name,
        )

        if not branch_ref:
            logger.warning("Failed to create branch from changes")
            return

        description = f"This PR adds tests for #{self.pr.number}\n\n" "### Commits:\n" + "\n".join(
            commit_messages
        )

        new_pr = self.repo_client.create_pr_from_branch(
            branch=branch_ref,
            title=pr_title,
            description=description,
            provided_base=self.pr.head.ref,
        )

        self.store_pr_context(new_pr=new_pr)

    def store_pr_context(self, new_pr: PullRequest):
        with Session() as session:
            run_info = DbPrContextToUnitTestGenerationRunIdMapping(
                provider="github",
                owner=self.repo_client.repo.owner.login,
                repo=self.repo_client.repo.name,
                pr_id=new_pr.number,
                iterations=0,
                run_id=self.unit_test_run_id,
                original_pr_url=self.pr.html_url,
            )
            session.add(run_info)
            session.commit()
