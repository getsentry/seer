from typing import cast

from github import GithubException

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest, CodebaseState
from seer.automation.autofix.state import ContinuationState
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import RepoDefinition
from seer.automation.preferences import GetSeerProjectPreferenceRequest, get_seer_project_preference
from seer.automation.state import DbState, DbStateRunTypes


def create_initial_autofix_run(request: AutofixRequest) -> DbState[AutofixContinuation]:
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
        t=DbStateRunTypes.AUTOFIX,
    )

    preference = get_seer_project_preference(
        GetSeerProjectPreferenceRequest(project_id=request.project_id)
    ).preference

    with state.update() as cur:
        if preference:
            cur.request.repos = preference.repositories

        create_missing_codebase_states(cur)
        set_accessible_repos(cur)

        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    event_manager.send_root_cause_analysis_will_start()

    return state


def validate_repo_branches_exist(
    repos: list[RepoDefinition], event_manager: AutofixEventManager
) -> bool:
    for repo in repos:
        if repo.provider == "github":
            if repo.branch_name:
                try:
                    RepoClient.from_repo_definition(repo, "read")
                except GithubException as e:
                    if e.status == 404:
                        event_manager.on_error(
                            f"The branch {repo.branch_name} does not exist in the repository {repo.full_name} or Autofix doesn't have access to it."
                        )
                    return False

    return True


def create_missing_codebase_states(cur: AutofixContinuation) -> None:
    for repo in cur.request.repos:
        if repo.external_id not in cur.codebases:
            cur.codebases[repo.external_id] = CodebaseState(
                file_changes=[],
                repo_external_id=repo.external_id,
            )


def set_accessible_repos(cur: AutofixContinuation) -> None:
    for repo in cur.request.repos:
        if repo.provider == "github":
            if RepoClient.check_repo_read_access(repo):
                cur.codebases[repo.external_id].is_readable = True
            if RepoClient.check_repo_write_access(repo):
                cur.codebases[repo.external_id].is_writeable = True
        else:
            cur.codebases[repo.external_id].is_readable = False
            cur.codebases[repo.external_id].is_writeable = False


def update_repo_access(state: ContinuationState) -> None:
    with state.update() as cur:
        set_accessible_repos(cur)
