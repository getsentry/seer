from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest, CodebaseState
from seer.automation.autofix.state import ContinuationState
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.state import DbState, DbStateRunTypes


def create_initial_autofix_run(request: AutofixRequest) -> DbState[AutofixContinuation]:
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
        t=DbStateRunTypes.AUTOFIX,
    )

    with state.update() as cur:
        create_initial_codebase_states(cur)
        set_accessible_repos(cur)

        cur.mark_triggered()

    event_manager = AutofixEventManager(state)
    event_manager.send_root_cause_analysis_will_start()

    return state


def create_initial_codebase_states(cur: AutofixContinuation) -> None:
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
