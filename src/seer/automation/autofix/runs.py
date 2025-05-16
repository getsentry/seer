import logging
from concurrent.futures import ThreadPoolExecutor

from github import GithubException

from seer.automation.autofix.event_manager import AutofixEventManager
from seer.automation.autofix.models import AutofixContinuation, AutofixRequest, CodebaseState
from seer.automation.autofix.state import ContinuationState
from seer.automation.codebase.repo_client import RepoClient
from seer.automation.models import RepoDefinition
from seer.automation.preferences import (
    MAX_REPOS_TOTAL,
    GetSeerProjectPreferenceRequest,
    create_initial_seer_project_preference_from_repos,
    get_seer_project_preference,
)
from seer.automation.state import DbState, DbStateRunTypes
from seer.dependency_injection import copy_modules_initializer

logger = logging.getLogger(__name__)


def create_initial_autofix_run(request: AutofixRequest) -> DbState[AutofixContinuation]:
    """
    Creates a new autofix run for an issue.
    Args:
        request (AutofixRequest): The autofix request containing issue and project details.
    Returns:
        DbState[AutofixContinuation]: Database state manager for the new autofix run.
    """
    state = ContinuationState.new(
        AutofixContinuation(request=request),
        group_id=request.issue.id,
        t=DbStateRunTypes.AUTOFIX,
    )

    main_project_id = request.project_id
    trace_connected_project_ids = (
        request.trace_tree.get_all_project_ids() if request.trace_tree else []
    )

    preference = get_seer_project_preference(
        GetSeerProjectPreferenceRequest(project_id=main_project_id)
    ).preference
    try:
        trace_connected_preferences = [
            get_seer_project_preference(
                GetSeerProjectPreferenceRequest(project_id=project_id)
            ).preference
            for project_id in trace_connected_project_ids
        ]
    except Exception as e:
        logger.exception(e)

    if not preference:
        # No preference found, create one from our list of code mapping repos.
        preference = create_initial_seer_project_preference_from_repos(
            organization_id=request.organization_id,
            project_id=main_project_id,
            repos=request.repos,
        )

    with state.update() as cur:
        if preference:
            cur.request.repos = preference.repositories

        try:
            for trace_connected_preference in trace_connected_preferences:
                if len(cur.request.repos) >= MAX_REPOS_TOTAL:
                    break
                if trace_connected_preference:
                    for repo in trace_connected_preference.repositories:
                        if not any(
                            existing_repo.external_id == repo.external_id
                            for existing_repo in cur.request.repos
                        ):
                            cur.request.repos.append(repo)
        except Exception as e:
            logger.exception(e)

    continuation_state = ContinuationState(state.id)

    # Add information about the git repositories to the autofix state
    update_repo_access_and_properties(continuation_state, set_branches_and_commits=True)

    with state.update() as cur:
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


def update_repo_access_and_properties(
    state: ContinuationState, set_branches_and_commits: bool = False
) -> None:
    """
    Updates repository access permissions and properties for each repository in the autofix state.
    For GitHub repositories, checks read/write access and optionally sets branch names and commit SHAs.
    Args:
        state: The ContinuationState object containing the autofix run state.
    """
    cur_state = state.get()
    # Create new codebases if needed.
    new_codebases = {}
    for repo in cur_state.request.repos:
        if repo.external_id not in cur_state.codebases:
            new_codebases[repo.external_id] = CodebaseState(
                file_changes=[],
                repo_external_id=repo.external_id,
            )

    # Set accesible repos and set branch_name and base_commit_sha if accessible.
    def _process_repo(repo):
        if repo.provider == "github":
            is_readable = RepoClient.check_repo_read_access(repo)
            is_writeable = RepoClient.check_repo_write_access(repo)
            if set_branches_and_commits:
                if is_readable and not repo.branch_name:
                    repo_client = RepoClient.from_repo_definition(repo, "read")
                    repo.branch_name = repo_client.base_branch
                    if not repo.base_commit_sha:
                        repo.base_commit_sha = repo_client.base_commit_sha
            update = {
                "is_readable": bool(is_readable),
                "is_writeable": bool(is_writeable),
            }
        else:
            update = {"is_readable": False, "is_writeable": False}
        return repo.external_id, update

    repos = cur_state.request.repos
    if len(repos) > 1:
        with ThreadPoolExecutor(initializer=copy_modules_initializer()) as executor:
            results = list(executor.map(_process_repo, repos))
        updates = dict(results)
    else:
        updates = {}
        for repo in repos:
            repo_id, update = _process_repo(repo)
            updates[repo_id] = update

    # Write updated state to postgres db.
    with state.update() as cur:
        if new_codebases:
            cur.codebases.update(new_codebases)
        for repo_id, update in updates.items():
            cur.codebases[repo_id].is_readable = update["is_readable"]
            cur.codebases[repo_id].is_writeable = update["is_writeable"]
        cur.request.repos = cur_state.request.repos
