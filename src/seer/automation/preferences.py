from pydantic import BaseModel

from seer.automation.models import RepoDefinition, SeerProjectPreference
from seer.db import DbSeerProjectPreference, Session

MAX_REPOS_PER_PROJECT = 8
MAX_REPOS_TOTAL = 12


class GetSeerProjectPreferenceRequest(BaseModel):
    project_id: int


class GetSeerProjectPreferenceResponse(BaseModel):
    preference: SeerProjectPreference | None


def get_seer_project_preference(
    data: GetSeerProjectPreferenceRequest,
) -> GetSeerProjectPreferenceResponse:
    with Session() as session:
        preference = session.get(DbSeerProjectPreference, data.project_id)
        if preference is None:
            return GetSeerProjectPreferenceResponse(preference=None)

        # correct prefs saved with too many repos; should only affect old prefs before we added a check before saving
        if len(preference.repositories) > MAX_REPOS_PER_PROJECT:
            preference.repositories = preference.repositories[:MAX_REPOS_PER_PROJECT]
            session.merge(preference)
            session.commit()

        return GetSeerProjectPreferenceResponse(
            preference=SeerProjectPreference.from_db_model(preference)
        )


class SetSeerProjectPreferenceRequest(BaseModel):
    preference: SeerProjectPreference


class SetSeerProjectPreferenceResponse(BaseModel):
    preference: SeerProjectPreference


def set_seer_project_preference(
    data: SetSeerProjectPreferenceRequest,
) -> SetSeerProjectPreferenceResponse:
    with Session() as session:
        if len(data.preference.repositories) > MAX_REPOS_PER_PROJECT:
            data.preference.repositories = data.preference.repositories[:MAX_REPOS_PER_PROJECT]
        session.merge(data.preference.to_db_model())
        session.commit()

    return SetSeerProjectPreferenceResponse(preference=data.preference)


def create_initial_seer_project_preference_from_repos(
    *,
    organization_id: int,
    project_id: int,
    repos: list[RepoDefinition],
) -> SeerProjectPreference:
    preference = SeerProjectPreference(
        organization_id=organization_id,
        project_id=project_id,
        repositories=repos[:MAX_REPOS_PER_PROJECT],
    )
    with Session() as session:
        session.add(preference.to_db_model())
        session.commit()
    return preference
