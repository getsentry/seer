from pydantic import BaseModel

from seer.automation.models import SeerProjectPreference
from seer.db import DbSeerProjectPreference, Session


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
        session.merge(data.preference.to_db_model())
        session.commit()

    return SetSeerProjectPreferenceResponse(preference=data.preference)
