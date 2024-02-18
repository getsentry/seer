from typing import Literal

from pydantic import BaseModel


class UpdateCodebaseTaskRequest(BaseModel):
    repo_id: int
