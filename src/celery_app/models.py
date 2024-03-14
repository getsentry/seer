from typing import Literal, ClassVar, List

from pydantic import BaseModel

class BaseModel(BaseModel):
    int_from_float_fields: ClassVar[List[str]] = []


class UpdateCodebaseTaskRequest(BaseModel):
    repo_id: int
