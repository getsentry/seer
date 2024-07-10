from pydantic import BaseModel


class SimpleEndpointResponse(BaseModel):
    success: bool
