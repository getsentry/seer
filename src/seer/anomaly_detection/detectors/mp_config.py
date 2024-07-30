from pydantic import BaseModel, Field


class MPConfig(BaseModel):
    """
    Class with configuration used for the Matrix Profile algorithm
    """

    ignore_trivial: bool = Field(
        ...,
        description="Flag that tells the stumpy library to ignore trivial matches to speed up MP computation",
    )
    normalize_mp: bool = Field(
        ...,
        description="Flag to control if the matrix profile is normalized first",
    )
