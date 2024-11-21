from pydantic import BaseModel, Field


class MPConfig(BaseModel):
    """
    Class with configuration used for the Matrix Profile algorithm
    """

    ignore_trivial: bool = Field(
        True,
        description="Flag that tells the stumpy library to ignore trivial matches to speed up MP computation",
    )
    normalize_mp: bool = Field(
        False,
        description="Flag to control if the matrix profile is normalized first",
    )
    fixed_window_size: int = Field(
        10,
        description="Fixed window size for the matrix profile",
    )
