from pydantic import BaseModel, Field


class AlgoConfig(BaseModel):
    """
    Class with configuration used for the Matrix Profile algorithm
    """

    mp_ignore_trivial: bool = Field(
        ...,
        description="Flag that tells the stumpy library to ignore trivial matches to speed up MP computation",
    )
    mp_normalize: bool = Field(
        ...,
        description="Flag to control if the matrix profile is normalized first",
    )

    prophet_uncertainty_samples: int = Field(
        ...,
        description="Number of montecarlo simulations to run to compute uncertainty intervals",
    )

    prophet_mcmc_samples: int = Field(
        ...,
        description="Number of montecarlo simulations to run to compute uncertainty intervals",
    )

    return_thresholds: bool = Field(
        False,
        description="Flag to control if the thresholds should be returned",
    )
