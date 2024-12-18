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

    return_predicted_range: bool = Field(
        False,
        description="Flag to control if the predicted range should be returned",
    )

    mp_fixed_window_size: int = Field(
        10,
        description="Fixed window size for the matrix profile",
    )

    direction_detection_num_timesteps_in_batch_mode: int = Field(
        12,
        description="Number of timesteps to do direction detection in batch mode",
    )

    period_to_smooth_size: dict[int, int] = Field(
        default={5: 19, 15: 11, 30: 7, 60: 5},
        description="Flag smoothing window size based on the function smooth_size = floor(43 / sqrt(time_period))",
    )

    stream_smooth_context_sizes: dict[int, int] = Field(
        default={5: 17, 15: 11, 30: 7, 60: 5},
        description="History size for stream smoothing based on the function smooth_size = floor(43 / sqrt(time_period))",
    )
