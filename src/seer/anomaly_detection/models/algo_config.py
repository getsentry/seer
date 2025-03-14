from typing import Dict, List

from pydantic import BaseModel, Field


class ProphetParams:
    """
    Parameter class for Prophet model

    This is the parameter class for prophet model, it contains all necessary
    parameters as definied in Prophet implementation:
    https://github.com/facebook/prophet/blob/master/python/prophet/forecaster.py

    Attributes:
        growth: String 'linear' or 'logistic' to specify a linear or logistic
            trend.
        changepoints: List of dates at which to include potential changepoints. If
            not specified, potential changepoints are selected automatically.
        n_changepoints: Number of potential changepoints to include. Not used
            if input `changepoints` is supplied. If `changepoints` is not supplied,
            then n_changepoints potential changepoints are selected uniformly from
            the first `changepoint_range` proportion of the history.
        changepoint_range: Proportion of history in which trend changepoints will
            be estimated. Defaults to 0.8 for the first 80%. Not used if
            `changepoints` is specified.
        yearly_seasonality: Fit yearly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        weekly_seasonality: Fit weekly seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        daily_seasonality: Fit daily seasonality.
            Can be 'auto', True, False, or a number of Fourier terms to generate.
        holidays: pd.DataFrame with columns holiday (string) and ds (date type)
            and optionally columns lower_window and upper_window which specify a
            range of days around the date to be included as holidays.
            lower_window=-2 will include 2 days prior to the date as holidays. Also
            optionally can have a column prior_scale specifying the prior scale for
            that holiday.
        seasonality_mode: 'additive' (default) or 'multiplicative'.
        seasonality_prior_scale: Parameter modulating the strength of the
            seasonality model. Larger values allow the model to fit larger seasonal
            fluctuations, smaller values dampen the seasonality. Can be specified
            for individual seasonalities using add_seasonality.
        holidays_prior_scale: Parameter modulating the strength of the holiday
            components model, unless overridden in the holidays input.
        changepoint_prior_scale: Parameter modulating the flexibility of the
            automatic changepoint selection. Large values will allow many
            changepoints, small values will allow few changepoints.
        mcmc_samples: Integer, if greater than 0, will do full Bayesian inference
            with the specified number of MCMC samples. If 0, will do MAP
            estimation.
        interval_width: Float, width of the uncertainty intervals provided
            for the forecast. If mcmc_samples=0, this will be only the uncertainty
            in the trend using the MAP estimate of the extrapolated generative
            model. If mcmc.samples>0, this will be integrated over all model
            parameters, which will include uncertainty in seasonality.
        uncertainty_samples: Number of simulated draws used to estimate
            uncertainty intervals. Settings this value to 0 or False will disable
            uncertainty estimation and speed up the calculation.
        cap: capacity, provided for logistic growth
        floor: floor, the fcst value must be greater than the specified floor
    """

    def __init__(
        self,
        growth="linear",
        changepoints=None,
        n_changepoints=25,
        changepoint_range=0.8,
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto",
        holidays=None,
        seasonality_mode="additive",
        seasonality_prior_scale=10.0,
        holidays_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        mcmc_samples=0,
        interval_width=0.80,
        uncertainty_samples=1000,
        cap=None,
        floor=None,
        custom_seasonalities: List[Dict] | None = None,
    ) -> None:
        self.growth = growth
        self.changepoints = changepoints
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.seasonality_mode = seasonality_mode
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.interval_width = interval_width
        self.uncertainty_samples = uncertainty_samples
        self.cap = cap
        self.floor = floor
        self.custom_seasonalities = [] if custom_seasonalities is None else custom_seasonalities


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

    period_to_smooth_size: dict[int, int] = Field(
        default={5: 19, 15: 11, 30: 7, 60: 5},
        description="Flag smoothing window size based on the function smooth_size = floor(43 / sqrt(time_period))",
    )

    stream_smooth_context_sizes: dict[int, int] = Field(
        default={5: 17, 15: 11, 30: 7, 60: 5},
        description="History size for stream smoothing based on the function smooth_size = floor(43 / sqrt(time_period))",
    )

    prophet_intervals: dict[str, float] = Field(
        default={
            "low": 0.975,
            "medium": 0.90,
            "high": 0.80,
        },
        description="Intervals for the Prophet model",
    )

    prophet_forecast_len: int = Field(
        36,
        description="Number of hours to forecast for the Prophet model",
    )

    max_stream_days_for_combo_detection: int = Field(
        3,
        description="Limit on the number of days we apply streaming to during combo detection",
    )

    max_batch_days_for_combo_detection: dict[int, int] = Field(
        default={5: 7, 15: 15, 30: 21, 60: 28},
        description="Limit on the number of days we apply batching to during combo detection",
    )

    combo_detection_prophet_batching_interval_days: float = Field(
        3,
        description="Number of days to batch prophet predictions for combo detection",
    )

    def get_prophet_params(self, sensitivity: str) -> ProphetParams:
        return ProphetParams(
            interval_width=self.prophet_intervals[sensitivity],
            changepoint_prior_scale=0.01,
            weekly_seasonality=14,
            daily_seasonality=False,
            uncertainty_samples=None,
        )
