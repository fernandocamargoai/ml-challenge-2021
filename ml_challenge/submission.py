from typing import List, Tuple, Callable

import numpy as np
import tweedie
from gluonts.model.forecast import SampleForecast
from scipy.stats import norm, beta, gamma
from statsmodels.distributions import ECDF


def pool_forecast_transform_fn(
    input_: Tuple[SampleForecast, int],
    forecast_transform_fn: Callable[[SampleForecast, int], List[float]],
) -> List[float]:
    forecast, stock = input_
    return forecast_transform_fn(forecast, stock)


def calculate_out_of_stock_days_from_samples(
    forecast: SampleForecast, stock: int, total_days: int = 30
) -> np.ndarray:
    sample_days = np.apply_along_axis(
        np.searchsorted, 1, np.cumsum(forecast.samples, axis=1) >= stock, True
    )
    sample_days[sample_days == total_days] -= 1
    return sample_days + 1


def cdf_to_probas(cdf: List[float]) -> List[float]:
    prob_array = np.array(np.ediff1d(cdf, to_begin=cdf[0]))
    return list(prob_array / np.sum(prob_array))


def apply_tweedie(
    sample_days: np.ndarray,
    std_multiplier: float = 1.0,
    phi: float = 2.0,
    power=1.3,
    min_std: float = 2.0,
    total_days: int = 30,
) -> List[float]:
    mu = sample_days.mean()
    if phi < 0:
        sigma = sample_days.std() * std_multiplier
        sigma = max(sigma, min_std)
        phi = (sigma ** 2) / mu ** power
    distro = tweedie.tweedie(p=power, mu=mu, phi=phi)

    cdf = [distro.cdf(i) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def apply_normal(
    sample_days: np.ndarray, std_multiplier: float = 1.0, total_days: int = 30
) -> List[float]:
    distro = norm(sample_days.mean(), sample_days.std() * std_multiplier)

    cdf = [distro.cdf(i) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def apply_ecdf(sample_days: np.ndarray, total_days: int = 30) -> List[float]:
    ecdf = ECDF(sample_days)
    cdf = [ecdf(i) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def apply_beta(
    sample_days: np.ndarray, std_multiplier: float = 1.0, total_days: int = 30
) -> List[float]:
    mu = sample_days.mean() / total_days
    sigma = (sample_days.std() * std_multiplier) / total_days

    a = mu ** 2 * ((1 - mu) / sigma ** 2 - 1 / mu)
    b = a * (1 / mu - 1)

    distro = beta(a, b)

    cdf = [distro.cdf(i / total_days) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)
