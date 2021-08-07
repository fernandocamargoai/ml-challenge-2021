import functools
import gzip
import os
from multiprocessing.dummy import Pool
from typing import List, Tuple, Callable, Dict

import numpy as np
import pandas as pd
import tweedie
import pytorch_lightning as pl
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import SampleForecast

from gluonts.torch.model.predictor import PyTorchPredictor
from scipy.stats import norm, beta
from statsmodels.distributions import ECDF
from tqdm import tqdm

from ml_challenge.lookahead_generator import LookaheadGenerator


def _pool_forecast_transform_fn(
    input_: Tuple[SampleForecast, int],
    forecast_transform_fn: Callable[[SampleForecast, int], List[float]],
) -> List[float]:
    forecast, stock = input_
    return forecast_transform_fn(forecast, stock)


def cdf_to_probas(cdf: List[float]) -> List[float]:
    prob_array = np.array(np.ediff1d(cdf, to_begin=cdf[0]))
    return list(prob_array / np.sum(prob_array))


def generate_submission(
    predictor: PyTorchPredictor,
    num_samples: int,
    seed: int,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    forecast_transform_fn: Callable[[SampleForecast, int], List[float]],
    suffix: str,
) -> str:
    pl.seed_everything(seed, workers=True)

    predictor.batch_size = 512
    with Pool(os.cpu_count()) as pool:
        with LookaheadGenerator(
            predictor.predict(dataset, num_samples=num_samples)
        ) as forecasts:
            all_probas = list(
                tqdm(
                    pool.imap(
                        functools.partial(
                            _pool_forecast_transform_fn,
                            forecast_transform_fn=forecast_transform_fn,
                        ),
                        zip(forecasts, df["target_stock"]),
                    ),
                    total=len(df),
                )
            )
    all_probas = [
        probas if sum(probas) > 0 else ([0.0] * (len(probas) - 1) + [1.0])
        for probas in all_probas
    ]

    submission_path = os.path.join(
        task_path, f"submission_{suffix}_num-samples={num_samples}_seed={seed}.csv.gz"
    )
    with gzip.open(submission_path, "wb") as f:
        for probas in all_probas:
            f.write(
                (",".join([str(round(proba, 4)) for proba in probas]) + "\n").encode(
                    "utf-8"
                )
            )
    return submission_path


def calculate_out_of_stock_cdf_with_quantiles(
    forecast: SampleForecast, stock: int, quantiles: List[str]
) -> List[float]:
    quantile_forecast = forecast.to_quantile_forecast(quantiles)
    out_of_stock = np.cumsum(quantile_forecast.forecast_array, axis=1) >= stock
    return [
        (1 - float(quantile_forecast.forecast_keys[out_of_stock_index]))
        if out_of_stock_num > 0
        else 0.0
        for out_of_stock_num, out_of_stock_index in zip(
            out_of_stock.sum(axis=0), np.argmax(out_of_stock, axis=0)
        )
    ]


def forecast_with_quantiles(
    forecast: SampleForecast, stock: int, quantiles: List[str]
) -> List[float]:
    cdf = calculate_out_of_stock_cdf_with_quantiles(forecast, stock, quantiles)
    return cdf_to_probas(cdf)


def generate_submission_with_quantiles(
    predictor: PyTorchPredictor,
    num_samples: int,
    seed: int,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    quantiles: List[float] = None,
) -> str:
    if quantiles is None:
        quantiles = [str(q) for q in np.arange(0.00, 1.01, 0.01)]

    forecast_transform_fn = functools.partial(
        forecast_with_quantiles, quantiles=quantiles
    )

    return generate_submission(
        predictor,
        num_samples,
        seed,
        dataset,
        df,
        task_path,
        forecast_transform_fn,
        "with_quantiles",
    )


@functools.lru_cache
def apply_tweedie(
    out_of_stock_days: int, phi: float = 2.0, power=1.3, total_days: int = 30
) -> List[float]:
    distro = tweedie.tweedie(p=power, mu=out_of_stock_days, phi=phi)

    cdf = [distro.cdf(i) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def calculate_out_of_stock_days_from_samples(
    forecast: SampleForecast, stock: int, total_days: int = 30
) -> np.ndarray:
    sample_days = np.apply_along_axis(
        np.searchsorted, 1, np.cumsum(forecast.samples, axis=1) >= stock, True
    )
    sample_days[sample_days == total_days] -= 1
    return sample_days + 1


def forecast_with_tweedie(
    forecast: SampleForecast,
    stock: int,
    phi: float = 2.0,
    power=1.3,
    total_days: int = 30,
) -> List[float]:
    sample_days = calculate_out_of_stock_days_from_samples(forecast, stock)
    return apply_tweedie(
        round(sample_days.mean()), phi=phi, power=power, total_days=total_days
    )


def generate_submission_with_tweedie(
    predictor: PyTorchPredictor,
    num_samples: int,
    seed: int,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    phi: float = 2.0,
    power: float = 1.3,
    total_days: int = 30,
) -> str:
    forecast_transform_fn = functools.partial(
        forecast_with_tweedie, phi=phi, power=power, total_days=total_days,
    )

    return generate_submission(
        predictor,
        num_samples,
        seed,
        dataset,
        df,
        task_path,
        forecast_transform_fn,
        f"with_tweedie_phi={phi}_power={power}",
    )


def apply_normal(days_mean: float, days_std: float, total_days: int = 30):
    distro = norm(days_mean, days_std)

    cdf = [distro.cdf(i) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def forecast_with_normal(
    forecast: SampleForecast, stock: int, total_days: int = 30
) -> List[float]:
    sample_days = calculate_out_of_stock_days_from_samples(forecast, stock)
    return apply_normal(sample_days.mean(), sample_days.std(), total_days=total_days)


def generate_submission_with_normal(
    predictor: PyTorchPredictor,
    num_samples: int,
    seed: int,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    total_days: int = 30,
) -> str:
    forecast_transform_fn = functools.partial(
        forecast_with_normal, total_days=total_days,
    )

    return generate_submission(
        predictor,
        num_samples,
        seed,
        dataset,
        df,
        task_path,
        forecast_transform_fn,
        "with_normal",
    )


def apply_ecdf(sample_days: List[int], total_days: int = 30):
    ecdf = ECDF(sample_days)
    cdf = [ecdf(i) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def forecast_with_ecdf(
    forecast: SampleForecast, stock: int, total_days: int = 30
) -> List[float]:
    sample_days = calculate_out_of_stock_days_from_samples(forecast, stock)
    return apply_ecdf(sample_days, total_days=total_days)


def generate_submission_with_ecdf(
    predictor: PyTorchPredictor,
    num_samples: int,
    seed: int,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    total_days: int = 30,
) -> str:
    forecast_transform_fn = functools.partial(
        forecast_with_ecdf, total_days=total_days,
    )

    return generate_submission(
        predictor,
        num_samples,
        seed,
        dataset,
        df,
        task_path,
        forecast_transform_fn,
        "with_ecdf",
    )


def apply_beta(days_mean: float, days_std: float, total_days: int = 30):
    mu = days_mean / total_days
    var = (days_std ** 2) / total_days
    a = ((1 - mu) / var - 1 / mu) * mu ** 2
    b = a * (1 / mu - 1)

    distro = beta(a, b)

    cdf = [distro.cdf(i / total_days) for i in range(1, total_days + 1)]
    return cdf_to_probas(cdf)


def forecast_with_beta(
    forecast: SampleForecast, stock: int, total_days: int = 30
) -> List[float]:
    sample_days = calculate_out_of_stock_days_from_samples(forecast, stock)
    return apply_normal(sample_days.mean(), sample_days.std(), total_days=total_days)


def generate_submission_with_beta(
    predictor: PyTorchPredictor,
    num_samples: int,
    seed: int,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    total_days: int = 30,
) -> str:
    forecast_transform_fn = functools.partial(
        forecast_with_beta, total_days=total_days,
    )

    return generate_submission(
        predictor,
        num_samples,
        seed,
        dataset,
        df,
        task_path,
        forecast_transform_fn,
        "with_beta",
    )
