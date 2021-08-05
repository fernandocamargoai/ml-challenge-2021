import functools
import gzip
import os
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import pandas as pd
from gluonts.dataset.common import Dataset
from gluonts.model.forecast import SampleForecast

from gluonts.torch.model.predictor import PyTorchPredictor
from tqdm import tqdm

from ml_challenge.lookahead_generator import LookaheadGenerator


def calculate_out_of_stock_cdf(
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


def calculate_out_of_stock_probas(
    forecast: SampleForecast, stock: int, quantiles: List[str]
) -> List[float]:
    cdf = calculate_out_of_stock_cdf(forecast, stock, quantiles)
    return list(np.ediff1d(cdf, to_begin=cdf[0]))


def _pool_calculate_out_of_stock_probas(
    input_: Tuple[SampleForecast, int], quantiles: List[str]
) -> List[float]:
    forecast, stock = input_
    return calculate_out_of_stock_probas(forecast, stock, quantiles)


def generate_submission(
    predictor: PyTorchPredictor,
    dataset: Dataset,
    df: pd.DataFrame,
    task_path: str,
    quantiles: List[float] = None,
) -> str:
    if quantiles is None:
        quantiles = [str(q) for q in np.arange(0.00, 1.01, 0.01)]

    predictor.batch_size = 512
    with Pool(os.cpu_count()) as pool:
        with LookaheadGenerator(predictor.predict(dataset)) as forecasts:
            all_probas = list(tqdm(
                pool.imap(
                    functools.partial(
                        _pool_calculate_out_of_stock_probas, quantiles=quantiles
                    ),
                    zip(forecasts, df["target_stock"]),
                ),
                total=len(df),
            ))
    all_probas = [
        probas if sum(probas) > 0 else ([0.0] * (len(probas) - 1) + [0.0001])
        for probas in all_probas
    ]

    submission_path = os.path.join(task_path, "submission.csv.gz")
    with gzip.open(submission_path, "wb") as f:
        for probas in all_probas:
            f.write(
                (",".join([str(round(proba, 4)) for proba in probas]) + "\n").encode(
                    "utf-8"
                )
            )
    return submission_path
