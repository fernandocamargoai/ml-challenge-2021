import functools
import gzip
import json
import os
from functools import cached_property
from multiprocessing import Pool
from typing import Union

import luigi
import numpy as np
import torch
import pytorch_lightning as pl
from gluonts.model.forecast import SampleForecast
from tqdm import tqdm

from ml_challenge.dataset import (
    SameSizeTransformedDataset,
    UseMinutesActiveForecastingTransformation,
    UseMeanOfLastMinutesActiveTransformation,
)
from ml_challenge.lookahead_generator import LookaheadGenerator
from ml_challenge.submission import (
    pool_forecast_transform_fn,
    calculate_out_of_stock_days_from_samples,
    apply_tweedie,
    apply_normal,
    apply_ecdf,
    apply_beta,
)
from ml_challenge.task.training import (
    DeepARTraining,
    DeepARForMinutesActiveTraining,
    CausalDeepARTraining,
)


def get_suffix(
    task: Union["GenerateOutOfStockDaySamplePredictions", "GenerateSubmission"]
) -> str:
    if task.minutes_active_task_path:
        return "_%s_%s" % (
            os.path.split(task.minutes_active_task_path)[-1],
            task.minutes_active_forecast_method,
        )
    elif task.use_mean_of_last_minutes_active:
        return "_using_mean_of_last_minutes_active"
    else:
        return ""


class GenerateOutOfStockDaySamplePredictions(luigi.Task):
    task_path: str = luigi.Parameter()

    minutes_active_task_path: str = luigi.Parameter(default=None)
    minutes_active_forecast_method: str = luigi.ChoiceParameter(
        choices=["mean", "max"], default="mean"
    )
    use_mean_of_last_minutes_active: bool = luigi.BoolParameter(default=False)

    num_samples: int = luigi.IntParameter(default=100)
    seed: int = luigi.IntParameter(default=42)

    @cached_property
    def training(self) -> DeepARTraining:
        with open(os.path.join(self.task_path, "params.json"), "r") as params_file:
            params = json.load(params_file)
        training_class = {
            DeepARTraining.__name__: DeepARTraining,
            CausalDeepARTraining.__name__: CausalDeepARTraining,
        }[os.path.split(os.path.split(self.task_path)[0])[1]]
        return training_class(**params)

    @cached_property
    def minutes_active_training(self) -> DeepARForMinutesActiveTraining:
        with open(
            os.path.join(self.minutes_active_task_path, "params.json"), "r"
        ) as params_file:
            params = json.load(params_file)
        return DeepARForMinutesActiveTraining(**params)

    def output(self):
        suffix = get_suffix(self)
        return luigi.LocalTarget(
            os.path.join(
                self.task_path,
                f"out_of_stock_day_sample_predictions_num-samples={self.num_samples}_seed={self.seed}{suffix}.npy",
            )
        )

    def process_minutes_active_forecast(self, forecast: SampleForecast) -> np.ndarray:
        if self.minutes_active_forecast_method == "mean":
            return np.clip(forecast.mean, 0.0, 1.0)
        elif self.minutes_active_forecast_method == "max":
            return np.clip(forecast.samples.max(axis=0), 0.0, 1.0)
        else:
            raise ValueError()

    def run(self):
        pl.seed_everything(self.seed, workers=True)

        if self.minutes_active_task_path:
            minutes_active_predictor = self.minutes_active_training.get_trained_predictor(
                torch.device("cuda")
            )
            minutes_active_predictor.batch_size = 512
            minutes_active_forecasts = [
                self.process_minutes_active_forecast(forecast)
                for forecast in tqdm(
                    minutes_active_predictor.predict(
                        self.minutes_active_training.test_dataset,
                        num_samples=self.num_samples,
                    ),
                    total=len(self.minutes_active_training.test_dataset),
                )
            ]
            minutes_active_forecasts_dict = {
                sku: forecast
                for sku, forecast in zip(
                    self.minutes_active_training.test_df["sku"],
                    minutes_active_forecasts,
                )
            }

            del minutes_active_predictor

            test_dataset = SameSizeTransformedDataset(
                self.training.test_dataset,
                transformation=UseMinutesActiveForecastingTransformation(
                    self.training.real_variables.index("minutes_active"),
                    minutes_active_forecasts_dict,
                ),
            )
        elif self.use_mean_of_last_minutes_active:
            test_dataset = SameSizeTransformedDataset(
                self.training.test_dataset,
                transformation=UseMeanOfLastMinutesActiveTransformation(
                    self.training.real_variables.index("minutes_active"),
                    self.training.test_steps,
                ),
            )
        else:
            test_dataset = self.training.test_dataset

        predictor = self.training.get_trained_predictor(torch.device("cuda"))
        predictor.batch_size = 512
        with Pool(os.cpu_count()) as pool:
            with LookaheadGenerator(
                predictor.predict(test_dataset, num_samples=self.num_samples)
            ) as forecasts:
                all_days_samples = list(
                    tqdm(
                        pool.imap(
                            functools.partial(
                                pool_forecast_transform_fn,
                                forecast_transform_fn=calculate_out_of_stock_days_from_samples,
                            ),
                            zip(forecasts, self.training.test_df["target_stock"]),
                        ),
                        total=len(self.training.test_df),
                    )
                )

        np.save(self.output().path, np.array(all_days_samples))


class GenerateSubmission(luigi.Task):
    task_path: str = luigi.Parameter()

    minutes_active_task_path: str = luigi.Parameter(default=None)
    minutes_active_forecast_method: str = luigi.ChoiceParameter(
        choices=["mean", "max"], default="mean"
    )
    use_mean_of_last_minutes_active: bool = luigi.BoolParameter(default=False)

    distribution: str = luigi.ChoiceParameter(
        choices=["tweedie", "normal", "ecdf", "beta"]
    )

    tweedie_phi: float = luigi.FloatParameter(default=2.0)
    tweedie_power: float = luigi.FloatParameter(default=1.3)

    num_samples: int = luigi.IntParameter(default=100)

    seed: int = luigi.IntParameter(default=42)

    def requires(self):
        return GenerateOutOfStockDaySamplePredictions(
            task_path=self.task_path,
            minutes_active_task_path=self.minutes_active_task_path,
            minutes_active_forecast_method=self.minutes_active_forecast_method,
            use_mean_of_last_minutes_active=self.use_mean_of_last_minutes_active,
            num_samples=self.num_samples,
            seed=self.seed,
        )

    def output(self):
        suffix = get_suffix(self)
        distribution = self.distribution
        if distribution == "tweedie":
            distribution += f"_phi={self.tweedie_phi}_power={self.tweedie_power}"
        return luigi.LocalTarget(
            os.path.join(
                self.task_path,
                f"submission_{distribution}_num-samples={self.num_samples}_seed={self.seed}{suffix}.csv.gz",
            )
        )

    def run(self):
        out_of_stock_day_sample_preds = np.load(self.input().path)

        if self.distribution == "tweedie":
            apply_dist_fn = functools.partial(
                apply_tweedie, phi=self.tweedie_phi, power=self.tweedie_power
            )
        elif self.distribution == "normal":
            apply_dist_fn = apply_normal
        elif self.distribution == "ecdf":
            apply_dist_fn = apply_ecdf
        else:  # self.distribution == "beta":
            apply_dist_fn = apply_beta

        with Pool(os.cpu_count()) as pool:
            all_probas = list(
                tqdm(
                    pool.imap(apply_dist_fn, out_of_stock_day_sample_preds,),
                    total=out_of_stock_day_sample_preds.shape[0],
                )
            )

        all_probas = [
            probas if sum(probas) > 0 else ([0.0] * (len(probas) - 1) + [1.0])
            for probas in all_probas
        ]

        with gzip.open(self.output().path, "wb") as f:
            for probas in all_probas:
                f.write(
                    (
                        ",".join([str(round(proba, 4)) for proba in probas]) + "\n"
                    ).encode("utf-8")
                )
