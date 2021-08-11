import functools
import gzip
import json
import os
from functools import cached_property
from multiprocessing import Pool

import luigi
import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm

from ml_challenge.lookahead_generator import LookaheadGenerator
from ml_challenge.submission import (
    pool_forecast_transform_fn,
    calculate_out_of_stock_days_from_samples,
    apply_tweedie,
    apply_normal,
    apply_ecdf,
    apply_beta,
)
from ml_challenge.task.training import DeepARTraining


class GenerateOutOfStockDaySamplePredictions(luigi.Task):
    task_path: str = luigi.Parameter()

    num_samples: int = luigi.IntParameter(default=100)
    seed: int = luigi.IntParameter(default=42)

    @cached_property
    def training(self) -> DeepARTraining:
        with open(os.path.join(self.task_path, "params.json"), "r") as params_file:
            params = json.load(params_file)
        return DeepARTraining(**params)

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.task_path,
                f"out_of_stock_day_sample_predictions_num-samples={self.num_samples}_seed={self.seed}.npy",
            )
        )

    def run(self):
        predictor = self.training.get_trained_predictor(torch.device("cuda"))

        pl.seed_everything(self.seed, workers=True)

        predictor.batch_size = 512
        with Pool(os.cpu_count()) as pool:
            with LookaheadGenerator(
                predictor.predict(
                    self.training.test_dataset, num_samples=self.num_samples
                )
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

    distribution: str = luigi.ChoiceParameter(
        choices=[
            "tweedie",
            "normal",
            "ecdf",
            "beta",
        ]
    )

    quantile_step: float = luigi.FloatParameter(default=0.01)
    tweedie_phi: float = luigi.FloatParameter(default=2.0)
    tweedie_power: float = luigi.FloatParameter(default=1.3)

    num_samples: int = luigi.IntParameter(default=100)

    seed: int = luigi.IntParameter(default=42)

    def requires(self):
        return GenerateOutOfStockDaySamplePredictions(
            task_path=self.task_path, num_samples=self.num_samples, seed=self.seed,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                self.task_path,
                f"submission_{self.distribution}_num-samples={self.num_samples}_seed={self.seed}.csv.gz",
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
