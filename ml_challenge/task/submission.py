import json
import os
from functools import cached_property

import luigi
import numpy as np
import torch

from ml_challenge.submission import (
    generate_submission_with_quantiles,
    generate_submission_with_tweedie,
    generate_submission_with_normal,
    generate_submission_with_ecdf, generate_submission_with_beta,
)
from ml_challenge.task.training import DeepARTraining


class GenerateSubmission(luigi.Task):
    task_path: str = luigi.Parameter()

    method: str = luigi.ChoiceParameter(
        choices=["with_quantiles", "with_tweedie", "with_normal", "with_ecdf", "with_beta"]
    )

    quantile_step: float = luigi.FloatParameter(default=0.01)
    tweedie_phi: float = luigi.FloatParameter(default=2.0)
    tweedie_power: float = luigi.FloatParameter(default=1.3)

    num_samples: int = luigi.IntParameter(default=100)

    seed: int = luigi.IntParameter(default=42)

    @cached_property
    def training(self) -> DeepARTraining:
        with open(os.path.join(self.task_path, "params.json"), "r") as params_file:
            params = json.load(params_file)
        return DeepARTraining(**params)

    def run(self):
        predictor = self.training.get_trained_predictor(torch.device("cuda"))

        if self.method == "with_quantiles":
            quantiles = [
                q for q in np.arange(0.00, 1.0 + self.quantile_step, self.quantile_step)
            ]
            generate_submission_with_quantiles(
                predictor,
                self.num_samples,
                self.seed,
                self.training.test_dataset,
                self.training.test_df,
                self.training.output().path,
                quantiles,
            )
        elif self.method == "with_tweedie":
            generate_submission_with_tweedie(
                predictor,
                self.num_samples,
                self.seed,
                self.training.test_dataset,
                self.training.test_df,
                self.training.output().path,
                phi=self.tweedie_phi,
                power=self.tweedie_power,
            )
        elif self.method == "with_normal":
            generate_submission_with_normal(
                predictor,
                self.num_samples,
                self.seed,
                self.training.test_dataset,
                self.training.test_df,
                self.training.output().path,
            )
        elif self.method == "with_ecdf":
            generate_submission_with_ecdf(
                predictor,
                self.num_samples,
                self.seed,
                self.training.test_dataset,
                self.training.test_df,
                self.training.output().path,
            )
        elif self.method == "with_beta":
            generate_submission_with_beta(
                predictor,
                self.num_samples,
                self.seed,
                self.training.test_dataset,
                self.training.test_df,
                self.training.output().path,
            )
