import json
import os
from functools import cached_property

import luigi
import numpy as np
import torch

from ml_challenge.submission import generate_submission
from ml_challenge.task.training import DeepARTraining


class GenerateSubmission(luigi.Task):
    task_path: str = luigi.Parameter()
    quantile_step: float = luigi.FloatParameter(default=0.01)

    @cached_property
    def training(self) -> DeepARTraining:
        with open(os.path.join(self.task_path, "params.json"), "r") as params_file:
            params = json.load(params_file)
        return DeepARTraining(**params)

    def output(self):
        return luigi.LocalTarget(os.path.join(self.task_path, "submission.csv.gz"))

    def run(self):
        predictor = self.training.get_trained_predictor(torch.device("cuda"))
        quantiles = [
            q for q in np.arange(0.00, 1.0 + self.quantile_step, self.quantile_step)
        ]
        generate_submission(
            predictor,
            self.training.test_dataset,
            self.training.test_df,
            self.training.output().path,
            quantiles,
        )
