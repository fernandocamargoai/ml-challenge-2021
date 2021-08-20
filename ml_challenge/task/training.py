import abc
import json
import os
import pickle
import shutil
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Dict, Type, cast, Any
from glob import glob

import luigi
import pandas as pd
import torch
import pytorch_lightning as pl
import wandb
from gluonts.dataset.common import Dataset
from gluonts.time_feature import (
    get_lags_for_frequency,
    TimeFeature,
    DayOfWeek,
    DayOfMonth,
    DayOfYear,
    WeekOfYear,
)
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    GammaOutput,
    BetaOutput,
)
from pts.modules import (
    NegativeBinomialOutput,
    PoissonOutput,
    ZeroInflatedPoissonOutput,
    ZeroInflatedNegativeBinomialOutput,
    NormalOutput,
    StudentTOutput,
)
from sklearn.preprocessing import OrdinalEncoder

from ml_challenge.dataset import (
    JsonGzDataset,
    SameSizeTransformedDataset,
    FilterTimeSeriesTransformation,
    TruncateTargetTransformation,
    ChangeTargetToMinutesActiveTransformation,
    MoveMinutesActiveToControlTransformation,
    TruncateControlTransformation,
)
from ml_challenge.gluonts.custom import CustomDeepAREstimator
from ml_challenge.gluonts.distribution import BimodalBetaOutput, BiStudentTMixtureOutput
from ml_challenge.gluonts.model.causal_deepar import CausalDeepAREstimator
from ml_challenge.gluonts.time_feature import (
    DayOfWeekSin,
    DayOfWeekCos,
    DayOfMonthSin,
    DayOfMonthCos,
    DayOfYearSin,
    DayOfYearCos,
    WeekOfYearSin,
    WeekOfYearCos,
)
from ml_challenge.path import get_assets_path
from ml_challenge.task.data_preparation import PrepareGluonTimeSeriesDatasets
from ml_challenge.utils import get_sku_from_data_entry_path, save_params
from ml_challenge.wandb import WandbWithBestMetricLogger

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_DISTRIBUTIONS: Dict[str, Type[DistributionOutput]] = {
    "negative_binomial": NegativeBinomialOutput,
    "zero_inflated_negative_binomial": ZeroInflatedNegativeBinomialOutput,
    "poisson": PoissonOutput,
    "zero_inflated_poisson": ZeroInflatedPoissonOutput,
    "normal": NormalOutput,
    "student_t": StudentTOutput,
    "beta": BetaOutput,
    "bimodal_beta": BimodalBetaOutput,
    "gamma": GammaOutput,
    "bi_student_t_mixture": BiStudentTMixtureOutput,
}

_TIME_FEATURES: Dict[str, Type[TimeFeature]] = {
    "day_of_week": DayOfWeek,
    "day_of_month": DayOfMonth,
    "day_of_year": DayOfYear,
    "week_of_year": WeekOfYear,
    "day_of_week_sin": DayOfWeekSin,
    "day_of_week_cos": DayOfWeekCos,
    "day_of_month_sin": DayOfMonthSin,
    "day_of_month_cos": DayOfMonthCos,
    "day_of_year_sin": DayOfYearSin,
    "day_of_year_cos": DayOfYearCos,
    "week_of_year_sin": WeekOfYearSin,
    "week_of_year_cos": WeekOfYearCos,
}


class BaseTraining(luigi.Task, metaclass=abc.ABCMeta):
    categorical_variables: List[str] = luigi.ListParameter(
        default=[
            "site_id",
            "currency",
            "listing_type",
            "shipping_logistic_type",
            "shipping_payment",
            "item_domain_id",
            "item_id",
            "sku",
        ]
    )
    real_variables: List[str] = luigi.ListParameter(
        default=[
            "minutes_active",
            "current_price",
            "currency_relative_price",
            "usd_relative_price",
            "minimum_salary_relative_price",
        ]
    )
    test_steps: int = luigi.IntParameter(default=30)
    validate_with_non_testing_skus: bool = luigi.BoolParameter(default=False)

    context_length: int = luigi.IntParameter(default=30)
    lags_seq_ub: int = luigi.IntParameter(default=60)
    time_features: List[str] = luigi.ListParameter(
        default=["day_of_week", "day_of_month", "day_of_year"]
    )

    precision: int = luigi.IntParameter(default=32)

    batch_size: int = luigi.IntParameter(default=32)
    accumulate_grad_batches: int = luigi.IntParameter(default=1)
    max_epochs: int = luigi.IntParameter(default=100)
    num_batches_per_epoch: int = luigi.IntParameter(default=-1)
    lr: float = luigi.FloatParameter(default=1e-3)
    weight_decay: float = luigi.FloatParameter(default=1e-8)
    gradient_clip_val: float = luigi.FloatParameter(default=10.0)
    early_stopping_patience: int = luigi.IntParameter(default=5)

    num_workers: int = luigi.IntParameter(default=0)
    num_prefetch: int = luigi.IntParameter(default=2)

    seed: int = luigi.IntParameter(default=42)
    use_gpu: bool = luigi.BoolParameter(default=False)

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", self.__class__.__name__, self.task_id)
        )

    @cached_property
    def input_path(self) -> str:
        return self.input().path

    @cached_property
    def holidays_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.input_path, "holidays.csv")).set_index(
            ["date", "site_id"]
        )

    @cached_property
    def test_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(get_assets_path(), "test_data.csv"))


class DeepARTraining(BaseTraining, metaclass=abc.ABCMeta):
    distribution: str = luigi.ChoiceParameter(
        choices=_DISTRIBUTIONS.keys(), default="negative_binomial"
    )

    num_layers: int = luigi.IntParameter(default=2)
    hidden_size: int = luigi.IntParameter(default=40)
    dropout_rate: float = luigi.FloatParameter(default=0.1)
    embedding_dimension: List[int] = luigi.ListParameter(default=None)
    num_parallel_samples: int = luigi.IntParameter(default=100)

    cache_dataset: bool = luigi.BoolParameter(default=False)
    preload_dataset: bool = luigi.BoolParameter(default=False)
    check_data: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return PrepareGluonTimeSeriesDatasets(
            categorical_variables=self.categorical_variables,
            real_variables=self.real_variables,
            test_steps=self.test_steps,
        )

    @cached_property
    def labels(self) -> Dict[str, List[Any]]:
        with open(os.path.join(self.input_path, "labels.json"), "r") as f:
            return json.load(f)

    @cached_property
    def cardinality(self) -> List[int]:
        return [
            len(self.labels[variable]) + 1  # last index is for unknown
            for variable in self.categorical_variables
        ]

    def create_dataset(self, paths) -> Dataset:
        return JsonGzDataset(paths, freq="D")

    @cached_property
    def train_dataset(self) -> Dataset:
        paths = glob(os.path.join(self.input_path, "*.json.gz"))
        if self.validate_with_non_testing_skus:
            testing_skus = set(self.test_df["sku"])
            paths = [
                path
                for path in paths
                if get_sku_from_data_entry_path(path) in testing_skus
            ]
        return SameSizeTransformedDataset(
            self.create_dataset(paths),
            transformation=FilterTimeSeriesTransformation(
                start=0, end=-self.test_steps
            ),
        )

    @cached_property
    def val_dataset(self) -> Optional[Dataset]:
        if self.validate_with_non_testing_skus:
            paths = glob(os.path.join(self.input_path, "*.json.gz"))
            testing_skus = set(self.test_df["sku"])
            paths = [
                path
                for path in paths
                if get_sku_from_data_entry_path(path) not in testing_skus
            ]
            return SameSizeTransformedDataset(
                self.create_dataset(paths),
                transformation=FilterTimeSeriesTransformation(
                    start=0, end=-self.test_steps
                ),
            )
        return None

    @cached_property
    def test_dataset(self) -> Dataset:
        paths = self.test_df["sku"].apply(
            lambda sku: os.path.join(self.input_path, f"{sku}.json.gz")
        )
        return SameSizeTransformedDataset(
            self.create_dataset(paths),
            transformation=TruncateTargetTransformation(self.test_steps),
        )

    def _serialize(self, predictor: PyTorchPredictor):
        print("Serializing predictor...")
        predictor_path = os.path.join(self.output().path, "predictor")
        os.mkdir(predictor_path)
        predictor.serialize(Path(predictor_path))
        print("Serialized predictor...")

    def get_trained_predictor(self, device: torch.device) -> PyTorchPredictor:
        predictor_path = os.path.join(self.output().path, "predictor")
        predictor = PyTorchPredictor.deserialize(Path(predictor_path), device=device,)
        predictor.prediction_net.to(device)
        return predictor

    @property
    def wandb_project(self) -> str:
        return "ml-challenge"

    @property
    def num_feat_dynamic_real(self) -> int:
        return len(self.real_variables) + len(self.holidays_df.columns)

    def get_estimator_params(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: pl.callbacks.EarlyStopping,
    ) -> Dict[str, Any]:
        return dict(
            freq="D",
            prediction_length=self.test_steps,
            context_length=self.context_length,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            num_feat_dynamic_real=self.num_feat_dynamic_real,
            num_feat_static_cat=len(self.categorical_variables),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            distr_output=_DISTRIBUTIONS[self.distribution](),
            scaling=True,
            lags_seq=get_lags_for_frequency("D", lag_ub=self.lags_seq_ub),
            time_features=[
                _TIME_FEATURES[time_feature]() for time_feature in self.time_features
            ],
            num_parallel_samples=self.num_parallel_samples,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            num_batches_per_epoch=self.num_batches_per_epoch
            if self.num_batches_per_epoch > 0
            else len(self.train_dataset) // self.batch_size,
            trainer_kwargs=dict(
                max_epochs=self.max_epochs,
                accumulate_grad_batches=self.accumulate_grad_batches,
                gradient_clip_val=self.gradient_clip_val,
                logger=wandb_logger,
                callbacks=[early_stopping],
                default_root_dir=self.output().path,
                gpus=torch.cuda.device_count() if self.use_gpu else 0,
                precision=self.precision,
                num_sanity_val_steps=0,
                deterministic=True,
            ),
        )

    def create_estimator(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: pl.callbacks.EarlyStopping,
    ) -> DeepAREstimator:
        return CustomDeepAREstimator(
            **self.get_estimator_params(wandb_logger, early_stopping)
        )

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        save_params(self.output().path, self.param_kwargs)

        pl.seed_everything(self.seed, workers=True)

        shutil.copy(
            os.path.join(self.input_path, "labels.json"),
            os.path.join(self.output().path, "labels.json"),
            follow_symlinks=True,
        )

        monitor = "train_loss" if self.val_dataset is None else "val_loss"

        wandb_logger = WandbWithBestMetricLogger(
            name=self.task_id,
            save_dir=self.output().path,
            project=self.wandb_project,
            log_model=False,
            monitor=monitor,
            mode="min",
        )
        wandb_logger.log_hyperparams(self.param_kwargs)

        early_stopping = pl.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=self.early_stopping_patience,
            verbose=True,
        )
        estimator = self.create_estimator(wandb_logger, early_stopping)

        train_output = estimator.train_model(
            self.train_dataset,
            validation_data=self.val_dataset,
            num_workers=self.num_workers,
            cache_data=self.cache_dataset,
        )

        self._serialize(train_output.predictor)
        predictor_artifact = wandb.Artifact(
            name=f"artifact-{wandb_logger.experiment.id}", type="model"
        )
        predictor_artifact.add_dir(os.path.join(self.output().path, "predictor"))
        wandb_logger.experiment.log_artifact(predictor_artifact)


class DeepARForMinutesActiveTraining(DeepARTraining):
    distribution: str = luigi.ChoiceParameter(
        choices=_DISTRIBUTIONS.keys(), default="student_t"
    )

    @property
    def wandb_project(self) -> str:
        return "ml-challenge-minutes-active"

    @property
    def num_feat_dynamic_real(self) -> int:
        return super().num_feat_dynamic_real - 1

    def create_dataset(self, paths) -> Dataset:
        return SameSizeTransformedDataset(
            super().create_dataset(paths),
            transformation=ChangeTargetToMinutesActiveTransformation(
                self.real_variables.index("minutes_active")
            ),
        )


class CausalDeepARTraining(DeepARTraining):
    control_distribution: str = luigi.ChoiceParameter(
        choices=_DISTRIBUTIONS.keys(), default="student_t"
    )
    control_loss_weight: float = luigi.FloatParameter(default=1.0)

    def create_estimator(
        self,
        wandb_logger: WandbWithBestMetricLogger,
        early_stopping: pl.callbacks.EarlyStopping,
    ) -> CausalDeepAREstimator:
        return CausalDeepAREstimator(
            control_output=_DISTRIBUTIONS[self.control_distribution](),
            min_control_value=0.0,
            max_control_value=1.0,
            **self.get_estimator_params(wandb_logger, early_stopping),
        )

    @property
    def wandb_project(self) -> str:
        return "ml-challenge-causal-deepar"

    @property
    def num_feat_dynamic_real(self) -> int:
        return super().num_feat_dynamic_real - 1

    def create_dataset(self, paths) -> Dataset:
        return SameSizeTransformedDataset(
            super().create_dataset(paths),
            transformation=MoveMinutesActiveToControlTransformation(
                self.real_variables.index("minutes_active")
            ),
        )

    @cached_property
    def test_dataset(self) -> Dataset:
        return SameSizeTransformedDataset(
            super().test_dataset,
            transformation=TruncateControlTransformation(self.test_steps),
        )
