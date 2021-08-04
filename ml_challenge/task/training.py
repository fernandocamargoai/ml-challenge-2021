import abc
import json
import os
import pickle
import shutil
from functools import cached_property
from pathlib import Path
from typing import List, Tuple, Optional
from glob import glob

import luigi
import pandas as pd
import torch
import pytorch_lightning as pl
import wandb
from gluonts.dataset.common import Dataset
from gluonts.time_feature import get_lags_for_frequency
from gluonts.torch.model.predictor import PyTorchPredictor
from pts.modules import NegativeBinomialOutput
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import OrdinalEncoder

from ml_challenge.dataset import (
    JsonGzDataset,
    SameSizeTransformedDataset,
    FilterTimeSeriesTransformation,
    TruncateTargetTransformation,
)
from ml_challenge.gluonts import CustomDeepAREstimator, CustomNegativeBinomialOutput
from ml_challenge.submission import generate_submission
from ml_challenge.task.data_preparation import PrepareGluonTimeSeriesDatasets
from ml_challenge.utils import get_sku_from_data_entry_path
from ml_challenge.wandb import WandbWithBestMetricLogger

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeepARTraining(luigi.Task, metaclass=abc.ABCMeta):
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
        default=["global_relative_price", "current_price", "minutes_active"]
    )
    test_steps: int = luigi.IntParameter(default=30)
    validate_with_non_testing_skus: bool = luigi.BoolParameter(default=False)

    context_length: int = luigi.IntParameter(default=30)
    num_layers: int = luigi.IntParameter(default=2)
    hidden_size: int = luigi.IntParameter(default=40)
    dropout_rate: float = luigi.FloatParameter(default=0.1)
    embedding_dimension: List[int] = luigi.ListParameter(default=None)
    num_parallel_samples: int = luigi.IntParameter(default=100)

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
    cache_dataset: bool = luigi.BoolParameter(default=False)
    preload_dataset: bool = luigi.BoolParameter(default=False)
    check_data: bool = luigi.BoolParameter(default=False)

    generate_submission: bool = luigi.BoolParameter(default=False)

    seed: int = luigi.IntParameter(default=42)
    use_gpu: bool = luigi.BoolParameter(default=False)

    def requires(self):
        return PrepareGluonTimeSeriesDatasets(
            categorical_variables=self.categorical_variables,
            real_variables=self.real_variables,
            test_steps=self.test_steps,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", self.__class__.__name__, self.task_id)
        )

    def _save_params(self):
        with open(os.path.join(self.output().path, "params.json"), "w") as params_file:
            json.dump(
                self.param_kwargs, params_file, default=lambda o: dict(o), indent=4
            )

    @cached_property
    def categorical_encoder(self) -> OrdinalEncoder:
        with open(
            os.path.join(self.input().path, "categorical_encoder.pkl"), "rb"
        ) as f:
            return pickle.load(f)

    @cached_property
    def holidays_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.input().path, "holidays.csv")).set_index(
            ["date", "site_id"]
        )

    @cached_property
    def cardinality(self) -> List[int]:
        return [
            len(categories) + 1 for categories in self.categorical_encoder.categories_
        ]

    @cached_property
    def test_df(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join("assets", "test_data.csv"))

    @cached_property
    def train_dataset(self) -> Dataset:
        paths = glob(os.path.join(self.input().path, "*.json.gz"))
        if self.validate_with_non_testing_skus:
            testing_skus = set(self.test_df["sku"])
            paths = [
                path
                for path in paths
                if get_sku_from_data_entry_path(path) in testing_skus
            ]
        return SameSizeTransformedDataset(
            JsonGzDataset(paths, freq="D"),
            transformation=FilterTimeSeriesTransformation(
                start=0, end=-self.test_steps
            ),
        )

    @cached_property
    def val_dataset(self) -> Optional[Dataset]:
        if self.validate_with_non_testing_skus:
            paths = glob(os.path.join(self.input().path, "*.json.gz"))[:1000] # TODO Remove [:1000]
            testing_skus = set(self.test_df["sku"])
            paths = [
                path
                for path in paths
                if get_sku_from_data_entry_path(path) not in testing_skus
            ]
            return SameSizeTransformedDataset(
                JsonGzDataset(paths, freq="D"),
                transformation=FilterTimeSeriesTransformation(
                    start=0, end=-self.test_steps
                ),
            )
        return None

    @cached_property
    def test_dataset(self) -> Dataset:
        paths = self.test_df["sku"].apply(
            lambda sku: os.path.join(self.input().path, f"{sku}.json.gz")
        )
        return SameSizeTransformedDataset(
            JsonGzDataset(paths, freq="D"),
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

    def run(self):
        os.makedirs(self.output().path, exist_ok=True)

        self._save_params()

        pl.seed_everything(self.seed, workers=True)

        shutil.copy(
            os.path.join(self.input().path, "categorical_encoder.pkl"),
            os.path.join(self.output().path, "categorical_encoder.pkl"),
            follow_symlinks=True,
        )

        monitor = "train_loss" if self.val_dataset is None else "val_loss"

        wandb_logger = WandbWithBestMetricLogger(
            name=self.task_id,
            save_dir=self.output().path,
            project="ml-challenge",
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
        estimator = CustomDeepAREstimator(
            freq="D",
            prediction_length=self.test_steps,
            context_length=self.context_length,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            dropout_rate=self.dropout_rate,
            num_feat_dynamic_real=len(self.real_variables)
            + len(self.holidays_df.columns),
            num_feat_static_cat=len(self.categorical_variables),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            distr_output=NegativeBinomialOutput(),
            scaling=True,
            lags_seq=get_lags_for_frequency("D", lag_ub=60),
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

        train_output = estimator.train_model(
            self.train_dataset,
            validation_data=self.val_dataset,
            num_workers=self.num_workers,
            cache_data=self.cache_dataset,
        )

        self._serialize(train_output.predictor)
        predictor_artifact = wandb.Artifact(name=f"artifact-{wandb_logger.experiment.id}", type="model")
        predictor_artifact.add_dir(os.path.join(self.output().path, "predictor"))
        wandb_logger.experiment.log_artifact(predictor_artifact)

        if self.generate_submission:
            generate_submission(
                train_output.predictor,
                self.test_dataset,
                self.test_df,
                self.output().path,
            )
