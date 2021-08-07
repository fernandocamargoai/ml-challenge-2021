import math
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import torch
from gluonts.time_feature import (
    TimeFeature,
    DayOfWeekIndex,
    DayOfMonthIndex,
    DayOfYearIndex,
    WeekOfYearIndex,
)
from gluonts.torch.model.deepar import (
    DeepAREstimator,
    DeepARLightningModule,
    DeepARModel,
)
from gluonts.torch.modules.distribution_output import (
    DistributionOutput,
    StudentTOutput,
)
from gluonts.torch.modules.loss import DistributionLoss, NegativeLogLikelihood


class CustomDeepAREstimator(DeepAREstimator):
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        context_length: Optional[int] = None,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinality: Optional[List[int]] = None,
        embedding_dimension: Optional[List[int]] = None,
        distr_output: DistributionOutput = StudentTOutput(),
        loss: DistributionLoss = NegativeLogLikelihood(),
        scaling: bool = True,
        lags_seq: Optional[List[int]] = None,
        time_features: Optional[List[TimeFeature]] = None,
        num_parallel_samples: int = 100,
        batch_size: int = 32,
        num_batches_per_epoch: int = 50,
        trainer_kwargs: Optional[Dict[str, Any]] = dict(),
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
    ) -> None:
        super().__init__(
            freq,
            prediction_length,
            context_length,
            num_layers,
            hidden_size,
            dropout_rate,
            num_feat_dynamic_real,
            num_feat_static_cat,
            num_feat_static_real,
            cardinality,
            embedding_dimension,
            distr_output,
            loss,
            scaling,
            lags_seq,
            time_features,
            num_parallel_samples,
            batch_size,
            num_batches_per_epoch,
            trainer_kwargs,
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def create_lightning_module(self) -> DeepARLightningModule:
        model = DeepARModel(
            freq=self.freq,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            num_feat_dynamic_real=(
                1 + self.num_feat_dynamic_real + len(self.time_features)
            ),
            num_feat_static_real=max(1, self.num_feat_static_real),
            num_feat_static_cat=max(1, self.num_feat_static_cat),
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            distr_output=self.distr_output,
            dropout_rate=self.dropout_rate,
            lags_seq=self.lags_seq,
            scaling=self.scaling,
            num_parallel_samples=self.num_parallel_samples,
        )

        return DeepARLightningModule(
            model=model, loss=self.loss, lr=self.lr, weight_decay=self.weight_decay
        )


class DayOfWeekSin(DayOfWeekIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_week = super().__call__(index)
        return np.sin(day_of_week * (2.0 * math.pi / 6))


class DayOfWeekCos(DayOfWeekIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_week = super().__call__(index)
        return np.cos(day_of_week * (2.0 * math.pi / 6))


class DayOfMonthSin(DayOfMonthIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_month = super().__call__(index)
        return np.sin(
            day_of_month * (2.0 * math.pi / (index.days_in_month - 1))
        )


class DayOfMonthCos(DayOfMonthIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_month = super().__call__(index)
        return np.cos(
            day_of_month * (2.0 * math.pi / (index.days_in_month - 1))
        )


class DayOfYearSin(DayOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_year = super().__call__(index)
        return np.sin(day_of_year * (2.0 * math.pi / 365))


class DayOfYearCos(DayOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_year = super().__call__(index)
        return np.cos(day_of_year * (2.0 * math.pi / 365))


class WeekOfYearSin(WeekOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        week_of_year = super().__call__(index)
        return np.sin(week_of_year * (2.0 * math.pi / 52))


class WeekOfYearCos(WeekOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        week_of_year = super().__call__(index)
        return np.cos(week_of_year * (2.0 * math.pi / 52))
