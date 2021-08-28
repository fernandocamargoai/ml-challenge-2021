# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch

from gluonts.torch.util import weighted_average
from pts.model.tft.tft_network import TemporalFusionTransformerNetwork


class TemporalFusionTransformerLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: TemporalFusionTransformerNetwork,
        lr: float = 1e-3,
        weight_decay: float = 1e-8,
        control_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.control_loss_weight = control_loss_weight

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        past_target = batch["past_target"]
        past_observed_values = batch["past_observed_values"]
        future_target = batch["future_target"]
        future_observed_values = batch["future_observed_values"]
        past_feat_dynamic_real = batch["past_feat_dynamic_real"]
        past_feat_dynamic_cat = batch["past_feat_dynamic_cat"]
        feat_dynamic_real = batch["feat_dynamic_real"]
        feat_dynamic_cat = batch["feat_dynamic_cat"]

        feat_static_real = batch["feat_static_real"]
        feat_static_cat = batch["feat_static_cat"]

        (
            past_covariates,
            future_covariates,
            static_covariates,
            offset,
            scale,
        ) = self.model._preprocess(
            past_target,
            past_observed_values,
            past_feat_dynamic_real,
            past_feat_dynamic_cat,
            feat_dynamic_real,
            feat_dynamic_cat,
            feat_static_real,
            feat_static_cat,
        )

        preds = self.model.forward(
            past_observed_values, past_covariates, future_covariates, static_covariates,
        )

        preds = self.model._postprocess(preds, offset, scale)

        loss = self.model.loss(future_target, preds)
        loss = weighted_average(loss, future_observed_values)
        return loss.mean()

    def training_step(self, batch, batch_idx: int):
        """Execute training step"""
        train_loss = self._compute_loss(batch)
        self.log(
            "train_loss", train_loss, on_epoch=True, on_step=False, prog_bar=True,
        )
        return train_loss

    def validation_step(self, batch, batch_idx: int):
        """Execute validation step"""
        val_loss = self._compute_loss(batch)
        self.log(
            "val_loss", val_loss, on_epoch=True, on_step=False, prog_bar=True,
        )
        return val_loss

    def configure_optimizers(self):
        """Returns the optimizer to use"""
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
