from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from gluonts.torch.modules.distribution_output import BetaOutput, DistributionOutput
from torch.distributions import (
    Distribution,
    Normal,
    MixtureSameFamily,
    Categorical,
    StudentT,
)


class BimodalBetaOutput(BetaOutput):
    @classmethod
    def domain_map(cls, concentration1, concentration0):
        concentration1 = F.sigmoid(concentration1)
        concentration0 = F.sigmoid(concentration0)
        return concentration1.squeeze(-1), concentration0.squeeze(-1)

    def distribution(
        self,
        distr_args,
        loc: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
    ) -> Distribution:
        return self.distr_cls(*distr_args)


def BiStudentTMixture(
    cls, logits: torch.Tensor, df: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> MixtureSameFamily:
    mix = Categorical(logits=logits)
    comp = StudentT(df, mean, std)
    return MixtureSameFamily(mix, comp)


class BiStudentTMixtureOutput(DistributionOutput):
    args_dim: Dict[str, int] = {"logits": 2, "df": 2, "mean": 2, "std": 2}
    distr_cls: type = BiStudentTMixture

    @classmethod
    def domain_map(cls, logits: torch.Tensor, df: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        std = F.softplus(std)
        df = 2.0 + F.softplus(df)
        return logits, df, mean, std

    @property
    def event_shape(self) -> Tuple:
        return ()
