import math
from typing import Dict, Optional

from pytorch_lightning.loggers import WandbLogger


class WandbWithBestMetricLogger(WandbLogger):
    def __init__(
        self,
        name: Optional[str] = None,
        save_dir: Optional[str] = None,
        offline: Optional[bool] = False,
        id: Optional[str] = None,
        anonymous: Optional[bool] = None,
        version: Optional[str] = None,
        project: Optional[str] = None,
        log_model: Optional[bool] = False,
        experiment=None,
        prefix: Optional[str] = "",
        sync_step: Optional[bool] = None,
        monitor: Optional[str] = None,
        mode: str = "min",
        **kwargs
    ):
        super().__init__(
            name,
            save_dir,
            offline,
            id,
            anonymous,
            version,
            project,
            log_model,
            experiment,
            prefix,
            sync_step,
            **kwargs
        )
        self._monitor = monitor
        self._mode = mode

        self._best_model_score = math.inf if self._mode == "min" else - math.inf

    def _is_better(self, current: float) -> bool:
        if self._mode == "min":
            return current < self._best_model_score
        else:
            return current > self._best_model_score

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        current = metrics.get(self._monitor)
        if current is not None:
            if self._is_better(current):
                self._best_model_score = current
            metrics["best_model_score"] = self._best_model_score

        super().log_metrics(metrics, step)
