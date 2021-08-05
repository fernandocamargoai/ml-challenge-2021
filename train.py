import os

import click
import luigi
from luigi.execution_summary import LuigiRunResult, LuigiStatusCode

from ml_challenge.task.training import DeepARTraining


@click.command()
@click.option("--gradient_clip_val", type=float)
@click.option("--context_length", type=int)
@click.option("--lags_seq_ub", type=int)
@click.option("--weight_decay", type=float)
@click.option("--dropout_rate", type=float)
@click.option("--hidden_size", type=int)
@click.option("--num_layers", type=int)
@click.option("--batch_size", type=int)
@click.option("--lr", type=float)
def train(
    gradient_clip_val: float,
    context_length: int,
    lags_seq_ub: int,
    weight_decay: float,
    dropout_rate: float,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    lr: float,
):
    result: LuigiRunResult = luigi.build(
        [
            DeepARTraining(
                use_gpu=True,
                validate_with_non_testing_skus=True,
                num_workers=max(os.cpu_count() - 1, 1),
                embedding_dimension=[2, 3, 2, 2, 2, 8, 16, 16],
                gradient_clip_val=gradient_clip_val,
                context_length=context_length,
                lags_seq_ub=lags_seq_ub,
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_size=batch_size,
                lr=lr,
                generate_submission=os.environ.get("GENERATE_SUBMISSION", False),
                precision=16
                if os.environ.get("ENABLE_MIXED_PRECISION", "False") == "True"
                else 32,
            )
        ],
        detailed_summary=True,
        local_scheduler=True,
    )
    if not result.status == LuigiStatusCode.SUCCESS:
        raise RuntimeError(result)


if __name__ == "__main__":
    train()
