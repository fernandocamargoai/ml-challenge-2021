import os
import ast

import click
import luigi
from luigi.execution_summary import LuigiRunResult, LuigiStatusCode

from ml_challenge.task.training import CausalDeepARTraining


@click.command()
@click.option("--distribution", type=str)
@click.option("--control_distribution", type=str)
@click.option("--embedding_dimension", type=str)
@click.option("--gradient_clip_val", type=float)
@click.option("--context_length", type=int)
@click.option("--lags_seq_ub", type=int)
@click.option("--time_features", type=str)
@click.option("--weight_decay", type=float)
@click.option("--dropout_rate", type=float)
@click.option("--hidden_size", type=int)
@click.option("--num_layers", type=int)
@click.option("--batch_size", type=int)
@click.option("--lr", type=float)
def train(
    distribution: str,
    control_distribution: str,
    embedding_dimension: str,
    gradient_clip_val: float,
    context_length: int,
    lags_seq_ub: int,
    time_features: str,
    weight_decay: float,
    dropout_rate: float,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    lr: float,
):
    result: LuigiRunResult = luigi.build(
        [
            CausalDeepARTraining(
                use_gpu=True,
                validate_with_non_testing_skus=True,
                num_workers=4,
                cache_dataset=False,
                distribution=distribution,
                control_distribution=control_distribution,
                control_loss_weight=0.1,
                embedding_dimension=ast.literal_eval(embedding_dimension),
                gradient_clip_val=gradient_clip_val,
                context_length=context_length,
                lags_seq_ub=lags_seq_ub,
                time_features=ast.literal_eval(time_features),
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_size=batch_size,
                lr=lr,
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
