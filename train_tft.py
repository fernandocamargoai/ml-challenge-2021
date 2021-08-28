import os
import ast

import click
import luigi
from luigi.execution_summary import LuigiRunResult, LuigiStatusCode

from ml_challenge.task.training import DeepARTraining, TemporalFusionTransformerTraining


@click.command()
@click.option("--embed_dim", type=int)
@click.option("--num_heads", type=int)
@click.option("--gradient_clip_val", type=float)
@click.option("--context_length", type=int)
@click.option("--time_features", type=str)
@click.option("--weight_decay", type=float)
@click.option("--dropout_rate", type=float)
@click.option("--batch_size", type=int)
@click.option("--lr", type=float)
def train(
    embed_dim: int,
    num_heads: int,
    gradient_clip_val: float,
    context_length: int,
    time_features: str,
    weight_decay: float,
    dropout_rate: float,
    batch_size: int,
    lr: float,
):
    result: LuigiRunResult = luigi.build(
        [
            TemporalFusionTransformerTraining(
                use_gpu=True,
                validate_with_non_testing_skus=True,
                num_workers=0,
                cache_dataset=True,
                embed_dim=embed_dim,
                num_heads=num_heads,
                gradient_clip_val=gradient_clip_val,
                context_length=context_length,
                time_features=ast.literal_eval(time_features),
                weight_decay=weight_decay,
                dropout_rate=dropout_rate,
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
