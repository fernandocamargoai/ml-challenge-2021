import functools
import os
import pickle
from typing import List, Tuple, Dict
import warnings

import luigi
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
from sklearn.preprocessing import OrdinalEncoder, scale
from tqdm import tqdm

from ml_challenge.holidays import create_holidays_df
from ml_challenge.path import get_assets_path
from ml_challenge.utils import save_json_gzip


def _default_real_variable(exog_column: str, last_value: float) -> float:
    return {"minutes_active": 1.0}.get(exog_column, last_value)


def _save_dataset_item(
    group: Tuple[str, pd.DataFrame],
    categorical_values_per_sku: Dict[str, np.ndarray],
    holidays_df: pd.DataFrame,
    real_variables: List[str],
    test_steps: int,
    global_current_price_mean: float,
    global_current_price_std: float,
    currency_current_price_mean: Dict[str, float],
    currency_current_price_std: Dict[str, float],
    output_dir: str,
):
    sku, df = group
    try:
        df = df.set_index("date").sort_index()
        df = pd.merge(
            df,
            holidays_df,
            how="left",
            left_on=[df.index, df["site_id"]],
            right_index=True,
        ).fillna(0)

        if "global_relative_price" in real_variables:
            df["global_relative_price"] = (
                df["current_price"] - global_current_price_mean
            ) / global_current_price_std

        if "currency_relative_price" in real_variables:
            currency = df.iloc[0]["currency"]
            df["currency_relative_price"] = (
                df["current_price"] - currency_current_price_mean[currency]
            ) / currency_current_price_std[currency]

        if "current_price" in real_variables:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df["current_price"] = scale(
                    df["current_price"], with_mean=True, with_std=True
                )
        if "minutes_active" in real_variables:
            df["minutes_active"] /= 1440.0

        categorical_values = categorical_values_per_sku[sku].tolist()

        exog_columns = list(real_variables) + list(holidays_df.columns)

        test_steps_df = pd.DataFrame(
            index=pd.date_range(df.index[-1] + pd.DateOffset(1), periods=test_steps)
        )
        test_steps_df["site_id"] = df.iloc[0]["site_id"]
        test_steps_df = pd.merge(
            test_steps_df,
            holidays_df,
            how="left",
            left_on=[test_steps_df.index, test_steps_df["site_id"]],
            right_index=True,
        ).fillna(0)
        for real_variable in real_variables:
            test_steps_df[real_variable] = _default_real_variable(
                real_variable, df.iloc[-1][real_variable]
            )

        real_df = pd.concat((df[exog_columns], test_steps_df))

        data_entry = {
            FieldName.ITEM_ID: sku,
            FieldName.TARGET: df["sold_quantity"].values.tolist() + [-1] * test_steps,
            FieldName.START: str(df.index[0]),
            FieldName.FEAT_STATIC_CAT: categorical_values,
            FieldName.FEAT_DYNAMIC_REAL: real_df[exog_columns].values.T.tolist(),
        }

        save_json_gzip(
            data_entry, os.path.join(output_dir, "{}.json.gz".format(sku)),
        )
    except Exception:
        print(f"Error when processing sku {sku}")
        raise


class PrepareGluonTimeSeriesDatasets(luigi.Task):
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
        default=["currency_relative_price", "current_price", "minutes_active"]
    )
    test_steps: int = luigi.IntParameter(default=30)

    def input(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget]:
        assets_path = get_assets_path()
        return (
            luigi.LocalTarget(os.path.join(assets_path, "train_data.parquet")),
            luigi.LocalTarget(os.path.join(assets_path, "items_static_metadata_full.jl")),
        )

    def output(self):
        if "DATA_PATH" in os.environ:
            return luigi.LocalTarget(os.path.join(os.environ["DATA_PATH"], self.task_id))
        else:
            return luigi.LocalTarget(os.path.join("output", self.__class__.__name__, self.task_id))

    def run(self):
        os.makedirs(self.output().path)

        df = pd.read_parquet(self.input()[0].path)
        metadata_df = pd.read_json(self.input()[1].path, lines=True)

        df = pd.merge(df, metadata_df, how="left", on="sku")
        df["date"] = pd.to_datetime(df["date"])

        holidays_df = create_holidays_df(set(df["date"].dt.year))

        categorical_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        sku_df = df.drop_duplicates(subset=["sku"])
        categorical_values_per_sku = dict(
            zip(
                sku_df["sku"],
                categorical_encoder.fit_transform(
                    sku_df[list(self.categorical_variables)]
                ).astype(int)
                + 1,  # 0 = unknown
            )
        )

        with open(
            os.path.join(self.output().path, "categorical_encoder.pkl"), "wb"
        ) as f:
            pickle.dump(categorical_encoder, f)

        holidays_df.to_csv(os.path.join(self.output().path, "holidays.csv"))

        global_current_price_mean = df["current_price"].mean()
        global_current_price_std = df["current_price"].std()

        currency_current_price_mean: Dict[str, float] = df.groupby("currency")["current_price"].mean().to_dict()
        currency_current_price_std: Dict[str, float] = df.groupby("currency")["current_price"].std().to_dict()

        list(
            tqdm(
                map(
                    functools.partial(
                        _save_dataset_item,
                        categorical_values_per_sku=categorical_values_per_sku,
                        holidays_df=holidays_df,
                        real_variables=self.real_variables,
                        test_steps=self.test_steps,
                        global_current_price_mean=global_current_price_mean,
                        global_current_price_std=global_current_price_std,
                        currency_current_price_mean=currency_current_price_mean,
                        currency_current_price_std=currency_current_price_std,
                        output_dir=self.output().path,
                    ),
                    df.groupby("sku"),
                ),
                total=df["sku"].nunique(),
            )
        )
