import abc
import functools
import json
import os
import pickle
import shutil
from glob import glob
from typing import List, Tuple, Dict
import warnings

import luigi
import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.sql import SparkSession, DataFrame, Window, WindowSpec
from pyspark.sql.functions import (
    col,
    lit,
    mean,
    stddev,
    coalesce,
    year,
    to_date,
    concat,
)
from pyspark.sql.types import StructType, StructField, DateType, ArrayType, IntegerType
import psutil
from gluonts.dataset.field_names import FieldName
from luigi.contrib.spark import PySparkTask
from sklearn.preprocessing import OrdinalEncoder, scale
from tqdm import tqdm

from ml_challenge.holidays import create_holidays_df
from ml_challenge.path import get_assets_path, get_extra_data_path
from ml_challenge.utils import save_json_gzip, save_params


class BasePySparkTask(PySparkTask, metaclass=abc.ABCMeta):
    def setup(self, conf: SparkConf):
        conf.set("spark.local.dir", os.path.join("output", "spark"))
        conf.set("spark.sql.warehouse.dir", os.path.join("output", "spark-warehouse"))
        conf.set("spark.driver.maxResultSize", f"{int(self._get_available_memory())}g")
        conf.set("spark.executor.memory", f"{int(self._get_available_memory())}g")
        conf.set(
            "spark.executor.memoryOverhead",
            f"{int(self._get_available_memory() * 0.25)}g",
        )

    @property
    def driver_memory(self):
        return f"{int(self._get_available_memory())}g"

    def _get_available_memory(self) -> str:
        return psutil.virtual_memory().available / (1024 * 1024 * 1024) * 0.85

    def main(self, sc: SparkContext, *args):
        self.spark = SparkSession(sc)
        self.run_with_spark()

    @abc.abstractmethod
    def run_with_spark(self):
        pass


def z_score(column: str, window: WindowSpec):
    return coalesce(
        (col(column) - mean(column).over(window)) / stddev(column).over(window),
        lit(0.0),
    )


class PreProcessRealVariables(BasePySparkTask):
    real_variables: List[str] = luigi.ListParameter(
        default=[
            "minutes_active",
            "current_price",
            "currency_relative_price",
            "usd_relative_price",
            "minimum_salary_relative_price",
        ]
    )

    def input(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
        assets_path = get_assets_path()
        extra_data_path = get_extra_data_path()
        return (
            luigi.LocalTarget(os.path.join(assets_path, "train_data.parquet")),
            luigi.LocalTarget(
                os.path.join(assets_path, "items_static_metadata_full.jl")
            ),
            luigi.LocalTarget(extra_data_path),
        )

    def output(self):
        dir_path = os.path.join("output", self.__class__.__name__, self.task_id)
        return (
            luigi.LocalTarget(dir_path),
            luigi.LocalTarget(os.path.join(dir_path, "data.parquet")),
        )

    def run_with_spark(self):
        os.makedirs(self.output()[0].path, exist_ok=True)

        save_params(self.output()[0].path, self.param_kwargs)

        df = self.spark.read.parquet(self.input()[0].path)
        df = df.withColumn("date", to_date("date"))

        metadata_df = self.spark.read.json(self.input()[1].path)

        df = df.join(metadata_df, how="left", on="sku").select("sku", "date", "site_id", "currency", "minutes_active", "current_price", "sold_quantity")

        df = df.withColumn("minutes_active", col("minutes_active") / 1440.0)

        if "global_relative_price" in self.real_variables:
            df = df.withColumn(
                "global_relative_price", z_score("current_price", Window.partitionBy()),
            )

        if "currency_relative_price" in self.real_variables:
            df = df.withColumn(
                "currency_relative_price",
                z_score("current_price", Window.partitionBy("currency")),
            )

        if (
            "usd_relative_price" in self.real_variables
            or "minimum_salary_relative_price" in self.real_variables
        ):
            usd_brl_df = (
                self.spark.read.csv(
                    os.path.join(self.input()[2].path, "USD_BRL.csv"),
                    inferSchema=True,
                    header=True,
                )
                .withColumn("date", to_date("Date", "MMM dd, yyyy"))
                .withColumn("currency", lit("REA"))
                .select("date", "currency", col("Price").alias("conversion_price"))
            )
            usd_ars_df = (
                self.spark.read.csv(
                    os.path.join(self.input()[2].path, "USD_BRL.csv"),
                    inferSchema=True,
                    header=True,
                )
                .withColumn("date", to_date("Date", "MMM dd, yyyy"))
                .withColumn("currency", lit("ARG"))
                .select("date", "currency", col("Price").alias("conversion_price"))
            )
            usd_mxn_df = (
                self.spark.read.csv(
                    os.path.join(self.input()[2].path, "USD_BRL.csv"),
                    inferSchema=True,
                    header=True,
                )
                .withColumn("date", to_date("Date", "MMM dd, yyyy"))
                .withColumn("currency", lit("MEX"))
                .select("date", "currency", col("Price").alias("conversion_price"))
            )
            usd_usd_df = usd_brl_df.withColumn("conversion_price", lit(1.0))
            conversion_df = (
                usd_brl_df.union(usd_ars_df).union(usd_mxn_df).union(usd_usd_df)
            ).cache()

            df = df.join(conversion_df, on=["date", "currency"], how="left").withColumn(
                "usd_relative_price", col("current_price") / col("conversion_price"),
            )

            if "minimum_salary_relative_price" in self.real_variables:
                minimum_salary_df = self.spark.read.csv(
                    os.path.join(self.input()[2].path, "MINIMUM_SALARY_2021.csv"),
                    inferSchema=True,
                    header=True,
                )
                usd_minimum_salary_df = (
                    conversion_df.join(minimum_salary_df, on="currency", how="inner")
                    .withColumn(
                        "usd_minimum_salary",
                        col("minimum_salary") / col("conversion_price"),
                    )
                    .select("date", "site_id", "usd_minimum_salary")
                )

                df = df.join(
                    usd_minimum_salary_df, on=["date", "site_id"], how="left"
                ).withColumn(
                    "minimum_salary_relative_price",
                    col("usd_relative_price") / col("usd_minimum_salary"),
                )

            df = df.withColumn(
                "usd_relative_price",
                z_score("usd_relative_price", Window.partitionBy()),
            )

        if "current_price" in self.real_variables:
            df = df.withColumn(
                "current_price", z_score("current_price", Window.partitionBy("sku")),
            )

        kept_features = set(
            ["date", "sku", "sold_quantity"]
            + list(self.real_variables)
        )
        df = df.select([column for column in df.columns if column in kept_features])

        df = df.withColumn("_sku", concat(col("sku"), lit(".parquet")))
        df.repartition("_sku").write.partitionBy("_sku").parquet(self.output()[1].path)


class PreProcessCategoricalVariables(BasePySparkTask):
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

    def input(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget]:
        assets_path = get_assets_path()
        return (
            luigi.LocalTarget(os.path.join(assets_path, "train_data.parquet")),
            luigi.LocalTarget(
                os.path.join(assets_path, "items_static_metadata_full.jl")
            ),
        )

    def output(self):
        dir_path = os.path.join("output", self.__class__.__name__, self.task_id)
        return (
            luigi.LocalTarget(dir_path),
            luigi.LocalTarget(os.path.join(dir_path, "data.parquet")),
            luigi.LocalTarget(os.path.join(dir_path, "labels.json")),
        )

    def run_with_spark(self):
        os.makedirs(self.output()[0].path, exist_ok=True)

        save_params(self.output()[0].path, self.param_kwargs)

        df = self.spark.read.parquet(self.input()[0].path)
        df = df.drop("date").dropDuplicates(["sku"])

        metadata_df = self.spark.read.json(self.input()[1].path)

        df = df.join(metadata_df, how="left", on="sku")

        index_pipeline = Pipeline(
            stages=[
                StringIndexer(
                    inputCol=variable,
                    outputCol=f"{variable}_index",
                    handleInvalid="keep",
                )
                for variable in self.categorical_variables
            ]
        )

        index_model = index_pipeline.fit(df)

        labels = {
            x.getInputCol(): x.labels
            for x in index_model.stages
            if isinstance(x, StringIndexerModel)
        }
        with open(self.output()[2].path, "w") as f:
            json.dump(labels, f)

        df = index_model.transform(df)
        kept_features = set(
            ["sku"]
            + list(self.categorical_variables)
            + [f"{variable}_index" for variable in self.categorical_variables]
        )
        df = df.select([column for column in df.columns if column in kept_features])

        df.write.parquet(self.output()[1].path)


# class PreProcessDataset(BasePySparkTask):
#     categorical_variables: List[str] = luigi.ListParameter(
#         default=[
#             "site_id",
#             "currency",
#             "listing_type",
#             "shipping_logistic_type",
#             "shipping_payment",
#             "item_domain_id",
#             "item_id",
#             "sku",
#         ]
#     )
#     real_variables: List[str] = luigi.ListParameter(
#         default=[
#             "minutes_active",
#             "current_price",
#             "currency_relative_price",
#             "usd_relative_price",
#             "minimum_salary_relative_price",
#         ]
#     )
#
#     def input(self) -> Tuple[luigi.LocalTarget, luigi.LocalTarget, luigi.LocalTarget]:
#         assets_path = get_assets_path()
#         extra_data_path = get_extra_data_path()
#         return (
#             luigi.LocalTarget(os.path.join(assets_path, "train_data.parquet")),
#             luigi.LocalTarget(
#                 os.path.join(assets_path, "items_static_metadata_full.jl")
#             ),
#             luigi.LocalTarget(extra_data_path),
#         )
#
#     def output(self):
#         dir_path = os.path.join("output", self.__class__.__name__, self.task_id)
#         return (
#             luigi.LocalTarget(dir_path),
#             luigi.LocalTarget(os.path.join(dir_path, "data.parquet")),
#             luigi.LocalTarget(os.path.join(dir_path, "labels.json")),
#         )
#
#     def run_with_spark(self):
#         os.makedirs(self.output()[0].path, exist_ok=True)
#
#         save_params(self.output()[0].path, self.param_kwargs)
#
#         df = self.spark.read.parquet(self.input()[0].path)
#         df = df.withColumn("date", to_date("date"))
#
#         metadata_df = self.spark.read.json(self.input()[1].path)
#
#         df = df.join(metadata_df, how="left", on="sku")
#
#         index_pipeline = Pipeline(
#             stages=[
#                 StringIndexer(
#                     inputCol=variable,
#                     outputCol=f"{variable}_index",
#                     handleInvalid="keep",
#                 )
#                 for variable in self.categorical_variables
#             ]
#         )
#
#         index_model = index_pipeline.fit(df)
#
#         labels = {
#             x.getInputCol(): x.labels
#             for x in index_model.stages
#             if isinstance(x, StringIndexerModel)
#         }
#         with open(self.output()[2].path, "w") as f:
#             json.dump(labels, f)
#
#         df = index_model.transform(df)
#
#         df = df.withColumn("minutes_active", col("minutes_active") / 1440.0)
#
#         if "global_relative_price" in self.real_variables:
#             df = df.withColumn(
#                 "global_relative_price", z_score("current_price", Window.partitionBy()),
#             )
#
#         if "currency_relative_price" in self.real_variables:
#             df = df.withColumn(
#                 "currency_relative_price",
#                 z_score("current_price", Window.partitionBy("currency")),
#             )
#
#         if (
#             "usd_relative_price" in self.real_variables
#             or "minimum_salary_relative_price" in self.real_variables
#         ):
#             usd_brl_df = (
#                 self.spark.read.csv(
#                     os.path.join(self.input()[2].path, "USD_BRL.csv"),
#                     inferSchema=True,
#                     header=True,
#                 )
#                 .withColumn("date", to_date("Date", "MMM dd, yyyy"))
#                 .withColumn("currency", lit("REA"))
#                 .select("date", "currency", col("Price").alias("conversion_price"))
#             )
#             usd_ars_df = (
#                 self.spark.read.csv(
#                     os.path.join(self.input()[2].path, "USD_BRL.csv"),
#                     inferSchema=True,
#                     header=True,
#                 )
#                 .withColumn("date", to_date("Date", "MMM dd, yyyy"))
#                 .withColumn("currency", lit("ARG"))
#                 .select("date", "currency", col("Price").alias("conversion_price"))
#             )
#             usd_mxn_df = (
#                 self.spark.read.csv(
#                     os.path.join(self.input()[2].path, "USD_BRL.csv"),
#                     inferSchema=True,
#                     header=True,
#                 )
#                 .withColumn("date", to_date("Date", "MMM dd, yyyy"))
#                 .withColumn("currency", lit("MEX"))
#                 .select("date", "currency", col("Price").alias("conversion_price"))
#             )
#             usd_usd_df = usd_brl_df.withColumn("conversion_price", lit(1.0))
#             conversion_df = (
#                 usd_brl_df.union(usd_ars_df).union(usd_mxn_df).union(usd_usd_df)
#             ).cache()
#
#             df = df.join(conversion_df, on=["date", "currency"], how="left").withColumn(
#                 "usd_relative_price", col("current_price") / col("conversion_price"),
#             )
#
#             if "minimum_salary_relative_price" in self.real_variables:
#                 minimum_salary_df = self.spark.read.csv(
#                     os.path.join(self.input()[2].path, "MINIMUM_SALARY_2021.csv"),
#                     inferSchema=True,
#                     header=True,
#                 )
#                 usd_minimum_salary_df = (
#                     conversion_df.join(minimum_salary_df, on="currency", how="inner")
#                     .withColumn(
#                         "usd_minimum_salary",
#                         col("minimum_salary") / col("conversion_price"),
#                     )
#                     .select("date", "site_id", "usd_minimum_salary")
#                 )
#
#                 df = df.join(
#                     usd_minimum_salary_df, on=["date", "site_id"], how="left"
#                 ).withColumn(
#                     "minimum_salary_relative_price",
#                     col("usd_relative_price") / col("usd_minimum_salary"),
#                 )
#
#             df = df.withColumn(
#                 "usd_relative_price",
#                 z_score("usd_relative_price", Window.partitionBy()),
#             )
#
#         if "current_price" in self.real_variables:
#             df = df.withColumn(
#                 "current_price", z_score("current_price", Window.partitionBy("sku")),
#             )
#
#         kept_features = set(
#             ["date", "sku", "sold_quantity", "site_id"]
#             + list(self.categorical_variables)
#             + list(self.real_variables)
#         )
#         df = df.select([column for column in df.columns if column in kept_features])
#
#         df.write.parquet(self.output()[1].path)


# class PartitionDataset(BasePySparkTask):
#     categorical_variables: List[str] = luigi.ListParameter(
#         default=[
#             "site_id",
#             "currency",
#             "listing_type",
#             "shipping_logistic_type",
#             "shipping_payment",
#             "item_domain_id",
#             "item_id",
#             "sku",
#         ]
#     )
#     real_variables: List[str] = luigi.ListParameter(
#         default=[
#             "minutes_active",
#             "current_price",
#             "currency_relative_price",
#             "usd_relative_price",
#             "minimum_salary_relative_price",
#         ]
#     )
#
#     def requires(self):
#         return PreProcessDataset(
#             categorical_variables=self.categorical_variables,
#             real_variables=self.real_variables,
#         )
#
#     def output(self):
#         dir_path = os.path.join("output", self.__class__.__name__, self.task_id)
#         return (
#             luigi.LocalTarget(dir_path),
#             luigi.LocalTarget(os.path.join(dir_path, "data.parquet")),
#             luigi.LocalTarget(os.path.join(dir_path, "labels.json")),
#         )
#
#     def run_with_spark(self):
#         os.makedirs(self.output()[0].path, exist_ok=True)
#
#         save_params(self.output()[0].path, self.param_kwargs)
#
#         shutil.copy(
#             self.input()[2].path, self.output()[2].path, follow_symlinks=True,
#         )
#
#         df = self.spark.read.parquet(self.input()[1].path)
#         df = df.withColumn("_sku", concat(col("sku"), lit(".parquet")))
#         df.repartition("_sku").write.partitionBy("_sku").parquet(self.output()[1].path)


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
        default=[
            "minutes_active",
            "current_price",
            "currency_relative_price",
            "usd_relative_price",
            "minimum_salary_relative_price",
        ]
    )
    test_steps: int = luigi.IntParameter(default=30)

    def requires(self):
        return PreProcessRealVariables(
            real_variables=self.real_variables,
        ), PreProcessCategoricalVariables(
            categorical_variables=self.categorical_variables,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join("output", self.__class__.__name__, self.task_id)
        )

    def run(self):
        os.makedirs(self.output().path)

        filepaths = glob(os.path.join(self.input()[0][1].path, "*.parquet"))

        shutil.copy(
            self.input()[1][2].path,
            os.path.join(self.output().path, "labels.json"),
            follow_symlinks=True,
        )
        
        holidays_df = create_holidays_df({2021})
        holidays_df.to_csv(os.path.join(self.output().path, "holidays.csv"))

        categorical_variables_df = pd.read_parquet(self.input()[1][1].path)

        list(
            tqdm(
                map(
                    functools.partial(
                        _save_dataset_item,
                        categorical_variables_df=categorical_variables_df,
                        categorical_variables=self.categorical_variables,
                        real_variables=self.real_variables,
                        holidays_df=holidays_df,
                        test_steps=self.test_steps,
                        output_dir=self.output().path,
                    ),
                    filepaths,
                ),
                total=len(filepaths),
            )
        )


def _default_real_variable(exog_column: str, last_value: float) -> float:
    return {"minutes_active": 1.0}.get(exog_column, last_value)


def _save_dataset_item(
    filepath: str,
    categorical_variables_df: pd.DataFrame,
    categorical_variables: List[str],
    real_variables: List[str],
    holidays_df: pd.DataFrame,
    test_steps: int,
    output_dir: str,
):
    df = pd.read_parquet(filepath)

    sku = df.iloc[0]["sku"]
    sku_cat_row = categorical_variables_df[categorical_variables_df["sku"] == sku].iloc[0]

    try:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        categorical_values = [
            int(sku_cat_row[f"{variable}_index"]) for variable in categorical_variables
        ]

        exog_columns = list(real_variables) + list(holidays_df.columns)

        test_steps_df = pd.DataFrame(
            index=pd.date_range(df.index[-1] + pd.DateOffset(1), periods=test_steps)
        )
        df["site_id"] = sku_cat_row["site_id"]
        test_steps_df["site_id"] = sku_cat_row["site_id"]
        df = pd.merge(
            df,
            holidays_df,
            how="left",
            left_on=[df.index, df["site_id"]],
            right_index=True,
        ).fillna(0)
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
            FieldName.ITEM_ID: int(sku),
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
