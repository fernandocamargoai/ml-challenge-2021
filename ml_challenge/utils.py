import gzip
import json
import os
from typing import cast, TypeVar, Any


def save_params(dir_path: str, params: dict):
    with open(os.path.join(dir_path, "params.json"), "w") as params_file:
        json.dump(params, params_file, default=lambda o: dict(o), indent=4)


T = TypeVar("T")


def load_json_gzip(filepath: str, expected_type: T) -> T:
    with gzip.open(filepath, "rt", encoding="utf-8") as zipfile:
        return cast(expected_type, json.load(zipfile))


def save_json_gzip(data: Any, filepath: str, compress_level: int = 9):
    with gzip.open(
        filepath, "wt", encoding="utf-8", compresslevel=compress_level
    ) as zipfile:
        json.dump(data, zipfile)


def get_sku_from_data_entry_path(path: str) -> int:
    return int(os.path.split(path)[1].replace(".json.gz", ""))
