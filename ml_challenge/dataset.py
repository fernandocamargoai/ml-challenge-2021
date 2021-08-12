import functools
import os
import warnings
from multiprocessing import Pool
from typing import Optional, Iterator, List, Dict

import numpy as np
from gluonts.dataset.common import DataEntry, Dataset, ProcessDataEntry, SourceContext
from gluonts.dataset.field_names import FieldName
from gluonts.transform import TransformedDataset, MapTransformation
from tqdm import tqdm

from ml_challenge.utils import load_json_gzip

PRICE_FIELD_NAME = "price"


class JsonGzFile(object):
    """
    A type that draws data from a JSON file.

    Parameters
    ----------
    path
        Path of the file to load data from. This should be a valid
        JSON file.
    """

    def __init__(
        self,
        path: str,
        process: ProcessDataEntry,
        cache: bool = False,
        preload: bool = False,
        check_data: bool = False,
    ) -> None:
        self.path = path
        self.process = process
        self.cache = cache or preload
        self.check_data = check_data
        self._data_cache: Optional[dict] = None
        if preload:
            self.get_data()

    def _check_data(self, data: DataEntry):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if not np.isfinite(value).all():
                    raise ValueError(
                        f"The key {key} of the source {data['source']} contains an invalid value"
                    )
                if key == FieldName.FEAT_DYNAMIC_REAL:
                    if value.max() > 1.0:
                        warnings.warn(
                            f"The key {key} of the source {data['source']} contains a value above 1.0"
                        )
                    if value.min() < 0.0:
                        warnings.warn(
                            f"The key {key} of the source {data['source']} contains a value bellow 0.0"
                        )

    def get_data(self) -> dict:
        if self._data_cache is not None:
            return self._data_cache
        else:
            data = self.process(load_json_gzip(self.path, DataEntry))
            data["source"] = SourceContext(source=self.path, row=0)
            if self.check_data:
                self._check_data(data)
            if self.cache:
                self._data_cache = data
            return data


class JsonGzDataset(Dataset):
    """
    Dataset that loads JSON files contained in a path.

    Parameters
    ----------
    files
        Paths for the dataset files. Each file should end with .json.gz.
        A file can be for
        instance: {"start": "2014-09-07", "target": [0.1, 0.2]}.
    freq
        Frequency of the observation in the time series.
        Must be a valid Pandas frequency.
    one_dim_target
        Whether to accept only univariate target time series.
    cache
        Indicates whether the dataset should be cached or not.
    """

    def __init__(
        self,
        files: List[str],
        freq: str,
        one_dim_target: bool = True,
        cache: bool = False,
        preload: bool = False,
        check_data: bool = False,
    ) -> None:
        self.cache = cache or preload
        self.check_data = check_data

        process = ProcessDataEntry(freq, one_dim_target=one_dim_target)
        if preload:
            with Pool(os.cpu_count()) as pool:
                print("Preloading dataset...")
                self._json_files = list(
                    tqdm(
                        pool.map(
                            functools.partial(
                                JsonGzFile,
                                process=process,
                                cache=cache,
                                preload=preload,
                                check_data=check_data,
                            ),
                            files,
                        ),
                        total=len(files),
                    )
                )
        else:
            self._json_files = [JsonGzFile(path, process, cache) for path in files]
        self._len = len(self._json_files)

    def __iter__(self) -> Iterator[DataEntry]:
        for json_file in self._json_files:
            yield json_file.get_data()

    def __len__(self):
        return self._len


class SameSizeTransformedDataset(TransformedDataset):
    def __len__(self):
        return len(self.base_dataset)


class TruncateTargetTransformation(MapTransformation):
    def __init__(self, prediction_length: int) -> None:
        super().__init__()
        self._prediction_length = prediction_length

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = data.copy()
        target = data[FieldName.TARGET]
        assert (
            target.shape[-1] >= self._prediction_length
        )  # handles multivariate case (target_dim, history_length)
        data[FieldName.TARGET] = target[..., : -self._prediction_length]
        return data


class FilterTimeSeriesTransformation(MapTransformation):
    def __init__(self, start: int, end: int) -> None:
        super().__init__()
        self._start = start
        self._end = end or None

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        new_data = data.copy()
        for field in (
            FieldName.TARGET,
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_DYNAMIC_REAL,
        ):
            if data.get(field) is not None:
                new_data[field] = np.array(data[field])[..., self._start : self._end]
        if data.get(PRICE_FIELD_NAME) is not None:
            new_data[PRICE_FIELD_NAME] = data[PRICE_FIELD_NAME][self._start : self._end]
        return new_data


class ChangeTargetToMinutesActiveTransformation(MapTransformation):
    def __init__(self, minutes_active_index: int) -> None:
        super().__init__()
        self._minutes_active_index = minutes_active_index

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = data.copy()
        data[FieldName.TARGET] = data[FieldName.FEAT_DYNAMIC_REAL][
            self._minutes_active_index
        ]
        data[FieldName.FEAT_DYNAMIC_REAL] = np.delete(
            data[FieldName.FEAT_DYNAMIC_REAL], self._minutes_active_index, axis=0
        )
        return data


class UseMinutesActiveForecastingTransformation(MapTransformation):
    def __init__(
        self,
        minutes_active_index: int,
        minutes_active_forecasts_dict: Dict[int, np.ndarray],
    ) -> None:
        super().__init__()
        self._minutes_active_index = minutes_active_index
        self._minutes_active_forecasts_dict = minutes_active_forecasts_dict

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = data.copy()
        minutes_active_forecast = self._minutes_active_forecasts_dict[
            data[FieldName.ITEM_ID]
        ]

        data[FieldName.FEAT_DYNAMIC_REAL][self._minutes_active_index][
            -len(minutes_active_forecast) :
        ] = np.clip(minutes_active_forecast, 0.0, 1.0)
        return data


class UseMeanOfLastMinutesActiveTransformation(MapTransformation):
    def __init__(self, minutes_active_index: int, test_steps: int,) -> None:
        super().__init__()
        self._minutes_active_index = minutes_active_index
        self._test_steps = test_steps

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = data.copy()

        data[FieldName.FEAT_DYNAMIC_REAL][self._minutes_active_index][
            -self._test_steps :
        ] = data[FieldName.FEAT_DYNAMIC_REAL][self._minutes_active_index][
            -(self._test_steps * 2) : -self._test_steps
        ].mean()
        return data
