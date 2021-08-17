import math

import numpy as np
import pandas as pd
from gluonts.time_feature import (
    DayOfWeekIndex,
    DayOfMonthIndex,
    DayOfYearIndex,
    WeekOfYearIndex,
)


class DayOfWeekSin(DayOfWeekIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_week = super().__call__(index)
        return np.sin(day_of_week * (2.0 * math.pi / 6))


class DayOfWeekCos(DayOfWeekIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_week = super().__call__(index)
        return np.cos(day_of_week * (2.0 * math.pi / 6))


class DayOfMonthSin(DayOfMonthIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_month = super().__call__(index)
        return np.sin(day_of_month * (2.0 * math.pi / (index.days_in_month - 1)))


class DayOfMonthCos(DayOfMonthIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_month = super().__call__(index)
        return np.cos(day_of_month * (2.0 * math.pi / (index.days_in_month - 1)))


class DayOfYearSin(DayOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_year = super().__call__(index)
        return np.sin(day_of_year * (2.0 * math.pi / 365))


class DayOfYearCos(DayOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        day_of_year = super().__call__(index)
        return np.cos(day_of_year * (2.0 * math.pi / 365))


class WeekOfYearSin(WeekOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        week_of_year = super().__call__(index)
        return np.sin(week_of_year * (2.0 * math.pi / 52))


class WeekOfYearCos(WeekOfYearIndex):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        week_of_year = super().__call__(index)
        return np.cos(week_of_year * (2.0 * math.pi / 52))
