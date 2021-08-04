from datetime import date
from typing import Dict, Set, Type

import pandas as pd
from convertdate.holidays import thanksgiving, mothers_day, fathers_day
from dateutil.relativedelta import relativedelta as rd, SU
from holidays import Brazil, Argentina, Mexico, HolidayBase
from holidays import JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC
from sklearn.preprocessing import MultiLabelBinarizer


class BrazilWithCommercialDates(Brazil):
    def _populate(self, year: int):
        super()._populate(year)

        self[date(year, FEB, 14)] = "Valentine's Day"
        self[date(year, MAR, 8)] = "Dia Internacional da Mulher"
        self[date(year, MAR, 15)] = "Dia do Consumidor"
        self[date(*mothers_day(year))] = "Dia das Mães"
        self[date(year, JUN, 12)] = "Dia dos Namorados"
        self[date(year, AUG, 1) + rd(weekday=SU(2))] = "Dia dos Pais"
        self[date(year, SEP, 15)] = "Dia do Cliente"
        self[date(year, OCT, 12)] = "Dia das Crianças"
        thanksgiving_ = date(*thanksgiving(year))
        self[thanksgiving_ + rd(days=1)] = "Black Friday"
        self[thanksgiving_ + rd(days=4)] = "Cyber Monday"


class ArgentinaWithCommercialDates(Argentina):
    def _populate(self, year):
        super()._populate(year)

        self[date(year, FEB, 14)] = "Valentine's Day"
        self[date(year, MAR, 8)] = "Dia Internacional da Mulher"
        self[date(year, MAR, 15)] = "Dia do Consumidor"
        self[date(year, OCT, 1) + rd(weekday=SU(3))] = "Dia das Mães"
        self[date(*fathers_day(year))] = "Dia dos Pais"
        self[date(year, SEP, 11)] = "Dia do Cliente"
        self[date(year, AUG, 1) + rd(weekday=SU(2))] = "Dia das Crianças"
        thanksgiving_ = date(*thanksgiving(year))
        self[thanksgiving_ + rd(days=1)] = "Black Friday"
        self[thanksgiving_ + rd(days=4)] = "Cyber Monday"


class MexicoWithCommercialDates(Mexico):
    def _populate(self, year):
        super()._populate(year)

        self[date(year, FEB, 14)] = "Valentine's Day"
        self[date(year, MAR, 8)] = "Dia Internacional da Mulher"
        self[date(year, MAY, 10) + rd(weekday=SU(3))] = "Dia das Mães"
        self[date(*fathers_day(year))] = "Dia dos Pais"
        self[date(year, APR, 30)] = "Dia das Crianças"
        thanksgiving_ = date(*thanksgiving(year))
        self[thanksgiving_ + rd(days=1)] = "Black Friday"
        self[thanksgiving_ + rd(days=4)] = "Cyber Monday"


def _create_holidays_df(years: Set[int], holidays_class: Type[HolidayBase]):
    holidays_: Dict[date, str] = holidays_class(years=years)
    holidays_exploded = [
        (ds, tuple(holiday.split(", ")))
        for ds, holiday in holidays_.items()
    ]
    holidays_df = pd.DataFrame(holidays_exploded, columns=["date", "holiday"])
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])
    return holidays_df


def create_holidays_df(years: Set[int]) -> pd.DataFrame:
    brazil_holidays_df = _create_holidays_df(years, BrazilWithCommercialDates)
    brazil_holidays_df["site_id"] = "MLB"
    argentina_holidays_df = _create_holidays_df(years, ArgentinaWithCommercialDates)
    argentina_holidays_df["site_id"] = "MLA"
    mexico_holidays_df = _create_holidays_df(years, MexicoWithCommercialDates)
    mexico_holidays_df["site_id"] = "MLM"

    holidays_df = pd.concat(
        [brazil_holidays_df, argentina_holidays_df, mexico_holidays_df]
    )
    holidays_df = holidays_df.set_index(["date", "site_id"]).sort_index()

    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(holidays_df["holiday"].values)

    return pd.DataFrame(
        index=holidays_df.index, columns=mlb.classes_, data=one_hot_encoded
    )
