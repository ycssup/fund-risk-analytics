import pandas as pd
from typing import Tuple

TRADING_DAYS = 252
WEEKS_PER_YEAR = 52
MONTHS_PER_YEAR = 12
CALENDAR_DAYS = 365.25


def infer_data_frequency(df: pd.DataFrame, strict: bool = False) -> Tuple[str, float]:
    """
    Infer NAV data frequency from the date column.

    Returns:
    - frequency label: daily, weekly, monthly, or custom
    - annualization factor: observations per year
    """
    if "date" not in df.columns:
        if strict:
            raise ValueError("Cannot infer data frequency: missing date column.")

        return "daily", TRADING_DAYS

    dates = pd.to_datetime(df["date"], errors="coerce").dropna().sort_values()
    day_diffs = dates.diff().dt.days.dropna()
    day_diffs = day_diffs[day_diffs > 0]

    if len(day_diffs) == 0:
        if strict:
            raise ValueError("Cannot infer data frequency: need at least two valid dates.")

        return "daily", TRADING_DAYS

    median_days = day_diffs.median()

    if median_days <= 3:
        return "daily", TRADING_DAYS

    if median_days <= 10:
        return "weekly", WEEKS_PER_YEAR

    if 25 <= median_days <= 35:
        return "monthly", MONTHS_PER_YEAR

    return "custom", CALENDAR_DAYS / median_days


def infer_periods_per_year(df: pd.DataFrame) -> float:
    """
    Infer the annualization factor from the date column.
    """
    return infer_data_frequency(df)[1]
