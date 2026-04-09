import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

from src.frequency import infer_periods_per_year


def rolling_config(freq: str) -> Tuple[List[int], float]:
    """
    Return rolling windows and annualization factor by data frequency.
    """
    if freq == "daily":
        return [60, 120, 252], np.sqrt(252)

    if freq == "weekly":
        return [12, 26, 52], np.sqrt(52)

    if freq == "monthly":
        return [6, 12, 24], np.sqrt(12)

    periods_per_year = 252
    return [60, 120, 252], np.sqrt(periods_per_year)


def rolling_volatility(
    df: pd.DataFrame,
    window: int,
    periods_per_year: Optional[float] = None,
    ann_factor: Optional[float] = None,
) -> pd.Series:
    """
    Rolling annualized volatility.
    """
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(df)

    if ann_factor is None:
        ann_factor = np.sqrt(periods_per_year)

    return df["fund_return"].rolling(window).std() * ann_factor


def rolling_sharpe(
    df: pd.DataFrame,
    window: int,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[float] = None,
    ann_factor: Optional[float] = None,
) -> pd.Series:
    """
    Rolling annualized Sharpe ratio.

    risk_free_rate is annualized. The default is 2%.
    """
    if periods_per_year is None:
        periods_per_year = infer_periods_per_year(df)

    if ann_factor is None:
        ann_factor = np.sqrt(periods_per_year)

    rolling_mean = df["fund_return"].rolling(window).mean() * periods_per_year
    rolling_std = df["fund_return"].rolling(window).std() * ann_factor
    rolling_std = rolling_std.replace(0, np.nan)

    return (rolling_mean - risk_free_rate) / rolling_std


def add_rolling_metrics(
    df: pd.DataFrame,
    freq: str,
    risk_free_rate: float = 0.02,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Add rolling volatility and rolling Sharpe columns based on data frequency.
    """
    df = df.copy()
    windows, ann_factor = rolling_config(freq)
    periods_per_year = ann_factor ** 2

    for window in windows:
        df[f"rolling_vol_{window}"] = rolling_volatility(
            df,
            window=window,
            periods_per_year=periods_per_year,
            ann_factor=ann_factor,
        )
        df[f"rolling_sharpe_{window}"] = rolling_sharpe(
            df,
            window=window,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
            ann_factor=ann_factor,
        )

    return df, windows
