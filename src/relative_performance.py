from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def _to_aligned_numeric_series(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Align fund and benchmark return series on a shared index and drop missing values.

    The caller is expected to pass return series that refer to the same native
    frequency. This helper makes the alignment explicit and raises when no
    overlapping observations remain.
    """
    aligned = pd.concat(
        [
            pd.to_numeric(fund_returns, errors="coerce").rename("fund_return"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark_return"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.empty:
        raise ValueError("Fund and benchmark return series do not have overlapping non-null observations.")

    return aligned["fund_return"], aligned["benchmark_return"]


def calculate_period_excess_returns(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.Series:
    """
    Calculate period excess returns as fund return minus benchmark return.
    """
    aligned_fund, aligned_benchmark = _to_aligned_numeric_series(fund_returns, benchmark_returns)
    excess = aligned_fund - aligned_benchmark
    excess.name = "period_excess_return"
    return excess


def calculate_cumulative_returns(return_series: pd.Series) -> pd.Series:
    """
    Convert a periodic return series into a cumulative return series.
    """
    cumulative_returns = (1 + pd.to_numeric(return_series, errors="coerce")).dropna().cumprod() - 1
    cumulative_returns.name = "cumulative_return"
    return cumulative_returns


def calculate_cumulative_excess_returns(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate cumulative fund, cumulative benchmark, and cumulative excess returns.

    Cumulative excess return is defined as:
    cumulative fund return minus cumulative benchmark return

    It is intentionally not defined as the cumulative product of period excess
    returns.
    """
    aligned_fund, aligned_benchmark = _to_aligned_numeric_series(fund_returns, benchmark_returns)
    cumulative_fund = calculate_cumulative_returns(aligned_fund).rename("cumulative_fund_return")
    cumulative_benchmark = calculate_cumulative_returns(aligned_benchmark).rename("cumulative_benchmark_return")
    cumulative_excess = (cumulative_fund - cumulative_benchmark).rename("cumulative_excess_return")

    computed_diff = cumulative_fund - cumulative_benchmark
    if not np.allclose(computed_diff.values, cumulative_excess.values, atol=1e-10, equal_nan=True):
        raise AssertionError(
            "Cumulative excess return must equal cumulative fund return minus cumulative benchmark return. "
            f"Max absolute difference: {np.nanmax(np.abs(computed_diff.values - cumulative_excess.values))}"
        )

    return cumulative_fund, cumulative_benchmark, cumulative_excess


def calculate_annualized_return_from_series(
    return_series: pd.Series,
    periods_per_year: float,
) -> float:
    """
    Annualize a periodic return series using geometric compounding.
    """
    clean_returns = pd.to_numeric(return_series, errors="coerce").dropna()
    if clean_returns.empty or periods_per_year <= 0:
        return np.nan

    cumulative_growth = float((1 + clean_returns).prod())
    if cumulative_growth <= 0:
        return np.nan

    return float(cumulative_growth ** (periods_per_year / len(clean_returns)) - 1)


def calculate_annualized_excess_return(
    fund_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float,
) -> float:
    """
    Calculate annualized excess return as annualized fund return minus annualized benchmark return.
    """
    aligned_fund, aligned_benchmark = _to_aligned_numeric_series(fund_returns, benchmark_returns)
    annualized_fund_return = calculate_annualized_return_from_series(aligned_fund, periods_per_year)
    annualized_benchmark_return = calculate_annualized_return_from_series(aligned_benchmark, periods_per_year)

    if pd.isna(annualized_fund_return) or pd.isna(annualized_benchmark_return):
        return np.nan

    return float(annualized_fund_return - annualized_benchmark_return)


def calculate_hit_rate(period_excess_returns: pd.Series) -> float:
    """
    Calculate the share of periods with positive excess return.
    """
    clean_excess_returns = pd.to_numeric(period_excess_returns, errors="coerce").dropna()
    if clean_excess_returns.empty:
        return np.nan

    return float((clean_excess_returns > 0).mean())


def resample_to_period_returns(
    df: pd.DataFrame,
    level_col: str,
    freq: str,
    output_col: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Resample an aligned level series into monthly or annual period returns.

    The first period uses the first observed level inside the period as the
    starting point. Subsequent periods use the prior period-end level.
    """
    required_columns = {date_col, level_col}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    level_data = df[[date_col, level_col]].copy()
    level_data[date_col] = pd.to_datetime(level_data[date_col], errors="coerce")
    level_data[level_col] = pd.to_numeric(level_data[level_col], errors="coerce")
    level_data = level_data.dropna(subset=[date_col, level_col]).sort_values(date_col).reset_index(drop=True)

    if level_data.empty:
        label_col = "month" if freq == "M" else "year"
        return pd.DataFrame(columns=[label_col, output_col])

    label_col = "month" if freq == "M" else "year"
    level_data["period"] = level_data[date_col].dt.to_period(freq)
    period_end_level = level_data.groupby("period")[level_col].last()
    period_start_level = level_data.groupby("period")[level_col].first()
    period_returns = period_end_level.pct_change()

    if len(period_returns) > 0:
        first_period = period_returns.index[0]
        start_level = period_start_level.loc[first_period]
        end_level = period_end_level.loc[first_period]
        period_returns.iloc[0] = float(end_level / start_level - 1) if start_level > 0 and end_level > 0 else np.nan

    result = period_returns.rename(output_col).reset_index()
    if freq == "M":
        result[label_col] = result["period"].astype(str)
    else:
        result[label_col] = result["period"].dt.year.astype(int)

    return result[[label_col, output_col]]


def build_relative_period_table(
    df: pd.DataFrame,
    freq: str,
    fund_level_col: str,
    benchmark_level_col: str,
    fund_output_col: str,
    benchmark_output_col: str,
    excess_output_col: str,
) -> pd.DataFrame:
    """
    Build a fund / benchmark / excess return table at a requested calendar frequency.
    """
    label_col = "month" if freq == "M" else "year"
    fund_period_table = resample_to_period_returns(
        df=df,
        level_col=fund_level_col,
        freq=freq,
        output_col=fund_output_col,
    )
    benchmark_period_table = resample_to_period_returns(
        df=df,
        level_col=benchmark_level_col,
        freq=freq,
        output_col=benchmark_output_col,
    )
    relative_period_table = (
        fund_period_table.merge(benchmark_period_table, on=label_col, how="outer")
        .sort_values(label_col)
        .reset_index(drop=True)
    )
    relative_period_table[excess_output_col] = calculate_period_excess_returns(
        relative_period_table[fund_output_col],
        relative_period_table[benchmark_output_col],
    ).reindex(relative_period_table.index)
    return relative_period_table
